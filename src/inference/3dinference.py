"""
3D Salt Detection Inference — PyVista Visualization
=====================================================
Runs inference on a single .npz cube and renders the results interactively
in 3D using PyVista, showing:
  • Seismic volume (clipped transparent volume rendering)
  • Ground-truth salt body (solid mesh)
  • Predicted salt body (semi-transparent mesh)
  • Side-by-side orthogonal slice panels

Dependencies:
    pip install torch numpy scipy pyvista tqdm
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import zoom, binary_erosion, label as ndlabel

# ── Optional: suppress PyVista GUI warnings on headless servers ──────────────
os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")
import pyvista as pv
pv.global_theme.background = "black"
pv.global_theme.font.color = "white"


# ============================================================
# CONFIGURATION — edit these paths before running
# ============================================================
MODEL_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\experiments\multi_dataset_multiscale_run_02\best_model.pth"

VAL_DIRS = [
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\keathley128unfiltered\test",
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi128unfiltered\test",
]

MIN_SALT_RATIO = 0.10   # only pick cubes with ≥10 % salt
MAX_SALT_RATIO = 0.80
THRESHOLD      = 0.70   # decision threshold for predicted probability

SAVE_SCREENSHOT = True  # also save a PNG screenshot
SCREENSHOT_PATH = "salt_inference_3d.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL ARCHITECTURE  (must match training weights exactly)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), nn.Dropout3d(p),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), nn.Dropout3d(p),
        )
    def forward(self, x): return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False),                              nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),       nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False),       nn.BatchNorm3d(out_ch), nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv3d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(), nn.Dropout3d(0.3),
        )
    def forward(self, x):
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


class SaltModel3D_MultiScale(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        p = dropout_rate
        self.enc1 = ConvBlock(1, 16, p);  self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(16, 32, p); self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(32, 64, p); self.pool3 = nn.MaxPool3d(2)
        self.ctx_enc = nn.Sequential(
            ConvBlock(1, 16, p), nn.MaxPool3d(2),
            ConvBlock(16, 32, p), nn.MaxPool3d(2),
            ConvBlock(32, 64, p), nn.MaxPool3d(2),
        )
        self.aspp = ASPP(128, 64)
        self.up3  = nn.ConvTranspose3d(64, 64, 2, stride=2); self.dec3 = ConvBlock(128, 64, p)
        self.up2  = nn.ConvTranspose3d(64, 32, 2, stride=2); self.dec2 = ConvBlock(64,  32, p)
        self.up1  = nn.ConvTranspose3d(32, 16, 2, stride=2); self.dec1 = ConvBlock(32,  16, p)
        self.final = nn.Conv3d(16, 1, 1)

    def forward(self, x_fine, x_ctx):
        x1 = self.enc1(x_fine)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        fine_f = self.pool3(x3)
        ctx_f  = self.ctx_enc(x_ctx)
        b  = self.aspp(torch.cat([fine_f, ctx_f], dim=1))
        d3 = self.dec3(torch.cat([self.up3(b),  x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        return self.final(d1)


# ============================================================
# HELPERS
# ============================================================
def make_context_patch(cube: np.ndarray) -> np.ndarray:
    padded  = np.pad(cube, pad_width=64, mode='reflect')   # 128 → 256
    context = zoom(padded, zoom=0.5, order=1)              # 256 → 128
    return context.astype(np.float32)


def volume_to_pyvista(arr: np.ndarray) -> pv.ImageData:
    """Wrap a (D,H,W) numpy array into a PyVista ImageData grid."""
    grid = pv.ImageData()
    grid.dimensions = np.array(arr.shape) + 1          # cell-centred → point dims
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["values"] = arr.flatten(order="F")  # Fortran (col-major) order
    return grid


def largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest CC in a binary volume."""
    labeled, n = ndlabel(binary)
    if n == 0:
        return binary
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    biggest = np.argmax(sizes) + 1
    return (labeled == biggest).astype(binary.dtype)


def iso_surface(binary_vol: np.ndarray, smooth_iter: int = 30) -> pv.PolyData:
    """Marching-cubes surface from a binary volume, with Laplacian smoothing.

    Pads with a 1-voxel border of zeros before contouring so that salt bodies
    touching any edge or corner of the cube are properly closed — without this,
    marching cubes leaves those boundary faces open.
    """
    # 1-voxel zero border forces a closed surface everywhere
    padded = np.pad(binary_vol.astype(np.float32), pad_width=1,
                    mode="constant", constant_values=0)
    grid = volume_to_pyvista(padded)
    grid = grid.cell_data_to_point_data()
    surf = grid.contour([0.5], scalars="values")
    if surf.n_points > 0 and smooth_iter > 0:
        surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.1)
    return surf


def compute_metrics(pred_prob, label, thr=0.5):
    pb  = (pred_prob > thr).astype(np.float32)
    tb  = label.astype(np.float32)
    inter = (pb * tb).sum();  union = pb.sum() + tb.sum() - inter
    iou  = (inter / union) if union else 1.0
    denom = pb.sum() + tb.sum()
    dice = (2 * inter / denom) if denom else 1.0
    return float(iou), float(dice)


# ============================================================
# MAIN
# ============================================================
def run():
    # ── 1. Load model ──────────────────────────────────────
    print(f"\n🔧  Device : {DEVICE}")
    print(f"📦  Loading model …")
    model = SaltModel3D_MultiScale(dropout_rate=0.2).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print("✅  Model ready\n")

    # ── 2. Pick one cube ───────────────────────────────────
    all_files = []
    for vd in VAL_DIRS:
        all_files.extend(glob.glob(os.path.join(vd, "*.npz")))
    if not all_files:
        sys.exit("❌  No .npz files found — check VAL_DIRS.")
    random.shuffle(all_files)

    seismic = label_vol = None
    for fpath in all_files:
        with np.load(fpath) as d:
            s, l = d["seismic"], d["label"]
        ratio = float(l.mean())
        if MIN_SALT_RATIO <= ratio <= MAX_SALT_RATIO:
            seismic, label_vol = s, l
            print(f"📂  Using cube : {Path(fpath).name}  (GT salt {ratio:.1%})")
            break
    if seismic is None:
        sys.exit("❌  No cube passed the salt-ratio filter.")

    # ── 3. Inference ───────────────────────────────────────
    context = make_context_patch(seismic)
    fine_t  = torch.from_numpy(seismic.copy()).float()[None, None].to(DEVICE)
    ctx_t   = torch.from_numpy(context).float()[None, None].to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(fine_t, ctx_t))[0, 0].cpu().numpy()

    pred_bin = (prob > THRESHOLD).astype(np.uint8)
    iou, dice = compute_metrics(prob, label_vol, THRESHOLD)
    print(f"📊  IoU={iou:.3f}   Dice={dice:.3f}\n")

    # ── 4. Build PyVista objects ───────────────────────────
    # Seismic volume (point data for slicing)
    seis_grid = volume_to_pyvista(seismic).cell_data_to_point_data()

    # Ground-truth surface (blue)
    truth_surf = iso_surface(label_vol.astype(np.uint8), smooth_iter=50)

    # Predicted surface (red/orange)
    pred_surf  = iso_surface(pred_bin, smooth_iter=50)

    # Probability volume (point data for slicing/contouring)
    prob_grid  = volume_to_pyvista(prob).cell_data_to_point_data()

    # ── 5. Build difference volume ────────────────────────────
    # TP=green, FP=red, FN=blue  (encoded as 1,2,3; 0=correct background)
    tp = (pred_bin == 1) & (label_vol == 1)
    fp = (pred_bin == 1) & (label_vol == 0)
    fn = (pred_bin == 0) & (label_vol == 1)
    diff = np.zeros_like(pred_bin, dtype=np.uint8)
    diff[tp] = 1   # True Positive
    diff[fp] = 2   # False Positive
    diff[fn] = 3   # False Negative

    tp_surf = iso_surface(tp.astype(np.uint8), smooth_iter=30)
    fp_surf = iso_surface(fp.astype(np.uint8), smooth_iter=30)
    fn_surf = iso_surface(fn.astype(np.uint8), smooth_iter=30)

    # ── 6. PyVista plotter — 2×2 grid of 3-D cubes ────────
    #   [0,0] Seismic (gray volume)   [0,1] Ground Truth (salt body)
    #   [1,0] Prediction (salt body)  [1,1] Difference (TP/FP/FN)

    # Shared camera position so all four cubes look the same angle
    CAM_POS = "iso"

    pl = pv.Plotter(
        shape=(2, 2),
        border=True,
        border_color="white",
        window_size=(1600, 1000),
        title="Salt Detection — 3D Inference",
    )

    def add_outline(renderer):
        renderer.add_mesh(seis_grid.outline(), color="white", line_width=1.5)

    # ── [0,0] Seismic cube ─────────────────────────────────
    pl.subplot(0, 0)
    pl.add_mesh(
        seis_grid,
        scalars="values",
        cmap="gray_r",
        clim=[-3, 3],
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Amplitude", "vertical": True,
                         "color": "white", "fmt": "%.1f"},
    )
    add_outline(pl)
    pl.add_text("Seismic", font_size=12, color="white", position="upper_left")
    pl.camera_position = CAM_POS
    pl.add_axes(interactive=False)

    # ── [0,1] Ground Truth cube ────────────────────────────
    pl.subplot(0, 1)
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    if truth_surf.n_points > 0:
        pl.add_mesh(
            truth_surf,
            color="#1e90ff",   # blue
            opacity=1.0,
            smooth_shading=True,
        )
    pl.add_text("Ground Truth (Salt)", font_size=12, color="white", position="upper_left")
    pl.camera_position = CAM_POS
    pl.add_axes(interactive=False)

    # ── [1,0] Prediction cube ──────────────────────────────
    pl.subplot(1, 0)
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    if pred_surf.n_points > 0:
        pl.add_mesh(
            pred_surf,
            color="#ff6600",   # orange
            opacity=1.0,
            smooth_shading=True,
        )
    pl.add_text(
        f"Prediction (p>0.5)   IoU={iou:.3f}  Dice={dice:.3f}",
        font_size=11, color="white", position="upper_left",
    )
    pl.camera_position = CAM_POS
    pl.add_axes(interactive=False)

    # ── [1,1] Difference cube ──────────────────────────────
    pl.subplot(1, 1)
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    if tp_surf.n_points > 0:
        pl.add_mesh(tp_surf, color="#00cc44", opacity=0.80,
                    smooth_shading=True, label="True Positive")
    if fp_surf.n_points > 0:
        pl.add_mesh(fp_surf, color="#ff2222", opacity=0.80,
                    smooth_shading=True, label="False Positive")
    if fn_surf.n_points > 0:
        pl.add_mesh(fn_surf, color="#2299ff", opacity=0.80,
                    smooth_shading=True, label="False Negative (Missed)")
    pl.add_legend(
        labels=[
            ("True Positive",        "#00cc44"),
            ("False Positive",       "#ff2222"),
            ("False Negative",       "#2299ff"),
        ],
        bcolor=(0.05, 0.05, 0.05),
        border=True,
        size=(0.38, 0.16),
    )
    pl.add_text("Prediction Error", font_size=12, color="white", position="upper_left")
    pl.camera_position = CAM_POS
    pl.add_axes(interactive=False)

    # ── 6. Screenshot & show ───────────────────────────────
    # screenshot() requires the window to be rendered first;
    # use auto_close=False, grab the frame, then close manually.
    print("\n🎮  Interactive window open — use mouse to rotate/zoom.")
    print("    Press  Q  or close the window to exit.\n")
    pl.show(auto_close=False)

    if SAVE_SCREENSHOT:
        pl.screenshot(SCREENSHOT_PATH, transparent_background=False)
        print(f"📸  Screenshot saved → {SCREENSHOT_PATH}")

    pl.close()


if __name__ == "__main__":
    run()