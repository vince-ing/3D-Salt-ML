"""
3D Salt Detection Inference — Mississippi 256³ Test Set
=========================================================
Loads the best model, runs inference on the center 128³ crop of each 256³
test cube, and renders a 2×2 PyVista panel showing:

  [0,0]  Seismic (128³ center crop, gray volume rendering)
  [0,1]  Ground Truth salt body  (blue mesh)
  [1,0]  Predicted salt body     (orange mesh)  + IoU / Dice
  [1,1]  Error map  TP=green · FP=red · FN=blue

Navigation:
    →  /  ←       next / previous cube
    R             pick a new random cube (within salt-ratio filter)
    S             save screenshot to disk
    Q             quit

Dependencies:
    pip install torch numpy scipy pyvista
"""

import os
import sys
import glob
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom, label as ndlabel
import pyvista as pv

os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")
pv.global_theme.background = "black"
pv.global_theme.font.color = "white"


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\outputs\experiments\multiscale_256_20260227_1346\best_model.pth"
)

TEST_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\data\processed\mississippi256\test"
)

SCREENSHOT_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\outputs\screenshots"
)

# ── Salt-ratio filter ────────────────────────────────────────
MIN_SALT_RATIO = 0.05    # skip nearly-empty cubes
MAX_SALT_RATIO = 0.90    # skip fully-saturated cubes

# ── Inference settings ───────────────────────────────────────
THRESHOLD  = 0.50        # probability cut-off for binary prediction
CUBE_SIZE  = 256         # stored patch size
FINE_SIZE  = 128         # center crop fed to the model
HALF       = FINE_SIZE // 2

# ── Visualisation settings ───────────────────────────────────
SMOOTH_ITER = 40         # Laplacian smoothing iterations on iso-surfaces
WINDOW_SIZE = (1700, 1020)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL  (must exactly match training architecture)
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
        self.b0 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv3d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(), nn.Dropout3d(0.3))
    def forward(self, x):
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


class SaltModel3D_MultiScale(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        p = dropout_rate
        self.enc1  = ConvBlock(1,  16, p); self.pool1 = nn.MaxPool3d(2)
        self.enc2  = ConvBlock(16, 32, p); self.pool2 = nn.MaxPool3d(2)
        self.enc3  = ConvBlock(32, 64, p); self.pool3 = nn.MaxPool3d(2)
        self.ctx_enc = nn.Sequential(
            ConvBlock(1, 16, p), nn.MaxPool3d(2),
            ConvBlock(16, 32, p), nn.MaxPool3d(2),
            ConvBlock(32, 64, p), nn.MaxPool3d(2),
        )
        self.aspp = ASPP(128, 64)
        self.up3  = nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64, p)
        self.up2  = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64,  32, p)
        self.up1  = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32,  16, p)
        self.final = nn.Conv3d(16, 1, 1)

    def forward(self, x_fine, x_ctx):
        x1     = self.enc1(x_fine)
        x2     = self.enc2(self.pool1(x1))
        x3     = self.enc3(self.pool2(x2))
        fine_f = self.pool3(x3)
        ctx_f  = self.ctx_enc(x_ctx)
        b      = self.aspp(torch.cat([fine_f, ctx_f], dim=1))
        d3     = self.dec3(torch.cat([self.up3(b),  x3], dim=1))
        d2     = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1     = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        return self.final(d1)


# ============================================================
# INFERENCE HELPERS
# ============================================================
def center_crop(cube256: np.ndarray):
    """Return the dead-center 128³ crop and the matching label crop."""
    c = CUBE_SIZE // 2  # 128
    return cube256[c - HALF : c + HALF,
                   c - HALF : c + HALF,
                   c - HALF : c + HALF]


def make_context(cube256: np.ndarray) -> np.ndarray:
    """Zoom the full 256³ block to 128³ (order=0 for speed)."""
    return zoom(cube256, zoom=0.5, order=0).astype(np.float32)


@torch.no_grad()
def run_inference(model, cube256: np.ndarray):
    """
    Returns:
        fine    (128,128,128) float32  — center seismic crop
        prob    (128,128,128) float32  — sigmoid probability
        pred    (128,128,128) uint8    — binary prediction
    """
    fine    = center_crop(cube256).astype(np.float32)
    context = make_context(cube256)

    fine_t = torch.from_numpy(fine[np.newaxis, np.newaxis]).to(DEVICE)
    ctx_t  = torch.from_numpy(context[np.newaxis, np.newaxis]).to(DEVICE)

    prob = torch.sigmoid(model(fine_t, ctx_t))[0, 0].cpu().numpy()
    pred = (prob > THRESHOLD).astype(np.uint8)
    return fine, prob, pred


# ============================================================
# PYVISTA HELPERS
# ============================================================
def to_grid(arr: np.ndarray, origin=(0., 0., 0.)) -> pv.ImageData:
    """Wrap a (D,H,W) numpy array into a PyVista ImageData (cell data)."""
    grid = pv.ImageData()
    grid.dimensions = np.array(arr.shape) + 1
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = origin
    grid.cell_data["values"] = arr.flatten(order="F")
    return grid


def iso_surface(binary_vol: np.ndarray, smooth_iter: int = SMOOTH_ITER,
                origin=(0., 0., 0.)) -> pv.PolyData:
    """
    Marching-cubes surface from a binary volume.
    Pads with a 1-voxel zero border so boundary-touching bodies close properly.
    """
    padded = np.pad(binary_vol.astype(np.float32), pad_width=1,
                    mode="constant", constant_values=0)
    # shift origin back by 1 voxel to compensate for padding
    shifted = tuple(o - 1.0 for o in origin)
    grid = to_grid(padded, origin=shifted).cell_data_to_point_data()
    surf = grid.contour([0.5], scalars="values")
    if surf.n_points > 0 and smooth_iter > 0:
        surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.1)
    return surf


def compute_metrics(prob: np.ndarray, label: np.ndarray, thr: float = THRESHOLD):
    pb  = (prob > thr).astype(np.float32)
    tb  = label.astype(np.float32)
    inter = (pb * tb).sum()
    union = pb.sum() + tb.sum() - inter
    iou  = float(inter / union) if union else 1.0
    denom = pb.sum() + tb.sum()
    dice = float(2 * inter / denom) if denom else 1.0
    return iou, dice


# ============================================================
# CUBE LIST MANAGEMENT
# ============================================================
def build_cube_list(test_dir: str,
                    min_ratio: float, max_ratio: float,
                    seed: int = 42) -> list:
    """
    Scan all .npz files in test_dir, filter by salt ratio,
    return a shuffled list of (filepath, salt_ratio) tuples.
    """
    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    if not files:
        sys.exit(f"❌  No .npz files found in:\n    {test_dir}")

    print(f"📂  Scanning {len(files)} test cubes …", end="", flush=True)
    candidates = []
    for fp in files:
        try:
            with np.load(fp) as d:
                ratio = float(d["label"].mean())
            if min_ratio <= ratio <= max_ratio:
                candidates.append((fp, ratio))
        except Exception as e:
            print(f"\n  ⚠ Skipping {Path(fp).name}: {e}")

    print(f"  {len(candidates)} pass the filter ({min_ratio:.0%}–{max_ratio:.0%})")
    if not candidates:
        sys.exit("❌  No cubes passed the salt-ratio filter — widen MIN/MAX_SALT_RATIO.")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates


# ============================================================
# RENDER ONE CUBE INTO AN EXISTING PLOTTER
# ============================================================
def render_cube(pl: pv.Plotter, model,
                fpath: str, salt_ratio: float,
                cube_index: int, total_cubes: int):
    """
    Load one cube, run inference, and fill all four subplots.
    Clears existing actors first so the plotter can be reused.
    """
    # ── load & infer ──
    with np.load(fpath) as d:
        cube256 = d["seismic"].astype(np.float32)
        label256 = d["label"].astype(np.float32)

    fine, prob, pred = run_inference(model, cube256)
    label_crop = center_crop(label256).astype(np.uint8)

    iou, dice = compute_metrics(prob, label_crop)

    gt_ratio  = float(label_crop.mean())
    pred_ratio = float(pred.mean())
    name = Path(fpath).name

    print(f"\n[{cube_index+1}/{total_cubes}]  {name}")
    print(f"  GT salt: {gt_ratio:.1%}   Pred salt: {pred_ratio:.1%}   "
          f"IoU: {iou:.3f}   Dice: {dice:.3f}")

    # ── build pyvista objects ──
    seis_grid  = to_grid(fine).cell_data_to_point_data()
    truth_surf = iso_surface(label_crop)
    pred_surf  = iso_surface(pred)

    tp = (pred == 1) & (label_crop == 1)
    fp = (pred == 1) & (label_crop == 0)
    fn = (pred == 0) & (label_crop == 1)
    tp_surf = iso_surface(tp.astype(np.uint8))
    fp_surf = iso_surface(fp.astype(np.uint8))
    fn_surf = iso_surface(fn.astype(np.uint8))

    # ── clear previous actors ──
    for i in range(2):
        for j in range(2):
            pl.subplot(i, j)
            pl.clear()

    CAM = "iso"

    # ── [0,0]  Seismic ─────────────────────────────────────
    pl.subplot(0, 0)
    pl.add_mesh(
        seis_grid,
        scalars="values",
        cmap="gray_r",
        clim=[-3, 3],
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Amplitude", "vertical": True,
            "color": "white", "fmt": "%.1f",
        },
    )
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    pl.add_text(
        f"Seismic  [{cube_index+1}/{total_cubes}]\n{name}",
        font_size=10, color="white", position="upper_left",
    )
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # ── [0,1]  Ground Truth ────────────────────────────────
    pl.subplot(0, 1)
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    if truth_surf.n_points > 0:
        pl.add_mesh(truth_surf, color="#1e90ff", opacity=1.0,
                    smooth_shading=True)
    pl.add_text(
        f"Ground Truth  (GT salt {gt_ratio:.1%})",
        font_size=10, color="white", position="upper_left",
    )
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # ── [1,0]  Prediction ──────────────────────────────────
    pl.subplot(1, 0)
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    if pred_surf.n_points > 0:
        pl.add_mesh(pred_surf, color="#ff6600", opacity=1.0,
                    smooth_shading=True)
    pl.add_text(
        f"Prediction (p>{THRESHOLD})   IoU={iou:.3f}   Dice={dice:.3f}",
        font_size=10, color="white", position="upper_left",
    )
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # ── [1,1]  Error map ───────────────────────────────────
    pl.subplot(1, 1)
    pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)
    if tp_surf.n_points > 0:
        pl.add_mesh(tp_surf, color="#00cc44", opacity=0.85,
                    smooth_shading=True)
    if fp_surf.n_points > 0:
        pl.add_mesh(fp_surf, color="#ff2222", opacity=0.85,
                    smooth_shading=True)
    if fn_surf.n_points > 0:
        pl.add_mesh(fn_surf, color="#2299ff", opacity=0.85,
                    smooth_shading=True)
    pl.add_legend(
        labels=[
            ("True Positive",  "#00cc44"),
            ("False Positive", "#ff2222"),
            ("False Negative", "#2299ff"),
        ],
        bcolor=(0.05, 0.05, 0.05),
        border=True,
        size=(0.40, 0.18),
    )
    pl.add_text(
        "Error Map",
        font_size=10, color="white", position="upper_left",
    )
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    pl.render()
    return name   # returned so screenshot can use cube name


# ============================================================
# MAIN
# ============================================================
def run():
    # ── load model ──────────────────────────────────────────
    print(f"\n🔧  Device : {DEVICE}")
    print(f"📦  Loading model …")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"❌  Model not found:\n    {MODEL_PATH}")

    model = SaltModel3D_MultiScale(dropout_rate=0.2).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print("✅  Model ready")

    # ── build filtered cube list ─────────────────────────────
    cubes = build_cube_list(TEST_DIR, MIN_SALT_RATIO, MAX_SALT_RATIO)
    total = len(cubes)

    # ── state shared with key callbacks ──────────────────────
    state = {"idx": 0, "current_name": ""}

    # ── create plotter ───────────────────────────────────────
    pl = pv.Plotter(
        shape=(2, 2),
        border=True,
        border_color="white",
        window_size=WINDOW_SIZE,
        title="Salt Detection — 3D Inference  |  ← → navigate · R random · S save · Q quit",
    )

    def refresh():
        fp, ratio = cubes[state["idx"]]
        state["current_name"] = render_cube(
            pl, model, fp, ratio, state["idx"], total)

    # ── key callbacks ────────────────────────────────────────
    def next_cube():
        state["idx"] = (state["idx"] + 1) % total
        refresh()

    def prev_cube():
        state["idx"] = (state["idx"] - 1) % total
        refresh()

    def random_cube():
        state["idx"] = random.randint(0, total - 1)
        refresh()

    def save_screenshot():
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        stem = Path(state["current_name"]).stem
        out  = os.path.join(SCREENSHOT_DIR, f"infer_{stem}.png")
        pl.screenshot(out, transparent_background=False)
        print(f"📸  Screenshot saved → {out}")

    pl.add_key_event("Right",  next_cube)
    pl.add_key_event("Left",   prev_cube)
    pl.add_key_event("r",      random_cube)
    pl.add_key_event("s",      save_screenshot)

    # ── render first cube & open window ─────────────────────
    refresh()

    print("\n" + "─" * 55)
    print("  →  /  ←     next / previous cube")
    print("  R           jump to a random cube")
    print("  S           save screenshot")
    print("  Q           quit")
    print("─" * 55 + "\n")

    pl.show(auto_close=False)
    pl.close()
    print("\n👋  Done.")


if __name__ == "__main__":
    run()