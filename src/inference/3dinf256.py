"""
3D Salt Detection Inference -- 256^3 Cube, 128^3 Center Crop Display
=====================================================================
Picks a random cube from the mississippi256/test directory that satisfies
the MIN/MAX salt ratio filter. The ratio is read directly from the filename
(no pre-scanning needed):

    mississippi_test_05316_01792_0249_0.187.npz
                                         ^^^^^  <- this value is used

Renders a 2x2 PyVista panel:
  [0,0]  Seismic outer faces  -- bounded to the 128^3 center region
  [0,1]  Ground-truth salt    -- solid blue mesh
  [1,0]  Model prediction     -- solid orange mesh
  [1,1]  Difference           -- TP (green), FP (red), FN (blue)

Keys: S = screenshot | Q = quit

Notes:
  - Only the center 128^3 crop of the 256^3 cube is shown/inferred on.
  - Inference: fine = center 128^3 crop, context = full 256^3 -> zoom 128^3.
  - All meshes share the same (0,0,0) -> (128,128,128) coordinate space.
"""

import gc
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")
import pyvista as pv

pv.global_theme.background = "black"
pv.global_theme.font.color = "white"


# ============================================================
# CONFIGURATION -- edit paths and filter values here
# ============================================================
MODEL_PATH = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\outputs\experiments\multiscale_256_2026-02-27_2007\best_model.pth"
)

TEST_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\data\processed\mississippi256\test"
)

SCREENSHOT_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\outputs\screenshots"
)

# Salt ratio filter -- compared against the value encoded in the filename.
# Set MIN=0.0 and MAX=1.0 to accept all cubes.
MIN_SALT_RATIO = 0.25
MAX_SALT_RATIO = 0.90

CUBE_SIZE   = 256   # full stored cube size
FINE_SIZE   = 128   # center crop size for inference and display
HALF        = FINE_SIZE // 2   # 64
THRESHOLD   = 0.75  # sigmoid probability threshold for binary prediction
SMOOTH_ITER = 40    # Laplacian smoothing iterations on iso-surfaces
WINDOW_SIZE = (1600, 900)

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
    def forward(self, x):
        return self.conv(x)


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
        self.enc1  = ConvBlock(1, 16, p);  self.pool1 = nn.MaxPool3d(2)
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
        self.dec2 = ConvBlock(64, 32, p)
        self.up1  = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16, p)
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
# HELPERS
# ============================================================
def parse_salt_ratio_from_filename(fpath: str) -> float:
    """
    Extract the salt ratio from the last underscore-separated field of the stem.
    e.g.  mississippi_test_05316_01792_0249_0.187.npz  ->  0.187
    Returns -1.0 on parse failure so the file is always excluded from filtering.
    """
    try:
        return float(Path(fpath).stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1.0


def pick_random_file(files: list) -> str:
    """
    Pick one random file whose filename salt ratio is within
    [MIN_SALT_RATIO, MAX_SALT_RATIO].  Exits cleanly if nothing qualifies.
    """
    candidates = [
        f for f in files
        if MIN_SALT_RATIO <= parse_salt_ratio_from_filename(f) <= MAX_SALT_RATIO
    ]
    if not candidates:
        sys.exit(
            f"No cubes found with salt ratio in "
            f"[{MIN_SALT_RATIO:.3f}, {MAX_SALT_RATIO:.3f}].\n"
            f"Adjust MIN_SALT_RATIO / MAX_SALT_RATIO in the config section."
        )
    chosen = random.choice(candidates)
    ratio  = parse_salt_ratio_from_filename(chosen)
    print(f"  Selected : {Path(chosen).name}")
    print(f"  Filename salt ratio : {ratio:.3f}")
    print(f"  Filter              : [{MIN_SALT_RATIO:.3f} - {MAX_SALT_RATIO:.3f}]  "
          f"({len(candidates)} matching cubes)")
    return chosen


def load_model():
    print(f"\nDevice : {DEVICE}")
    print(f"Loading model from:\n  {MODEL_PATH}")
    model = SaltModel3D_MultiScale(dropout_rate=0.2).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Model ready  ({n:,} parameters)\n")
    return model


def extract_center_crop(cube256: np.ndarray) -> tuple:
    """Return the center 128^3 crop and the slice bounds (z0,z1,y0,y1,x0,x1)."""
    c = CUBE_SIZE // 2   # 128
    z0, z1 = c - HALF, c + HALF   # 64:192
    y0, y1 = c - HALF, c + HALF
    x0, x1 = c - HALF, c + HALF
    return cube256[z0:z1, y0:y1, x0:x1].copy(), (z0, z1, y0, y1, x0, x1)


def run_inference(model, cube256: np.ndarray):
    """
    fine    = center 128^3 crop  (full-res local detail)
    context = full 256^3 -> zoom 0.5 -> 128^3  (half-res wide context)

    Returns:
      prob     (128,128,128) float32  sigmoid probability map
      pred_bin (128,128,128) uint8    binary mask at THRESHOLD
    """
    fine, _ = extract_center_crop(cube256)
    context = zoom(cube256, zoom=0.5, order=0).astype(np.float32)

    fine_t = torch.from_numpy(fine.copy()).float()[None, None].to(DEVICE)
    ctx_t  = torch.from_numpy(context).float()[None, None].to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(fine_t, ctx_t))[0, 0].cpu().numpy()

    pred_bin = (prob > THRESHOLD).astype(np.uint8)
    return prob, pred_bin


def make_seismic_grid(fine: np.ndarray) -> pv.ImageData:
    """Wrap 128^3 seismic array into a PyVista ImageData at origin (0,0,0)."""
    grid = pv.ImageData()
    grid.dimensions = np.array(fine.shape) + 1
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["values"] = fine.flatten(order="F")
    return grid.cell_data_to_point_data()


def iso_surface(binary_vol: np.ndarray) -> pv.PolyData:
    """
    Marching-cubes surface from a binary volume.
    A 1-voxel zero border closes any open faces at cube edges.
    Origin is shifted by -1 on each axis to keep the mesh aligned
    to the (0,0,0)-(128,128,128) seismic coordinate space.
    """
    padded = np.pad(binary_vol.astype(np.float32), pad_width=1,
                    mode="constant", constant_values=0)
    grid = pv.ImageData()
    grid.dimensions = np.array(padded.shape) + 1
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (-1.0, -1.0, -1.0)
    grid.cell_data["values"] = padded.flatten(order="F")
    grid = grid.cell_data_to_point_data()
    surf = grid.contour([0.5], scalars="values")
    if surf.n_points > 0 and SMOOTH_ITER > 0:
        surf = surf.smooth(n_iter=SMOOTH_ITER, relaxation_factor=0.1)
    return surf


def compute_metrics(pred_prob: np.ndarray, label: np.ndarray, thr: float = 0.5):
    pb    = (pred_prob > thr).astype(np.float32)
    tb    = label.astype(np.float32)
    inter = (pb * tb).sum()
    union = pb.sum() + tb.sum() - inter
    iou   = float(inter / union) if union else 1.0
    denom = pb.sum() + tb.sum()
    dice  = float(2 * inter / denom) if denom else 1.0
    return iou, dice


# ============================================================
# RENDER
# ============================================================
def render(pl: pv.Plotter, fpath: str, model) -> str:
    """Load cube, run inference, populate the 2x2 panel."""

    stem       = Path(fpath).stem
    file_ratio = parse_salt_ratio_from_filename(fpath)

    print(f"\n{'='*60}")
    print(f"Cube              : {stem}")
    print(f"Filename ratio    : {file_ratio:.3f}")

    # -- Load 256^3 arrays ----------------------------------------
    with np.load(fpath) as d:
        cube256  = d["seismic"].astype(np.float32)
        label256 = d["label"].astype(np.float32)

    # Center 128^3 crops for display and ground-truth label
    fine,  (z0, z1, y0, y1, x0, x1) = extract_center_crop(cube256)
    label_fine = label256[z0:z1, y0:y1, x0:x1].copy()
    salt_center = float(label_fine.mean())
    print(f"Center crop salt  : {salt_center:.1%}")

    # -- Inference ------------------------------------------------
    prob, pred_bin = run_inference(model, cube256)
    iou, dice = compute_metrics(prob, label_fine, THRESHOLD)
    print(f"IoU={iou:.3f}   Dice={dice:.3f}   threshold={THRESHOLD}")

    del cube256, label256
    gc.collect()

    # -- PyVista meshes -------------------------------------------
    seis_grid  = make_seismic_grid(fine)
    seis_outer = seis_grid.extract_geometry()
    outline    = seis_grid.outline()

    truth_surf = iso_surface(label_fine.astype(np.uint8))
    pred_surf  = iso_surface(pred_bin)

    tp = ((pred_bin == 1) & (label_fine == 1)).astype(np.uint8)
    fp = ((pred_bin == 1) & (label_fine == 0)).astype(np.uint8)
    fn = ((pred_bin == 0) & (label_fine == 1)).astype(np.uint8)
    tp_surf = iso_surface(tp)
    fp_surf = iso_surface(fp)
    fn_surf = iso_surface(fn)

    del fine, label_fine, prob, pred_bin, tp, fp, fn
    gc.collect()

    vmin = float(np.percentile(seis_outer.point_data["values"], 2))
    vmax = float(np.percentile(seis_outer.point_data["values"], 98))
    CAM  = "iso"

    # -- [0,0]  Seismic outer faces (128^3 crop) ------------------
    pl.subplot(0, 0)
    pl.add_mesh(
        seis_outer,
        scalars="values", cmap="gray_r", clim=[vmin, vmax],
        show_scalar_bar=True,
        scalar_bar_args={"title": "Amplitude", "vertical": True,
                         "color": "white", "fmt": "%.2f"},
    )
    pl.add_mesh(outline, color="cyan", line_width=2.0)
    pl.add_text(stem,                   font_size=7, color="yellow", position="upper_left")
    pl.add_text("Seismic (128^3 crop)", font_size=9, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # -- [0,1]  Ground Truth --------------------------------------
    pl.subplot(0, 1)
    pl.add_mesh(outline, color="cyan", line_width=2.0)
    if truth_surf.n_points > 0:
        pl.add_mesh(truth_surf, color="#1e90ff", opacity=1.0, smooth_shading=True)
    else:
        pl.add_text("(no salt in crop)", font_size=10, color="gray", position="lower_right")
    pl.add_text(stem, font_size=7, color="yellow", position="upper_left")
    pl.add_text(f"Ground Truth  ({salt_center:.1%})",
                font_size=9, color="white", position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # -- [1,0]  Model Prediction ----------------------------------
    pl.subplot(1, 0)
    pl.add_mesh(outline, color="cyan", line_width=2.0)
    if pred_surf.n_points > 0:
        pl.add_mesh(pred_surf, color="#ff6600", opacity=1.0, smooth_shading=True)
    else:
        pl.add_text("(nothing predicted)", font_size=10, color="gray", position="lower_right")
    pl.add_text(stem, font_size=7, color="yellow", position="upper_left")
    pl.add_text(f"Prediction (thr={THRESHOLD})   IoU={iou:.3f}  Dice={dice:.3f}",
                font_size=9, color="white", position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # -- [1,1]  Difference (TP / FP / FN) ------------------------
    pl.subplot(1, 1)
    pl.add_mesh(outline, color="cyan", line_width=2.0)
    if tp_surf.n_points > 0:
        pl.add_mesh(tp_surf, color="#00cc44", opacity=0.85, smooth_shading=True)
    if fp_surf.n_points > 0:
        pl.add_mesh(fp_surf, color="#ff2222", opacity=0.85, smooth_shading=True)
    if fn_surf.n_points > 0:
        pl.add_mesh(fn_surf, color="#2299ff", opacity=0.85, smooth_shading=True)
    pl.add_legend(
        labels=[
            ("True Positive",  "#00cc44"),
            ("False Positive", "#ff2222"),
            ("False Negative", "#2299ff"),
        ],
        bcolor=(0.05, 0.05, 0.05),
        border=True,
        size=(0.38, 0.16),
    )
    pl.add_text(stem,               font_size=7, color="yellow", position="upper_left")
    pl.add_text("Prediction Error", font_size=9, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    pl.render()

    del seis_grid, seis_outer, outline, truth_surf, pred_surf, tp_surf, fp_surf, fn_surf
    gc.collect()

    return stem


# ============================================================
# MAIN
# ============================================================
def main():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No .npz files found in:\n  {TEST_DIR}\n"
            "Check that TEST_DIR is correct."
        )
    print(f"Found {len(files)} cubes in test directory.")

    fpath = pick_random_file(files)
    model = load_model()

    pl = pv.Plotter(
        shape=(2, 2),
        border=True,
        border_color="white",
        window_size=WINDOW_SIZE,
        title="Salt Detection Inference  |  S=screenshot  Q=quit",
    )

    stem = render(pl, fpath, model)

    def save_screenshot(_=None):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        out = os.path.join(SCREENSHOT_DIR, f"inference_{stem[:50]}.png")
        pl.screenshot(out)
        print(f"Screenshot saved -> {out}")

    pl.add_key_event("s", save_screenshot)

    print("\n  S           -- save screenshot")
    print("  Q / close   -- quit\n")

    pl.show(auto_close=False)
    pl.close()
    print("Done.")


if __name__ == "__main__":
    main()