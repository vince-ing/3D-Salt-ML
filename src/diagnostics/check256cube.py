"""
3D PyVista viewer — single 256³ cube from mississippi256/train
==============================================================
Picks one random cube and shows a 2x2 panel:
  [0,0]  Seismic — full cube (outer edges/faces)
  [0,1]  Salt label — solid blue mesh
  [1,0]  Seismic — three orthogonal slices (inline, crossline, time middle planes)
  [1,1]  Seismic + label overlay (transparent salt on middle slices)

Keys: R = new random cube, S = screenshot, Q = quit

Memory fixes vs original:
  - RemoveAllViewProps() on every sub-renderer before each redraw — this is
    the correct VTK way to release actors/mappers without destroying the plotter.
  - extract_geometry() instead of extract_surface() so point scalars (amplitude)
    are preserved on the cube faces — fixes the flat-gray rendering bug.
  - All intermediate numpy and VTK objects explicitly deleted after rendering.
  - gc.collect() called after each cube load.
  - Single long-lived Plotter reused (same as original), so key events and the
    window all work exactly as before.
"""

import gc
import glob
import os
import random
from pathlib import Path

import numpy as np
import pyvista as pv

pv.global_theme.background = "black"
pv.global_theme.font.color = "white"

# ── Config ────────────────────────────────────────────────────
TRAIN_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\data\processed\mississippi256\train"
)
SCREENSHOT_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\outputs\screenshots"
)
WINDOW_SIZE = (1600, 900)
SMOOTH_ITER = 40
# ─────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.npz")))
if not files:
    raise FileNotFoundError(f"No .npz files found in:\n  {TRAIN_DIR}")
print(f"Found {len(files)} cubes in train dir")


def load_cube(fpath: str):
    with np.load(fpath) as d:
        seis  = d["seismic"].astype(np.float32)
        label = d["label"].astype(np.float32)
    return seis, label


def make_grid(arr: np.ndarray) -> pv.ImageData:
    """Wrap (D, H, W) numpy array into PyVista ImageData with cell data."""
    grid = pv.ImageData()
    grid.dimensions = np.array(arr.shape) + 1
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["values"] = arr.flatten(order="F")
    return grid


def iso_surface(binary_vol: np.ndarray) -> pv.PolyData:
    padded = np.pad(
        binary_vol.astype(np.float32),
        pad_width=1,
        mode="constant",
        constant_values=0,
    )
    grid = make_grid(padded)
    grid.origin = (-1.0, -1.0, -1.0)
    surf = grid.cell_data_to_point_data().contour([0.5], scalars="values")
    del grid, padded
    if surf.n_points > 0 and SMOOTH_ITER > 0:
        surf = surf.smooth(n_iter=SMOOTH_ITER, relaxation_factor=0.1)
    return surf


def render(pl: pv.Plotter, fpath: str) -> str:
    """
    Clear all renderers properly, load one cube, populate the 2×2 panel,
    then free every heavyweight object. Returns the cube stem name.
    """
    # ── Release all actors/mappers from the previous cube ─
    # RemoveAllViewProps() is the correct VTK call — it drops the renderer's
    # reference counts immediately, unlike pl.clear() which only works per
    # active subplot and leaves ghost references in other renderers.
    for renderer in pl.renderers:
        renderer.RemoveAllViewProps()

    seis, label = load_cube(fpath)
    stem        = Path(fpath).stem
    salt_ratio  = float(label.mean())

    vmin = float(np.percentile(seis, 2))
    vmax = float(np.percentile(seis, 98))

    print(f"\n{stem}")
    print(
        f"  shape={seis.shape}  salt={salt_ratio:.1%}  "
        f"range=[{seis.min():.3f}, {seis.max():.3f}]"
    )

    # ── Build grids ────────────────────────────────────────
    seis_grid = make_grid(seis).cell_data_to_point_data()
    outline   = seis_grid.outline()

    # extract_geometry() preserves the point scalar array that
    # cell_data_to_point_data() just computed, so amplitude colours render
    # correctly on the cube faces. extract_surface() on an ImageData discards
    # point data and was the cause of the flat-gray appearance.
    seis_outer = seis_grid.extract_geometry()

    sl_inline    = seis_grid.slice(normal="x", origin=seis_grid.center)
    sl_crossline = seis_grid.slice(normal="y", origin=seis_grid.center)
    sl_time      = seis_grid.slice(normal="z", origin=seis_grid.center)

    salt_surf = iso_surface(label.astype(np.uint8))

    # Raw numpy arrays no longer needed past this point
    del seis, label

    # ── Shared kwargs ──────────────────────────────────────
    seis_kw = dict(
        scalars="values",
        cmap="gray_r",
        clim=[vmin, vmax],
        show_scalar_bar=False,
    )
    CAM = "iso"

    # ── [0,0]  Seismic Cube ────────────────────────────────
    pl.subplot(0, 0)
    pl.add_mesh(
        seis_outer,
        scalars="values",
        cmap="gray_r",
        clim=[vmin, vmax],
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Amplitude",
            "vertical": True,
            "color": "white",
            "fmt": "%.2f",
        },
    )
    pl.add_mesh(outline, color="white", line_width=1.5)
    pl.add_text(stem,           font_size=8,  color="yellow", position="upper_left")
    pl.add_text("Seismic Cube", font_size=10, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # ── [0,1]  Salt label ──────────────────────────────────
    pl.subplot(0, 1)
    pl.add_mesh(outline, color="white", line_width=1.5)
    if salt_surf.n_points > 0:
        pl.add_mesh(salt_surf, color="#1e90ff", opacity=1.0, smooth_shading=True)
    pl.add_text(stem,                              font_size=8,  color="yellow", position="upper_left")
    pl.add_text(f"Salt Label  ({salt_ratio:.1%})", font_size=10, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # ── [1,0]  Middle slices ───────────────────────────────
    pl.subplot(1, 0)
    pl.add_mesh(sl_inline,    **seis_kw)
    pl.add_mesh(sl_crossline, **seis_kw)
    pl.add_mesh(sl_time,      **seis_kw)
    pl.add_mesh(outline, color="white", line_width=1.5)
    pl.add_text(stem,            font_size=8,  color="yellow", position="upper_left")
    pl.add_text("Middle Slices", font_size=10, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    # ── [1,1]  Slices + salt overlay ──────────────────────
    pl.subplot(1, 1)
    pl.add_mesh(sl_inline,    **seis_kw)
    pl.add_mesh(sl_crossline, **seis_kw)
    pl.add_mesh(sl_time,      **seis_kw)
    if salt_surf.n_points > 0:
        pl.add_mesh(salt_surf, color="#ff4400", opacity=0.35, smooth_shading=True)
    pl.add_mesh(outline, color="white", line_width=1.5)
    pl.add_text(stem,            font_size=8,  color="yellow", position="upper_left")
    pl.add_text("Slices + Salt", font_size=10, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    pl.render()

    # ── Release all heavyweight VTK + numpy objects ────────
    del seis_grid, seis_outer, sl_inline, sl_crossline, sl_time, salt_surf, outline
    gc.collect()

    return stem


# ── Main ──────────────────────────────────────────────────────
app = {
    "idx":  random.randint(0, len(files) - 1),
    "stem": "",
}

pl = pv.Plotter(
    shape=(2, 2),
    border=True,
    border_color="white",
    window_size=WINDOW_SIZE,
    title="256³ Cube Viewer  |  R=random  S=screenshot  Q=quit",
)


def refresh():
    app["stem"] = render(pl, files[app["idx"]])


def new_random():
    app["idx"] = random.randint(0, len(files) - 1)
    refresh()


def save_screenshot():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    out = os.path.join(SCREENSHOT_DIR, f"cube_{app['stem'][:50]}.png")
    pl.screenshot(out)
    print(f"Screenshot → {out}")


pl.add_key_event("r", new_random)
pl.add_key_event("s", save_screenshot)

refresh()

print("\n  R   new random cube")
print("  S   screenshot")
print("  Q / close window   quit\n")

pl.show(auto_close=False)
pl.close()
print("Done.")