# /code
"""
3D PyVista viewer — single 256³ multi-class cube
==============================================================
Picks one random cube of a specified category and shows a 2x2 panel:
  [0,0]  Seismic — full cube (outer edges/faces)
  [0,1]  Labels — 4-class multi-color rendering
  [1,0]  Seismic — three orthogonal slices
  [1,1]  Seismic + label overlay

Memory fixes vs original:
  - RemoveAllViewProps() used to correctly release actors.
  - extract_geometry() preserves point scalars (amplitude).
  - Explicit garbage collection after rendering.
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
# Choose which type of cube to view: 'salt', 'boundary', 'rock', or 'empty'
CUBE_TYPE = "salt" 

TRAIN_DIR = (
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection"
    r"\data\processed\mississippi256seafloor\train"
)
WINDOW_SIZE = (1600, 900)
SMOOTH_ITER = 40
# ─────────────────────────────────────────────────────────────

def load_cube(fpath: str):
    with np.load(fpath) as d:
        seis  = d["seismic"].astype(np.float32)
        label = d["label"].astype(np.float32)
        
    # Flip the depth axis so the seafloor renders at the top
    seis = np.flip(seis, axis=2)
    label = np.flip(label, axis=2)
    
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
    Clear all renderers properly, load one cube, populate the 2x2 panel,
    then free every heavyweight object.
    """
    for renderer in pl.renderers:
        renderer.RemoveAllViewProps()

    seis, label = load_cube(fpath)
    stem        = Path(fpath).stem
    
    # Calculate ratios for the display text
    r_rock  = float((label == 0).mean())
    r_salt  = float((label == 1).mean())
    r_water = float((label == 2).mean())
    r_blank = float((label == 3).mean())

    vmin = float(np.percentile(seis, 2))
    vmax = float(np.percentile(seis, 98))

    print(f"\nViewing: {stem}")
    print(f"  shape={seis.shape}  range=[{seis.min():.3f}, {seis.max():.3f}]")
    print(f"  Rock: {r_rock:.1%} | Salt: {r_salt:.1%} | Water: {r_water:.1%} | Blank: {r_blank:.1%}")

    # ── Build grids ────────────────────────────────────────
    seis_grid = make_grid(seis).cell_data_to_point_data()
    outline   = seis_grid.outline()
    seis_outer = seis_grid.extract_geometry()

    sl_inline    = seis_grid.slice(normal="x", origin=seis_grid.center)
    sl_crossline = seis_grid.slice(normal="y", origin=seis_grid.center)
    sl_time      = seis_grid.slice(normal="z", origin=seis_grid.center)

    # Extract independent surfaces for all 4 classes
    rock_surf  = iso_surface((label == 0).astype(np.uint8))
    salt_surf  = iso_surface((label == 1).astype(np.uint8))
    water_surf = iso_surface((label == 2).astype(np.uint8))
    blank_surf = iso_surface((label == 3).astype(np.uint8))

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

    # ── [0,1]  Multi-Class Labels ──────────────────────────
    pl.subplot(0, 1)
    pl.add_mesh(outline, color="white", line_width=1.5)
    
    # Render meshes from back to front (roughly) for better alpha blending
    if blank_surf.n_points > 0:
        pl.add_mesh(blank_surf, color="#ff00ff", opacity=0.15, smooth_shading=True) # Magenta
    if rock_surf.n_points > 0:
        pl.add_mesh(rock_surf, color="#8b7355", opacity=0.1, smooth_shading=True)   # Muted Sand/Brown
    if water_surf.n_points > 0:
        pl.add_mesh(water_surf, color="#00ffff", opacity=0.3, smooth_shading=True)  # Cyan
    if salt_surf.n_points > 0:
        pl.add_mesh(salt_surf, color="#1e90ff", opacity=1.0, smooth_shading=True)   # Dodger Blue
        
    pl.add_text(stem, font_size=8, color="yellow", position="upper_left")
    pl.add_text(f"Labels (R:{r_rock:.0%} S:{r_salt:.0%} W:{r_water:.0%} B:{r_blank:.0%})", 
                font_size=10, color="white", position="upper_right")
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

    # ── [1,1]  Slices + overlay ────────────────────────────
    pl.subplot(1, 1)
    pl.add_mesh(sl_inline,    **seis_kw)
    pl.add_mesh(sl_crossline, **seis_kw)
    pl.add_mesh(sl_time,      **seis_kw)
    
    # Slightly lower opacities for the overlay so seismic is visible
    if blank_surf.n_points > 0:
        pl.add_mesh(blank_surf, color="#ff00ff", opacity=0.1, smooth_shading=True)
    if rock_surf.n_points > 0:
        pl.add_mesh(rock_surf, color="#8b7355", opacity=0.05, smooth_shading=True)
    if water_surf.n_points > 0:
        pl.add_mesh(water_surf, color="#00ffff", opacity=0.2, smooth_shading=True)
    if salt_surf.n_points > 0:
        pl.add_mesh(salt_surf, color="#ff4400", opacity=0.4, smooth_shading=True) # Orange-Red contrast
        
    pl.add_mesh(outline, color="white", line_width=1.5)
    pl.add_text(stem,            font_size=8,  color="yellow", position="upper_left")
    pl.add_text("Slices + Labels", font_size=10, color="white",  position="upper_right")
    pl.camera_position = CAM
    pl.add_axes(interactive=False)

    pl.render()

    del seis_grid, seis_outer, sl_inline, sl_crossline, sl_time, outline
    del rock_surf, salt_surf, water_surf, blank_surf
    gc.collect()

    return stem


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Scanning for '{CUBE_TYPE}' cubes...")
    # Updated glob to match new naming convention: e.g., mississippi_train_boundary_i00128_x03328_s00505.npz
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, f"*_{CUBE_TYPE}_*.npz")))
    
    if not files:
        raise FileNotFoundError(f"No '{CUBE_TYPE}' .npz files found in:\n  {TRAIN_DIR}")
        
    print(f"Found {len(files)} '{CUBE_TYPE}' cubes in train dir.")
    
    target_file = random.choice(files)

    pl = pv.Plotter(
        shape=(2, 2),
        border=True,
        border_color="white",
        window_size=WINDOW_SIZE,
        title=f"256³ Cube Viewer  |  {CUBE_TYPE.capitalize()} Cube",
    )

    render(pl, target_file)
    
    print("\nClose the window to exit.")
    pl.show()
    pl.close()
    print("Done.")