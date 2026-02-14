import numpy as np
import pyvista as pv
import os
import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "processed_data/mckinley/train" 
BATCH_FILE_STEP = 4      # load every Nth batch file
CUBE_IN_BATCH_STEP = 8   # load every Nth cube inside a batch
SHOW_ONLY_SALT = True

def visualize_reassembly_batched():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    
    if not files:
        print(f"No .npz files found in {DATA_DIR}")
        return

    selected_files = files[::BATCH_FILE_STEP]
    print(f"Found {len(files)} batch files.")
    print(f"Loading every {BATCH_FILE_STEP}th batch file ({len(selected_files)} files)...")

    blocks = pv.MultiBlock()
    
    print("Reassembling survey geometry...")
    
    for filepath in tqdm(selected_files):
        try:
            data = np.load(filepath)
            seismic_batch = data["seismic"]    # (B, 128, 128, 128)
            label_batch   = data["label"]
            meta_batch    = data["metadata"]  # (B, 5)
        except Exception as e:
            print(f"⚠️ Skipping {filepath}: {e}")
            continue
        
        B = seismic_batch.shape[0]
        
        for b in range(0, B, CUBE_IN_BATCH_STEP):
            seismic = seismic_batch[b]
            label   = label_batch[b]
            meta    = meta_batch[b]
            
            i_start = int(meta[0])
            x_start = int(meta[1])
            t_start = int(meta[2])
            has_salt = bool(meta[4])
            
            if SHOW_ONLY_SALT and not has_salt:
                continue
            
            grid = pv.ImageData()
            grid.dimensions = seismic.shape
            grid.origin = (i_start, x_start, t_start)
            grid.spacing = (1, 1, 1)

            grid.point_data["seismic"] = seismic.flatten(order="F")
            grid.point_data["label"]   = label.flatten(order="F")
            
            blocks.append(grid)

    if len(blocks) == 0:
        print("No cubes passed the filter!")
        return

    print(f"Rendering {len(blocks)} cubes...")

    plotter = pv.Plotter()
    plotter.set_background("black")

    print(" - Rendering Salt bodies...")
    for block in blocks:
        salt = block.threshold(0.5, scalars="label")
        if salt.n_points > 0:
            plotter.add_mesh(salt, color="red", opacity=1.0)

    if not SHOW_ONLY_SALT:
        print(" - Rendering Seismic context...")
        for block in blocks:
            plotter.add_mesh(block.outline(), color="white", opacity=0.1)

            center = np.array(block.center)
            slices = block.slice_orthogonal(x=center[0], y=center[1], z=center[2])
            plotter.add_mesh(slices, scalars="seismic", cmap="gray", opacity=0.25, show_scalar_bar=False)

    plotter.add_axes(xlabel="Inline", ylabel="Crossline", zlabel="Time")
    plotter.add_text(f"Batched NPZ Reassembly\nBatch step={BATCH_FILE_STEP}, Cube step={CUBE_IN_BATCH_STEP}", font_size=10)
    plotter.show()

if __name__ == "__main__":
    visualize_reassembly_batched()