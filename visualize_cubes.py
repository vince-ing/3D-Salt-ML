import numpy as np
import pyvista as pv
import os
import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your processed data
DATA_DIR = "processed_data/train" 

# VISUALIZATION STRIDE
# "10" means we only load every 10th cube file.
#   - Lower (e.g., 5) = More dense, higher RAM usage.
#   - Higher (e.g., 20) = Faster, more gaps between cubes.
CUBE_STEP = 50 

# FILTERING
# Set to True to ONLY show cubes that contain salt.
# This helps you check if the salt body looks continuous across cubes.
SHOW_ONLY_SALT = False

def visualize_reassembly():
    # 1. Find all files
    search_path = os.path.join(DATA_DIR, "*.npz")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No .npz files found in {DATA_DIR}")
        return

    # 2. Subsample the list
    selected_files = files[::CUBE_STEP]
    print(f"Found {len(files)} total cubes.")
    print(f"Loading every {CUBE_STEP}th cube ({len(selected_files)} cubes to render)...")

    # 3. Create a MultiBlock dataset (Container for many meshes)
    blocks = pv.MultiBlock()
    
    print("Reassembling survey geometry...")
    
    for filepath in tqdm(selected_files):
        try:
            # Load Data
            data = np.load(filepath)
            
            # Extract Metadata (Saved by your script)
            # Format: [i_start, x_start, t_start, mean_salt, has_salt]
            meta = data['metadata']
            i_start = int(meta[0])
            x_start = int(meta[1])
            t_start = int(meta[2])
            has_salt = bool(meta[4])

            # Optional: Skip empty rock cubes if requested
            if SHOW_ONLY_SALT and not has_salt:
                continue

            seismic = data['seismic'] # 128x128x128
            label   = data['label']   # 128x128x128

            # 4. Create a PyVista Grid for THIS cube
            # origin=(Z, X, Y) -> We map Inline to X, Crossline to Y, Time to Z
            # Note: PyVista is (x, y, z). 
            # We map: Inline -> X, Crossline -> Y, Time -> Z (depth)
            grid = pv.ImageData()
            grid.dimensions = seismic.shape
            grid.origin = (i_start, x_start, t_start)
            grid.spacing = (1, 1, 1) # 1 unit = 1 pixel

            # Add data to grid
            grid.point_data["seismic"] = seismic.flatten(order="F")
            grid.point_data["label"]   = label.flatten(order="F")

            # 5. Thresholding (Optimization)
            # Instead of rendering the whole solid cube, we only render:
            #  a) The Salt (Red)
            #  b) The Strong Seismic Reflectors (Grey) - filtered by opacity later
            
            blocks.append(grid)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    # 6. Visualization
    if len(blocks) == 0:
        print("No cubes passed the filter!")
        return

    print("Rendering... (This may take a moment)")
    
    plotter = pv.Plotter()
    plotter.set_background("black")

    # Combine all blocks into one mesh for faster rendering? 
    # No, MultiBlock is fine for this scale.
    
    # We iterate blocks to apply specific styles
    # (PyVista doesn't support 'volume' rendering on MultiBlock directly easily, 
    #  so we render isosurfaces or thresholded meshes)
    
    # A. Render Salt (RED)
    # We want to see the Salt body form a continuous shape across the gaps
    print(" - Generating Salt Meshes...")
    for block in blocks:
        # Extract salt (label == 1)
        salt = block.threshold(0.5, scalars="label")
        if salt.n_points > 0:
             plotter.add_mesh(salt, color="red", show_edges=False)

    # B. Render Seismic (GREY)
    # Since Volume Rendering many blocks is heavy, we use Orthogonal Slices 
    # OR a threshold of strong reflectors.
    if not SHOW_ONLY_SALT:
        print(" - Generating Seismic Context...")
        for block in blocks:
            # Option 1: Outline of the cube (Shows the 'grid' structure)
            plotter.add_mesh(block.outline(), color="white", opacity=0.1)
            
            # Option 2: Slices (Cheaper than volume)
            # Show a slice through the middle of the cube
            center = np.array(block.center)
            slices = block.slice_orthogonal(x=center[0], y=center[1], z=center[2])
            plotter.add_mesh(slices, scalars="seismic", cmap="gray", opacity=0.3, show_scalar_bar=False)

    plotter.add_axes(xlabel="Inline", ylabel="Crossline", zlabel="Time")
    plotter.add_text(f"Reassembled {len(blocks)} Cubes\nStep: {CUBE_STEP}", font_size=10)
    
    plotter.show()

if __name__ == "__main__":
    visualize_reassembly()