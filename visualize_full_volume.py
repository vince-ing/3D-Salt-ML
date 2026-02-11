import pyvista as pv
import numpy as np

# --- CONFIGURATION ---
MASK_BIN = "mckinley_mask.bin"  
SHAPE = (5391, 7076, 1001)   # Full Dimensions
STRIDE = 20                  # Take 1 voxel every 20 steps
# ---------------------

def visualize_full_mask():
    print(f"1. Mapping massive volume {SHAPE}...")
    # Mode='r' ensures we don't load 38GB into RAM
    full_vol = np.memmap(MASK_BIN, dtype='int8', mode='r', shape=SHAPE)

    print(f"2. Downsampling (Stride={STRIDE})...")
    # Slicing with [::STRIDE] creates a copy in RAM, but it's small!
    # New size approx: 270 x 350 x 50 voxels (~4.7 million points)
    small_vol = full_vol[::STRIDE, ::STRIDE, ::STRIDE]
    
    print(f"   - Reduced shape: {small_vol.shape}")
    print(f"   - Memory footprint: {small_vol.nbytes / 1024**2:.2f} MB")

    print("3. creating 3D Mesh...")
    # Create grid
    grid = pv.UniformGrid()
    grid.dimensions = np.array(small_vol.shape) + 1
    
    # Set spacing so the shape looks correct (not squashed)
    # We multiply original spacing (12.5, 20, 4) by the stride
    grid.spacing = (12.5 * STRIDE, 20.0 * STRIDE, 4.0 * STRIDE) 
    
    grid.cell_data["Salt"] = small_vol.flatten(order="F")

    print("4. Rendering...")
    p = pv.Plotter()
    p.set_background("black")

    # Threshold > 0 to see the salt
    # Since we downsampled, we might lose thin salt stems.
    # If it looks too empty, lower the STRIDE to 10.
    salt_body = grid.threshold(0.5, scalars="Salt")

    if salt_body.n_cells > 0:
        p.add_mesh(salt_body, 
                   color="red", 
                   show_edges=False,
                   smooth_shading=True,
                   label="Salt Body")
    else:
        print("Warning: No salt found in downsampled volume. Try reducing Stride.")

    # Add a bounding box of the survey
    p.add_mesh(grid.outline(), color="white", opacity=0.3)
    
    # Add axes (scaled to real coordinates)
    p.show_axes()
    p.show_grid()
    
    p.show()

if __name__ == "__main__":
    visualize_full_mask()
