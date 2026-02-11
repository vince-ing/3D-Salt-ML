import pyvista as pv
import numpy as np
import os
import random

# --- CONFIGURATION ---
CROP_DIR = "training_crops/mckinley"  # Where your .npz files are
# ---------------------

def visualize_random_crop():
    # 1. Pick a random file
    files = [f for f in os.listdir(CROP_DIR) if f.endswith(".npz")]
    if not files:
        print(f"No .npz files found in {CROP_DIR}")
        return

    filename = random.choice(files)
    filepath = os.path.join(CROP_DIR, filename)
    print(f"Visualizing: {filename}")

    # 2. Load the data
    data = np.load(filepath)
    mask = data['mask']     # Shape: (128, 128, 128)
    
    # Optional: Load seismic if you want to toggle it later
    # seismic = data['seismic'] 

    # 3. Create the PyVista Grid
    # We must use Fortran order ('F') flattening because seismic data 
    # is usually stored column-major (z-axis fast), matching VTK's expectation.
    grid = pv.UniformGrid()
    grid.dimensions = np.array(mask.shape) + 1 # +1 for points vs cells
    grid.cell_data["Label"] = mask.flatten(order="F")

    # 4. Create the Plotter
    p = pv.Plotter(window_size=[1000, 800])
    p.set_background("black") # Dark background makes Red/White pop

    # --- VISUALIZATION LOGIC ---

    # PART A: The Salt (Red & Solid)
    # We "threshold" the grid to extract only cells where Label >= 0.5 (Salt)
    salt_mesh = grid.threshold(0.5, scalars="Label")
    
    if salt_mesh.n_cells > 0:
        p.add_mesh(salt_mesh, 
                   color="red", 
                   show_edges=False,
                   lighting=True,  # Gives it a 3D shiny look
                   label="Salt Body")
    else:
        print("Note: This crop contains no salt.")

    # PART B: The Background (White & Transparent)
    # We threshold for Label < 0.5 to get the empty space
    # (Or we can just plot the outline of the whole box)
    bg_mesh = grid.threshold(0.5, invert=True, scalars="Label")
    
    if bg_mesh.n_cells > 0:
        p.add_mesh(bg_mesh, 
                   color="white", 
                   opacity=0.05,       # Very faint (5% opacity)
                   style='surface',    # 'wireframe' also looks cool here
                   show_edges=False, 
                   label="Background")

    # Add a bounding box so you can see the cube limits clearly
    p.add_mesh(grid.outline(), color="white", opacity=0.3)

    # 5. Add text and show
    p.add_text(f"Crop: {filename}", position='upper_left', font_size=12)
    p.add_legend()
    p.show()

if __name__ == "__main__":
    # Keep visualizing until user stops
    while True:
        visualize_random_crop()
        user_input = input("Press Enter for another crop, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
