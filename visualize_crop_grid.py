import pyvista as pv
import numpy as np

# --- CONFIGURATION ---
SURVEY_SHAPE = (5391, 7076, 1001)  # (Inlines, Crosslines, Samples)
CROP_SIZE = (128, 128, 128)
STRIDE = (64, 64, 64)              # 50% overlap
# ---------------------

def visualize_crop_grid():
    print("Generating crop locations...")
    
    d_il, d_xl, d_z = CROP_SIZE
    s_il, s_xl, s_z = STRIDE
    n_il, n_xl, n_z = SURVEY_SHAPE

    # Calculate starting points (top-left corner of each cube)
    il_starts = range(0, n_il - d_il + 1, s_il)
    xl_starts = range(0, n_xl - d_xl + 1, s_xl)
    z_starts  = range(0, n_z  - d_z  + 1, s_z)

    # Calculate total number of cubes
    total_cubes = len(il_starts) * len(xl_starts) * len(z_starts)
    print(f"Total Cubes to Render: {total_cubes}")
    
    if total_cubes > 100000:
        print("WARNING: That is a lot of cubes. Rendering might be slow.")

    # --- EFFICIENT RENDERING TRICK ---
    # Instead of creating 50,000 separate mesh objects, we create ONE mesh
    # containing all the points, but they are disconnected.
    
    # We will build a "Glyph" visualization.
    # 1. Create a single point at the center of every crop.
    # 2. Assign a single 128x128x128 box to those points.
    
    centers = []
    
    # Pre-calculate half-sizes to find centers
    h_il, h_xl, h_z = d_il / 2, d_xl / 2, d_z / 2
    
    for il in il_starts:
        for xl in xl_starts:
            for z in z_starts:
                # Center = Start + Half_Size
                c = [il + h_il, xl + h_xl, z + h_z]
                centers.append(c)
    
    centers = np.array(centers)
    
    # Create a PolyData object from these center points
    cloud = pv.PolyData(centers)
    
    # Create the single "Template" box (Wireframe)
    # We make a box of size 128x128x128 centered at (0,0,0)
    cube = pv.Cube(x_length=d_il, y_length=d_xl, z_length=d_z)
    
    # Glyph: Put a cube at every point in the cloud
    print("Building geometry...")
    grid = cloud.glyph(geom=cube, scale=False, orient=False)
    
    # --- VISUALIZATION ---
    print("Rendering...")
    p = pv.Plotter()
    p.set_background("black")
    
    # Render the cubes
    # style='wireframe' makes them look like empty cages
    p.add_mesh(grid, 
               style='wireframe', 
               color='cyan', 
               opacity=0.1,  # Very faint so you can see through the density
               line_width=1)
               
    # Add the Survey Bounding Box (The "Container")
    # This helps you see if your crops reach the edges
    survey_box = pv.Box(bounds=(0, n_il, 0, n_xl, 0, n_z))
    p.add_mesh(survey_box, color='white', style='wireframe', opacity=1.0, line_width=2, label="Survey Bounds")

    p.add_text(f"Grid: {len(il_starts)}x{len(xl_starts)}x{len(z_starts)} crops", position='upper_left')
    p.show_grid()
    p.show()

if __name__ == "__main__":
    visualize_crop_grid()
