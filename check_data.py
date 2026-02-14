import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "processed_data/mississippi/train"  # Ensure this path is correct
N_FILES_TO_CHECK = 3

def check_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not files:
        print(f"No files found in {DATA_DIR}! Check your path.")
        return

    print(f"Found {len(files)} batch files. Checking {N_FILES_TO_CHECK}...")
    
    samples = random.sample(files, min(len(files), N_FILES_TO_CHECK))

    for fpath in samples:
        data = np.load(fpath)
        seismic_batch = data['seismic']  # Shape: (64, 128, 128, 128)
        label_batch = data['label']      # Shape: (64, 128, 128, 128)
        
        # Pick the FIRST cube in the batch to visualize
        # (You can change 0 to random.randint(0, len(seismic_batch)-1) to see others)
        cube_idx = 0
        seismic_cube = seismic_batch[cube_idx]
        label_cube = label_batch[cube_idx]

        print(f"\n--- File: {os.path.basename(fpath)} | Cube Index: {cube_idx} ---")
        print(f"Cube Shape:   {seismic_cube.shape}")
        print(f"Seismic Range: {seismic_cube.min():.2f} to {seismic_cube.max():.2f}")
        print(f"Label Values:  {np.unique(label_cube)}")
        
        # SLICING: We want the middle Inline of this specific cube
        # Shape is (Inline, Xline, Time) -> (128, 128, 128)
        mid_idx = seismic_cube.shape[0] // 2  # Index 64
        
        # Extract 2D slices
        # Transpose (.T) puts Time on the Y-axis (standard seismic view)
        seismic_slice = seismic_cube[mid_idx, :, :].T
        label_slice = label_cube[mid_idx, :, :].T

        # PLOTTING
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
        
        # 1. Seismic
        # vmin/vmax clips outliers so the image isn't washed out by that -17/+20 range
        ax1.imshow(seismic_slice, cmap='gray', aspect='auto', vmin=-3, vmax=3)
        ax1.set_title("Seismic (Clipped -3 to +3)")
        ax1.set_xlabel("Crossline")
        ax1.set_ylabel("Time")
        
        # 2. Label
        ax2.imshow(label_slice, cmap='jet', interpolation='nearest', aspect='auto')
        ax2.set_title("Label (Salt Mask)")
        
        # 3. Overlay
        ax3.imshow(seismic_slice, cmap='gray', aspect='auto', vmin=-3, vmax=3)
        # Create a masked alpha array so '0' (rock) is transparent
        mask_alpha = np.ma.masked_where(label_slice == 0, label_slice)
        ax3.imshow(mask_alpha, cmap='jet', alpha=0.5, aspect='auto') 
        ax3.set_title("Overlay Check")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    check_data()