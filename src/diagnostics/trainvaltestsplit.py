import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from tqdm import tqdm

try:
    import segyio
except ImportError:
    sys.exit("segyio not found. Run: pip install segyio tqdm")

# Pointing to your specified label data
LABEL_SGY = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelkeathley_seafloor.sgy"
CUBE_SIZE = 256
STRIDE = 128
STEP = 100  # Decimation factor for the fast scan

def generate_cube_estimate_map():
    print(f"Opening {os.path.basename(LABEL_SGY)}...")
    try:
        with segyio.open(LABEL_SGY, "r", ignore_geometry=False) as f:
            NI = len(f.ilines)
            NX = len(f.xlines)
            
            print(f"Volume dimensions: {NI}x{NX} (Inline x Crossline)")
            print(f"Mapping valid survey area (fast scan: 1 every {STEP} lines)...")
            
            sampled_ilines = f.ilines[::STEP]
            sampled_xline_idx = np.arange(0, NX, STEP)
            
            valid_map = np.zeros((len(sampled_ilines), len(sampled_xline_idx)), dtype=bool)
            
            for i, iline_num in enumerate(tqdm(sampled_ilines, desc="Reading Inlines (Decimated)")):
                inline_data = f.iline[iline_num]
                decimated_inline = inline_data[sampled_xline_idx, :]
                valid_map[i, :] = (decimated_inline == 2).any(axis=1)
                
    except Exception as e:
        sys.exit(f"Failed to process SEG-Y file: {e}")

    # Calculate boundaries
    b1 = int(NI * 0.20)
    b2 = int(NI * 0.35)
    b3 = int(NI * 0.70)
    b4 = int(NI * 0.85)

    # Define the blocks explicitly - Validation and Test are now swapped!
    blocks = [
        ("train", 0, b1, 'lightgreen', 'darkgreen'),
        ("test", b1, b2, 'lightcoral', 'darkred'),     # Previously Val
        ("train", b2, b3, 'lightgreen', 'darkgreen'),
        ("val", b3, b4, 'lightblue', 'darkblue'),      # Previously Test
        ("train", b4, NI, 'lightgreen', 'darkgreen')
    ]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(valid_map, cmap='gray', aspect='auto', alpha=0.3, extent=[0, NX, NI, 0])

    print("\nSimulating dense block-local extraction overlay...")
    
    # Global crossline starts remain the same
    x_starts = list(range(0, NX - CUBE_SIZE + 1, STRIDE))
    
    cube_counts = {"train": 0, "val": 0, "test": 0}
    
    for split, start, end, bg_color, dot_color in blocks:
        # Draw the background block
        ax.add_patch(patches.Rectangle((0, start), NX, end-start, fill=True, color=bg_color, alpha=0.3))
        ax.axhline(end, color='black', linestyle='--', linewidth=2)
        
        # Create a local grid starting exactly at the boundary line
        block_i_starts = list(range(start, end - CUBE_SIZE + 1, STRIDE))
        
        # Highlight the tiny unavoidable remainder gap at the end of the block
        if block_i_starts:
            last_end = block_i_starts[-1] + CUBE_SIZE
            if last_end < end:
                ax.axhspan(last_end, end, color='black', alpha=0.5, hatch='//')
        
        # Plot the cubes
        for i in block_i_starts:
            for x in x_starts:
                i_center = i + (CUBE_SIZE // 2)
                x_center = x + (CUBE_SIZE // 2)
                
                # Check if it falls inside the valid data footprint
                idx_i = min(i_center // STEP, valid_map.shape[0] - 1)
                idx_x = min(x_center // STEP, valid_map.shape[1] - 1)
                
                if valid_map[idx_i, idx_x]:
                    ax.scatter(x_center, i_center, c=dot_color, s=2, alpha=0.8)
                    cube_counts[split] += 1

    ax.set_title(f"Optimized Cube Extraction Density (Swapped Val/Test)\n(NI={NI}, NX={NX}, CUBE={CUBE_SIZE}, STRIDE={STRIDE})")
    ax.set_ylabel("Inline Index (NI)")
    ax.set_xlabel("Crossline Index (NX)")

    # Custom legend
    handles = [
        mpatches.Patch(color='darkgreen', label=f"Train Cubes (~{cube_counts['train']})"),
        mpatches.Patch(color='darkblue', label=f"Val Cubes (~{cube_counts['val']})"),
        mpatches.Patch(color='darkred', label=f"Test Cubes (~{cube_counts['test']})"),
        mpatches.Patch(color='black', alpha=0.5, hatch='//', label='Unused Remainder Gap\n(Too small for 256³ cube)'),
    ]
    
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), markerscale=5)

    plt.tight_layout()
    out_path = 'survey_cube_estimate_optimized_keathley.png'
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved optimized visualization to: {out_path}")

if __name__ == "__main__":
    generate_cube_estimate_map()