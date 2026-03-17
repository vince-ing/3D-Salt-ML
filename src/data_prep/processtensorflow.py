"""
Extract 128x128x100 Training Patches with Multi-Class Balancing & Swapped Block-Striping
================================================================================
Classes after dynamic masking:
    0 = Rock
    1 = Salt
    2 = Water
    3 = Blank (Dead traces outside the valid survey area)

Features:
    - Zero Data Leakage: Rigid geographic boundaries between Train/Val/Test.
    - Swapped Splits: Validation is now the 3rd block (35-70%) for a larger sample size.
    - Bottom-Anchoring: Ensures patches perfectly hug the Max Depth (Z), Max Xline (X), 
      and Block boundaries (I) to eliminate wasted data gaps.
    - Rolling Slab Buffer: Prevents Out-of-Memory crashes on 130GB+ volumes.
    - Visualizations: Generates correctly oriented Map View and Middle Crossline View.
    - Zero-Padded Naming: 'mississippi_train_salt_i00128_x03328_s00505.npz'
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from tqdm import tqdm

try:
    import segyio
except ImportError:
    sys.exit("segyio not found. Run: pip install segyio tqdm")

# ============================================================
# CONFIGURATION
# ============================================================
SURVEY_NAME = "mississippi"
SEISMIC_SGY = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\raw\raw_seismic_mississippi.sgy"
LABEL_SGY   = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelmississippi_seafloor.sgy"
OUT_DIR     = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi128x100seafloor"

# Independent dimensions for Inline, Crossline, and Depth/Sample
PATCH_I = 128
PATCH_X = 128
PATCH_S = 100

# Strides (set to 50% overlap to match original script's behavior)
STRIDE_I = 64
STRIDE_X = 64
STRIDE_S = 50

# Keep Probabilities
KEEP_SALT     = 1.0  # 100%
KEEP_BOUNDARY = 0.20 # 20%
KEEP_ROCK     = 0.10 # 10%
KEEP_EMPTY    = 0.05 # 5%

# ============================================================
# HELPERS
# ============================================================
def get_starts(size, max_val, stride):
    """
    Calculates extraction anchors. If a remainder exists at the end of the 
    dimension, forces a final anchor perfectly flush with the boundary.
    """
    if max_val < size:
        return []
    starts = list(range(0, max_val - size + 1, stride))
    if not starts or starts[-1] != max_val - size:
        starts.append(max_val - size)  # The bottom-anchor
    return starts

# ============================================================
# VISUALIZATION GENERATOR
# ============================================================
def generate_visualizations(f_lbl, NI, NX, NS, blocks):
    print("\nGenerating Visualizations...")
    
    # Calculate boundaries explicitly for the buffer drawing
    b1 = int(NI * 0.20)
    b2 = int(NI * 0.35)
    b3 = int(NI * 0.70)
    b4 = int(NI * 0.85)
    
    # --------------------------------------------------------
    # 1. Map View (Inline vs Crossline) - Fast Decimated Scan
    # --------------------------------------------------------
    STEP = 100
    sampled_ilines = f_lbl.ilines[::STEP]
    sampled_xline_idx = np.arange(0, NX, STEP)
    valid_map = np.zeros((len(sampled_ilines), len(sampled_xline_idx)), dtype=bool)
    
    for i, iline_num in enumerate(tqdm(sampled_ilines, desc="  Scanning Map View")):
        inline_data = f_lbl.iline[iline_num]
        decimated_inline = inline_data[sampled_xline_idx, :]
        valid_map[i, :] = (decimated_inline == 2).any(axis=1)

    fig1, ax1 = plt.subplots(figsize=(10, 12))
    
    # standard top-down Y-axis 
    ax1.imshow(valid_map, cmap='gray', aspect='auto', alpha=0.3, extent=[0, NX, NI, 0], origin='upper')
    
    # FLIP HORIZONTALLY: Invert the X-axis so it matches Petrel view
    ax1.invert_xaxis()
    
    for split, start, end, bg_color, _ in blocks:
        ax1.add_patch(patches.Rectangle((0, start), NX, end-start, fill=True, color=bg_color, alpha=0.3))

    # Only draw the black buffers at the 4 internal block boundaries
    for b in [b1, b2, b3, b4]:
        ax1.axhline(b, color='black', linestyle='--', linewidth=2)
        buffer_start = max(0, b - PATCH_I)
        ax1.axhspan(buffer_start, b, color='black', alpha=0.5, hatch='//')

    ax1.set_title(f"Optimized Map View (Swapped Val/Test & Horizontally Flipped)\n(NI={NI}, NX={NX}, PATCH=({PATCH_I},{PATCH_X}))")
    ax1.set_ylabel("Inline Index (NI)")
    ax1.set_xlabel("Crossline Index (NX)")
    
    handles = [
        mpatches.Patch(color='darkgreen', alpha=0.5, label="Train Block"),
        mpatches.Patch(color='darkblue', alpha=0.5, label="Val Block"),
        mpatches.Patch(color='darkred', alpha=0.5, label="Test Block"),
        mpatches.Patch(color='black', alpha=0.5, hatch='//', label='Dropped (Leakage Buffer)'),
    ]
    ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    fig1.savefig('survey_map_view_keathley.png', dpi=200)
    print("  -> Saved 'survey_map_view_keathley.png'")
    
    # --------------------------------------------------------
    # 2. Middle Crossline Slice (Depth vs Inline)
    # --------------------------------------------------------
    print("  Extracting Middle Crossline Slice...")
    mid_x = NX // 2
    xl_data = f_lbl.xline[f_lbl.xlines[mid_x]] # Shape: (NI, NS)
    
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.imshow(xl_data.T, cmap='nipy_spectral', aspect='auto', extent=[0, NI, NS, 0], origin='upper')
    
    for split, start, end, bg_color, _ in blocks:
        ax2.axvspan(start, end, color=bg_color, alpha=0.3, label=f"{split.upper()} Block")

    for b in [b1, b2, b3, b4]:
        ax2.axvline(b, color='black', linestyle='--', linewidth=2)
        buffer_start = max(0, b - PATCH_I)
        ax2.axvspan(buffer_start, b, color='black', alpha=0.5, hatch='//')

    ax2.set_title(f"Middle Crossline Slice ({mid_x}) - Vertical Block Splitting")
    ax2.set_xlabel("Inline Index (NI)")
    ax2.set_ylabel("Depth Sample (NS)")
    
    h, l = ax2.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    ax2.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.tight_layout()
    fig2.savefig('survey_middle_crossline_keathley.png', dpi=200)
    print("  -> Saved 'survey_middle_crossline_keathley.png'\n")


# ============================================================
# MAIN EXTRACTION
# ============================================================
def extract_cubes():
    os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "test"), exist_ok=True)

    print("Opening SEG-Y volumes...")
    f_seis = segyio.open(SEISMIC_SGY, "r", ignore_geometry=False)
    f_lbl  = segyio.open(LABEL_SGY, "r", ignore_geometry=False)

    NI, NX, NS = len(f_seis.ilines), len(f_seis.xlines), len(f_seis.samples)
    print(f"Volume loaded. Shape: {NI}x{NX}x{NS}")

    # Calculate boundaries
    b1 = int(NI * 0.20)
    b2 = int(NI * 0.35)
    b3 = int(NI * 0.70)
    b4 = int(NI * 0.85)

    # Swapped Validation and Test blocks
    blocks = [
        ("train", 0, b1, 'lightgreen', 'darkgreen'),
        ("test", b1, b2, 'lightcoral', 'darkred'),
        ("train", b2, b3, 'lightgreen', 'darkgreen'),
        ("val", b3, b4, 'lightblue', 'darkblue'),
        ("train", b4, NI, 'lightgreen', 'darkgreen')
    ]

    generate_visualizations(f_lbl, NI, NX, NS, blocks)

    # Global anchors for X and Z axes (includes bottom-anchor logic)
    x_starts = get_starts(PATCH_X, NX, STRIDE_X)
    s_starts = get_starts(PATCH_S, NS, STRIDE_S)
    
    stats = {
        'train': 0, 'val': 0, 'test': 0,
        'cat_salt': 0, 'cat_boundary': 0, 'cat_rock': 0, 'cat_empty': 0
    }

    print("Commencing Patch Extraction (Rolling Slab Buffer)...")
    
    for split, b_start, b_end, _, _ in blocks:
        if b_end - b_start < PATCH_I:
            continue
            
        # Get localized inline anchors for this block (includes bottom-anchor logic)
        local_i_starts = get_starts(PATCH_I, b_end - b_start, STRIDE_I)
        i_starts = [b_start + i for i in local_i_starts]

        # Allocate rolling slab buffers
        seis_slab = np.zeros((PATCH_I, NX, NS), dtype=np.float32)
        lbl_slab  = np.zeros((PATCH_I, NX, NS), dtype=np.uint8)
        current_loaded_start = -1

        for i in tqdm(i_starts, desc=f"Extracting {split.upper()} block"):
            
            # --- 1. Load / Roll the Slab Buffer ---
            if current_loaded_start == -1 or i >= current_loaded_start + PATCH_I or i < current_loaded_start:
                # Full fresh read
                for offset in range(PATCH_I):
                    seis_slab[offset] = f_seis.iline[f_seis.ilines[i + offset]]
                    lbl_slab[offset]  = f_lbl.iline[f_lbl.ilines[i + offset]]
            else:
                # Roll existing buffer and read only what's new
                shift = i - current_loaded_start
                seis_slab[:-shift] = seis_slab[shift:]
                lbl_slab[:-shift]  = lbl_slab[shift:]
                for offset in range(PATCH_I - shift, PATCH_I):
                    seis_slab[offset] = f_seis.iline[f_seis.ilines[i + offset]]
                    lbl_slab[offset]  = f_lbl.iline[f_lbl.ilines[i + offset]]
            
            current_loaded_start = i

            # --- 2. Dynamic Blank Masking for the active slab ---
            # Any trace with Water(2) is valid. Reassign invalid Rock(0) to Blank(3).
            valid_traces = (lbl_slab == 2).any(axis=2)
            invalid_traces_3d = ~valid_traces[:, :, None]
            mask_to_blank = (lbl_slab == 0) & invalid_traces_3d
            lbl_slab[mask_to_blank] = 3

            # --- 3. Extract Patches ---
            for x in x_starts:
                for s in s_starts:
                    # Slice from the buffer (relative coordinates)
                    l_patch = lbl_slab[:, x:x+PATCH_X, s:s+PATCH_S]
                    unique_vals = np.unique(l_patch)
                    
                    has_salt  = 1 in unique_vals
                    has_rock  = 0 in unique_vals
                    has_water = 2 in unique_vals
                    has_blank = 3 in unique_vals
                    
                    keep = False
                    category = ""
                    
                    if has_salt:
                        keep = (random.random() <= KEEP_SALT)
                        category = "cat_salt"
                    elif has_rock and (has_water or has_blank):
                        keep = (random.random() <= KEEP_BOUNDARY)
                        category = "cat_boundary"
                    elif has_rock and not (has_water or has_blank):
                        keep = (random.random() <= KEEP_ROCK)
                        category = "cat_rock"
                    elif not has_rock and not has_salt:
                        keep = (random.random() <= KEEP_EMPTY)
                        category = "cat_empty"
                        
                    if keep:
                        cat_name = category.split('_')[1] # salt, boundary, rock, empty
                        stats[category] += 1
                        stats[split] += 1
                        s_patch = seis_slab[:, x:x+PATCH_X, s:s+PATCH_S]
                        
                        # New Naming Convention: mississippi_train_salt_i00128_x03328_s00505.npz
                        filename = f"{SURVEY_NAME}_{split}_{cat_name}_i{i:05d}_x{x:05d}_s{s:05d}.npz"
                        filepath = os.path.join(OUT_DIR, split, filename)
                        
                        np.savez_compressed(
                            filepath,
                            seismic=s_patch.astype(np.float32),
                            label=l_patch.astype(np.uint8)
                        )

    f_seis.close()
    f_lbl.close()

    print("\nExtraction Complete!")
    print(f"  Train patches: {stats['train']}")
    print(f"  Val patches:   {stats['val']}")
    print(f"  Test patches:  {stats['test']}")
    print("\nComposition:")
    print(f"  Salt:     {stats['cat_salt']}")
    print(f"  Boundary: {stats['cat_boundary']}")
    print(f"  Rock:     {stats['cat_rock']}")
    print(f"  Empty:    {stats['cat_empty']}")

if __name__ == "__main__":
    extract_cubes()