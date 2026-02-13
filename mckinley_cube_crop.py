import segyio
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import random
from collections import defaultdict
import pickle

# ==========================================
# CONFIGURATION
# ==========================================
SEISMIC_PATH = "raw_seismic_mckinley.sgy"
LABEL_PATH = "labelmckinley.sgy"
OUTPUT_DIR = "processed_data"

GLOBAL_MEAN = 0.0003
GLOBAL_STD = 35.9071

CUBE_SIZE = 128
STRIDE = 64

SALT_THRESHOLD = 0.05
ROCK_KEEP_PROBABILITY = 0.10

random.seed(42)
np.random.seed(42)


# ==========================================
# HELPER FUNCTIONS (DEFINE FIRST)
# ==========================================

def build_trace_index(segy_file):
    """
    Build a mapping from (inline, crossline) → trace_index.
    
    This is CRITICAL for SEG-Y files with non-sequential trace numbering.
    
    Returns:
        trace_map: dict[(inline, xline)] = trace_idx
        inline_range: (min_inline, max_inline)
        xline_range: (min_xline, max_xline)
    """
    print("Building trace index (this may take a few minutes)...")
    
    trace_map = {}
    inline_numbers = []
    xline_numbers = []
    
    # Read headers for all traces
    for trace_idx in tqdm(range(len(segy_file.trace)), desc="Indexing traces"):
        header = segy_file.header[trace_idx]
        inline = header[segyio.TraceField.INLINE_3D]
        xline = header[segyio.TraceField.CROSSLINE_3D]
        
        trace_map[(inline, xline)] = trace_idx
        inline_numbers.append(inline)
        xline_numbers.append(xline)
    
    inline_range = (min(inline_numbers), max(inline_numbers))
    xline_range = (min(xline_numbers), max(xline_numbers))
    
    print(f"✅ Indexed {len(trace_map):,} traces")
    print(f"   Inline range: {inline_range[0]} to {inline_range[1]}")
    print(f"   Crossline range: {xline_range[0]} to {xline_range[1]}")
    
    return trace_map, inline_range, xline_range


def build_or_load_trace_index(segy_file, cache_path="trace_index.pkl"):
    """
    Build trace index and cache it to disk.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached trace index from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Build fresh
    trace_map, inline_range, xline_range = build_trace_index(segy_file)
    
    # Save for next time
    print(f"Saving trace index to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump((trace_map, inline_range, xline_range), f)
    
    return trace_map, inline_range, xline_range


def extract_cube_with_index(segy_file, trace_map, 
                            inline_start, inline_end,
                            xline_start, xline_end,
                            time_start, time_end):
    """
    Extract cube using trace index mapping.
    
    Args:
        trace_map: dict from build_trace_index()
        inline_start/end: Inline numbers (NOT indices)
        xline_start/end: Crossline numbers (NOT indices)
        time_start/end: Sample indices (0-1000)
    """
    n_inlines = inline_end - inline_start
    n_xlines = xline_end - xline_start
    n_samples = time_end - time_start
    
    cube = np.zeros((n_inlines, n_xlines, n_samples), dtype=np.float32)
    
    for i, inline in enumerate(range(inline_start, inline_end)):
        for x, xline in enumerate(range(xline_start, xline_end)):
            
            # Look up trace index
            trace_idx = trace_map.get((inline, xline))
            
            if trace_idx is not None:
                try:
                    trace = segy_file.trace[trace_idx]
                    cube[i, x, :] = trace[time_start:time_end]
                except (IndexError, KeyError):
                    pass  # Leave as zeros
    
    return cube


def normalize_seismic(cube):
    """Z-score normalization."""
    cube = cube.astype(np.float32)
    return (cube - GLOBAL_MEAN) / GLOBAL_STD


def binarize_label(cube):
    """Convert to strict binary."""
    return (cube > 0).astype(np.uint8)


def should_keep_cube(label_cube):
    """
    Intelligent sampling: keep all salt, discard 90% of rock.
    """
    salt_ratio = label_cube.mean()
    if salt_ratio >= SALT_THRESHOLD:
        return True, "salt"
    if random.random() < ROCK_KEEP_PROBABILITY:
        return True, "rock_sampled"
    return False, "rock_discarded"


# ==========================================
# MAIN PROCESSING FUNCTION
# ==========================================

def process_segy_to_cubes_v3():
    """
    Fixed version using header-based trace indexing.
    """
    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    print("="*70)
    print("SEISMIC SALT SEGMENTATION - DATA PREPROCESSING v3.0")
    print("(Header-Based Indexing)")
    print("="*70)
    
    # Open both files and build indices
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f_seis, \
         segyio.open(LABEL_PATH, 'r', ignore_geometry=True) as f_label:
        
        # Build trace index with caching
        trace_map_seis, inline_range, xline_range = build_or_load_trace_index(
            f_seis, "seismic_trace_index.pkl"
        )
        
        print("\nBuilding/loading label trace index...")
        trace_map_label, _, _ = build_or_load_trace_index(
            f_label, "label_trace_index.pkl"
        )
        
        # Define geographic regions using ACTUAL inline/crossline numbers
        min_inline, max_inline = inline_range
        min_xline, max_xline = xline_range
        
        print(f"\n" + "-"*70)
        print("SURVEY EXTENT:")
        print(f"  Inlines: {min_inline} to {max_inline} "
              f"(range: {max_inline - min_inline + 1})")
        print(f"  Crosslines: {min_xline} to {max_xline} "
              f"(range: {max_xline - min_xline + 1})")
        print("-"*70)
        
        # Define splits (using actual inline numbers)
        inline_span = max_inline - min_inline
        
        # 70% train, 10% val, 20% test (by inline number)
        train_end = min_inline + int(0.70 * inline_span)
        val_end = min_inline + int(0.80 * inline_span)
        
        TRAIN_REGION = {
            "inline_range": (min_inline, train_end),
            "xline_range": (min_xline, max_xline)
        }
        VAL_REGION = {
            "inline_range": (train_end, val_end),
            "xline_range": (min_xline, max_xline)
        }
        TEST_REGION = {
            "inline_range": (val_end, max_inline),
            "xline_range": (min_xline, max_xline)
        }
        
        print(f"\nGEOGRAPHIC SPLIT:")
        print(f"  Train: Inline {TRAIN_REGION['inline_range']}")
        print(f"  Val:   Inline {VAL_REGION['inline_range']}")
        print(f"  Test:  Inline {TEST_REGION['inline_range']}")
        print("-"*70)
        
        def get_split(inline_start):
            """Determine which split this inline belongs to."""
            inline_end = inline_start + CUBE_SIZE
            
            # Check each region
            if (TRAIN_REGION["inline_range"][0] <= inline_start and 
                inline_end <= TRAIN_REGION["inline_range"][1]):
                return "train"
            elif (VAL_REGION["inline_range"][0] <= inline_start and 
                  inline_end <= VAL_REGION["inline_range"][1]):
                return "val"
            elif (TEST_REGION["inline_range"][0] <= inline_start and 
                  inline_end <= TEST_REGION["inline_range"][1]):
                return "test"
            return None
        
        # Statistics tracking
        stats = {split: {
            "saved": 0,
            "skipped_boundary": 0,
            "skipped_empty": 0,
            "skipped_rock_sampling": 0,
            "salt_cubes": 0,
            "rock_cubes": 0
        } for split in ["train", "val", "test"]}
        
        # Generate cube positions (using ACTUAL inline/xline numbers)
        inline_positions = list(range(min_inline, max_inline - CUBE_SIZE + 1, STRIDE))
        xline_positions = list(range(min_xline, max_xline - CUBE_SIZE + 1, STRIDE))
        time_positions = list(range(0, 1001 - CUBE_SIZE + 1, STRIDE))
        
        total_candidates = len(inline_positions) * len(xline_positions) * len(time_positions)
        
        print(f"\nScanning ~{total_candidates:,} candidate positions...\n")
        
        pbar = tqdm(total=total_candidates, desc="Extracting cubes")
        
        # Main extraction loop
        for i_start in inline_positions:
            i_end = i_start + CUBE_SIZE
            
            split = get_split(i_start)
            if split is None:
                # Skip entire inline if it spans regions
                stats["train"]["skipped_boundary"] += len(xline_positions) * len(time_positions)
                pbar.update(len(xline_positions) * len(time_positions))
                continue
            
            for x_start in xline_positions:
                x_end = x_start + CUBE_SIZE
                
                for t_start in time_positions:
                    t_end = t_start + CUBE_SIZE
                    
                    # Extract cubes using index
                    seismic_cube = extract_cube_with_index(
                        f_seis, trace_map_seis,
                        i_start, i_end, x_start, x_end, t_start, t_end
                    )
                    
                    label_cube = extract_cube_with_index(
                        f_label, trace_map_label,
                        i_start, i_end, x_start, x_end, t_start, t_end
                    )
                    
                    # Check if cube has data
                    if np.std(seismic_cube) < 1e-6:
                        stats[split]["skipped_empty"] += 1
                        pbar.update(1)
                        continue
                    
                    # Normalize and binarize
                    seismic_cube = normalize_seismic(seismic_cube)
                    label_cube = binarize_label(label_cube)
                    
                    # Sampling decision
                    keep, reason = should_keep_cube(label_cube)
                    
                    if not keep:
                        stats[split]["skipped_rock_sampling"] += 1
                        pbar.update(1)
                        continue
                    
                    # Track statistics
                    if reason == "salt":
                        stats[split]["salt_cubes"] += 1
                    else:
                        stats[split]["rock_cubes"] += 1
                    
                    # Save cube
                    cube_id = stats[split]["saved"]
                    save_path = os.path.join(OUTPUT_DIR, split, f"cube_{cube_id:06d}.npz")
                    
                    np.savez_compressed(
                        save_path,
                        seismic=seismic_cube,
                        label=label_cube,
                        metadata=np.array([
                            i_start, x_start, t_start,
                            label_cube.mean(),
                            int(reason == "salt")
                        ])
                    )
                    
                    stats[split]["saved"] += 1
                    pbar.update(1)
        
        pbar.close()
    
    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    
    for split in ["train", "val", "test"]:
        s = stats[split]
        total = (s["saved"] + s["skipped_boundary"] + s["skipped_empty"] + 
                s["skipped_rock_sampling"])
        
        if total > 0:
            print(f"\n{split.upper()}:")
            print(f"  Saved:  {s['saved']:6,} cubes "
                  f"({s['saved']/total*100:.1f}% of {total:,} candidates)")
            print(f"    ├─ Salt cubes (≥5%):      {s['salt_cubes']:6,}")
            print(f"    └─ Rock cubes (<5%, 10%): {s['rock_cubes']:6,}")
            print(f"  Skipped:")
            print(f"    ├─ Region boundary:       {s['skipped_boundary']:6,}")
            print(f"    ├─ Empty/padding:         {s['skipped_empty']:6,}")
            print(f"    └─ Rock sampling (90%):   {s['skipped_rock_sampling']:6,}")
            
            if s['saved'] > 0:
                salt_pct = s['salt_cubes'] / s['saved'] * 100
                print(f"  Final balance: {salt_pct:.1f}% salt, {100-salt_pct:.1f}% rock")
    
    print("="*70)


# ==========================================
# RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    process_segy_to_cubes_v3()