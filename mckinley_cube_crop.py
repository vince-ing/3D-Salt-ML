import segyio
import numpy as np
import os
from tqdm import tqdm
import random

# ==========================================
# CONFIGURATION
# ==========================================
SEISMIC_PATH = "raw_seismic_mckinley.sgy"
LABEL_PATH = "labelmckinley.sgy"
OUTPUT_DIR = "processed_data/mckinley"

GLOBAL_MEAN = 0.0003
GLOBAL_STD = 35.9071

CUBE_SIZE = 128
STRIDE = 64

SALT_THRESHOLD = 0.05
ROCK_KEEP_PROBABILITY = 0.10

# Survey geometry (from your statistics - CRITICAL!)
N_INLINES = 5391
N_CROSSLINES = 7076
N_SAMPLES = 1001

INLINE_START = 1960      # First inline number
INLINE_END = 7350        # Last inline number
XLINE_START = 1875       # First crossline number
XLINE_END = 8950         # Last crossline number

# Memory management
CHUNK_SIZE = 256  # Process 256 inlines at a time

random.seed(42)
np.random.seed(42)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_seismic(cube):
    """Z-score normalization using global statistics."""
    cube = cube.astype(np.float32)
    return (cube - GLOBAL_MEAN) / GLOBAL_STD


def binarize_label(cube):
    """Convert label to strict binary (0 or 1)."""
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


def load_cube_manual(segy_file, inline_start_idx, inline_end_idx):
    """
    Manually load a chunk of the survey by reading traces.
    
    Args:
        inline_start_idx: Starting inline INDEX (0-5390), not inline NUMBER
        inline_end_idx: Ending inline INDEX (0-5390), not inline NUMBER
    
    Returns:
        3D numpy array of shape (n_inlines, n_crosslines, n_samples)
    """
    n_inlines = inline_end_idx - inline_start_idx
    chunk = np.zeros((n_inlines, N_CROSSLINES, N_SAMPLES), dtype=np.float32)
    
    # Read traces
    for i in range(n_inlines):
        global_inline_idx = inline_start_idx + i
        
        for x in range(N_CROSSLINES):
            # Compute trace index (assuming row-major order)
            trace_idx = global_inline_idx * N_CROSSLINES + x
            
            try:
                if trace_idx < len(segy_file.trace):
                    chunk[i, x, :] = segy_file.trace[trace_idx]
            except (IndexError, KeyError):
                pass  # Leave as zeros
    
    return chunk


# ==========================================
# MAIN PROCESSING FUNCTION
# ==========================================

def process_segy_chunked_manual():
    """
    Manual trace-by-trace loading (works with any SEG-Y structure).
    """
    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    print("="*70)
    print("SEISMIC SALT SEGMENTATION - MANUAL TRACE EXTRACTION")
    print("="*70)
    print(f"Survey: {N_INLINES} inlines × {N_CROSSLINES} xlines × {N_SAMPLES} samples")
    print(f"Cube size: {CUBE_SIZE}³")
    print(f"Stride: {STRIDE}")
    print(f"Chunk size: {CHUNK_SIZE} inlines")
    print("="*70)
    
    # Define splits by inline index
    train_end_idx = int(0.70 * N_INLINES)
    val_end_idx = int(0.80 * N_INLINES)
    
    TRAIN_RANGE = (0, train_end_idx)
    VAL_RANGE = (train_end_idx, val_end_idx)
    TEST_RANGE = (val_end_idx, N_INLINES)
    
    print(f"\nSPLIT BY INLINE INDEX:")
    print(f"  Train: {TRAIN_RANGE[0]:4d} to {TRAIN_RANGE[1]:4d} ({TRAIN_RANGE[1]-TRAIN_RANGE[0]:4d} inlines)")
    print(f"  Val:   {VAL_RANGE[0]:4d} to {VAL_RANGE[1]:4d} ({VAL_RANGE[1]-VAL_RANGE[0]:4d} inlines)")
    print(f"  Test:  {TEST_RANGE[0]:4d} to {TEST_RANGE[1]:4d} ({N_INLINES-TEST_RANGE[0]:4d} inlines)")
    print("="*70 + "\n")
    
    def get_split(inline_idx):
        """Determine split based on inline INDEX."""
        inline_end = inline_idx + CUBE_SIZE
        
        if TRAIN_RANGE[0] <= inline_idx and inline_end <= TRAIN_RANGE[1]:
            return "train"
        elif VAL_RANGE[0] <= inline_idx and inline_end <= VAL_RANGE[1]:
            return "val"
        elif TEST_RANGE[0] <= inline_idx and inline_end <= TEST_RANGE[1]:
            return "test"
        return None
    
    # Statistics
    stats = {split: {
        "saved": 0,
        "skipped_boundary": 0,
        "skipped_empty": 0,
        "skipped_rock_sampling": 0,
        "salt_cubes": 0,
        "rock_cubes": 0
    } for split in ["train", "val", "test"]}
    
    # Pre-compute positions
    xline_positions = list(range(0, N_CROSSLINES - CUBE_SIZE + 1, STRIDE))
    time_positions = list(range(0, N_SAMPLES - CUBE_SIZE + 1, STRIDE))
    
    total_inline_positions = len(range(0, N_INLINES - CUBE_SIZE + 1, STRIDE))
    total_candidates = total_inline_positions * len(xline_positions) * len(time_positions)
    
    print(f"Total candidate positions: {total_candidates:,}\n")
    
    # Open files
    print("Opening SEG-Y files...")
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f_seis, \
         segyio.open(LABEL_PATH, 'r', ignore_geometry=True) as f_label:
        
        print(f"  Seismic: {len(f_seis.trace):,} traces")
        print(f"  Label:   {len(f_label.trace):,} traces")
        print(f"\nProcessing in chunks of {CHUNK_SIZE} inlines...\n")
        
        # Progress bar
        pbar = tqdm(total=total_candidates, desc="Extracting cubes")
        
        # Process in chunks
        num_chunks = (N_INLINES + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, N_INLINES)
            chunk_size = chunk_end - chunk_start
            
            if chunk_size < CUBE_SIZE:
                # Skip small chunks at end
                continue
            
            # Load chunk
            print(f"\n  Chunk {chunk_idx+1}/{num_chunks}: Loading inlines {chunk_start}-{chunk_end}...")
            
            seismic_chunk = load_cube_manual(f_seis, chunk_start, chunk_end)
            label_chunk = load_cube_manual(f_label, chunk_start, chunk_end)
            
            print(f"    Loaded: seismic {seismic_chunk.shape}, label {label_chunk.shape}")
            
            # Extract cubes from this chunk
            local_inline_positions = range(0, chunk_size - CUBE_SIZE + 1, STRIDE)
            
            for local_i_idx in local_inline_positions:
                global_i_idx = chunk_start + local_i_idx
                split = get_split(global_i_idx)
                
                if split is None:
                    stats["train"]["skipped_boundary"] += len(xline_positions) * len(time_positions)
                    pbar.update(len(xline_positions) * len(time_positions))
                    continue
                
                for x_idx in xline_positions:
                    for t_idx in time_positions:
                        
                        # Extract cube
                        seismic_cube = seismic_chunk[
                            local_i_idx:local_i_idx + CUBE_SIZE,
                            x_idx:x_idx + CUBE_SIZE,
                            t_idx:t_idx + CUBE_SIZE
                        ]
                        
                        label_cube = label_chunk[
                            local_i_idx:local_i_idx + CUBE_SIZE,
                            x_idx:x_idx + CUBE_SIZE,
                            t_idx:t_idx + CUBE_SIZE
                        ]
                        
                        # Validation
                        if np.std(seismic_cube) < 1e-6:
                            stats[split]["skipped_empty"] += 1
                            pbar.update(1)
                            continue
                        
                        # Normalize
                        seismic_cube = normalize_seismic(seismic_cube)
                        label_cube = binarize_label(label_cube)
                        
                        # Sampling decision
                        keep, reason = should_keep_cube(label_cube)
                        
                        if not keep:
                            stats[split]["skipped_rock_sampling"] += 1
                            pbar.update(1)
                            continue
                        
                        # Track
                        if reason == "salt":
                            stats[split]["salt_cubes"] += 1
                        else:
                            stats[split]["rock_cubes"] += 1
                        
                        # Save
                        cube_id = stats[split]["saved"]
                        save_path = os.path.join(OUTPUT_DIR, split, f"cube_{cube_id:06d}.npz")
                        
                        np.savez_compressed(
                            save_path,
                            seismic=seismic_cube,
                            label=label_cube,
                            metadata=np.array([
                                global_i_idx,
                                x_idx,
                                t_idx,
                                label_cube.mean(),
                                int(reason == "salt")
                            ])
                        )
                        
                        stats[split]["saved"] += 1
                        pbar.update(1)
            
            # Free memory
            del seismic_chunk
            del label_chunk
        
        pbar.close()
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    
    for split in ["train", "val", "test"]:
        s = stats[split]
        total = (s["saved"] + s["skipped_boundary"] + s["skipped_empty"] + 
                s["skipped_rock_sampling"])
        
        if total > 0:
            print(f"\n{split.upper()}:")
            print(f"  Saved:  {s['saved']:6,} cubes")
            print(f"    ├─ Salt cubes (≥5%):      {s['salt_cubes']:6,}")
            print(f"    └─ Rock cubes (<5%, 10%): {s['rock_cubes']:6,}")
            
            if s['saved'] > 0:
                salt_pct = s['salt_cubes'] / s['saved'] * 100
                print(f"  Final balance: {salt_pct:.1f}% salt, {100-salt_pct:.1f}% rock")
    
    print("="*70)


if __name__ == "__main__":
    process_segy_chunked_manual()