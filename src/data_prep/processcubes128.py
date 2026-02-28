import segyio
import numpy as np
import os
from tqdm import tqdm
import random
import sys

"""
Extracts 128x128x128 seismic patches for multi-scale salt detection training.

Parallel usage (split by inline range across machines):
    py src/data_prep/processcubes128.py 0 1954
    python src/data_prep/processcubes128.py 1954 3908
    py src/data_prep/processcubes128.py 3908 5862

Single machine:
    python src/data_prep/processcubes128.py
"""

# ==========================================
# CONFIGURATION
# ==========================================
SURVEY_NAME  = "mississippi"

SEISMIC_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\raw\raw_seismic_mississippi.sgy"
LABEL_PATH   = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelmississippi.sgy"
OUTPUT_DIR   = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi128/"

# Cube size — all dimensions 128
CUBE_SIZE = 128         # inline, crossline, and time all 128

# Strides — denser in Z since it is the shallowest dimension (only 1081 samples)
STRIDE_INLINE     = 64
STRIDE_CROSSLINE  = 64
STRIDE_Z          = 64

# Volume dimensions
N_INLINES     = 5862
N_CROSSLINES  = 5289
N_SAMPLES     = 1081

# Train / val / test split boundaries (by inline index)
# 60% train, 30% val, 10% test
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.30
# TEST_RATIO  = 0.10 (remainder)

# Filtering
SALT_THRESHOLD        = 0.05   # keep cube if >= 5% of voxels are salt
ROCK_KEEP_PROBABILITY = 0.10   # keep 10% of cubes with < 5% salt

# Chunk size for memory-efficient SEG-Y loading (inlines per read)
CHUNK_SIZE = 64

# ==========================================
# PARALLEL PROCESSING
# ==========================================
if len(sys.argv) == 3:
    PROCESS_INLINE_START = int(sys.argv[1])
    PROCESS_INLINE_END   = int(sys.argv[2])
else:
    PROCESS_INLINE_START = 0
    PROCESS_INLINE_END   = N_INLINES

print(f"Survey:  {SURVEY_NAME}")
print(f"Inlines: {PROCESS_INLINE_START} → {PROCESS_INLINE_END}")

JOB_ID = f"{PROCESS_INLINE_START}_{PROCESS_INLINE_END}"

random.seed(42)
np.random.seed(42)

# ==========================================
# SPLIT BOUNDARIES
# ==========================================
TRAIN_END_INLINE = int(TRAIN_RATIO * N_INLINES)                    # 0     → 3517
VAL_END_INLINE   = int((TRAIN_RATIO + VAL_RATIO) * N_INLINES)      # 3517  → 5279
# TEST             = 5279 → 5862

def get_split(inline_start):
    """
    Assign a cube to train/val/test based on where its first inline falls.
    A cube must fit entirely within the volume so inline_end = inline_start + CUBE_SIZE.
    """
    inline_end = inline_start + CUBE_SIZE
    if inline_end <= TRAIN_END_INLINE:
        return "train"
    elif inline_start >= TRAIN_END_INLINE and inline_end <= VAL_END_INLINE:
        return "val"
    elif inline_start >= VAL_END_INLINE and inline_end <= N_INLINES:
        return "test"
    return None  # cube straddles a boundary — skip to avoid data leakage

# ==========================================
# NORMALIZATION
# ==========================================
def normalize_cube(cube):
    """
    Per-cube normalization:
      1. Clip to 1st–99th percentile to remove amplitude spikes
      2. Z-score using the clipped cube's mean and std
    """
    cube = cube.astype(np.float32)
    p1  = np.percentile(cube, 1)
    p99 = np.percentile(cube, 99)
    cube = np.clip(cube, p1, p99)
    mean = cube.mean()
    std  = cube.std()
    if std < 1e-8:
        return cube - mean          # flat cube, avoid divide by zero
    return (cube - mean) / std

def binarize_label(cube):
    return (cube > 0).astype(np.uint8)

# ==========================================
# FILTERING
# ==========================================
def should_keep_cube(label_cube):
    salt_ratio = label_cube.mean()
    if salt_ratio >= SALT_THRESHOLD:
        return True, "salt"
    if random.random() < ROCK_KEEP_PROBABILITY:
        return True, "rock_sampled"
    return False, "rock_discarded"

# ==========================================
# SEG-Y LOADING
# ==========================================
def load_chunk_fast(segy_file, inline_start_idx, inline_end_idx):
    """
    Load a block of inlines from a SEG-Y file into a numpy array.
    Shape: (n_inlines, N_CROSSLINES, N_SAMPLES)
    """
    n_inlines = inline_end_idx - inline_start_idx
    chunk = np.zeros((n_inlines, N_CROSSLINES, N_SAMPLES), dtype=np.float32)
    total_traces = len(segy_file.trace)

    for i in range(n_inlines):
        global_inline_idx = inline_start_idx + i
        trace_start = global_inline_idx * N_CROSSLINES
        if trace_start >= total_traces:
            break
        for x in range(N_CROSSLINES):
            trace_idx = trace_start + x
            if trace_idx < total_traces:
                chunk[i, x, :] = segy_file.trace[trace_idx]

    return chunk

# ==========================================
# MAIN
# ==========================================
def process_segy_parallel():

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # Pre-calculate all grid positions
    xline_positions = list(range(0, N_CROSSLINES - CUBE_SIZE + 1, STRIDE_CROSSLINE))
    # Anchor to bottom of survey so the base is always covered
    # The top few samples get clipped rather than the base where salt lives
    time_positions  = sorted(range(N_SAMPLES - CUBE_SIZE, -1, -STRIDE_Z))

    inline_positions_local = list(range(
        PROCESS_INLINE_START,
        min(PROCESS_INLINE_END, N_INLINES - CUBE_SIZE + 1),
        STRIDE_INLINE
    ))

    total_candidates = len(inline_positions_local) * len(xline_positions) * len(time_positions)
    print(f"\nGrid positions:")
    print(f"  Inline:     {len(inline_positions_local)} positions (stride {STRIDE_INLINE})")
    print(f"  Crossline:  {len(xline_positions)} positions (stride {STRIDE_CROSSLINE})")
    print(f"  Time/Z:     {len(time_positions)} positions (stride {STRIDE_Z})")
    print(f"  Total candidates: {total_candidates:,}\n")

    # Track counts and skips per split
    cube_counts  = {"train": 0, "val": 0, "test": 0}
    skip_counts  = {"boundary": 0, "flat": 0, "rock_discarded": 0}

    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f_seis, \
         segyio.open(LABEL_PATH,   'r', ignore_geometry=True) as f_label:

        pbar = tqdm(total=total_candidates, desc="Extracting cubes")

        # Iterate over inlines in memory-efficient chunks
        for chunk_start_idx in range(PROCESS_INLINE_START, PROCESS_INLINE_END, CHUNK_SIZE):

            processing_end_idx = min(chunk_start_idx + CHUNK_SIZE, PROCESS_INLINE_END)

            # Load enough extra inlines to satisfy CUBE_SIZE at the chunk boundary
            load_end_idx = min(processing_end_idx + CUBE_SIZE, N_INLINES)

            current_load_size = load_end_idx - chunk_start_idx
            if current_load_size < CUBE_SIZE:
                pbar.update(
                    len([i for i in inline_positions_local
                         if chunk_start_idx <= i < processing_end_idx])
                    * len(xline_positions) * len(time_positions)
                )
                continue

            seismic_chunk = load_chunk_fast(f_seis,  chunk_start_idx, load_end_idx)
            label_chunk   = load_chunk_fast(f_label, chunk_start_idx, load_end_idx)

            # Process inline positions that fall within this chunk
            for global_i_idx in inline_positions_local:

                if not (chunk_start_idx <= global_i_idx < processing_end_idx):
                    continue

                local_i_idx = global_i_idx - chunk_start_idx

                if local_i_idx + CUBE_SIZE > seismic_chunk.shape[0]:
                    pbar.update(len(xline_positions) * len(time_positions))
                    continue

                split = get_split(global_i_idx)
                if split is None:
                    skip_counts["boundary"] += len(xline_positions) * len(time_positions)
                    pbar.update(len(xline_positions) * len(time_positions))
                    continue

                for x_idx in xline_positions:
                    for t_idx in time_positions:

                        # Extract 256 x 256 x 256 cube
                        seismic_cube = seismic_chunk[
                            local_i_idx : local_i_idx + CUBE_SIZE,
                            x_idx       : x_idx       + CUBE_SIZE,
                            t_idx       : t_idx       + CUBE_SIZE
                        ]
                        label_cube = label_chunk[
                            local_i_idx : local_i_idx + CUBE_SIZE,
                            x_idx       : x_idx       + CUBE_SIZE,
                            t_idx       : t_idx       + CUBE_SIZE
                        ]

                        # Skip flat/empty cubes
                        if (seismic_cube.max() - seismic_cube.min()) < 1e-6:
                            skip_counts["flat"] += 1
                            pbar.update(1)
                            continue

                        # Binarize label first (needed for salt ratio check)
                        label_cube = binarize_label(label_cube)

                        # Filter by salt content
                        keep, reason = should_keep_cube(label_cube)
                        if not keep:
                            skip_counts["rock_discarded"] += 1
                            pbar.update(1)
                            continue

                        # Per-cube normalization: clip 1st-99th percentile then z-score
                        seismic_cube = normalize_cube(seismic_cube)

                        # Compute salt percentage for filename
                        salt_pct = float(label_cube.mean())

                        # Metadata array saved alongside cube
                        metadata = np.array([
                            global_i_idx,           # inline start
                            x_idx,                  # crossline start
                            t_idx,                  # time start
                            salt_pct,               # fraction of voxels that are salt
                            int(reason == "salt")   # 1 = salt cube, 0 = sampled rock cube
                        ])

                        # Naming: {survey}_{split}_{inline}_{xline}_{time}_{salt_pct}.npz
                        fname = (
                            f"{SURVEY_NAME}_{split}"
                            f"_{global_i_idx:05d}"
                            f"_{x_idx:05d}"
                            f"_{t_idx:04d}"
                            f"_{salt_pct:.3f}"
                            f".npz"
                        )
                        save_path = os.path.join(OUTPUT_DIR, split, fname)

                        np.savez_compressed(
                            save_path,
                            seismic=seismic_cube,
                            label=label_cube,
                            metadata=metadata
                        )
                        cube_counts[split] += 1
                        pbar.update(1)

            del seismic_chunk
            del label_chunk

        pbar.close()

    # ==========================================
    # SUMMARY
    # ==========================================
    total_saved = sum(cube_counts.values())
    print(f"\n{'='*60}")
    print(f"DONE — {SURVEY_NAME} | Job {JOB_ID}")
    print(f"{'='*60}")
    print(f"Cubes saved:")
    print(f"  Train: {cube_counts['train']:,}")
    print(f"  Val:   {cube_counts['val']:,}")
    print(f"  Test:  {cube_counts['test']:,}")
    print(f"  Total: {total_saved:,}")
    print(f"\nSkipped:")
    print(f"  Boundary (straddles split): {skip_counts['boundary']:,}")
    print(f"  Flat/empty:                 {skip_counts['flat']:,}")
    print(f"  Rock discarded:             {skip_counts['rock_discarded']:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_segy_parallel()
