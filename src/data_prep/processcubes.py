import segyio
import numpy as np
import os
from tqdm import tqdm
import random
import sys

"""
python processcubes.py 0 1954
py -m processmiss 1954 3908
py -m processmiss 3908 5862
"""

# ==========================================
# CONFIGURATION
# ==========================================
SEISMIC_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\filtering\mississippi_filtered_3_37Hz.sgy"
LABEL_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelmississippi.sgy"
OUTPUT_DIR = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi256/"

GLOBAL_MEAN = -0.0003
GLOBAL_STD = 1.1787

CUBE_XY = 128
CUBE_Z = 256
STRIDE = 64

SALT_THRESHOLD = 0.05
ROCK_KEEP_PROBABILITY = 0.10

N_INLINES = 5862
N_CROSSLINES = 5289
N_SAMPLES = 1081

#CHUNK_SIZE = 64 No chunks, individual .npz files

# ==========================================
# PARALLEL PROCESSING CONFIGURATION
# ==========================================
if len(sys.argv) == 3:
    PROCESS_INLINE_START = int(sys.argv[1])
    PROCESS_INLINE_END = int(sys.argv[2])
else:
    PROCESS_INLINE_START = 0
    PROCESS_INLINE_END = N_INLINES

print(f"This machine will process inlines {PROCESS_INLINE_START} to {PROCESS_INLINE_END}")

random.seed(42)
np.random.seed(42)

JOB_ID = f"{PROCESS_INLINE_START}_{PROCESS_INLINE_END}"

# ==========================================
# HELPERS
# ==========================================
def normalize_seismic(cube):
    cube = cube.astype(np.float32)
    return (cube - GLOBAL_MEAN) / GLOBAL_STD

def binarize_label(cube):
    return (cube > 0).astype(np.uint8)

def should_keep_cube(label_cube):
    salt_ratio = label_cube.mean()
    if salt_ratio >= SALT_THRESHOLD:
        return True, "salt"
    if random.random() < ROCK_KEEP_PROBABILITY:
        return True, "rock_sampled"
    return False, "rock_discarded"

def load_chunk_fast(segy_file, inline_start_idx, inline_end_idx):
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
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # Define Split Boundaries
    train_end_idx = int(0.70 * N_INLINES)
    val_end_idx = int(0.80 * N_INLINES)

    def get_split(inline_idx):
        inline_end = inline_idx + CUBE_XY
        if 0 <= inline_idx and inline_end <= train_end_idx:
            return "train"
        elif train_end_idx <= inline_idx and inline_end <= val_end_idx:
            return "val"
        elif val_end_idx <= inline_idx and inline_end <= N_INLINES:
            return "test"
        return None

    # Counters per split for unique file naming
    cube_counts = {"train": 0, "val": 0, "test": 0}

    # Pre-calculate positions
    xline_positions = list(range(0, N_CROSSLINES - CUBE_XY + 1, STRIDE))
    time_positions  = list(range(0, N_SAMPLES  - CUBE_Z  + 1, STRIDE))

    inline_positions_local = list(range(
        PROCESS_INLINE_START,
        min(PROCESS_INLINE_END, N_INLINES - CUBE_XY + 1),
        STRIDE
    ))

    total_candidates = len(inline_positions_local) * len(xline_positions) * len(time_positions)
    print(f"\nThis machine's work: {total_candidates:,} candidates\n")

    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f_seis, \
         segyio.open(LABEL_PATH,   'r', ignore_geometry=True) as f_label:

        pbar = tqdm(total=total_candidates, desc="Extracting cubes")

        for chunk_start_idx in range(PROCESS_INLINE_START, PROCESS_INLINE_END, CHUNK_SIZE):

            processing_end_idx = min(chunk_start_idx + CHUNK_SIZE, PROCESS_INLINE_END)
            load_end_idx       = min(processing_end_idx + CUBE_XY, N_INLINES)

            current_load_size = load_end_idx - chunk_start_idx
            if current_load_size < CUBE_XY:
                continue

            seismic_chunk = load_chunk_fast(f_seis,  chunk_start_idx, load_end_idx)
            label_chunk   = load_chunk_fast(f_label, chunk_start_idx, load_end_idx)

            processing_len = processing_end_idx - chunk_start_idx

            for local_i_idx in range(0, processing_len, STRIDE):

                if local_i_idx + CUBE_XY > seismic_chunk.shape[0]:
                    continue

                global_i_idx = chunk_start_idx + local_i_idx

                if global_i_idx >= PROCESS_INLINE_END:
                    continue

                split = get_split(global_i_idx)
                if split is None:
                    pbar.update(len(xline_positions) * len(time_positions))
                    continue

                for x_idx in xline_positions:
                    for t_idx in time_positions:

                        # Extract 128 x 128 x 256 cube
                        seismic_cube = seismic_chunk[
                            local_i_idx : local_i_idx + CUBE_XY,
                            x_idx       : x_idx       + CUBE_XY,
                            t_idx       : t_idx       + CUBE_Z
                        ]

                        label_cube = label_chunk[
                            local_i_idx : local_i_idx + CUBE_XY,
                            x_idx       : x_idx       + CUBE_XY,
                            t_idx       : t_idx       + CUBE_Z
                        ]

                        # Skip near-empty cubes
                        if (seismic_cube.max() - seismic_cube.min()) < 1e-6:
                            pbar.update(1)
                            continue

                        seismic_cube = normalize_seismic(seismic_cube)
                        label_cube   = binarize_label(label_cube)

                        keep, reason = should_keep_cube(label_cube)
                        if not keep:
                            pbar.update(1)
                            continue

                        metadata = np.array([
                            global_i_idx,
                            x_idx,
                            t_idx,
                            label_cube.mean(),
                            int(reason == "salt")
                        ])

                        # Save each cube as its own .npz
                        cube_id   = cube_counts[split]
                        save_path = os.path.join(
                            OUTPUT_DIR, split,
                            f"{JOB_ID}_cube_{cube_id:08d}.npz"
                        )
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

    print(f"\nDONE. Cubes saved — train: {cube_counts['train']}, "
          f"val: {cube_counts['val']}, test: {cube_counts['test']}")

if __name__ == "__main__":
    process_segy_parallel()