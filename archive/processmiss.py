import segyio
import numpy as np
import os
from tqdm import tqdm
import random
import sys

"""
python processmiss.py 0 1954
py -m processmiss 1954 3908
py -m processmiss 3908 5862
"""

# ==========================================
# CONFIGURATION
# ==========================================
SEISMIC_PATH = "raw_seismic_mississippi.sgy"
LABEL_PATH = "labelmississippi.sgy"
OUTPUT_DIR = "processed_data/mississippi/"

GLOBAL_MEAN = -0.0003
GLOBAL_STD = 1.1787

CUBE_SIZE = 128
STRIDE = 64

SALT_THRESHOLD = 0.05
ROCK_KEEP_PROBABILITY = 0.10

N_INLINES = 5862
N_CROSSLINES = 5289
N_SAMPLES = 1081

CHUNK_SIZE = 64

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
# BATCHED WRITING
# ==========================================
class BatchedWriter:
    def __init__(self, batch_size=64, job_id="job"):
        self.batch_size = batch_size
        self.job_id = job_id
        self.batches = {"train": [], "val": [], "test": []}
        self.counts = {"train": 0, "val": 0, "test": 0}

    def add_cube(self, split, seismic, label, metadata):
        self.batches[split].append({
            "seismic": seismic,
            "label": label,
            "metadata": metadata
        })

        if len(self.batches[split]) >= self.batch_size:
            self.flush_batch(split)

    def flush_batch(self, split):
        if len(self.batches[split]) == 0:
            return

        batch_id = self.counts[split] // self.batch_size
        save_path = os.path.join(
            OUTPUT_DIR, split,
            f"{self.job_id}_batch_{batch_id:06d}.npz"
        )

        seismic_batch = np.stack([c["seismic"] for c in self.batches[split]])
        label_batch = np.stack([c["label"] for c in self.batches[split]])
        metadata_batch = np.stack([c["metadata"] for c in self.batches[split]])

        np.savez_compressed(
            save_path,
            seismic=seismic_batch,
            label=label_batch,
            metadata=metadata_batch
        )

        self.counts[split] += len(self.batches[split])
        self.batches[split].clear()

    def flush_all(self):
        for split in ["train", "val", "test"]:
            self.flush_batch(split)

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
        inline_end = inline_idx + CUBE_SIZE
        if 0 <= inline_idx and inline_end <= train_end_idx:
            return "train"
        elif train_end_idx <= inline_idx and inline_end <= val_end_idx:
            return "val"
        elif val_end_idx <= inline_idx and inline_end <= N_INLINES:
            return "test"
        return None

    writer = BatchedWriter(batch_size=64, job_id=JOB_ID)

    # Pre-calculate positions
    xline_positions = list(range(0, N_CROSSLINES - CUBE_SIZE + 1, STRIDE))
    time_positions = list(range(0, N_SAMPLES - CUBE_SIZE + 1, STRIDE))
    
    # Calculate strictly valid start positions for this machine
    inline_positions_local = list(range(
        PROCESS_INLINE_START, 
        min(PROCESS_INLINE_END, N_INLINES - CUBE_SIZE + 1), 
        STRIDE
    ))

    total_candidates = len(inline_positions_local) * len(xline_positions) * len(time_positions)
    print(f"\nThis machine's work: {total_candidates:,} candidates\n")

    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f_seis, \
         segyio.open(LABEL_PATH, 'r', ignore_geometry=True) as f_label:
        
        pbar = tqdm(total=total_candidates, desc="Extracting cubes")

        # === THE FIX IS IN THIS LOOP ===
        # We iterate through "Processing Windows"
        for chunk_start_idx in range(PROCESS_INLINE_START, PROCESS_INLINE_END, CHUNK_SIZE):
            
            # 1. Determine the range of START POSITIONS we handle in this pass
            #    We are responsible for cubes starting at [chunk_start, chunk_end)
            processing_end_idx = min(chunk_start_idx + CHUNK_SIZE, PROCESS_INLINE_END)
            
            # 2. Determine how much data we need to LOAD to support those cubes.
            #    To process a cube starting at 'processing_end_idx - 1', 
            #    we need data extending to 'processing_end_idx - 1 + CUBE_SIZE'.
            #    So we create an Overlap/Buffer.
            load_end_idx = min(processing_end_idx + CUBE_SIZE, N_INLINES)

            current_load_size = load_end_idx - chunk_start_idx
            
            # Skip if we are at the very end and can't even form one cube
            if current_load_size < CUBE_SIZE:
                continue

            # print(f"DEBUG: Processing Starts {chunk_start_idx}-{processing_end_idx} | Loading Data {chunk_start_idx}-{load_end_idx}")

            # 3. Load the EXTENDED chunk (with overlap)
            seismic_chunk = load_chunk_fast(f_seis, chunk_start_idx, load_end_idx)
            label_chunk = load_chunk_fast(f_label, chunk_start_idx, load_end_idx)

            # 4. Iterate strictly through the processing window
            #    We only create cubes that START in our assigned window.
            processing_len = processing_end_idx - chunk_start_idx
            
            for local_i_idx in range(0, processing_len, STRIDE):
                
                # Verify we have enough data (boundary check)
                if local_i_idx + CUBE_SIZE > seismic_chunk.shape[0]:
                    continue

                global_i_idx = chunk_start_idx + local_i_idx
                
                # Double check we don't exceed the machine's assigned task
                if global_i_idx >= PROCESS_INLINE_END:
                    continue

                split = get_split(global_i_idx)
                if split is None:
                    pbar.update(len(xline_positions) * len(time_positions))
                    continue

                # Inner Loops (Xline / Time)
                for x_idx in xline_positions:
                    for t_idx in time_positions:
                        
                        # Slicing from the loaded chunk
                        seismic_cube = seismic_chunk[
                            local_i_idx : local_i_idx + CUBE_SIZE,
                            x_idx : x_idx + CUBE_SIZE,
                            t_idx : t_idx + CUBE_SIZE
                        ]
                        
                        label_cube = label_chunk[
                            local_i_idx : local_i_idx + CUBE_SIZE,
                            x_idx : x_idx + CUBE_SIZE,
                            t_idx : t_idx + CUBE_SIZE
                        ]

                        # --- (Standard Checks) ---
                        if (seismic_cube.max() - seismic_cube.min()) < 1e-6:
                            pbar.update(1)
                            continue

                        seismic_cube = normalize_seismic(seismic_cube)
                        label_cube = binarize_label(label_cube)

                        keep, reason = should_keep_cube(label_cube)
                        if not keep:
                            pbar.update(1)
                            continue

                        writer.add_cube(
                            split,
                            seismic_cube,
                            label_cube,
                            np.array([
                                global_i_idx, 
                                x_idx, 
                                t_idx,
                                label_cube.mean(),
                                int(reason == "salt")
                            ])
                        )
                        pbar.update(1)
            
            # Clean up memory immediately
            del seismic_chunk
            del label_chunk

        pbar.close()
        print("\nFlushing remaining batches...")
        writer.flush_all()

    print("\nDONE.")

if __name__ == "__main__":
    process_segy_parallel()