import numpy as np
import os
import segyio
from tqdm import tqdm

# --- CONFIGURATION ---------------------------------------------------------
SEISMIC_BIN = "mckinley_normalized.bin"
MASK_SEGY = "labelmckinley.sgy"  # Your label file
MASK_BIN = "mckinley_mask.bin"          # We will create this
OUTPUT_DIR = "training_crops/mckinley"

# Dimensions from your earlier metadata
# INLINES = 5391  <-- CHECK THIS: Is your mask exactly the same size?
# CROSSLINES = 7076
# SAMPLES = 1001
SHAPE = (5391, 7076, 1001) 

CROP_SIZE = (128, 128, 128)  # (Inlines, Crosslines, Samples)
STRIDE = (64, 64, 64)        # 50% overlap
KEEP_EMPTY_RATIO = 0.10      # Keep 10% of background-only crops
# ---------------------------------------------------------------------------

def convert_mask_to_bin(segy_path, bin_path):
    """
    Converts a SEG-Y mask to a simple binary file (0s and 1s).
    Assumes mask values are already 0 or 1.
    """
    if os.path.exists(bin_path):
        print(f"Mask binary {bin_path} already exists. Skipping conversion.")
        return

    print(f"Converting Mask SEG-Y to Binary: {segy_path}")
    with open(bin_path, "wb") as f_out:
        with segyio.open(segy_path, "r", ignore_geometry=True) as src:
            n_traces = src.tracecount
            for i in range(0, n_traces, 5000):
                end = min(i + 5000, n_traces)
                # Read raw, convert to int8 to save space (0 or 1)
                # If your mask uses 255 for salt, change this logic!
                traces = src.trace.raw[i:end]
                
                # Binarize just in case (e.g., if salt is 255 or arbitrary ID)
                # Any value > 0 becomes 1
                traces_bin = (traces > 0).astype(np.int8)
                
                traces_bin.tofile(f_out)
                print(f"  Processed {end}/{n_traces} traces...", end="\r")
    print("\nMask conversion done.")

def generate_crops():
    # 1. Ensure Mask Binary Exists
    convert_mask_to_bin(MASK_SEGY, MASK_BIN)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Memory Map the Volumes (Instant load, low RAM)
    print("Mapping volumes...")
    # Seismic is float32 (4 bytes)
    seismic_vol = np.memmap(SEISMIC_BIN, dtype='float32', mode='r', shape=SHAPE)
    
    # Mask is int8 (1 byte) - much smaller!
    mask_vol = np.memmap(MASK_BIN, dtype='int8', mode='r', shape=SHAPE)

    # 3. Define Sliding Window Ranges
    d_il, d_xl, d_z = CROP_SIZE
    s_il, s_xl, s_z = STRIDE
    n_il, n_xl, n_z = SHAPE

    # Calculate steps
    il_steps = range(0, n_il - d_il + 1, s_il)
    xl_steps = range(0, n_xl - d_xl + 1, s_xl)
    z_steps  = range(0, n_z  - d_z  + 1, s_z)

    total_cubes = len(il_steps) * len(xl_steps) * len(z_steps)
    print(f"Scanning volume. Max potential crops: {total_cubes}")

    saved_count = 0
    discarded_count = 0

    # 4. The Loop
    with tqdm(total=total_cubes, desc="Cropping") as pbar:
        for il in il_steps:
            for xl in xl_steps:
                for z in z_steps:
                    
                    # Cut the mask first (it's smaller/faster to check)
                    m_crop = mask_vol[il:il+d_il, xl:xl+d_xl, z:z+d_z]
                    
                    # Logic: Does this crop have salt?
                    has_salt = np.any(m_crop == 1)
                    
                    keep = False
                    if has_salt:
                        keep = True
                    elif np.random.rand() < KEEP_EMPTY_RATIO:
                        keep = True
                    
                    if keep:
                        # Now cut the seismic
                        s_crop = seismic_vol[il:il+d_il, xl:xl+d_xl, z:z+d_z]
                        
                        # Save compressed .npz
                        # Naming convention: survey_IL_XL_Z.npz
                        fname = f"mck_{il}_{xl}_{z}.npz"
                        save_path = os.path.join(OUTPUT_DIR, fname)
                        
                        # Use float16 for seismic to save disk space? 
                        # float32 is safer for training, let's stick to float32.
                        np.savez_compressed(
                            save_path, 
                            seismic=s_crop,  # Already normalized float32
                            mask=m_crop      # int8
                        )
                        saved_count += 1
                    else:
                        discarded_count += 1
                    
                    pbar.update(1)

    print("\n--- Summary ---")
    print(f"Saved Cubes: {saved_count}")
    print(f"Discarded:   {discarded_count}")
    if saved_count + discarded_count > 0:
        ratio = saved_count / (saved_count + discarded_count)
        print(f"Yield Rate:  {ratio:.2%}")

if __name__ == "__main__":
    generate_crops()
