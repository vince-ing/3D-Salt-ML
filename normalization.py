import segyio
import numpy as np
import os

def normalize_segy(input_filename, output_filename, chunk_size=1000):
    """
    Reads a SEG-Y, calculates global stats (Mean/Std) incrementally,
    and writes a normalized float32 binary volume.
    """
    print(f"Processing: {input_filename}")
    
    # --- PASS 1: Calculate Mean and Std ---
    # We use Welford's algorithm or simple sum accumulation for stability
    # Here we use simple accumulation for speed on massive files
    
    sum_val = 0.0
    sum_sq_val = 0.0
    count = 0
    
    with segyio.open(input_filename, "r", ignore_geometry=True) as src:
        n_traces = src.tracecount
        print(f"  - Traces: {n_traces}")
        
        # Process in chunks to save RAM
        for i in range(0, n_traces, chunk_size):
            # Read a block of traces
            end = min(i + chunk_size, n_traces)
            traces = src.trace.raw[i:end] # Returns a numpy array (chunk, samples)
            
            # Flatten to 1D for stats
            data_chunk = traces.flatten()
            
            sum_val += np.sum(data_chunk, dtype=np.float64)
            sum_sq_val += np.sum(data_chunk**2, dtype=np.float64)
            count += data_chunk.size
            
            if i % 10000 == 0:
                print(f"    Scanning trace {i}/{n_traces}...", end="\r")

    # Calculate Global Stats
    global_mean = sum_val / count
    global_variance = (sum_sq_val / count) - (global_mean ** 2)
    global_std = np.sqrt(global_variance)
    
    print(f"\n  - Global Mean: {global_mean:.4f}")
    print(f"  - Global Std:  {global_std:.4f}")
    
    # --- PASS 2: Normalize and Write ---
    print(f"  - Writing normalized volume to: {output_filename}")
    
    # We will write this as a simple binary float32 file (Raw format)
    # This is much faster to read for the cropper than SEG-Y
    with open(output_filename, "wb") as f_out:
         with segyio.open(input_filename, "r", ignore_geometry=True) as src:
            for i in range(0, n_traces, chunk_size):
                end = min(i + chunk_size, n_traces)
                traces = src.trace.raw[i:end].astype(np.float32)
                
                # Apply Normalization
                traces = (traces - global_mean) / global_std
                
                # Write to disk
                traces.tofile(f_out)
                
                if i % 10000 == 0:
                    print(f"    Writing trace {i}/{n_traces}...", end="\r")
                    
    print("\nDone!")

# --- Usage ---
# normalize_segy("Survey_A.sgy", "Survey_A_Normalized.bin")
# normalize_segy("Survey_B.sgy", "Survey_B_Normalized.bin")
normalize_segy("raw_seismic_mckinley.sgy", "mckinley_normalized.bin")
