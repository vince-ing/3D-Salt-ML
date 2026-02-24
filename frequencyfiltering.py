import os
import segyio
import numpy as np
import shutil
import time
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Designs a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_to_segy_fast(input_file, output_file, lowcut, highcut, chunk_size=10000):
    """Applies a bandpass filter to a SEGY file using high-speed chunking."""
    
    # 1. Check if the output file already exists to avoid redundant copying
    if not os.path.exists(output_file):
        print(f"Copying {input_file} to {output_file} (this may take a few minutes for large files)...")
        shutil.copyfile(input_file, output_file)
    else:
        print(f"Found existing file at {output_file}. Skipping copy step...")
    
    # 2. Open the file in read/write mode ("r+")
    with segyio.open(output_file, "r+", ignore_geometry=True) as f:
        dt = segyio.tools.dt(f) / 1e6
        fs = 1.0 / dt
        b, a = butter_bandpass(lowcut, highcut, fs, order=4)
        
        trace_count = f.tracecount
        print(f"Sample rate: {dt} s (Fs = {fs} Hz)")
        print(f"Filtering {trace_count} traces in chunks of {chunk_size}...")
        
        start_time = time.time()
        
        # Loop through traces in large chunks
        for start_idx in range(0, trace_count, chunk_size):
            end_idx = min(start_idx + chunk_size, trace_count)
            
            # --- THE FIX ---
            # 1. Read block of traces and cast the generator into a 2D numpy array
            tr_chunk = np.array(list(f.trace[start_idx:end_idx]))
            
            # 2. Filter the entire block at once along the time axis (axis=1)
            filtered_chunk = filtfilt(b, a, tr_chunk, axis=1).astype(np.float32)
            
            # 3. Write the block back to the file iteratively to ensure safety
            for i, tr_idx in enumerate(range(start_idx, end_idx)):
                f.trace[tr_idx] = filtered_chunk[i]
            
            # --- Progress reporting ---
            if (start_idx % (chunk_size * 10)) == 0 and start_idx > 0:
                elapsed = time.time() - start_time
                percent_done = (end_idx / trace_count) * 100
                print(f"Processed {end_idx}/{trace_count} traces ({percent_done:.1f}%) - {elapsed:.1f} sec elapsed")
                
        total_time = (time.time() - start_time) / 60
        print(f"Finished processing {output_file} in {total_time:.2f} minutes.\n")


# --- Run the processing for both surveys ---

if __name__ == "__main__":
    # Define your files (Update these paths to match your actual file names)
    file_A_in = "raw_seismic_mckinley.sgy"
    file_A_out = "keathley_filtered_3_37Hz.sgy"

    file_B_in = "raw_seismic_mississippi.sgy"
    file_B_out = "mississippi_filtered_3_37Hz.sgy"

    # Filter parameters
    low_freq = 3.0   # Hz
    high_freq = 37.0 # Hz

    # chunk_size=10000 is a good balance for RAM vs Speed.
    #apply_bandpass_to_segy_fast(file_A_in, file_A_out, low_freq, high_freq, chunk_size=10000)
    apply_bandpass_to_segy_fast(file_B_in, file_B_out, low_freq, high_freq, chunk_size=10000)
    
    print("All filtering complete!")