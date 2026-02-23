import segyio
import numpy as np
import shutil
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Designs a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_to_segy(input_file, output_file, lowcut, highcut):
    """Copies a SEGY file and applies a zero-phase bandpass filter to the traces."""
    print(f"Copying {input_file} to {output_file}...")
    # Copy the file first to preserve all headers perfectly
    shutil.copyfile(input_file, output_file)
    
    # Open the new file in read/write mode ("r+")
    with segyio.open(output_file, "r+", ignore_geometry=True) as f:
        # Get sample rate in seconds
        dt = segyio.tools.dt(f) / 1e6
        fs = 1.0 / dt
        print(f"Sample rate: {dt} s (Fs = {fs} Hz)")
        
        # Design the filter
        b, a = butter_bandpass(lowcut, highcut, fs, order=4)
        
        trace_count = f.tracecount
        print(f"Filtering {trace_count} traces. This may take a moment...")
        
        # Loop through and filter each trace
        for i in range(trace_count):
            tr = f.trace[i]
            
            # Apply zero-phase filter
            filtered_tr = filtfilt(b, a, tr)
            
            # Write the filtered trace back into the file
            # Ensure it is saved as a 32-bit float, which is standard for SEGY
            f.trace[i] = filtered_tr.astype(np.float32)
            
        print(f"Finished processing {output_file}.\n")

# --- Run the processing for both surveys ---

# Define your files (Update these paths to match your actual file names)
file_A_in = "raw_seismic_mckinley.sgy"
file_A_out = "keathley_filtered_3_37Hz.sgy"

file_B_in = "raw_seismic_mississippi.sgy"
file_B_out = "mississippi_filtered_3_37Hz.sgy"

# Filter parameters
low_freq = 3.0   # Hz
high_freq = 37.0 # Hz

# Execute
apply_bandpass_to_segy(file_A_in, file_A_out, low_freq, high_freq)
apply_bandpass_to_segy(file_B_in, file_B_out, low_freq, high_freq)

print("All filtering complete!")