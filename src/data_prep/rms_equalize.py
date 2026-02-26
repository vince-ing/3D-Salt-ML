import os
import segyio
import numpy as np
import shutil
import time

def calculate_survey_rms(filename, step=1000):
    """
    Estimates the overall RMS amplitude of a SEGY file by sampling every Nth trace.
    This is drastically faster for 30M+ trace volumes than reading the whole file.
    """
    print(f"Calculating RMS amplitude for {filename} (sampling every {step} traces)...")
    rms_values = []
    
    with segyio.open(filename, "r", ignore_geometry=True) as f:
        trace_count = f.tracecount
        
        # Read a subset of traces to get a representative energy average
        for i in range(0, trace_count, step):
            tr = f.trace[i]
            # Avoid dividing by zero or calculating empty traces
            if np.any(tr):
                # Calculate RMS of this trace: sqrt(mean(trace^2))
                tr_rms = np.sqrt(np.mean(tr**2))
                rms_values.append(tr_rms)
                
    # Return the average RMS of all sampled traces
    global_rms = np.mean(rms_values)
    print(f"--> Estimated RMS: {global_rms:.2f}\n")
    return global_rms

def apply_scalar_to_segy(input_file, output_file, scalar, chunk_size=10000):
    """Multiplies all traces in a SEGY file by a scalar using high-speed chunking."""
    if not os.path.exists(output_file):
        print(f"Copying {input_file} to {output_file}...")
        shutil.copyfile(input_file, output_file)
    else:
        print(f"Found existing file at {output_file}. Skipping copy step...")

    with segyio.open(output_file, "r+", ignore_geometry=True) as f:
        trace_count = f.tracecount
        print(f"Applying scalar of {scalar:.4f} to {trace_count} traces in chunks of {chunk_size}...")
        
        start_time = time.time()
        
        for start_idx in range(0, trace_count, chunk_size):
            end_idx = min(start_idx + chunk_size, trace_count)
            
            # Read chunk, multiply by scalar, and format as 32-bit float
            tr_chunk = np.array(list(f.trace[start_idx:end_idx]))
            scaled_chunk = (tr_chunk * scalar).astype(np.float32)
            
            # Write back to file
            for i, tr_idx in enumerate(range(start_idx, end_idx)):
                f.trace[tr_idx] = scaled_chunk[i]
                
            # Progress reporting
            if (start_idx % (chunk_size * 10)) == 0 and start_idx > 0:
                elapsed = time.time() - start_time
                percent_done = (end_idx / trace_count) * 100
                print(f"Processed {end_idx}/{trace_count} traces ({percent_done:.1f}%) - {elapsed:.1f} sec elapsed")
                
        total_time = (time.time() - start_time) / 60
        print(f"Finished scaling {output_file} in {total_time:.2f} minutes.\n")

# --- Run the Equalization ---

if __name__ == "__main__":
    # We use the ALREADY FILTERED files
    ref_file = "keathley_filtered_3_37Hz.sgy"
    target_file_in = "mississippi_filtered_3_37Hz.sgy"
    target_file_out = "mississippi_filtered_equalized.sgy"

    # 1. Calculate RMS for both surveys
    rms_ref = calculate_survey_rms(ref_file, step=1000)
    rms_target = calculate_survey_rms(target_file_in, step=1000)

    # 2. Calculate the matching scalar
    # If Mississippi is 2x louder than Keathley, this scalar will be 0.5
    matching_scalar = rms_ref / rms_target
    print(f"*** Matching Scalar to apply to Mississippi: {matching_scalar:.4f} ***\n")

    # 3. Apply the scalar to the target volume
    apply_scalar_to_segy(target_file_in, target_file_out, matching_scalar, chunk_size=10000)
    
    print("Energy equalization complete! Datasets are now matched in frequency and amplitude.")