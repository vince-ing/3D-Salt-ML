import os
import sys
import random
import numpy as np
from tqdm import tqdm

try:
    import segyio
except ImportError:
    sys.exit("segyio not found. Run: pip install segyio tqdm")

# ============================================================
# 1. CONFIGURATION
# ============================================================
K_SEIS = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\raw\raw_seismic_keathley.sgy"
K_LBL  = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelkeathley_seafloor.sgy"
M_SEIS = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\raw\raw_seismic_mississippi.sgy"
M_LBL  = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelmississippi_seafloor.sgy"

OUT_DIR = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\salt3dnetoutstyle"

PATCH_I, PATCH_X, PATCH_S = 100, 128, 128
STRIDE_I, STRIDE_X, STRIDE_S = 50, 64, 64 # Overlap to ensure we find enough dense patches

# Target Quotas (Sums to 495)
QUOTAS_K = {'salt': 124, 'boundary': 74, 'rock': 50}
QUOTAS_M = {'salt': 124, 'boundary': 74, 'rock': 49}

# Parameter Thresholds
MIN_SALT_FOR_SALT_CUBE = 0.540   
MAX_IMPURITY_FOR_ROCK = 0.10    
MIN_SALT_FOR_BOUNDARY = 0.02    
MIN_ROCK_FOR_BOUNDARY = 0.02    

# ============================================================
# 2. HELPERS & EXTRACTION LOGIC
# ============================================================
def get_starts(size, max_val, stride):
    if max_val < size: return []
    starts = list(range(0, max_val - size + 1, stride))
    if not starts or starts[-1] != max_val - size:
        starts.append(max_val - size)
    return starts

def categorize_patch(l_patch):
    valid_traces = (l_patch == 2).any(axis=2)
    invalid_traces_3d = ~valid_traces[:, :, None]
    mask_to_blank = (l_patch == 0) & invalid_traces_3d
    l_patch[mask_to_blank] = 3

    total_voxels = l_patch.size
    salt_ratio = np.count_nonzero(l_patch == 1) / total_voxels
    rock_ratio = np.count_nonzero(l_patch == 0) / total_voxels

    if salt_ratio >= MIN_SALT_FOR_SALT_CUBE:
        return "salt", l_patch
    elif rock_ratio >= (1.0 - MAX_IMPURITY_FOR_ROCK):
        return "rock", l_patch
    elif salt_ratio >= MIN_SALT_FOR_BOUNDARY and rock_ratio >= MIN_ROCK_FOR_BOUNDARY:
        return "boundary", l_patch
    else:
        return "empty", l_patch

def sequential_quota_extraction(seis_path, lbl_path, quotas, survey_name):
    target_n = sum(quotas.values())
    print(f"\nScanning {survey_name} Sequentially for {target_n} cubes...")
    
    f_seis = segyio.open(seis_path, "r", ignore_geometry=False)
    f_lbl  = segyio.open(lbl_path, "r", ignore_geometry=False)

    NI, NX, NS = len(f_seis.ilines), len(f_seis.xlines), len(f_seis.samples)

    res_seis = np.zeros((target_n, PATCH_I, PATCH_X, PATCH_S), dtype=np.float32)
    res_lbl  = np.zeros((target_n, PATCH_I, PATCH_X, PATCH_S), dtype=np.float32)
    
    current_quotas = {k: 0 for k in quotas.keys()}
    total_kept = 0
    failed_attempts = 0  
    PATIENCE_LIMIT = 2000 

    i_starts = get_starts(PATCH_I, NI, STRIDE_I)
    x_starts = get_starts(PATCH_X, NX, STRIDE_X)
    s_starts = get_starts(PATCH_S, NS, STRIDE_S)

    lbl_slab  = np.zeros((PATCH_I, NX, NS), dtype=np.uint8)
    seis_slab = np.zeros((PATCH_I, NX, NS), dtype=np.float32)
    
    lbl_loaded_start = -1
    seis_loaded_start = -1 # Tracks if we have the seismic data for the current block

    pbar = tqdm(total=target_n, desc=f"Finding valid cubes")

    for i in i_starts:
        if total_kept >= target_n:
            break 

        pbar.set_postfix({"Scouting Inline": i})

        # 1. SCOUTING: Load ONLY the Labels (Extremely Fast)
        if lbl_loaded_start == -1 or i >= lbl_loaded_start + PATCH_I or i < lbl_loaded_start:
            for offset in range(PATCH_I):
                lbl_slab[offset]  = f_lbl.iline[f_lbl.ilines[i + offset]]
        else:
            shift = i - lbl_loaded_start
            lbl_slab[:-shift]  = lbl_slab[shift:]
            for offset in range(PATCH_I - shift, PATCH_I):
                lbl_slab[offset]  = f_lbl.iline[f_lbl.ilines[i + offset]]
        
        lbl_loaded_start = i

        # Quick Check: If the entire 100-inline block has NO salt and NO water, it is pure dead space. Skip it!
        unique_in_slab = np.unique(lbl_slab)
        if 1 not in unique_in_slab and 2 not in unique_in_slab:
            continue

        random.shuffle(x_starts)
        random.shuffle(s_starts)

        for x in x_starts:
            if total_kept >= target_n: break
            for s in s_starts:
                if total_kept >= target_n: break
                
                raw_l_patch = lbl_slab[:, x:x+PATCH_X, s:s+PATCH_S].copy()
                category, processed_l_patch = categorize_patch(raw_l_patch)
                
                keep_patch = False
                
                if category in quotas and current_quotas[category] < quotas[category]:
                    keep_patch = True
                    failed_attempts = 0
                elif failed_attempts > PATIENCE_LIMIT and category in quotas and category != 'empty':
                    keep_patch = True
                    failed_attempts = 0
                    tqdm.write(f"[Geographic Bottleneck] Accepting extra '{category}'.")
                else:
                    failed_attempts += 1

                # 2. LAZY LOADING: Only load Seismic if we found a patch we actually want
                if keep_patch:
                    if seis_loaded_start != i:
                        # We need the seismic data for this block, and we don't have it yet! Load it now.
                        for offset in range(PATCH_I):
                            seis_slab[offset] = f_seis.iline[f_seis.ilines[i + offset]]
                        seis_loaded_start = i
                        
                    s_patch = seis_slab[:, x:x+PATCH_X, s:s+PATCH_S]
                    res_seis[total_kept] = s_patch
                    res_lbl[total_kept]  = processed_l_patch.astype(np.float32)
                    
                    current_quotas[category] += 1
                    total_kept += 1
                    pbar.update(1)

    pbar.close()
    f_seis.close()
    f_lbl.close()
    
    if total_kept < target_n:
        missing = {k: quotas[k] - current_quotas[k] for k in quotas if quotas[k] - current_quotas[k] > 0}
        print(f"\n[WARNING] Hit end of file. Still missing: {missing}")
        res_seis = res_seis[:total_kept]
        res_lbl = res_lbl[:total_kept]
    else:
        print(f"  -> Finished {survey_name}. Final makeup: {current_quotas}")

    return res_seis, res_lbl

# ============================================================
# 3. MAIN AGGREGATION
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    k_seis, k_lbl = sequential_quota_extraction(K_SEIS, K_LBL, QUOTAS_K, "Keathley")
    m_seis, m_lbl = sequential_quota_extraction(M_SEIS, M_LBL, QUOTAS_M, "Mississippi Canyon")
    
    print("\nConcatenating and Shuffling Datasets...")
    final_seis = np.concatenate((k_seis, m_seis), axis=0)
    final_lbl  = np.concatenate((k_lbl, m_lbl), axis=0)
    
    indices = np.arange(final_seis.shape[0])
    

    np.random.shuffle(indices)
    final_seis = final_seis[indices]
    final_lbl  = final_lbl[indices]
    
    print(f"Final Shapes -> Seismic: {final_seis.shape}, Labels: {final_lbl.shape}")
    
    seis_out = os.path.join(OUT_DIR, "samples.bin")
    lbl_out  = os.path.join(OUT_DIR, "labels.bin")
    
    print("Writing to binary files...")
    final_seis.tofile(seis_out)
    final_lbl.tofile(lbl_out)
    
    print("\nProcess Complete! Ready for TensorFlow training.")

if __name__ == "__main__":
    main()