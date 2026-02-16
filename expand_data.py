import os
import glob
import numpy as np
from tqdm import tqdm

# SETUP PATHS
SOURCE_DIR = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mckinley\train" # Where your .npz files are
DEST_DIR   = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mckinley_expand\train" # New fast folder

def explode_dataset(source, dest):
    os.makedirs(dest, exist_ok=True)
    files = glob.glob(os.path.join(source, "*.npz"))
    
    print(f"Exploding {len(files)} files from {source}...")
    
    for fpath in tqdm(files):
        # Load the big compressed file
        try:
            with np.load(fpath) as data:
                seismic = data['seismic']
                labels = data['label']
                
                # Save each cube individually
                base_name = os.path.basename(fpath).replace('.npz', '')
                
                for i in range(len(seismic)):
                    # Save as uncompressed .npy (Super fast to load)
                    # We save seismic and label together in one .npz to keep them paired, 
                    # BUT we don't compress it (savez, not savez_compressed)
                    save_path = os.path.join(dest, f"{base_name}_{i}.npz")
                    np.savez(save_path, seismic=seismic[i], label=labels[i])
                    
        except Exception as e:
            print(f"Error on {fpath}: {e}")

if __name__ == "__main__":
    # 1. Explode Training Data
    explode_dataset(SOURCE_DIR, DEST_DIR)

    VAL_SOURCE = r"C:\Users\ig-gbds\ML_Data\val"
    VAL_DEST   = r"C:\Users\ig-gbds\ML_Data_unpacked\val"
    
    explode_dataset(VAL_SOURCE, VAL_DEST)
    
    # 2. Explode Validation Data (Update paths!)
    # explode_dataset(r"C:\Users\ig-gbds\processed_data\val", r"C:\Users\ig-gbds\processed_data_exploded\val")