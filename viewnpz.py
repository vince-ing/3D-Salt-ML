import numpy as np
from pathlib import Path
import zipfile

def check_all_cubes_batched():
    for split in ["train", "val", "test"]:
        cube_dir = Path(f"processed_data/mckinley/{split}")
        
        if not cube_dir.exists():
            print(f"{split.upper()}: Directory not found")
            continue
        
        batch_files = sorted(cube_dir.glob("*.npz"))
        print(f"\n{split.upper()}: Analyzing {len(batch_files)} batch files...")
        
        salt_ratios = []
        salt_cubes_count = 0
        rock_cubes_count = 0
        total_cubes = 0
        bad_files = 0
        
        for batch_file in batch_files:
            try:
                data = np.load(batch_file)
                labels = data["label"]   # (B, D, H, W)
            except (zipfile.BadZipFile, ValueError, OSError) as e:
                print(f"  ⚠️ Skipping corrupted file: {batch_file.name}")
                bad_files += 1
                continue
            
            B = labels.shape[0]
            total_cubes += B
            
            batch_ratios = labels.reshape(B, -1).mean(axis=1)
            salt_ratios.extend(batch_ratios.tolist())
            
            salt_cubes_count += (batch_ratios >= 0.05).sum()
            rock_cubes_count += (batch_ratios < 0.05).sum()
        
        if total_cubes == 0:
            print("  ❌ No valid cubes found.")
            continue
        
        salt_ratios = np.array(salt_ratios)
        
        print(f"  Total cubes: {total_cubes}")
        print(f"  Salt cubes (≥5% salt): {salt_cubes_count}")
        print(f"  Rock cubes (<5% salt): {rock_cubes_count}")
        print(f"  Bad batch files skipped: {bad_files}")
        
        print(f"\n  Salt ratio stats:")
        print(f"    Min:    {salt_ratios.min():.4f}")
        print(f"    Max:    {salt_ratios.max():.4f}")
        print(f"    Mean:   {salt_ratios.mean():.4f}")
        print(f"    Median: {np.median(salt_ratios):.4f}")

if __name__ == "__main__":
    check_all_cubes_batched()