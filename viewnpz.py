import numpy as np
import os
from pathlib import Path

def check_all_cubes():
    """Check salt distribution across all extracted cubes."""
    
    for split in ["train", "val", "test"]:
        cube_dir = Path(f"processed_data/mckinley/{split}")
        
        if not cube_dir.exists():
            print(f"{split.upper()}: Directory not found")
            continue
        
        cube_files = sorted(cube_dir.glob("*.npz"))
        
        if len(cube_files) == 0:
            print(f"{split.upper()}: No cubes found")
            continue
        
        print(f"\n{split.upper()}: Analyzing {len(cube_files)} cubes...")
        
        salt_ratios = []
        salt_cubes_count = 0
        rock_cubes_count = 0
        
        for cube_file in cube_files:
            data = np.load(cube_file)
            label = data['label']
            metadata = data['metadata']
            
            salt_ratio = label.mean()
            salt_ratios.append(salt_ratio)
            
            # Check if cube has any salt
            if salt_ratio >= 0.05:
                salt_cubes_count += 1
            else:
                rock_cubes_count += 1
        
        # Statistics
        salt_ratios = np.array(salt_ratios)
        
        print(f"  Total cubes: {len(cube_files)}")
        print(f"  Salt cubes (≥5% salt): {salt_cubes_count}")
        print(f"  Rock cubes (<5% salt): {rock_cubes_count}")
        print(f"\n  Salt ratio distribution:")
        print(f"    Min:  {salt_ratios.min():.4f} ({salt_ratios.min()*100:.2f}%)")
        print(f"    Max:  {salt_ratios.max():.4f} ({salt_ratios.max()*100:.2f}%)")
        print(f"    Mean: {salt_ratios.mean():.4f} ({salt_ratios.mean()*100:.2f}%)")
        print(f"    Median: {np.median(salt_ratios):.4f} ({np.median(salt_ratios)*100:.2f}%)")
        
        # Count how many cubes have ANY salt
        cubes_with_salt = np.sum(salt_ratios > 0)
        print(f"\n  Cubes with ANY salt (>0%): {cubes_with_salt} ({cubes_with_salt/len(cube_files)*100:.1f}%)")
        
        # Histogram
        print(f"\n  Salt ratio histogram:")
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        hist, _ = np.histogram(salt_ratios, bins=bins)
        for i in range(len(bins)-1):
            print(f"    {bins[i]*100:5.1f}%-{bins[i+1]*100:5.1f}%: {hist[i]:6d} cubes")

if __name__ == "__main__":
    check_all_cubes()
