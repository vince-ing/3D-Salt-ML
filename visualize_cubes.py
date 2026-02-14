import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "processed_data/mckinley/train"

# How many batches to sample (for speed)
MAX_BATCHES = 492  # Sample 50 batches across the survey


def create_survey_overview():
    """
    Create a clean 2D map showing where salt is located in the survey.
    Much clearer than 3D visualization.
    """
    # Find all batch files
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_batch_*.npz")))
    
    if not files:
        print(f"❌ No batch files found in {DATA_DIR}")
        return
    
    print("="*70)
    print("SURVEY SALT DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"Found {len(files)} batch files")
    print(f"Sampling {min(MAX_BATCHES, len(files))} batches across survey...")
    print("="*70 + "\n")
    
    # Sample files evenly across survey
    step = max(1, len(files) // MAX_BATCHES)
    sampled_files = files[::step]
    
    # Storage for cube locations and properties
    cube_data = []
    
    print("Loading cube metadata...")
    for filepath in tqdm(sampled_files):
        try:
            data = np.load(filepath)
            meta_batch = data["metadata"]
            
            for meta in meta_batch:
                inline_idx = meta[0]
                xline_idx = meta[1]
                time_idx = meta[2]
                salt_ratio = meta[3]
                
                cube_data.append({
                    'inline': inline_idx,
                    'xline': xline_idx,
                    'time': time_idx,
                    'salt_ratio': salt_ratio
                })
        except:
            continue
    
    if len(cube_data) == 0:
        print("❌ No data loaded!")
        return
    
    print(f"✅ Loaded {len(cube_data):,} cube locations\n")
    
    # Convert to arrays
    inlines = np.array([c['inline'] for c in cube_data])
    xlines = np.array([c['xline'] for c in cube_data])
    times = np.array([c['time'] for c in cube_data])
    salt_ratios = np.array([c['salt_ratio'] for c in cube_data])
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # ==========================================
    # PLOT 1: Map View (Inline vs Crossline)
    # ==========================================
    ax1 = fig.add_subplot(131)
    
    # Color by salt ratio
    scatter = ax1.scatter(xlines, inlines, c=salt_ratios, 
                         cmap='RdYlGn_r', s=20, alpha=0.6,
                         vmin=0, vmax=1, edgecolors='none')
    
    ax1.set_xlabel('Crossline Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inline Index', fontsize=12, fontweight='bold')
    ax1.set_title('Map View: Salt Distribution\n(Red = More Salt)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Salt Ratio', fontsize=10)
    
    # Add survey bounds
    ax1.axhline(y=inlines.min(), color='white', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axhline(y=inlines.max(), color='white', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axvline(x=xlines.min(), color='white', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axvline(x=xlines.max(), color='white', linestyle='--', linewidth=2, alpha=0.5)
    
    # ==========================================
    # PLOT 2: Depth Section (Inline vs Time)
    # ==========================================
    ax2 = fig.add_subplot(132)
    
    scatter2 = ax2.scatter(inlines, times, c=salt_ratios,
                          cmap='RdYlGn_r', s=20, alpha=0.6,
                          vmin=0, vmax=1, edgecolors='none')
    
    ax2.set_xlabel('Inline Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time Sample', fontsize=12, fontweight='bold')
    ax2.set_title('Depth Section: Salt vs Time\n(Deeper = Higher Time)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Time increases downward
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Salt Ratio', fontsize=10)
    
    # ==========================================
    # PLOT 3: Histogram of Salt Ratios
    # ==========================================
    ax3 = fig.add_subplot(133)
    
    # Create histogram
    counts, bins, patches = ax3.hist(salt_ratios, bins=50, 
                                     color='steelblue', edgecolor='black',
                                     alpha=0.7)
    
    # Color bars by salt ratio
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.RdYlGn_r(bins[i]))
    
    ax3.set_xlabel('Salt Ratio', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Cubes', fontsize=12, fontweight='bold')
    ax3.set_title('Salt Distribution Histogram', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_salt = salt_ratios.mean()
    median_salt = np.median(salt_ratios)
    
    ax3.axvline(mean_salt, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_salt:.2%}')
    ax3.axvline(median_salt, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {median_salt:.2%}')
    ax3.legend()
    
    # Add text box with statistics
    stats_text = (
        f"Survey Statistics:\n"
        f"Cubes analyzed: {len(cube_data):,}\n"
        f"Mean salt: {mean_salt:.1%}\n"
        f"Median salt: {median_salt:.1%}\n"
        f"Inline range: {inlines.min():.0f} - {inlines.max():.0f}\n"
        f"Xline range: {xlines.min():.0f} - {xlines.max():.0f}\n"
        f"Time range: {times.min():.0f} - {times.max():.0f}"
    )
    
    fig.text(0.98, 0.02, stats_text, 
            fontsize=10, 
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('survey_salt_overview.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: survey_salt_overview.png")
    plt.show()


def create_salt_density_heatmap():
    """
    Create a 2D heatmap showing salt density across the survey.
    Clearest way to see where salt is located.
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_batch_*.npz")))
    
    if not files:
        print(f"❌ No batch files found")
        return
    
    print("\nCreating salt density heatmap...")
    
    # Sample files
    step = max(1, len(files) // MAX_BATCHES)
    sampled_files = files[::step]
    
    # Load data
    cube_data = []
    for filepath in tqdm(sampled_files, desc="Loading"):
        try:
            data = np.load(filepath)
            meta_batch = data["metadata"]
            for meta in meta_batch:
                cube_data.append({
                    'inline': int(meta[0]),
                    'xline': int(meta[1]),
                    'salt_ratio': meta[3]
                })
        except:
            continue
    
    if len(cube_data) == 0:
        return
    
    # Create 2D grid
    inlines = np.array([c['inline'] for c in cube_data])
    xlines = np.array([c['xline'] for c in cube_data])
    salt_ratios = np.array([c['salt_ratio'] for c in cube_data])
    
    # Grid resolution
    inline_bins = 100
    xline_bins = 100
    
    # Create 2D histogram (average salt in each bin)
    H, xedges, yedges = np.histogram2d(
        xlines, inlines, 
        bins=[xline_bins, inline_bins],
        weights=salt_ratios
    )
    
    counts, _, _ = np.histogram2d(
        xlines, inlines,
        bins=[xline_bins, inline_bins]
    )
    
    # Average salt per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        salt_density = H / counts
        salt_density[~np.isfinite(salt_density)] = 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(salt_density.T, 
                   cmap='RdYlGn_r',
                   aspect='auto',
                   origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   vmin=0, vmax=1,
                   interpolation='bilinear')
    
    ax.set_xlabel('Crossline Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inline Index', fontsize=14, fontweight='bold')
    ax.set_title('Salt Density Heatmap\n(Brighter Red = More Salt)', 
                fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Salt Ratio', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(True, color='white', alpha=0.3, linestyle='--')
    
    # Add statistics
    stats_text = (
        f"Survey Coverage:\n"
        f"Sampled cubes: {len(cube_data):,}\n"
        f"Overall avg salt: {salt_ratios.mean():.1%}\n"
        f"Inline: {inlines.min():.0f} - {inlines.max():.0f}\n"
        f"Crossline: {xlines.min():.0f} - {xlines.max():.0f}"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('salt_density_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: salt_density_heatmap.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SEISMIC SALT VISUALIZATION")
    print("="*70)
    print("\nSelect visualization type:")
    print("  1. Survey Overview (3 plots: map, depth, histogram)")
    print("  2. Salt Density Heatmap (clearest view)")
    print("  3. Both")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        create_survey_overview()
    elif choice == "2":
        create_salt_density_heatmap()
    else:
        create_survey_overview()
        create_salt_density_heatmap()