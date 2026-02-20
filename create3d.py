"""
3D Salt Segmentation on SEGY File with PyVista Visualization

This script:
1. Loads a SEGY file
2. Normalizes the seismic data
3. Tiles it into 128³ patches with overlap
4. Runs inference using a trained model
5. Stitches predictions back together
6. Displays interactive 3D visualization of the salt body
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv
from tqdm import tqdm
import segyio
from scipy.ndimage import zoom

# ==========================================
# CONFIGURATION
# ==========================================
SEGY_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\Seismic_data.sgy"  # UPDATE THIS
MODEL_PATH = "experiments/multi_dataset_run_01/best_modelcrash.pth"
OUTPUT_DIR = "segy_inference_results"

# Inference settings
PATCH_SIZE = 128        # Model was trained on 128³ patches
OVERLAP = 32            # Overlap between patches for smooth stitching
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5         # Salt probability threshold

# Visualization settings
SHOW_SEISMIC = True     # Show seismic data as slices
SHOW_SALT = True        # Show predicted salt body
OPACITY = 0.8           # Salt body opacity (0-1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL ARCHITECTURE (Must match training!)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p)
        )
    def forward(self, x): 
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv3d(out_ch*3, out_ch, 1, bias=False), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(),
            nn.Dropout3d(0.3)
        )
    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        return self.project(torch.cat([x0, x1, x2], dim=1))

class SaltModel3D(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.enc1 = ConvBlock(1, 16, p=dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(16, 32, p=dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(32, 64, p=dropout_rate)
        self.pool3 = nn.MaxPool3d(2)
        self.aspp = ASPP(64, 64)
        self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64+64, 64, p=dropout_rate)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32+32, 32, p=dropout_rate)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16+16, 16, p=dropout_rate)
        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        b = self.aspp(p3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        return self.final(d1)

# ==========================================
# SEGY LOADING & PREPROCESSING
# ==========================================
def load_segy(filepath):
    """
    Load SEGY file and convert to numpy array.
    Returns: (depth, inline, crossline) array
    """
    print(f"Loading SEGY file: {filepath}")
    
    with segyio.open(filepath, ignore_geometry=True) as f:
        # Get dimensions
        n_traces = f.tracecount
        n_samples = f.samples.size
        
        print(f"  Traces: {n_traces}")
        print(f"  Samples per trace: {n_samples}")
        
        # Try to infer inline/crossline dimensions
        try:
            f.sorting
            n_inline = len(f.ilines)
            n_crossline = len(f.xlines)
            print(f"  Inlines: {n_inline}")
            print(f"  Crosslines: {n_crossline}")
            
            # Load as structured 3D volume
            data = segyio.tools.cube(f)  # (inline, crossline, depth)
            data = np.transpose(data, (2, 0, 1))  # (depth, inline, crossline)
            
        except:
            print("  ⚠ Could not infer geometry, attempting manual reshape...")
            # Fall back to manual reshape (assumes square grid)
            data = np.array([f.trace[i] for i in range(n_traces)])
            
            # Try to make it square
            n_inline = int(np.sqrt(n_traces))
            n_crossline = n_traces // n_inline
            
            if n_inline * n_crossline != n_traces:
                raise ValueError(f"Cannot reshape {n_traces} traces into a grid")
            
            data = data.reshape(n_inline, n_crossline, n_samples)
            data = np.transpose(data, (2, 0, 1))  # (depth, inline, crossline)
            
            print(f"  Reshaped to: {data.shape}")
    
    print(f"✅ Loaded volume: {data.shape}")
    return data

def normalize_seismic(data, clip_percentile=99):
    """
    Normalize seismic data to [-1, 1] range with outlier clipping.
    """
    print("Normalizing seismic data...")
    
    # Clip outliers
    lower = np.percentile(data, 100 - clip_percentile)
    upper = np.percentile(data, clip_percentile)
    data_clipped = np.clip(data, lower, upper)
    
    # Normalize to [-1, 1]
    data_min = data_clipped.min()
    data_max = data_clipped.max()
    data_norm = 2 * (data_clipped - data_min) / (data_max - data_min) - 1
    
    print(f"  Original range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Normalized range: [{data_norm.min():.2f}, {data_norm.max():.2f}]")
    
    return data_norm.astype(np.float32)

# ==========================================
# TILED INFERENCE WITH OVERLAP
# ==========================================
def predict_with_overlap(model, volume, patch_size=128, overlap=32):
    """
    Run inference on a large volume using overlapping patches.
    Averages predictions in overlapping regions for smooth stitching.
    """
    print("\nRunning tiled inference with overlap...")
    
    D, H, W = volume.shape
    stride = patch_size - overlap
    
    # Output accumulator
    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    count_volume = np.zeros((D, H, W), dtype=np.float32)
    
    # Calculate number of patches
    n_d = (D - overlap) // stride + (1 if (D - overlap) % stride > 0 else 0)
    n_h = (H - overlap) // stride + (1 if (H - overlap) % stride > 0 else 0)
    n_w = (W - overlap) // stride + (1 if (W - overlap) % stride > 0 else 0)
    
    total_patches = n_d * n_h * n_w
    print(f"  Volume shape: {volume.shape}")
    print(f"  Patch grid: {n_d} × {n_h} × {n_w} = {total_patches} patches")
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=total_patches, desc="Inference")
        
        for d_idx in range(n_d):
            d_start = d_idx * stride
            d_end = min(d_start + patch_size, D)
            d_start = max(0, d_end - patch_size)  # Adjust if near edge
            
            for h_idx in range(n_h):
                h_start = h_idx * stride
                h_end = min(h_start + patch_size, H)
                h_start = max(0, h_end - patch_size)
                
                for w_idx in range(n_w):
                    w_start = w_idx * stride
                    w_end = min(w_start + patch_size, W)
                    w_start = max(0, w_end - patch_size)
                    
                    # Extract patch
                    patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # Pad if needed (for edge cases)
                    if patch.shape != (patch_size, patch_size, patch_size):
                        padded = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
                        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                        patch = padded
                    
                    # Run inference
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    pred = torch.sigmoid(model(patch_tensor))
                    pred_np = pred[0, 0].cpu().numpy()
                    
                    # Add to accumulator (only the valid region)
                    valid_d = min(patch_size, d_end - d_start)
                    valid_h = min(patch_size, h_end - h_start)
                    valid_w = min(patch_size, w_end - w_start)
                    
                    pred_volume[d_start:d_start+valid_d,
                                h_start:h_start+valid_h,
                                w_start:w_start+valid_w] += pred_np[:valid_d, :valid_h, :valid_w]
                    
                    count_volume[d_start:d_start+valid_d,
                                 h_start:h_start+valid_h,
                                 w_start:w_start+valid_w] += 1
                    
                    pbar.update(1)
        
        pbar.close()
    
    # Average overlapping predictions
    pred_volume = pred_volume / np.maximum(count_volume, 1)
    
    print(f"✅ Inference complete!")
    print(f"  Salt coverage: {(pred_volume > THRESHOLD).mean()*100:.2f}%")
    
    return pred_volume

# ==========================================
# 3D VISUALIZATION WITH PYVISTA
# ==========================================
def visualize_3d(seismic, salt_prob, threshold=0.5):
    """
    Create interactive 3D visualization with PyVista.
    Shows seismic slices and salt body isosurface.
    """
    print("\nCreating 3D visualization...")
    
    D, H, W = seismic.shape
    
    # Create PyVista plotter
    plotter = pv.Plotter()
    plotter.set_background("white")
    
    # --- 1. Add seismic data as orthogonal slices ---
    if SHOW_SEISMIC:
        grid = pv.ImageData()
        grid.dimensions = (W, H, D)
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        grid["seismic"] = seismic.flatten(order='F')
        
        # Add three orthogonal slices
        slices = grid.slice_orthogonal(x=W//2, y=H//2, z=D//2)
        plotter.add_mesh(slices, cmap='gray', opacity=0.6, 
                        scalar_bar_args={'title': 'Seismic Amplitude'})
    
    # --- 2. Add salt body as isosurface ---
    if SHOW_SALT:
        # Create binary mask
        salt_mask = (salt_prob > threshold).astype(np.uint8)
        
        # Create PyVista grid for salt
        salt_grid = pv.ImageData()
        salt_grid.dimensions = (W, H, D)
        salt_grid.spacing = (1, 1, 1)
        salt_grid.origin = (0, 0, 0)
        salt_grid["salt_probability"] = salt_prob.flatten(order='F')
        salt_grid["salt_binary"] = salt_mask.flatten(order='F')
        
        # Extract isosurface at threshold
        salt_surface = salt_grid.contour([threshold], scalars="salt_probability")
        
        if salt_surface.n_points > 0:
            plotter.add_mesh(salt_surface, color='red', opacity=OPACITY,
                           label='Salt Body', smooth_shading=True)
            print(f"  Salt surface vertices: {salt_surface.n_points:,}")
        else:
            print("  ⚠ No salt detected above threshold")
    
    # --- 3. Add annotations ---
    plotter.add_text(f"3D Salt Segmentation\nThreshold: {threshold}", 
                    position='upper_left', font_size=10)
    
    # Add axes
    plotter.add_axes()
    plotter.add_bounding_box(color='black', line_width=1)
    
    # Set camera
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Add legend if salt is shown
    if SHOW_SALT and salt_surface.n_points > 0:
        plotter.add_legend()
    
    print("✅ Opening interactive viewer...")
    print("\nControls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - 'q': Quit")
    
    plotter.show()

def save_results(seismic, salt_prob, threshold=0.5):
    """
    Save results to disk for later analysis.
    """
    print("\nSaving results...")
    
    # Save probability volume
    np.save(f"{OUTPUT_DIR}/salt_probability.npy", salt_prob)
    print(f"  Saved: {OUTPUT_DIR}/salt_probability.npy")
    
    # Save binary mask
    salt_mask = (salt_prob > threshold).astype(np.uint8)
    np.save(f"{OUTPUT_DIR}/salt_mask.npy", salt_mask)
    print(f"  Saved: {OUTPUT_DIR}/salt_mask.npy")
    
    # Save metadata
    with open(f"{OUTPUT_DIR}/metadata.txt", 'w') as f:
        f.write(f"3D Salt Segmentation Results\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Input SEGY: {SEGY_PATH}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Volume shape: {seismic.shape}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Salt coverage: {(salt_prob > threshold).mean()*100:.2f}%\n")
        f.write(f"Total voxels: {salt_prob.size:,}\n")
        f.write(f"Salt voxels: {(salt_prob > threshold).sum():,}\n")
    
    print(f"  Saved: {OUTPUT_DIR}/metadata.txt")
    print("✅ Results saved!")

# ==========================================
# MAIN SCRIPT
# ==========================================
def main():
    print("="*70)
    print("3D SALT SEGMENTATION FROM SEGY")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"SEGY: {SEGY_PATH}")
    print(f"Model: {MODEL_PATH}")
    print("="*70 + "\n")
    
    # 1. Load model
    print("Loading model...")
    model = SaltModel3D(dropout_rate=0.2).to(DEVICE)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded (new format)")
            if 'val_iou' in checkpoint:
                print(f"   Model IoU: {checkpoint['val_iou']:.3f}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded (old format)")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    model.eval()
    
    # 2. Load and preprocess SEGY
    seismic = load_segy(SEGY_PATH)
    seismic_norm = normalize_seismic(seismic)
    
    # 3. Run inference
    salt_prob = predict_with_overlap(model, seismic_norm, 
                                      patch_size=PATCH_SIZE, 
                                      overlap=OVERLAP)
    
    # 4. Save results
    save_results(seismic_norm, salt_prob, threshold=THRESHOLD)
    
    # 5. Visualize
    visualize_3d(seismic_norm, salt_prob, threshold=THRESHOLD)
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)

if __name__ == "__main__":
    main()