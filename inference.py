import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "experiments/pilot_run_01/best_model.pth"
VAL_DIR = r"C:\Users\ig-gbds\ML_Data_unpacked\val"
OUTPUT_DIR = "inference_results"

MIN_SALT_RATIO = 0.10  # Only check slices with ≥10% salt
NUM_SAMPLES = 9        # Number of random slices to check

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL ARCHITECTURE (UPDATED - Must match training script!)
# ==========================================
class ConvBlock(nn.Module):
    """Basic 3D Conv -> BN -> ReLU with Dropout"""
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p),  # Dropout layer
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p)   # Dropout layer
        )
    def forward(self, x): 
        return self.conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with Dropout"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU()
        )
        self.b1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv3d(out_ch*3, out_ch, 1, bias=False), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(),
            nn.Dropout3d(0.3)  # Dropout in bottleneck
        )

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        return self.project(torch.cat([x0, x1, x2], dim=1))


class SaltModel3D(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 16, p=dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock(16, 32, p=dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock(32, 64, p=dropout_rate)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bridge
        self.aspp = ASPP(64, 64)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64+64, 64, p=dropout_rate)
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32+32, 32, p=dropout_rate)
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16+16, 16, p=dropout_rate)
        
        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        
        # Middle
        b = self.aspp(p3)
        
        # Up
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        
        return self.final(d1)


# ==========================================
# METRICS
# ==========================================
def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU)"""
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient"""
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    
    if pred_binary.sum() + target_binary.sum() == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2. * intersection) / (pred_binary.sum() + target_binary.sum())


def compute_boundary_f1(pred, target, threshold=0.5):
    """Compute F1 score on boundaries only"""
    from scipy.ndimage import binary_erosion
    
    pred_binary = pred > threshold
    target_binary = target > 0.5
    
    # Extract boundaries
    pred_boundary = pred_binary & ~binary_erosion(pred_binary)
    target_boundary = target_binary & ~binary_erosion(target_binary)
    
    tp = (pred_boundary & target_boundary).sum()
    fp = (pred_boundary & ~target_boundary).sum()
    fn = (~pred_boundary & target_boundary).sum()
    
    if tp + fp + fn == 0:
        return 1.0
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# ==========================================
# FIND SALT-RICH CUBES
# ==========================================
def find_salt_cubes(data_dir, min_salt_ratio=0.10):
    """
    Scan validation set and find cubes with sufficient salt.
    
    Returns: List of (filepath, salt_ratio, metadata) tuples
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    salt_cubes = []
    
    print(f"Scanning {len(files)} validation cubes for salt content...")
    
    for fpath in files:
        try:
            with np.load(fpath) as data:
                label = data['label']
                salt_ratio = label.mean()
                
                if salt_ratio >= min_salt_ratio:
                    # Also get metadata if available
                    metadata = data.get('metadata', None)
                    salt_cubes.append((fpath, salt_ratio, metadata))
        except Exception as e:
            print(f"Warning: Could not load {fpath}: {e}")
            continue
    
    print(f"✅ Found {len(salt_cubes)} cubes with ≥{min_salt_ratio*100:.0f}% salt")
    
    return salt_cubes


# ==========================================
# VISUALIZATION
# ==========================================
def visualize_prediction(seismic, ground_truth, prediction, metrics, output_path):
    """
    Create comprehensive visualization with colorbars and legends.
    
    Shows 3 orthogonal slices (inline, crossline, time) for each of:
    - Seismic input
    - Ground truth
    - AI prediction
    - Overlay with legend
    """
    # Get middle slices
    d, h, w = seismic.shape
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid: 3 rows (inline, xline, time) × 4 columns (seismic, truth, pred, overlay)
    
    for row, (axis_name, slice_idx, axis) in enumerate([
        ("Inline Slice", mid_d, 0),
        ("Crossline Slice", mid_h, 1),
        ("Time Slice", mid_w, 2)
    ]):
        # Extract slices
        if axis == 0:
            seismic_slice = seismic[slice_idx, :, :]
            truth_slice = ground_truth[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
        elif axis == 1:
            seismic_slice = seismic[:, slice_idx, :]
            truth_slice = ground_truth[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
        else:
            seismic_slice = seismic[:, :, slice_idx]
            truth_slice = ground_truth[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
        
        # Column 1: Seismic
        ax1 = plt.subplot(3, 4, row*4 + 1)
        im1 = ax1.imshow(seismic_slice.T, cmap='gray', vmin=-3, vmax=3)
        ax1.set_title(f'{axis_name}: Seismic Input', fontweight='bold')
        ax1.axis('off')
        
        # Add colorbar for seismic (only for first row)
        if row == 0:
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Normalized\nAmplitude', fontsize=9)
        
        # Column 2: Ground Truth
        ax2 = plt.subplot(3, 4, row*4 + 2)
        im2 = ax2.imshow(truth_slice.T, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax2.set_title(f'{axis_name}: Ground Truth', fontweight='bold')
        ax2.axis('off')
        
        # Add colorbar for ground truth (only for first row)
        if row == 0:
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Salt\nProbability', fontsize=9)
            cbar2.set_ticks([0, 0.5, 1])
            cbar2.set_ticklabels(['Rock\n(0)', 'Uncertain\n(0.5)', 'Salt\n(1)'])
        
        # Column 3: Prediction
        ax3 = plt.subplot(3, 4, row*4 + 3)
        im3 = ax3.imshow(pred_slice.T, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax3.set_title(f'{axis_name}: AI Prediction', fontweight='bold')
        ax3.axis('off')
        
        # Add colorbar for prediction (only for first row)
        if row == 0:
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Predicted\nProbability', fontsize=9)
            cbar3.set_ticks([0, 0.5, 1])
            cbar3.set_ticklabels(['Rock\n(0)', 'Uncertain\n(0.5)', 'Salt\n(1)'])
        
        # Column 4: Overlay with Error Visualization
        ax4 = plt.subplot(3, 4, row*4 + 4)
        
        # Show seismic as grayscale background
        ax4.imshow(seismic_slice.T, cmap='gray', vmin=-3, vmax=3, alpha=0.6)
        
        # Create error map: Green=correct, Red=false positive, Blue=false negative
        pred_binary = (pred_slice > 0.5).astype(float)
        truth_binary = (truth_slice > 0.5).astype(float)
        
        # Create RGB overlay
        overlay = np.zeros((*pred_slice.T.shape, 4))  # RGBA
        
        # True Positives (correct salt) - Green
        tp_mask = (pred_binary.T == 1) & (truth_binary.T == 1)
        overlay[tp_mask] = [0, 1, 0, 0.6]  # Green, 60% opacity
        
        # False Positives (predicted salt, actually rock) - Red
        fp_mask = (pred_binary.T == 1) & (truth_binary.T == 0)
        overlay[fp_mask] = [1, 0, 0, 0.6]  # Red, 60% opacity
        
        # False Negatives (missed salt) - Blue
        fn_mask = (pred_binary.T == 0) & (truth_binary.T == 1)
        overlay[fn_mask] = [0, 0.5, 1, 0.6]  # Blue, 60% opacity
        
        ax4.imshow(overlay)
        ax4.set_title(f'{axis_name}: Prediction Quality', fontweight='bold')
        ax4.axis('off')
        
        # Add legend (only for first row)
        if row == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.6, label='Correct (True Positive)'),
                Patch(facecolor='red', alpha=0.6, label='False Alarm (False Positive)'),
                Patch(facecolor='dodgerblue', alpha=0.6, label='Missed Salt (False Negative)')
            ]
            ax4.legend(handles=legend_elements, loc='upper right', 
                      fontsize=8, framealpha=0.9)
    
    plt.suptitle('3D Salt Segmentation Inference Results', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 0.98, 0.97])
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {output_path}")


# ==========================================
# MAIN INFERENCE
# ==========================================
def run_inference():
    """
    Load model and run inference on random salt-rich cubes.
    """
    print("="*70)
    print("3D SALT SEGMENTATION - INFERENCE")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Validation Dir: {VAL_DIR}")
    print("="*70 + "\n")
    
    # 1. Load Model
    print("Loading model...")
    model = SaltModel3D(dropout_rate=0.2).to(DEVICE)  # IMPORTANT: Must match training dropout
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
        print("   Note: Dropout layers are automatically disabled during inference (eval mode)\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    model.eval()  # CRITICAL: Sets dropout to eval mode (disabled)
    
    # 2. Find salt-rich cubes
    salt_cubes = find_salt_cubes(VAL_DIR, min_salt_ratio=MIN_SALT_RATIO)
    
    if len(salt_cubes) == 0:
        print(f"❌ No cubes found with ≥{MIN_SALT_RATIO*100:.0f}% salt")
        return
    
    # 3. Select random samples
    num_samples = min(NUM_SAMPLES, len(salt_cubes))
    selected_cubes = random.sample(salt_cubes, num_samples)
    
    print(f"\n{'='*70}")
    print(f"Selected {num_samples} random cubes for inference:")
    print(f"{'='*70}\n")
    
    # 4. Run inference on each sample
    all_metrics = []
    
    for idx, (fpath, salt_ratio, metadata) in enumerate(selected_cubes, 1):
        print(f"Sample {idx}/{num_samples}")
        print(f"  File: {Path(fpath).name}")
        print(f"  Ground truth salt: {salt_ratio:.1%}")
        
        # Load data
        try:
            with np.load(fpath) as data:
                seismic = data['seismic']
                label = data['label']
        except Exception as e:
            print(f"  ❌ Error loading: {e}\n")
            continue
        
        # Prepare input
        seismic_tensor = torch.from_numpy(seismic).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        seismic_tensor = seismic_tensor.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            pred_logits = model(seismic_tensor)
            pred_prob = torch.sigmoid(pred_logits)
        
        # Convert to numpy
        pred_prob = pred_prob[0, 0].cpu().numpy()  # Remove batch and channel dims
        
        # Compute metrics
        iou = compute_iou(pred_prob, label)
        dice = compute_dice(pred_prob, label)
        boundary_f1 = compute_boundary_f1(pred_prob, label)
        pred_salt_ratio = pred_prob.mean()
        
        metrics = {
            'iou': iou,
            'dice': dice,
            'boundary_f1': boundary_f1,
            'gt_salt_ratio': salt_ratio,
            'pred_salt_ratio': pred_salt_ratio
        }
        
        all_metrics.append(metrics)
        
        print(f"  IoU: {iou:.3f}")
        print(f"  Dice: {dice:.3f}")
        print(f"  Boundary F1: {boundary_f1:.3f}")
        print(f"  Predicted salt: {pred_salt_ratio:.1%}\n")
        
        # Visualize
        output_path = os.path.join(OUTPUT_DIR, f"inference_sample_{idx}.png")
        visualize_prediction(seismic, label, pred_prob, metrics, output_path)
    
    # 5. Summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    if len(all_metrics) > 0:
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_dice = np.mean([m['dice'] for m in all_metrics])
        avg_boundary_f1 = np.mean([m['boundary_f1'] for m in all_metrics])
        
        print(f"Average IoU: {avg_iou:.3f}")
        print(f"Average Dice: {avg_dice:.3f}")
        print(f"Average Boundary F1: {avg_boundary_f1:.3f}")
        print(f"\nResults saved to: {OUTPUT_DIR}/")
    
    print("="*70)
    print("✅ Inference complete!")
    print("="*70)


if __name__ == "__main__":
    run_inference()