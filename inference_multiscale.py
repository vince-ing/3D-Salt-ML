import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import zoom
import random
import datetime

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\experiments\multi_dataset_multiscale_run_02\best_model.pth"
VAL_DIRS = [
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mckinley_expand\test",
    r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mississippi_expand\test"
]
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
run_name = Path(MODEL_PATH).parts[-2]
OUTPUT_DIR = f"inference_results/{run_name}_{timestamp}"

# FILTERING OPTIONS
NUM_SAMPLES = 25
MIN_SALT_RATIO = 0.1
MAX_SALT_RATIO = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# MODEL ARCHITECTURE — Multi-Scale (matches training!)
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
            nn.Conv3d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Dropout3d(0.3)
        )

    def forward(self, x):
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


class SaltModel3D_MultiScale(nn.Module):
    """
    Must exactly match the training model.
    Takes x_fine (B,1,128,128,128) and x_context (B,1,128,128,128).
    Context is the 256³ neighbourhood downsampled to 128³ via reflection padding.
    """
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        # Fine encoder
        self.enc1 = ConvBlock(1, 16, p=dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(16, 32, p=dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(32, 64, p=dropout_rate)
        self.pool3 = nn.MaxPool3d(2)

        # Context encoder (separate weights)
        self.ctx_enc = nn.Sequential(
            ConvBlock(1, 16, p=dropout_rate),
            nn.MaxPool3d(2),
            ConvBlock(16, 32, p=dropout_rate),
            nn.MaxPool3d(2),
            ConvBlock(32, 64, p=dropout_rate),
            nn.MaxPool3d(2),
        )

        # Bridge — receives 128 ch (64 fine + 64 context)
        self.aspp = ASPP(128, 64)

        # Decoder
        self.up3  = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 64, 64, p=dropout_rate)
        self.up2  = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32 + 32, 32, p=dropout_rate)
        self.up1  = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16 + 16, 16, p=dropout_rate)

        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x_fine, x_context):
        x1 = self.enc1(x_fine)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        fine_features = self.pool3(x3)

        ctx_features = self.ctx_enc(x_context)

        fused = torch.cat([fine_features, ctx_features], dim=1)
        b = self.aspp(fused)

        d3 = self.dec3(torch.cat([self.up3(b),  x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        return self.final(d1)


# ==========================================
# CONTEXT PATCH HELPER  (mirrors training)
# ==========================================
def make_context_patch(cube: np.ndarray) -> np.ndarray:
    """Pad 128³ -> 256³ with reflection, then downsample back to 128³."""
    pad = 64
    padded = np.pad(cube, pad_width=pad, mode='reflect')   # (256,256,256)
    context = zoom(padded, zoom=0.5, order=1)              # (128,128,128)
    return context.astype(np.float32)


# ==========================================
# METRICS
# ==========================================
def compute_iou(pred, target, threshold=0.5):
    pred_b  = (pred > threshold).astype(np.float32)
    tgt_b   = target.astype(np.float32)
    inter   = (pred_b * tgt_b).sum()
    union   = pred_b.sum() + tgt_b.sum() - inter
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def compute_dice(pred, target, threshold=0.5):
    pred_b  = (pred > threshold).astype(np.float32)
    tgt_b   = target.astype(np.float32)
    inter   = (pred_b * tgt_b).sum()
    denom   = pred_b.sum() + tgt_b.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2.0 * inter / denom)


def compute_boundary_f1(pred, target, threshold=0.5):
    from scipy.ndimage import binary_erosion
    pred_b  = pred > threshold
    tgt_b   = target > 0.5
    pb = pred_b & ~binary_erosion(pred_b)
    tb = tgt_b  & ~binary_erosion(tgt_b)
    tp = (pb & tb).sum()
    fp = (pb & ~tb).sum()
    fn = (~pb & tb).sum()
    if tp + fp + fn == 0:
        return 1.0
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# ==========================================
# VISUALIZATION
# ==========================================
def visualize_prediction(seismic, ground_truth, prediction, metrics, output_path):
    d, h, w = seismic.shape
    mids = [d // 2, h // 2, w // 2]
    labels = ["Inline Slice", "Crossline Slice", "Time Slice"]

    fig = plt.figure(figsize=(20, 12))

    for row, (axis_name, mid) in enumerate(zip(labels, mids)):
        if row == 0:
            seis_sl  = seismic[mid, :, :]
            truth_sl = ground_truth[mid, :, :]
            pred_sl  = prediction[mid, :, :]
        elif row == 1:
            seis_sl  = seismic[:, mid, :]
            truth_sl = ground_truth[:, mid, :]
            pred_sl  = prediction[:, mid, :]
        else:
            seis_sl  = seismic[:, :, mid]
            truth_sl = ground_truth[:, :, mid]
            pred_sl  = prediction[:, :, mid]

        ax1 = plt.subplot(3, 4, row * 4 + 1)
        im1 = ax1.imshow(seis_sl.T, cmap='gray', vmin=-3, vmax=3)
        ax1.set_title(f'{axis_name}: Seismic Input', fontweight='bold')
        ax1.axis('off')
        if row == 0:
            cb = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cb.set_label('Normalized\nAmplitude', fontsize=9)

        ax2 = plt.subplot(3, 4, row * 4 + 2)
        im2 = ax2.imshow(truth_sl.T, cmap='bwr', vmin=0, vmax=1)
        ax2.set_title(f'{axis_name}: Ground Truth', fontweight='bold')
        ax2.axis('off')
        if row == 0:
            cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cb2.set_label('Salt\nProbability', fontsize=9)
            cb2.set_ticks([0, 0.5, 1])
            cb2.set_ticklabels(['Rock\n(0)', 'Uncertain\n(0.5)', 'Salt\n(1)'])

        ax3 = plt.subplot(3, 4, row * 4 + 3)
        im3 = ax3.imshow(pred_sl.T, cmap='bwr', vmin=0, vmax=1)
        ax3.set_title(f'{axis_name}: AI Prediction', fontweight='bold')
        ax3.axis('off')
        if row == 0:
            cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cb3.set_label('Predicted\nProbability', fontsize=9)
            cb3.set_ticks([0, 0.5, 1])
            cb3.set_ticklabels(['Rock\n(0)', 'Uncertain\n(0.5)', 'Salt\n(1)'])

        ax4 = plt.subplot(3, 4, row * 4 + 4)
        ax4.imshow(seis_sl.T, cmap='gray', vmin=-3, vmax=3, alpha=0.6)
        pb  = (pred_sl  > 0.5).astype(float)
        tb  = (truth_sl > 0.5).astype(float)
        ov  = np.zeros((*pred_sl.T.shape, 4))
        ov[(pb.T == 1) & (tb.T == 1)] = [0,   1,   0,   0.6]
        ov[(pb.T == 1) & (tb.T == 0)] = [1,   0,   0,   0.6]
        ov[(pb.T == 0) & (tb.T == 1)] = [0, 0.5,   1,   0.6]
        ax4.imshow(ov)
        ax4.set_title(f'{axis_name}: Prediction Quality', fontweight='bold')
        ax4.axis('off')
        if row == 0:
            from matplotlib.patches import Patch
            ax4.legend(handles=[
                Patch(facecolor='green',      alpha=0.6, label='Correct (TP)'),
                Patch(facecolor='red',        alpha=0.6, label='False Alarm (FP)'),
                Patch(facecolor='dodgerblue', alpha=0.6, label='Missed Salt (FN)'),
            ], loc='upper right', fontsize=8, framealpha=0.9)

    iou_str  = f"IoU={metrics['iou']:.3f}"
    dice_str = f"Dice={metrics['dice']:.3f}"
    bf1_str  = f"BF1={metrics['boundary_f1']:.3f}"
    gt_str   = f"GT salt={metrics['gt_salt_ratio']:.1%}"
    pred_str = f"Pred salt={metrics['pred_salt_ratio']:.1%}"
    plt.suptitle(
        f'3D Salt Segmentation — {iou_str}  {dice_str}  {bf1_str}  |  {gt_str}  {pred_str}',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 0.98, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


# ==========================================
# MAIN INFERENCE
# ==========================================
def run_inference():
    print("=" * 70)
    print("SALT SEGMENTATION — MULTI-SCALE INFERENCE")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH}")
    for vd in VAL_DIRS:
        print(f"  - {vd}")
    print("=" * 70 + "\n")

    # 1. Load model
    print("Loading model (SaltModel3D_MultiScale)...")
    model = SaltModel3D_MultiScale(dropout_rate=0.2).to(DEVICE)

    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        print("✅ Model loaded successfully")
        if isinstance(ckpt, dict):
            for key in ('epoch', 'val_iou', 'val_dice', 'val_loss', 'model_type', 'data_sources'):
                if key in ckpt:
                    print(f"   {key}: {ckpt[key]}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    model.eval()

    # 2. Collect files
    print("\nFinding .npz files...")
    all_files = []
    for vd in VAL_DIRS:
        files = glob.glob(os.path.join(vd, "*.npz"))
        all_files.extend(files)
        print(f"  {Path(vd).name}: {len(files)} cubes")

    if not all_files:
        print("❌ No .npz files found.")
        return

    print(f"✅ {len(all_files)} total cubes")

    random.shuffle(all_files)

    print(f"\nSearching for {NUM_SAMPLES} cubes with "
          f"{MIN_SALT_RATIO*100:.0f}%–{MAX_SALT_RATIO*100:.0f}% salt\n")

    all_metrics     = []
    samples_found   = 0
    files_checked   = 0

    for fpath in all_files:
        if samples_found >= NUM_SAMPLES:
            break

        files_checked += 1

        try:
            with np.load(fpath) as data:
                seismic = data['seismic']
                label   = data['label']
        except Exception as e:
            print(f"  ❌ Error loading {Path(fpath).name}: {e}")
            continue

        salt_ratio = float(label.mean())

        # Salt filter
        if salt_ratio < MIN_SALT_RATIO:
            continue
        if MAX_SALT_RATIO is not None and salt_ratio > MAX_SALT_RATIO:
            continue

        samples_found += 1
        print(f"\n✓ Sample {samples_found}/{NUM_SAMPLES}  —  {Path(fpath).name}  (GT salt: {salt_ratio:.1%})")

        # Build context patch (same as training)
        context = make_context_patch(seismic)

        fine_t = torch.from_numpy(seismic.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        ctx_t  = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_logits = model(fine_t, ctx_t)
            pred_prob   = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()

        iou          = compute_iou(pred_prob, label)
        dice         = compute_dice(pred_prob, label)
        boundary_f1  = compute_boundary_f1(pred_prob, label)
        pred_salt    = float(pred_prob.mean())

        metrics = {
            'iou':            iou,
            'dice':           dice,
            'boundary_f1':    boundary_f1,
            'gt_salt_ratio':  salt_ratio,
            'pred_salt_ratio': pred_salt,
        }
        all_metrics.append(metrics)

        print(f"  IoU={iou:.3f}  Dice={dice:.3f}  BF1={boundary_f1:.3f}  "
              f"PredSalt={pred_salt:.1%}")

        out_path = os.path.join(OUTPUT_DIR, f"inference_sample_{samples_found:03d}.png")
        visualize_prediction(seismic, label, pred_prob, metrics, out_path)

    # 3. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_metrics:
        print(f"Samples evaluated : {len(all_metrics)}")
        print(f"Average IoU       : {np.mean([m['iou']         for m in all_metrics]):.3f}")
        print(f"Average Dice      : {np.mean([m['dice']        for m in all_metrics]):.3f}")
        print(f"Average BF1       : {np.mean([m['boundary_f1'] for m in all_metrics]):.3f}")
        print(f"Results saved to  : {OUTPUT_DIR}/")
    else:
        print("No valid samples found — check MIN/MAX_SALT_RATIO and VAL_DIRS.")

    if samples_found < NUM_SAMPLES:
        print(f"\n⚠ Only {samples_found}/{NUM_SAMPLES} valid samples found "
              f"after checking {files_checked} files.")

    print("=" * 70)
    print("✅ Inference complete!")


if __name__ == "__main__":
    run_inference()