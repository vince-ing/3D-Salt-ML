import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.amp
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# PATHS (Update these!)
TRAIN_DIR = r"C:\Users\ig-gbds\ML_Data_unpacked\train"  # Point to your train folder
VAL_DIR   = r"C:\Users\ig-gbds\ML_Data_unpacked\val"    # Point to your val folder
SAVE_DIR  = "experiments/pilot_run_01"

# HYPERPARAMETERS
BATCH_SIZE = 6          # Start small (2 or 4) to avoid Out-Of-Memory
LR = 1e-4               # Learning Rate
EPOCHS = 20
#NUM_WORKERS = 4         # Number of CPU cores for loading data

# HARDWARE SETUP


os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. DATASET (The Loader)
# ==========================================
class SaltDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        # Now we look for the exploded files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        
        try:
            # Load the single tiny file
            with np.load(fpath) as data:
                cube = data['seismic']
                mask = data['label']

            # --- Augmentation ---
            if self.augment:
                if np.random.rand() > 0.5: # Flip Z
                    cube = np.flip(cube, axis=2)
                    mask = np.flip(mask, axis=2)
                if np.random.rand() > 0.5: # Flip X/Y
                    cube = np.flip(cube, axis=1)
                    mask = np.flip(mask, axis=1)

            # Copy allows negative strides (from flips) to work
            cube = torch.from_numpy(cube.copy()).float().unsqueeze(0)
            mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)
            
            return cube, mask

        except Exception as e:
            return torch.zeros((1,128,128,128)), torch.zeros((1,128,128,128))

# ==========================================
# 3. MODEL: ResNet-UNet + ASPP (3D)
# ==========================================
class ConvBlock(nn.Module):
    """Basic 3D Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (The Context Extractor)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Dilations: 1 (normal), 2 (small gap), 4 (medium gap)
        # We use smaller dilations than 2D because 128^3 is tight
        self.b0 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.project = nn.Sequential(nn.Conv3d(out_ch*3, out_ch, 1, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        return self.project(torch.cat([x0, x1, x2], dim=1))

class SaltModel3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bridge (ASPP)
        self.aspp = ASPP(64, 64)
        
        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64+64, 64) # 64 from up3 + 64 from enc3
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32+32, 32)
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16+16, 16)
        
        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = self.enc1(x)       # 128 -> 128
        p1 = self.pool1(x1)     # 128 -> 64
        
        x2 = self.enc2(p1)      # 64 -> 64
        p2 = self.pool2(x2)     # 64 -> 32
        
        x3 = self.enc3(p2)      # 32 -> 32
        p3 = self.pool3(x3)     # 32 -> 16
        
        # Middle (Context)
        b = self.aspp(p3)       # 16 -> 16
        
        # Up
        u3 = self.up3(b)        # 16 -> 32
        # Concat skip connection
        d3 = self.dec3(torch.cat([u3, x3], dim=1))
        
        u2 = self.up2(d3)       # 32 -> 64
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        
        u1 = self.up1(d2)       # 64 -> 128
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        
        return self.final(d1)

# ==========================================
# 4. LOSS & UTILS
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = raw logits (no sigmoid applied yet)
        # targets = 0 or 1 (ground truth)
        
        # 1. BCE with Logits (This is the Safe/Fused version for Mixed Precision)
        # It applies sigmoid internally in a numerically stable way
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 2. Dice Loss (We still need probabilities for Dice, so we apply sigmoid here)
        inputs_prob = torch.sigmoid(inputs)
        
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return bce + dice_loss

def save_visual_report(model, loader, epoch, device):
    model.eval()
    
    target_image = None
    target_mask = None
    
    # 1. Hunt for a batch that actually has salt
    print("  > Hunting for a visual example with salt...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            # Check if there is any salt in this batch (sum > 100 pixels)
            if masks.sum() > 100: 
                target_image = images
                target_mask = masks
                print(f"  > Found salt in batch {i}!")
                break
        
        # Fallback: If no salt found in entire validation set (unlikely), just take the first one
        if target_image is None:
            print("  > Warning: No salt found in validation set. Showing empty rock.")
            target_image, target_mask = next(iter(loader))

        # Move to GPU
        img = target_image.to(device)
        mask = target_mask.to(device)

        # 2. Run Inference
        output = model(img)
        pred = torch.sigmoid(output)  # Convert logits to probability (0-1)
        
        # 3. Create the Plot (Slice 64 - Middle of the cube)
        # We take the first item in the batch [0]
        # We take the middle slice in depth [:, 64, :, :]
        slice_idx = 64
        
        input_slice = img[0, 0, slice_idx, :, :].cpu().numpy()
        mask_slice  = mask[0, 0, slice_idx, :, :].cpu().numpy()
        pred_slice  = pred[0, 0, slice_idx, :, :].cpu().numpy()
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input Seismic
        ax[0].imshow(input_slice, cmap='gray')
        ax[0].set_title(f"Input Seismic (Epoch {epoch})")
        
        # Ground Truth
        ax[1].imshow(mask_slice, cmap='gray')
        ax[1].set_title("Ground Truth (Target)")
        
        # Model Prediction
        # We overlay the prediction in Red (with transparency)
        ax[2].imshow(input_slice, cmap='gray')
        ax[2].imshow(pred_slice, cmap='jet', alpha=0.5) # Jet heatmap over seismic
        ax[2].set_title(f"Prediction (Prob > 0.5)")
        
        # Save
        os.makedirs("visual_reports", exist_ok=True)
        plt.savefig(f"visual_reports/epoch_{epoch}.png")
        plt.close()

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def run_training():
    # Load Data
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}") # This should only print ONCE now
    NUM_WORKERS = 0

    train_ds = SaltDataset(TRAIN_DIR, augment=True)
    val_ds = SaltDataset(VAL_DIR, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,          # Try 4 workers
        pin_memory=True,        # Speed up CPU-to-GPU transfer
        persistent_workers=True # KEEP WORKERS ALIVE (Crucial for Windows)
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,          # Fewer workers needed for val
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training Data: {len(train_ds)} cubes")
    print(f"Validation Data: {len(val_ds)} cubes")

    # Init Model
    model = SaltModel3D().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision

    # Loop
    best_loss = 999.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, leave=True)
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            # Forward (Mixed Precision)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, masks)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        print(f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # Save Checkpoint
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(">>> Saved Best Model!")
            
        # Visualize
        save_visual_report(model, val_loader, epoch+1, DEVICE)

if __name__ == "__main__":
    run_training()