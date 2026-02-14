import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# PATHS (Update these!)
TRAIN_DIR = "processed_data/mississippi/train"  # Point to your train folder
VAL_DIR   = "processed_data/mississippi/val"    # Point to your val folder
SAVE_DIR  = "experiments/pilot_run_01"

# HYPERPARAMETERS
BATCH_SIZE = 4          # Start small (2 or 4) to avoid Out-Of-Memory
LR = 1e-4               # Learning Rate
EPOCHS = 20
NUM_WORKERS = 4         # Number of CPU cores for loading data

# HARDWARE SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. DATASET (The Loader)
# ==========================================
class SaltDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        # Assume 64 cubes per file for speed
        self.per_file = 64 

    def __len__(self):
        return len(self.files) * self.per_file

    def __getitem__(self, idx):
        file_idx = idx // self.per_file
        local_idx = idx % self.per_file
        
        # Clamp index just in case
        if file_idx >= len(self.files): file_idx = len(self.files) - 1
        
        fpath = self.files[file_idx]
        
        try:
            with np.load(fpath) as data:
                seismic_batch = data['seismic']
                label_batch = data['label']
                
                if local_idx >= len(seismic_batch): local_idx = 0
                
                cube = seismic_batch[local_idx]
                mask = label_batch[local_idx]

            # --- Augmentation (Simple Flips) ---
            if self.augment:
                # 50% chance to flip Left-Right
                if np.random.rand() > 0.5:
                    cube = np.flip(cube, axis=2)
                    mask = np.flip(mask, axis=2)
                # 50% chance to flip Front-Back (Crossline)
                if np.random.rand() > 0.5:
                    cube = np.flip(cube, axis=1)
                    mask = np.flip(mask, axis=1)

            # Convert to Tensor (Add Channel Dim: [1, D, H, W])
            cube = torch.from_numpy(cube.copy()).float().unsqueeze(0)
            mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)
            
            return cube, mask

        except Exception as e:
            print(f"Error loading {fpath}: {e}")
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
        # Flatten
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        # BCE
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return bce + dice_loss

def save_visual_report(model, loader, epoch):
    """Saves a PNG showing Input vs Truth vs Pred"""
    model.eval()
    with torch.no_grad():
        # Grab one batch
        img, mask = next(iter(loader))
        img = img.to(DEVICE)
        
        # Predict
        pred = model(img)
        pred = torch.sigmoid(pred)
        
        # Convert to CPU numpy
        img = img[0, 0].cpu().numpy()       # First cube, channel 0
        mask = mask[0, 0].cpu().numpy()
        pred = pred[0, 0].cpu().numpy()
        
        # Pick middle slice
        mid = img.shape[0] // 2
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Seismic
        ax[0].imshow(img[mid,:,:].T, cmap='gray')
        ax[0].set_title(f"Seismic (Epoch {epoch})")
        
        # 2. Truth
        ax[1].imshow(mask[mid,:,:].T, cmap='jet', interpolation='nearest')
        ax[1].set_title("Ground Truth")
        
        # 3. Prediction
        ax[2].imshow(img[mid,:,:].T, cmap='gray')
        ax[2].imshow(pred[mid,:,:].T, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax[2].set_title("AI Prediction")
        
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/epoch_{epoch}_check.png")
        plt.close()
    model.train()

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def run_training():
    # Load Data
    train_ds = SaltDataset(TRAIN_DIR, augment=True)
    val_ds = SaltDataset(VAL_DIR, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Training Data: {len(train_ds)} cubes")
    print(f"Validation Data: {len(val_ds)} cubes")

    # Init Model
    model = SaltModel3D().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceBCELoss()
    scaler = GradScaler() # Mixed Precision

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
            with autocast():
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
        save_visual_report(model, val_loader, epoch+1)

if __name__ == "__main__":
    run_training()