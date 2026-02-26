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
DROPOUT_RATE = 0.2      # Dropout probability (NEW)
WEIGHT_DECAY = 1e-4     # L2 regularization (NEW)
EARLY_STOP_PATIENCE = 3 # Epochs to wait before stopping (NEW)

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

            # --- ENHANCED Augmentation ---
            if self.augment:
                # Geometric augmentations
                if np.random.rand() > 0.5:  # Flip Z
                    cube = np.flip(cube, axis=2)
                    mask = np.flip(mask, axis=2)
                if np.random.rand() > 0.5:  # Flip X/Y
                    cube = np.flip(cube, axis=1)
                    mask = np.flip(mask, axis=1)
                
                # Amplitude scaling (NEW - prevents overfitting to specific amplitudes)
                scale = 0.9 + 0.2 * np.random.rand()
                cube = cube * scale
                
                # Gaussian noise (NEW - makes model robust to noise)
                if np.random.rand() < 0.3:
                    noise = np.random.normal(0, 0.02, cube.shape)
                    cube = cube + noise

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
    """Basic 3D Conv -> BN -> ReLU with Dropout"""
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p),  # Regularization
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p)   # Regularization
        )
    def forward(self, x): 
        return self.conv(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with Dropout (The Context Extractor)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Dilations: 1 (normal), 2 (small gap), 4 (medium gap)
        # We use smaller dilations than 2D because 128^3 is tight
        self.b0 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        
        self.project = nn.Sequential(
            nn.Conv3d(out_ch*3, out_ch, 1, bias=False), 
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(),
            nn.Dropout3d(0.3)  # Higher dropout in bottleneck (NEW)
        )

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        return self.project(torch.cat([x0, x1, x2], dim=1))

class SaltModel3D(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(1, 16, p=dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock(16, 32, p=dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock(32, 64, p=dropout_rate)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bridge (ASPP)
        self.aspp = ASPP(64, 64)
        
        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64+64, 64, p=dropout_rate) # 64 from up3 + 64 from enc3
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32+32, 32, p=dropout_rate)
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16+16, 16, p=dropout_rate)
        
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
    def __init__(self, pos_weight=2.0):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1):
        # inputs = raw logits (no sigmoid applied yet)
        # targets = 0 or 1 (ground truth)
        
        # 1. BCE with Logits + Class Imbalance Handling (NEW)
        # pos_weight gives higher weight to salt class
        bce = F.binary_cross_entropy_with_logits(
            inputs, 
            targets, 
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='mean'
        )
        
        # 2. Dice Loss (We still need probabilities for Dice, so we apply sigmoid here)
        inputs_prob = torch.sigmoid(inputs)
        
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return bce + dice_loss

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def run_training():
    # Load Data
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}")

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

    # Init Model with dropout
    model = SaltModel3D(dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    # Optimizer with weight decay (L2 regularization) (NEW)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY  # Penalizes large weights
    )
    
    # Loss with class imbalance handling (NEW)
    criterion = DiceBCELoss(pos_weight=2.0)  # 2x weight for salt class
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision

    # Early stopping tracking (NEW)
    best_loss = 999.0
    patience_counter = 0
    
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
        
        # Save Checkpoint & Early Stopping Logic (UPDATED)
        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0  # Reset counter
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(">>> Saved Best Model!")
        else:
            patience_counter += 1
            print(f">>> No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered - validation loss stopped improving")
                break
            

    
    print("\n" + "="*50)
    print(f"Training Complete! Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {SAVE_DIR}/best_model.pth")
    print("="*50)

if __name__ == "__main__":
    run_training()