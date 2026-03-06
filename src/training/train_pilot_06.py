"""
Multi-Class Salt Detection (Rock=0, Salt=1, Water=2, Blank=3)
Dual-Survey Training with Boundary-Aware Cross Entropy & Salt Dice
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.amp
from tqdm import tqdm
from scipy.ndimage import zoom
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_SOURCES = [
    {
        'name': 'mississippi',
        'train': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi256seafloor\train",
        'val':   r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi256_multiclass\val"
    },
    {
        'name': 'keathley',
        'train': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\keathley256seafloor\train",
        'val':   r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\keathley256seafloor\val"
    },
]

EXPERIMENTS_BASE = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\experiments"

# PATCH SIZES
CUBE_SIZE  = 256    
FINE_SIZE  = 128    
HALF       = FINE_SIZE // 2          
MAX_OFFSET = 64     

CROPS_PER_CUBE = 2

# HYPERPARAMETERS
BATCH_SIZE           = 6
LR                   = 1e-4
EPOCHS               = 50
DROPOUT_RATE         = 0.2
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 10
LR_PATIENCE          = 5
LR_FACTOR            = 0.5
USE_TTA              = False
SYNTH_AUG_PROB       = 0.1   


# ==========================================
# 2. DATASET
# ==========================================

class SaltDataset(Dataset):
    def __init__(self, data_dir, augment=False, dataset_name="unknown",
                 crops_per_cube=CROPS_PER_CUBE):
        self.files          = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment        = augment
        self.dataset_name   = dataset_name
        self.crops_per_cube = crops_per_cube if augment else 1
        print(f"  [{dataset_name}] {len(self.files)} cubes | "
              f"effective dataset size: {len(self)} | augment={augment}")

    def __len__(self):
        return len(self.files) * self.crops_per_cube

    def add_synthetic_water_edges(self, cube, mask):
        cube_aug = cube.copy()
        mask_aug = mask.copy()
        S = FINE_SIZE  

        # Water layer at top (Class 2)
        if np.random.rand() < 0.5:
            thickness = np.random.randint(5, 20)
            cube_aug[:thickness, :, :] = np.random.normal(0, 0.01, (thickness, S, S))
            mask_aug[:thickness, :, :] = 2

        # Survey edge dead zone (Class 3)
        if np.random.rand() < 0.5:
            width = np.random.randint(3, 15)
            side  = np.random.choice(['left', 'right', 'front', 'back'])
            if side == 'left':
                cube_aug[:, :, :width]  = np.random.normal(0, 0.01, (S, S, width))
                mask_aug[:, :, :width]  = 3
            elif side == 'right':
                cube_aug[:, :, -width:] = np.random.normal(0, 0.01, (S, S, width))
                mask_aug[:, :, -width:] = 3
            elif side == 'front':
                cube_aug[:, :width, :]  = np.random.normal(0, 0.01, (S, width, S))
                mask_aug[:, :width, :]  = 3
            elif side == 'back':
                cube_aug[:, -width:, :] = np.random.normal(0, 0.01, (S, width, S))
                mask_aug[:, -width:, :] = 3

        # Dead traces (Class 3)
        if np.random.rand() < 0.3:
            for _ in range(np.random.randint(1, 5)):
                x = np.random.randint(0, S)
                y = np.random.randint(0, S)
                ys, ye = max(0, y - 2), min(S, y + 2)
                xs, xe = max(0, x - 2), min(S, x + 2)
                cube_aug[:, ys:ye, xs:xe] = np.random.normal(0, 0.005, (S, ye - ys, xe - xs))
                mask_aug[:, ys:ye, xs:xe] = 3

        return cube_aug, mask_aug

    def __getitem__(self, idx):
        file_idx = idx // self.crops_per_cube

        try:
            with np.load(self.files[file_idx]) as data:
                cube256 = data['seismic'].astype(np.float32)  
                mask256 = data['label'].astype(np.float32)    

            if self.augment:
                if np.random.rand() > 0.5:   
                    cube256 = np.flip(cube256, axis=0)
                    mask256 = np.flip(mask256, axis=0)
                if np.random.rand() > 0.5:   
                    cube256 = np.flip(cube256, axis=1)
                    mask256 = np.flip(mask256, axis=1)
                if np.random.rand() > 0.5:   
                    cube256 = np.flip(cube256, axis=2)
                    mask256 = np.flip(mask256, axis=2)

                cube256 = cube256 * (0.9 + 0.2 * np.random.rand())
                if np.random.rand() < 0.3:
                    cube256 = cube256 + np.random.normal(0, 0.02, cube256.shape)

            context = zoom(cube256, zoom=0.5, order=0).astype(np.float32)
            center = CUBE_SIZE // 2  

            if self.augment:
                oz = np.random.randint(-MAX_OFFSET, MAX_OFFSET + 1)
                oy = np.random.randint(-MAX_OFFSET, MAX_OFFSET + 1)
                ox = np.random.randint(-MAX_OFFSET, MAX_OFFSET + 1)
            else:
                oz, oy, ox = 0, 0, 0  

            cz, cy, cx = center + oz, center + oy, center + ox

            fine = cube256[cz - HALF : cz + HALF, cy - HALF : cy + HALF, cx - HALF : cx + HALF].copy()   
            mask = mask256[cz - HALF : cz + HALF, cy - HALF : cy + HALF, cx - HALF : cx + HALF].copy()   

            if self.augment and np.random.rand() < SYNTH_AUG_PROB:
                fine, mask = self.add_synthetic_water_edges(fine, mask)

            fine_t    = torch.from_numpy(fine).float().unsqueeze(0)     # (1, 128, 128, 128)
            context_t = torch.from_numpy(context).float().unsqueeze(0)  # (1, 128, 128, 128)
            
            # CE Loss expects targets without channel dim if they are class indices
            mask_t    = torch.from_numpy(mask).long()                   # (128, 128, 128)

            return fine_t, context_t, mask_t

        except Exception as e:
            print(f"  [WARNING] Failed to load {self.files[file_idx]}: {e}")
            zeros = torch.zeros((1, FINE_SIZE, FINE_SIZE, FINE_SIZE))
            return zeros, zeros, torch.zeros((FINE_SIZE, FINE_SIZE, FINE_SIZE)).long()


# ==========================================
# 3. MODEL: Multi-Scale UNet + ASPP (3D)
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
            nn.Dropout3d(p),
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
            nn.BatchNorm3d(out_ch), nn.ReLU(),
            nn.Dropout3d(0.3))

    def forward(self, x):
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


class SaltModel3D_MultiScale(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.enc1  = ConvBlock(1,  16, p=dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2  = ConvBlock(16, 32, p=dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3  = ConvBlock(32, 64, p=dropout_rate)
        self.pool3 = nn.MaxPool3d(2)

        self.ctx_enc = nn.Sequential(
            ConvBlock(1, 16, p=dropout_rate), nn.MaxPool3d(2),
            ConvBlock(16, 32, p=dropout_rate), nn.MaxPool3d(2),
            ConvBlock(32, 64, p=dropout_rate), nn.MaxPool3d(2),
        )

        self.aspp = ASPP(128, 64)

        self.up3  = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 64, 64, p=dropout_rate)
        self.up2  = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32 + 32, 32, p=dropout_rate)
        self.up1  = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16 + 16, 16, p=dropout_rate)

        # 4 Classes: 0=Rock, 1=Salt, 2=Water, 3=Blank
        self.final = nn.Conv3d(16, 4, kernel_size=1)

    def forward(self, x_fine, x_context):
        x1          = self.enc1(x_fine)          
        x2          = self.enc2(self.pool1(x1))  
        x3          = self.enc3(self.pool2(x2))  
        fine_feat   = self.pool3(x3)             

        ctx_feat = self.ctx_enc(x_context)       

        fused = torch.cat([fine_feat, ctx_feat], dim=1)  
        b     = self.aspp(fused)                          

        d3 = self.dec3(torch.cat([self.up3(b),  x3], dim=1))  
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))  
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))  

        return self.final(d1)                                  


# ==========================================
# 4. LOSS & METRICS
# ==========================================

class MultiClassBoundaryLoss(nn.Module):
    def __init__(self, class_weights=[0.5, 3.0, 0.5, 0.1], boundary_weight=0.5):
        super().__init__()
        # Register as buffer so it naturally moves to the correct device
        self.register_buffer('class_weights', torch.tensor(class_weights))
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets, smooth=1):
        # 1. Standard Weighted Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)
        
        # 2. Salt-Specific Dice Loss (Optimizes directly for Class 1)
        probs = torch.softmax(inputs, dim=1)
        salt_probs = probs[:, 1, ...]
        salt_targets = (targets == 1).float()
        
        inter = (salt_probs * salt_targets).sum(dim=(1,2,3))
        union = salt_probs.sum(dim=(1,2,3)) + salt_targets.sum(dim=(1,2,3))
        dice_loss = 1 - (2. * inter + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()

        # 3. Boundary Penalty (Fast 3D Morphological Ops via MaxPool)
        with torch.no_grad():
            salt_targets_5d = salt_targets.unsqueeze(1) # (B, 1, D, H, W)
            dilated = F.max_pool3d(salt_targets_5d, kernel_size=3, stride=1, padding=1)
            eroded  = -F.max_pool3d(-salt_targets_5d, kernel_size=3, stride=1, padding=1)
            boundary_mask = (dilated - eroded).squeeze(1) # (B, D, H, W)

        # Apply CE penalty specifically to the shell where the boundary is incorrect
        ce_none = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        boundary_loss = (ce_none * boundary_mask).sum() / (boundary_mask.sum() + 1e-7)

        return ce_loss + dice_loss + (self.boundary_weight * boundary_loss)


def compute_salt_metrics(preds, targets):
    """Calculates IoU and Dice specifically for the Salt class (1)."""
    preds_class = torch.argmax(preds, dim=1)
    
    pred_salt   = (preds_class == 1).float()
    target_salt = (targets == 1).float()
    
    inter = (pred_salt * target_salt).sum().item()
    union = pred_salt.sum().item() + target_salt.sum().item() - inter
    
    iou  = (inter + 1e-7) / (union + 1e-7)
    dice = (2. * inter + 1e-7) / (pred_salt.sum().item() + target_salt.sum().item() + 1e-7)
    
    return iou, dice


def compute_metrics_with_tta(model, imgs, contexts, masks):
    """Average Softmax predictions over 4 flip configurations."""
    preds = []
    for dims in [[], [4], [3], [2]]:
        with torch.no_grad():
            f = torch.flip(imgs, dims=dims)     if dims else imgs
            c = torch.flip(contexts, dims=dims) if dims else contexts
            p = torch.softmax(model(f, c), dim=1)
            if dims:
                p = torch.flip(p, dims=dims)
            preds.append(p)
            
    avg_probs = torch.stack(preds).mean(0)
    
    preds_class = torch.argmax(avg_probs, dim=1)
    pred_salt   = (preds_class == 1).float()
    target_salt = (masks == 1).float()
    
    inter = (pred_salt * target_salt).sum().item()
    union = pred_salt.sum().item() + target_salt.sum().item() - inter
    iou  = (inter + 1e-7) / (union + 1e-7)
    dice = (2. * inter + 1e-7) / (pred_salt.sum().item() + target_salt.sum().item() + 1e-7)
    
    return iou, dice, avg_probs


# ==========================================
# 5. CSV EPOCH LOGGER
# ==========================================

CSV_FIELDS = [
    'epoch', 'timestamp', 'train_loss', 'val_loss', 
    'val_salt_iou', 'val_salt_dice', 'learning_rate', 
    'is_best', 'best_val_iou_so_far', 'early_stop_counter', 'epoch_duration_sec',
]

def init_csv_log(save_dir):
    csv_path = os.path.join(save_dir, "epoch_log.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
    return csv_path

def append_csv_log(csv_path, row: dict):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ==========================================
# 6. TRAINING
# ==========================================

def run_training():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    save_dir  = os.path.join(EXPERIMENTS_BASE, f"multiclass_256_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {DEVICE}")

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_datasets, val_datasets = [], []

    for source in DATA_SOURCES:
        print(f"\nSource: {source['name']}")
        print("─" * 70)

        if not os.path.exists(source['train']):
            print(f"  ⚠ Train dir not found: {source['train']} — skipping")
            continue
        if not os.path.exists(source['val']):
            print(f"  ⚠ Val dir not found:   {source['val']} — skipping")
            continue

        train_ds = SaltDataset(source['train'], augment=True,
                               dataset_name=source['name'],
                               crops_per_cube=CROPS_PER_CUBE)
        val_ds   = SaltDataset(source['val'],   augment=False,
                               dataset_name=source['name'],
                               crops_per_cube=1)

        if len(train_ds.files) > 0: train_datasets.append(train_ds)
        if len(val_ds.files) > 0:   val_datasets.append(val_ds)

    if not train_datasets:
        print("\n❌ No training data found."); return
    if not val_datasets:
        print("\n❌ No validation data found."); return

    combined_train = ConcatDataset(train_datasets)
    combined_val   = ConcatDataset(val_datasets)

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Train — {len(combined_train)} effective samples")
    for ds in train_datasets: print(f"    [{ds.dataset_name}]: {len(ds.files)} cubes")
    print(f"Val   — {len(combined_val)} samples (center crop, fixed)")
    for ds in val_datasets: print(f"    [{ds.dataset_name}]: {len(ds.files)} cubes")

    train_loader = DataLoader(
        combined_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        combined_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True)

    print("\n" + "=" * 70)
    print("INITIALISING MODEL")
    print("=" * 70)

    model     = SaltModel3D_MultiScale(dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=1e-7)
    
    criterion = MultiClassBoundaryLoss(class_weights=[0.5, 3.0, 0.5, 0.1], boundary_weight=0.5)
    scaler    = torch.amp.GradScaler('cuda')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    csv_path = init_csv_log(save_dir)

    best_iou         = 0.0
    best_loss        = 999.0
    patience_counter = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou':    [], 'val_dice': [], 'lr': []
    }

    for epoch in range(EPOCHS):
        epoch_start = datetime.now()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")

        # ── Train ──
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, leave=True, desc="Train")

        for fine, context, mask in loop:
            fine    = fine.to(DEVICE)
            context = context.to(DEVICE)
            mask    = mask.to(DEVICE)

            with torch.amp.autocast('cuda'):
                preds = model(fine, context)
                loss  = criterion(preds, mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = val_iou_total = val_dice_total = 0.0
        val_loop = tqdm(val_loader, leave=True, desc="Val  ")

        for fine, context, mask in val_loop:
            fine    = fine.to(DEVICE)
            context = context.to(DEVICE)
            mask    = mask.to(DEVICE)

            if USE_TTA:
                iou, dice, _ = compute_metrics_with_tta(model, fine, context, mask)
                with torch.no_grad():
                    preds = model(fine, context)
                    loss  = criterion(preds, mask)
            else:
                with torch.no_grad():
                    preds = model(fine, context)
                    loss  = criterion(preds, mask)
                    iou, dice = compute_salt_metrics(preds, mask)

            val_loss       += loss.item()
            val_iou_total  += iou
            val_dice_total += dice
            val_loop.set_postfix(loss=f"{loss.item():.4f}", salt_iou=f"{iou:.4f}")

        avg_val_loss  = val_loss       / len(val_loader)
        avg_val_iou   = val_iou_total  / len(val_loader)
        avg_val_dice  = val_dice_total / len(val_loader)
        current_lr    = optimizer.param_groups[0]['lr']
        epoch_secs    = (datetime.now() - epoch_start).total_seconds()

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)

        print(f"\n{'─'*60}")
        print(f"Train Loss:     {avg_train:.4f}")
        print(f"Val Loss:       {avg_val_loss:.4f}")
        print(f"Val Salt IoU:   {avg_val_iou:.4f}")
        print(f"Val Salt Dice:  {avg_val_dice:.4f}")
        print(f"Learning Rate:  {current_lr:.2e}")
        print(f"Epoch Time:     {epoch_secs:.1f}s")
        print(f"{'─'*60}")

        scheduler.step(avg_val_loss)

        # ── Checkpoint ──
        is_best = avg_val_iou > best_iou
        if is_best:
            best_iou         = avg_val_iou
            best_loss        = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':             avg_val_loss,
                'val_salt_iou':         avg_val_iou,
                'val_salt_dice':        avg_val_dice,
                'model_type':           'multiclass_256',
                'data_sources':         [s['name'] for s in DATA_SOURCES],
                'config': {
                    'cube_size':      CUBE_SIZE,
                    'fine_size':      FINE_SIZE,
                    'max_offset':     MAX_OFFSET,
                    'crops_per_cube': CROPS_PER_CUBE,
                    'batch_size':     BATCH_SIZE,
                    'lr':             LR,
                    'dropout':        DROPOUT_RATE,
                }
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"✓ Saved best model (Salt IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        append_csv_log(csv_path, {
            'epoch':               epoch + 1,
            'timestamp':           datetime.now().strftime("%Y-%m-%d_%H%M"),
            'train_loss':          round(avg_train,     6),
            'val_loss':            round(avg_val_loss,  6),
            'val_salt_iou':        round(avg_val_iou,   6),
            'val_salt_dice':       round(avg_val_dice,  6),
            'learning_rate':       f"{current_lr:.2e}",
            'is_best':             is_best,
            'best_val_iou_so_far': round(best_iou,      6),
            'early_stop_counter':  patience_counter,
            'epoch_duration_sec':  round(epoch_secs,    1),
        })

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("\n⚠ Early stopping triggered")
            break

    np.savez(os.path.join(save_dir, "training_history.npz"), **history)

    with open(os.path.join(save_dir, "run_info.txt"), 'w') as f:
        f.write(f"Run timestamp:  {timestamp}\n")
        f.write(f"Model type:     Multi-Class (4 channel) 256³ → fine 128³ + context 256³→128³\n\n")
        f.write(f"DATA SOURCES\n{'='*50}\n")
        for source in DATA_SOURCES:
            f.write(f"  {source['name']}\n")
            f.write(f"    train: {source['train']}\n")
            f.write(f"    val:   {source['val']}\n\n")
        f.write(f"\nCONFIG\n{'='*50}\n")
        f.write(f"  batch_size:     {BATCH_SIZE}\n")
        f.write(f"  lr:             {LR}\n")
        f.write(f"  epochs:         {EPOCHS}\n")
        f.write(f"  dropout:        {DROPOUT_RATE}\n")
        f.write(f"\nRESULTS\n{'='*50}\n")
        f.write(f"  Best Val Salt IoU: {best_iou:.4f}\n")
        f.write(f"  Best Val Loss:     {best_loss:.4f}\n")

    try:
        plot_training_curves(history, save_dir)
    except Exception as e:
        print(f"Warning: could not save training curves: {e}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best Val Salt IoU: {best_iou:.4f}")
    print(f"Results in:        {save_dir}")
    print("=" * 60)


# ==========================================
# 7. PLOTTING
# ==========================================

def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'],   label='Val Loss')
    axes[0, 0].set_title('Loss');  axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['val_iou'], color='green', label='Val Salt IoU')
    axes[0, 1].set_title('Validation Salt IoU'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(history['val_dice'], color='orange', label='Val Salt Dice')
    axes[1, 0].set_title('Validation Salt Dice'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['lr'], color='red', label='LR')
    axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_yscale('log'); axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    print(f"Training curves saved → {out}")
    plt.close()

if __name__ == "__main__":
    run_training()