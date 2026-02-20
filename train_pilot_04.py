import os
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

# ==========================================
# 1. CONFIGURATION
# ==========================================
# PATHS - Now supports MULTIPLE data sources!
DATA_SOURCES = [
    {
        'name': 'mckinley',
        'train': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mckinley_expand\train",
        'val': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mckinley_expand\val"
    },
    {
        'name': 'mississippi',
        'train': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mississippi_expand\train",
        'val': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mississippi_expand\val"
    },
    # Add more datasets here as needed:
    # {
    #     'name': 'third_dataset',
    #     'train': r"G:\path\to\third\train",
    #     'val': r"G:\path\to\third\val"
    # },
]

SAVE_DIR = "experiments/multi_dataset_multiscale_run_02"

# HYPERPARAMETERS
BATCH_SIZE = 6          # Reduced from 6 - multi-scale uses more VRAM
LR = 1e-4               # Learning Rate
EPOCHS = 50
DROPOUT_RATE = 0.2      # Dropout probability
WEIGHT_DECAY = 1e-4     # L2 regularization
EARLY_STOP_PATIENCE = 10 # Epochs to wait before stopping
LR_PATIENCE = 5         # Epochs to wait before reducing LR
LR_FACTOR = 0.5         # Factor to reduce LR by
USE_TTA = False          # Test-Time Augmentation during validation
SYNTH_AUG_PROB = 0.1    # Probability of adding synthetic water/edges

# Multi-scale settings
FINE_SIZE = 128         # Fine patch size (same as before)
CONTEXT_SIZE = 256      # Context patch size (2x, sees wider area)

# HARDWARE SETUP
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. DATASET (The Loader)
# ==========================================
class SaltDataset(Dataset):
    def __init__(self, data_dir, augment=False, dataset_name="unknown"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        self.dataset_name = dataset_name
        
        print(f"  [{dataset_name}] Loaded {len(self.files)} cubes from {data_dir}")

    def __len__(self):
        return len(self.files)
    
    def add_synthetic_water_edges(self, cube, mask):
        """
        Inject synthetic water layers and survey edges.
        Applied to BOTH fine and context patches via the mask.
        """
        cube_aug = cube.copy()
        mask_aug = mask.copy()
        
        if np.random.rand() < 0.5:
            water_thickness = np.random.randint(5, 20)
            cube_aug[:water_thickness, :, :] = np.random.normal(0, 0.01, 
                                                                 (water_thickness, 128, 128))
            mask_aug[:water_thickness, :, :] = 0
        
        if np.random.rand() < 0.5:
            edge_width = np.random.randint(3, 15)
            edge_side = np.random.choice(['left', 'right', 'front', 'back'])
            
            if edge_side == 'left':
                cube_aug[:, :, :edge_width] = np.random.normal(0, 0.01, (128, 128, edge_width))
                mask_aug[:, :, :edge_width] = 0
            elif edge_side == 'right':
                cube_aug[:, :, -edge_width:] = np.random.normal(0, 0.01, (128, 128, edge_width))
                mask_aug[:, :, -edge_width:] = 0
            elif edge_side == 'front':
                cube_aug[:, :edge_width, :] = np.random.normal(0, 0.01, (128, edge_width, 128))
                mask_aug[:, :edge_width, :] = 0
            elif edge_side == 'back':
                cube_aug[:, -edge_width:, :] = np.random.normal(0, 0.01, (128, edge_width, 128))
                mask_aug[:, -edge_width:, :] = 0
        
        if np.random.rand() < 0.3:
            num_dead = np.random.randint(1, 5)
            for _ in range(num_dead):
                x = np.random.randint(0, 128)
                y = np.random.randint(0, 128)
                y_start, y_end = max(0, y-2), min(128, y+2)
                x_start, x_end = max(0, x-2), min(128, x+2)
                cube_aug[:, y_start:y_end, x_start:x_end] = np.random.normal(0, 0.005, 
                                                     (128, y_end-y_start, x_end-x_start))
                mask_aug[:, y_start:y_end, x_start:x_end] = 0
        
        return cube_aug, mask_aug

    def make_context_patch(self, cube):
        """
        Create a context patch by padding the 128³ cube to simulate a 256³ view,
        then downsampling back to 128³. This gives the model awareness of the
        surrounding area at half resolution.
        
        In a real pipeline where you have access to the full volume, you would
        extract a true 256³ patch centred on the same location. Here we use
        reflection padding as a practical approximation for the pre-tiled dataset.
        """
        # Pad 128³ -> 256³ using reflection (mirrors edges outward)
        # This is a reasonable approximation when the full volume isn't available
        pad = 64  # 64 on each side to go from 128 -> 256
        padded = np.pad(cube, pad_width=pad, mode='reflect')  # (256, 256, 256)
        
        # Downsample 256³ -> 128³ (each voxel now represents 2x the real-world area)
        context = zoom(padded, zoom=0.5, order=1)  # (128, 128, 128)
        
        return context.astype(np.float32)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        
        try:
            with np.load(fpath) as data:
                cube = data['seismic']
                mask = data['label']

            # --- AUGMENTATION ---
            if self.augment:
                # Geometric augmentations
                if np.random.rand() > 0.5:  # Flip Z
                    cube = np.flip(cube, axis=2)
                    mask = np.flip(mask, axis=2)
                if np.random.rand() > 0.5:  # Flip X/Y
                    cube = np.flip(cube, axis=1)
                    mask = np.flip(mask, axis=1)
                
                # Amplitude scaling
                scale = 0.9 + 0.2 * np.random.rand()
                cube = cube * scale
                
                # Gaussian noise
                if np.random.rand() < 0.3:
                    noise = np.random.normal(0, 0.02, cube.shape)
                    cube = cube + noise
                
                # Synthetic water/edge injection
                if np.random.rand() < SYNTH_AUG_PROB:
                    cube, mask = self.add_synthetic_water_edges(cube, mask)

            # Build context patch AFTER augmentation so both see the same transforms
            context = self.make_context_patch(cube)

            # Convert to tensors
            cube_t    = torch.from_numpy(cube.copy()).float().unsqueeze(0)     # (1, 128, 128, 128)
            context_t = torch.from_numpy(context.copy()).float().unsqueeze(0)  # (1, 128, 128, 128)
            mask_t    = torch.from_numpy(mask.copy()).float().unsqueeze(0)     # (1, 128, 128, 128)
            
            return cube_t, context_t, mask_t

        except Exception as e:
            # Return zeros for all three outputs on failure
            zeros = torch.zeros((1, 128, 128, 128))
            return zeros, zeros, zeros

# ==========================================
# 3. MODEL: Multi-Scale ResNet-UNet + ASPP (3D)
# ==========================================
class ConvBlock(nn.Module):
    """Basic 3D Conv -> BN -> ReLU with Dropout"""
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
    """Atrous Spatial Pyramid Pooling with Dropout"""
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

class SaltModel3D_MultiScale(nn.Module):
    """
    Multi-Scale UNet: processes a fine 128³ patch AND a context patch
    (256³ area downsampled to 128³) simultaneously. The context encoder
    gives the model awareness of geological structures beyond the 128³ window,
    helping it correctly classify complex/low-contrast salt bodies.
    
    Data flow:
        x_fine    (B, 1, 128, 128, 128) -- full resolution, local detail
        x_context (B, 1, 128, 128, 128) -- half resolution, 2x wider area
    """
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # --- Fine encoder (full resolution, same as original model) ---
        self.enc1 = ConvBlock(1, 16, p=dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock(16, 32, p=dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock(32, 64, p=dropout_rate)
        self.pool3 = nn.MaxPool3d(2)
        # After pool3: (B, 64, 16, 16, 16)

        # --- Context encoder (lightweight - processes the wider view) ---
        # Uses the same architecture but is a separate set of weights,
        # so it can specialise on low-resolution, large-scale features
        self.ctx_enc = nn.Sequential(
            ConvBlock(1, 16, p=dropout_rate),
            nn.MaxPool3d(2),
            ConvBlock(16, 32, p=dropout_rate),
            nn.MaxPool3d(2),
            ConvBlock(32, 64, p=dropout_rate),
            nn.MaxPool3d(2),
        )
        # After ctx_enc: (B, 64, 16, 16, 16)

        # --- Bridge ---
        # ASPP now receives 128 channels (64 fine + 64 context fused)
        self.aspp = ASPP(128, 64)
        # After ASPP: (B, 64, 16, 16, 16)
        
        # --- Decoder (unchanged from original) ---
        self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64+64, 64, p=dropout_rate)   # 64 up + 64 skip from enc3
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32+32, 32, p=dropout_rate)   # 32 up + 32 skip from enc2
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16+16, 16, p=dropout_rate)   # 16 up + 16 skip from enc1
        
        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x_fine, x_context):
        # === Fine path ===
        x1 = self.enc1(x_fine)          # (B, 16, 128, 128, 128)
        x2 = self.enc2(self.pool1(x1))  # (B, 32,  64,  64,  64)
        x3 = self.enc3(self.pool2(x2))  # (B, 64,  32,  32,  32)
        fine_features = self.pool3(x3)  # (B, 64,  16,  16,  16)
        
        # === Context path ===
        ctx_features = self.ctx_enc(x_context)  # (B, 64, 16, 16, 16)
        
        # === Fuse at bottleneck ===
        fused = torch.cat([fine_features, ctx_features], dim=1)  # (B, 128, 16, 16, 16)
        b = self.aspp(fused)                                      # (B,  64, 16, 16, 16)
        
        # === Decode using fine skip connections ===
        d3 = self.dec3(torch.cat([self.up3(b), x3], dim=1))  # (B, 64, 32, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1)) # (B, 32, 64, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1)) # (B, 16, 128, 128, 128)
        
        return self.final(d1)                                 # (B,  1, 128, 128, 128)

# ==========================================
# 4. METRICS, LOSS & UTILS
# ==========================================
def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU) metric"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    return ((intersection + 1e-7) / (union + 1e-7)).item()

def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient metric"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    intersection = (pred_binary * target_binary).sum()
    return ((2. * intersection + 1e-7) / (pred_binary.sum() + target_binary.sum() + 1e-7)).item()

def compute_metrics_with_tta(model, imgs, contexts, masks, device):
    """
    Test-Time Augmentation - averages predictions across flipped versions.
    Both fine and context patches are flipped together.
    """
    model.eval()
    predictions = []
    
    flip_configs = [
        [],      # Original
        [4],     # Flip Z
        [3],     # Flip Y
        [2],     # Flip X
    ]
    
    for dims in flip_configs:
        with torch.no_grad():
            if dims:
                imgs_f = torch.flip(imgs, dims=dims)
                ctx_f  = torch.flip(contexts, dims=dims)
            else:
                imgs_f = imgs
                ctx_f  = contexts
            
            pred = torch.sigmoid(model(imgs_f, ctx_f))
            
            if dims:
                pred = torch.flip(pred, dims=dims)  # Flip back
            
            predictions.append(pred)
    
    avg_pred = torch.stack(predictions).mean(dim=0)
    iou  = compute_iou(avg_pred, masks)
    dice = compute_dice(avg_pred, masks)
    
    return iou, dice, avg_pred

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1):
        bce = F.binary_cross_entropy_with_logits(
            inputs, 
            targets, 
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='mean'
        )
        
        inputs_prob  = torch.sigmoid(inputs)
        inputs_flat  = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return bce + dice_loss

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def run_training():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}")
    
    print("\n" + "="*70)
    print("LOADING DATA FROM MULTIPLE SOURCES")
    print("="*70)
    
    train_datasets = []
    val_datasets   = []
    
    for source in DATA_SOURCES:
        print(f"\nSource: {source['name']}")
        print("-" * 70)
        
        train_exists = os.path.exists(source['train'])
        val_exists   = os.path.exists(source['val'])
        
        if not train_exists:
            print(f"  ⚠ WARNING: Training directory not found: {source['train']}")
            print(f"  Skipping this data source...")
            continue
            
        if not val_exists:
            print(f"  ⚠ WARNING: Validation directory not found: {source['val']}")
            print(f"  Skipping this data source...")
            continue
        
        train_ds = SaltDataset(source['train'], augment=True,  dataset_name=source['name'])
        val_ds   = SaltDataset(source['val'],   augment=False, dataset_name=source['name'])
        
        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        else:
            print(f"  ⚠ WARNING: No training files found in {source['train']}")
            
        if len(val_ds) > 0:
            val_datasets.append(val_ds)
        else:
            print(f"  ⚠ WARNING: No validation files found in {source['val']}")
    
    if len(train_datasets) == 0:
        print("\n❌ ERROR: No training data found in any source!")
        return
        
    if len(val_datasets) == 0:
        print("\n❌ ERROR: No validation data found in any source!")
        return
    
    print("\n" + "="*70)
    print("COMBINING DATASETS")
    print("="*70)
    
    combined_train = ConcatDataset(train_datasets)
    combined_val   = ConcatDataset(val_datasets)
    
    print(f"Total Training Cubes: {len(combined_train)}")
    for ds in train_datasets:
        print(f"    [{ds.dataset_name}]: {len(ds)} cubes")
    
    print(f"\nTotal Validation Cubes: {len(combined_val)}")
    for ds in val_datasets:
        print(f"    [{ds.dataset_name}]: {len(ds)} cubes")
    
    train_loader = DataLoader(
        combined_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        combined_val, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print("\n" + "="*70)
    print("INITIALIZING MULTI-SCALE MODEL")
    print("="*70)

    model     = SaltModel3D_MultiScale(dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, verbose=True, min_lr=1e-7
    )
    
    criterion = DiceBCELoss(pos_weight=2.0)
    scaler    = torch.amp.GradScaler('cuda')

    best_loss      = 999.0
    best_iou       = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou': [],    'val_dice': [], 'lr': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, leave=True)
        for imgs, contexts, masks in loop:  # Now unpacks 3 values
            imgs, contexts, masks = imgs.to(DEVICE), contexts.to(DEVICE), masks.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                preds = model(imgs, contexts)  # Pass both fine and context
                loss  = criterion(preds, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss      = 0
        val_iou_total = 0
        val_dice_total = 0
        
        print("\nRunning validation...")
        val_loop = tqdm(val_loader, leave=True)
        
        for imgs, contexts, masks in val_loop:  # Now unpacks 3 values
            imgs, contexts, masks = imgs.to(DEVICE), contexts.to(DEVICE), masks.to(DEVICE)
            
            if USE_TTA:
                iou, dice, _ = compute_metrics_with_tta(model, imgs, contexts, masks, DEVICE)
                with torch.no_grad():
                    preds = model(imgs, contexts)
                    loss  = criterion(preds, masks)
            else:
                with torch.no_grad():
                    preds       = model(imgs, contexts)
                    loss        = criterion(preds, masks)
                    preds_sig   = torch.sigmoid(preds)
                    iou         = compute_iou(preds_sig, masks)
                    dice        = compute_dice(preds_sig, masks)
            
            val_loss       += loss.item()
            val_iou_total  += iou
            val_dice_total += dice
            
            val_loop.set_postfix(loss=loss.item(), iou=iou, dice=dice)
        
        avg_val_loss  = val_loss       / len(val_loader)
        avg_val_iou   = val_iou_total  / len(val_loader)
        avg_val_dice  = val_dice_total / len(val_loader)
        current_lr    = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)
        
        print(f"\n{'─'*60}")
        print(f"Train Loss:    {avg_train:.4f}")
        print(f"Val Loss:      {avg_val_loss:.4f}")
        print(f"Val IoU:       {avg_val_iou:.4f}")
        print(f"Val Dice:      {avg_val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"{'─'*60}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_iou > best_iou:
            best_iou  = avg_val_iou
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_iou':  avg_val_iou,
                'val_dice': avg_val_dice,
                'model_type': 'multiscale',
                'data_sources': [s['name'] for s in DATA_SOURCES],
            }, f"{SAVE_DIR}/best_model.pth")
            print(f"✓ Saved Best Model! (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("\n⚠ Early stopping triggered")
                break
    
    np.savez(f"{SAVE_DIR}/training_history.npz", **history)
    
    with open(f"{SAVE_DIR}/dataset_info.txt", 'w') as f:
        f.write("DATA SOURCES USED IN TRAINING\n")
        f.write("="*70 + "\n\n")
        for source in DATA_SOURCES:
            f.write(f"Source: {source['name']}\n")
            f.write(f"  Train: {source['train']}\n")
            f.write(f"  Val:   {source['val']}\n\n")
        f.write(f"\nTotal Training Cubes: {len(combined_train)}\n")
        f.write(f"Total Validation Cubes: {len(combined_val)}\n")
        f.write(f"\nModel Type: Multi-Scale (fine 128³ + context 256³→128³)\n")
    
    try:
        plot_training_curves(history, SAVE_DIR)
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")
    
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Best Validation IoU:  {best_iou:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Model saved to: {SAVE_DIR}/best_model.pth")
    print("="*60)

def plot_training_curves(history, save_dir):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_dice'], label='Val Dice', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['lr'], label='Learning Rate', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    print(f"Training curves saved to {save_dir}/training_curves.png")
    plt.close()

if __name__ == "__main__":
    run_training()