import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.amp
from tqdm import tqdm

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
        'name': 'mississippi',  # UPDATE THIS NAME
        'train': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mississippi_expand\train",  # UPDATE THIS PATH
        'val': r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\processed_data\mississippi_expand\val"      # UPDATE THIS PATH
    },
    # Add more datasets here as needed:
    # {
    #     'name': 'third_dataset',
    #     'train': r"G:\path\to\third\train",
    #     'val': r"G:\path\to\third\val"
    # },
]

SAVE_DIR = "experiments/multi_dataset_run_01"
USE_SUBSET = 0.5

# HYPERPARAMETERS
BATCH_SIZE = 6          # Start small (2 or 4) to avoid Out-Of-Memory
LR = 1e-4               # Learning Rate
EPOCHS = 50
DROPOUT_RATE = 0.2      # Dropout probability
WEIGHT_DECAY = 1e-4     # L2 regularization
EARLY_STOP_PATIENCE = 10 # Epochs to wait before stopping
LR_PATIENCE = 5         # Epochs to wait before reducing LR
LR_FACTOR = 0.5         # Factor to reduce LR by
USE_TTA = True          # Test-Time Augmentation during validation
SYNTH_AUG_PROB = 0.1    # Probability of adding synthetic water/edges

# HARDWARE SETUP
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. DATASET (The Loader)
# ==========================================
class SaltDataset(Dataset):
    def __init__(self, data_dir, augment=False, dataset_name="unknown"):
        # Now we look for the exploded files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        self.dataset_name = dataset_name
        
        print(f"  [{dataset_name}] Loaded {len(self.files)} cubes from {data_dir}")

    def __len__(self):
        return len(self.files)
    
    def add_synthetic_water_edges(self, cube, mask):
        """
        Inject synthetic water layers and survey edges.
        This teaches the model: low-amplitude regions are NOT always salt.
        Prevents false positives in water and at survey boundaries.
        """
        cube_aug = cube.copy()
        mask_aug = mask.copy()
        
        # Option 1: Add water layer at top (50% chance)
        if np.random.rand() < 0.5:
            water_thickness = np.random.randint(5, 20)  # 5-20 slices
            # Zero out top slices (simulates water)
            cube_aug[:water_thickness, :, :] = np.random.normal(0, 0.01, 
                                                                 (water_thickness, 128, 128))
            # Ensure no salt predicted in water
            mask_aug[:water_thickness, :, :] = 0
        
        # Option 2: Add blank survey edges (50% chance)
        if np.random.rand() < 0.5:
            edge_width = np.random.randint(3, 15)
            edge_side = np.random.choice(['left', 'right', 'front', 'back'])
            
            if edge_side == 'left':
                cube_aug[:, :, :edge_width] = np.random.normal(0, 0.01, 
                                                                (128, 128, edge_width))
                mask_aug[:, :, :edge_width] = 0
            elif edge_side == 'right':
                cube_aug[:, :, -edge_width:] = np.random.normal(0, 0.01, 
                                                                 (128, 128, edge_width))
                mask_aug[:, :, -edge_width:] = 0
            elif edge_side == 'front':
                cube_aug[:, :edge_width, :] = np.random.normal(0, 0.01, 
                                                                (128, edge_width, 128))
                mask_aug[:, :edge_width, :] = 0
            elif edge_side == 'back':
                cube_aug[:, -edge_width:, :] = np.random.normal(0, 0.01, 
                                                                 (128, edge_width, 128))
                mask_aug[:, -edge_width:, :] = 0
        
        # Option 3: Add random "dead trace" columns (simulates acquisition gaps)
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
                
                # Amplitude scaling (prevents overfitting to specific amplitudes)
                scale = 0.9 + 0.2 * np.random.rand()
                cube = cube * scale
                
                # Gaussian noise (makes model robust to noise)
                if np.random.rand() < 0.3:
                    noise = np.random.normal(0, 0.02, cube.shape)
                    cube = cube + noise
                
                # NEW: Synthetic water/edge injection
                if np.random.rand() < SYNTH_AUG_PROB:
                    cube, mask = self.add_synthetic_water_edges(cube, mask)

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
            nn.Dropout3d(0.3)  # Higher dropout in bottleneck
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
# 4. METRICS, LOSS & UTILS
# ==========================================
def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU) metric"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # Avoid division by zero
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()

def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient metric"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection + 1e-7) / (pred_binary.sum() + target_binary.sum() + 1e-7)
    
    return dice.item()

def compute_metrics_with_tta(model, imgs, masks, device):
    """
    Compute metrics using Test-Time Augmentation (TTA).
    Averages predictions across multiple augmented versions.
    """
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(imgs))
        predictions.append(pred)
    
    # Flip Z
    with torch.no_grad():
        imgs_flipped = torch.flip(imgs, dims=[4])  # Flip along Z dimension
        pred = torch.sigmoid(model(imgs_flipped))
        pred = torch.flip(pred, dims=[4])  # Flip back
        predictions.append(pred)
    
    # Flip Y
    with torch.no_grad():
        imgs_flipped = torch.flip(imgs, dims=[3])  # Flip along Y dimension
        pred = torch.sigmoid(model(imgs_flipped))
        pred = torch.flip(pred, dims=[3])  # Flip back
        predictions.append(pred)
    
    # Flip X
    with torch.no_grad():
        imgs_flipped = torch.flip(imgs, dims=[2])  # Flip along X dimension
        pred = torch.sigmoid(model(imgs_flipped))
        pred = torch.flip(pred, dims=[2])  # Flip back
        predictions.append(pred)
    
    # Average all predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    
    # Compute metrics
    iou = compute_iou(avg_pred, masks)
    dice = compute_dice(avg_pred, masks)
    
    return iou, dice, avg_pred

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1):
        # inputs = raw logits (no sigmoid applied yet)
        # targets = 0 or 1 (ground truth)
        
        # 1. BCE with Logits + Class Imbalance Handling
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
    
    print("\n" + "="*70)
    print("LOADING DATA FROM MULTIPLE SOURCES")
    print("="*70)
    
    # Create datasets for each source
    train_datasets = []
    val_datasets = []
    
    for source in DATA_SOURCES:
        print(f"\nSource: {source['name']}")
        print("-" * 70)
        
        # Check if directories exist
        train_exists = os.path.exists(source['train'])
        val_exists = os.path.exists(source['val'])
        
        if not train_exists:
            print(f"  ⚠ WARNING: Training directory not found: {source['train']}")
            print(f"  Skipping this data source...")
            continue
            
        if not val_exists:
            print(f"  ⚠ WARNING: Validation directory not found: {source['val']}")
            print(f"  Skipping this data source...")
            continue
        
        # Create datasets
        train_ds = SaltDataset(source['train'], augment=True, dataset_name=source['name'])
        val_ds = SaltDataset(source['val'], augment=False, dataset_name=source['name'])
        
        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        else:
            print(f"  ⚠ WARNING: No training files found in {source['train']}")
            
        if len(val_ds) > 0:
            val_datasets.append(val_ds)
        else:
            print(f"  ⚠ WARNING: No validation files found in {source['val']}")
    
    # Check if we have any data
    if len(train_datasets) == 0:
        print("\n❌ ERROR: No training data found in any source!")
        print("Please check your DATA_SOURCES paths and ensure .npz files exist.")
        return
        
    if len(val_datasets) == 0:
        print("\n❌ ERROR: No validation data found in any source!")
        print("Please check your DATA_SOURCES paths and ensure .npz files exist.")
        return
    
    if USE_SUBSET < 1.0:
        print(f"\n⚡ Using {USE_SUBSET*100:.0f}% of training data for faster iteration")
        for ds in train_datasets:
            original_count = len(ds.files)
            random.shuffle(ds.files)
            ds.files = ds.files[:int(len(ds.files) * USE_SUBSET)]
            print(f"  [{ds.dataset_name}]: {original_count} → {len(ds.files)} cubes")
        
        for ds in val_datasets:
            original_count = len(ds.files)
            random.shuffle(ds.files)
            ds.files = ds.files[:int(len(ds.files) * USE_SUBSET)]
            print(f"  [{ds.dataset_name}] val: {original_count} → {len(ds.files)} cubes")
    
    # Combine all datasets using ConcatDataset
    print("\n" + "="*70)
    print("COMBINING DATASETS")
    print("="*70)
    
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    print(f"Total Training Cubes: {len(combined_train)}")
    print(f"  Breakdown:")
    for ds in train_datasets:
        print(f"    [{ds.dataset_name}]: {len(ds)} cubes")
    
    print(f"\nTotal Validation Cubes: {len(combined_val)}")
    print(f"  Breakdown:")
    for ds in val_datasets:
        print(f"    [{ds.dataset_name}]: {len(ds)} cubes")
    
    # Create DataLoaders
    train_loader = DataLoader(
        combined_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True,           # Shuffle mixes all datasets together!
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
    print("INITIALIZING MODEL AND TRAINING")
    print("="*70)

    # Init Model with dropout
    model = SaltModel3D(dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True,
        min_lr=1e-7
    )
    
    # Loss with class imbalance handling
    criterion = DiceBCELoss(pos_weight=2.0)
    scaler = torch.amp.GradScaler('cuda')

    # Early stopping tracking
    best_loss = 999.0
    best_iou = 0.0
    patience_counter = 0
    
    # Metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'lr': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
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
        
        # Validation with metrics
        model.eval()
        val_loss = 0
        val_iou_total = 0
        val_dice_total = 0
        
        print("\nRunning validation...")
        val_loop = tqdm(val_loader, leave=True)
        
        for imgs, masks in val_loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            if USE_TTA:
                # Test-Time Augmentation (slower but more accurate)
                iou, dice, _ = compute_metrics_with_tta(model, imgs, masks, DEVICE)
                
                # Still compute loss on original prediction
                with torch.no_grad():
                    preds = model(imgs)
                    loss = criterion(preds, masks)
            else:
                # Standard validation (faster)
                with torch.no_grad():
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                    preds_sigmoid = torch.sigmoid(preds)
                    
                    iou = compute_iou(preds_sigmoid, masks)
                    dice = compute_dice(preds_sigmoid, masks)
            
            val_loss += loss.item()
            val_iou_total += iou
            val_dice_total += dice
            
            val_loop.set_postfix(loss=loss.item(), iou=iou, dice=dice)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_total / len(val_loader)
        avg_val_dice = val_dice_total / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"\n{'─'*60}")
        print(f"Train Loss: {avg_train:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Val IoU:    {avg_val_iou:.4f}")
        print(f"Val Dice:   {avg_val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"{'─'*60}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save Checkpoint & Early Stopping Logic
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_loss = avg_val_loss
            patience_counter = 0  # Reset counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_iou': avg_val_iou,
                'val_dice': avg_val_dice,
                'data_sources': [s['name'] for s in DATA_SOURCES],  # Track what data was used
            }, f"{SAVE_DIR}/best_model.pth")
            print(f"✓ Saved Best Model! (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("\n⚠ Early stopping triggered - validation metrics stopped improving")
                break
    
    # Save training history
    np.savez(f"{SAVE_DIR}/training_history.npz", **history)
    
    # Save dataset info
    with open(f"{SAVE_DIR}/dataset_info.txt", 'w') as f:
        f.write("DATA SOURCES USED IN TRAINING\n")
        f.write("="*70 + "\n\n")
        for source in DATA_SOURCES:
            f.write(f"Source: {source['name']}\n")
            f.write(f"  Train: {source['train']}\n")
            f.write(f"  Val:   {source['val']}\n\n")
        f.write(f"\nTotal Training Cubes: {len(combined_train)}\n")
        f.write(f"Total Validation Cubes: {len(combined_val)}\n")
    
    # Plot training curves
    try:
        plot_training_curves(history, SAVE_DIR)
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")
    
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Best Validation IoU:  {best_iou:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Model saved to: {SAVE_DIR}/best_model.pth")
    print(f"Dataset info saved to: {SAVE_DIR}/dataset_info.txt")
    print("="*60)

def plot_training_curves(history, save_dir):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(history['val_dice'], label='Val Dice', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
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