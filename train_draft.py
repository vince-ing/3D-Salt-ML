import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # <--- NEW: Saves VRAM
from tqdm import tqdm
import os
import numpy as np

# --- CONFIG ---
BATCH_SIZE = 4            # Reduced from 12 for 3D
LEARNING_RATE = 1e-4      
EPOCHS = 20               
MODEL_SAVE_DIR = "models/3d_unet"
# --------------

def robust_normalize_3d(imgs):
    """
    3D Adaptation: Calculates stats over Depth(2), Height(3), Width(4)
    Input shape: (Batch, Channel, D, H, W)
    """
    # Calculate stats over spatial dims (2, 3, 4) instead of just (2, 3)
    mean = imgs.mean(dim=(2, 3, 4), keepdim=True)
    std = imgs.std(dim=(2, 3, 4), keepdim=True)
    
    # Clip at 2.5 std devs
    lower = mean - 2.5 * std
    upper = mean + 2.5 * std
    imgs = torch.clamp(imgs, min=lower, max=upper)
    
    # Min-Max Scale to [-1, 1]
    min_val = imgs.amin(dim=(2, 3, 4), keepdim=True)
    max_val = imgs.amax(dim=(2, 3, 4), keepdim=True)
    
    imgs = 2 * (imgs - min_val) / (max_val - min_val + 1e-6) - 1.0
    return imgs

class FocalTverskyLoss3D(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, D, H, W) -> Logits
        # targets: (B, D, H, W)   -> Indices (0 or 1)
        
        # 1. Softmax over Channel dim (1)
        inputs_soft = torch.softmax(inputs, dim=1)
        
        # 2. Extract Salt Class (Assuming Class 1 is Salt)
        # Your mask is binary (0=Background, 1=Salt), so we take index 1
        salt_inputs = inputs_soft[:, 1, :, :, :] 
        salt_targets = (targets == 1).float()
        
        # 3. Flatten for Tversky (Batch, -1)
        inputs_flat = salt_inputs.contiguous().view(salt_inputs.size(0), -1)
        targets_flat = salt_targets.contiguous().view(salt_targets.size(0), -1)
        
        # 4. Calculate Tversky
        TP = (inputs_flat * targets_flat).sum(1)
        FP = ((1-targets_flat) * inputs_flat).sum(1)
        FN = (targets_flat * (1-inputs_flat)).sum(1)
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        # 5. Focal Modulation
        loss = (1 - tversky)**self.gamma
        return loss.mean()

def train_3d():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(n_channels=1, n_classes=2).to(device) # <--- Make sure this is 3D!
    
    criterion = FocalTverskyLoss3D()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # <--- For Mixed Precision
    
    # Dataloaders (Assume you created these using the code from before)
    # train_loader = ... 
    
    print("Starting 3D Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for imgs, masks in loop:
            imgs = imgs.to(device).float()
            masks = masks.to(device).long()
            
            # Normalization
            imgs = robust_normalize_3d(imgs)
            
            # --- Mixed Precision Training ---
            # This runs the forward pass in float16 (fast) 
            # and backward pass in float32 (stable)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # -------------------------------
            
            loop.set_postfix(loss=loss.item())
            
        # Validation Loop would go here...
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_3d()
