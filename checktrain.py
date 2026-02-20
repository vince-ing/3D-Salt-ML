import numpy as np
import torch
from pathlib import Path

MODEL_PATH = "experiments/multi_dataset_run_01/best_model.pth"

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print("="*70)
print("CHECKPOINT ANALYSIS")
print("="*70)

# Basic info
if isinstance(checkpoint, dict):
    print(f"\nEpoch saved: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'Unknown')}")
    print(f"Val IoU: {checkpoint.get('val_iou', 'Unknown')}")
    print(f"Val Dice: {checkpoint.get('val_dice', 'Unknown')}")
    
    if 'data_sources' in checkpoint:
        print(f"\nDatasets used: {checkpoint['data_sources']}")
else:
    print("Old checkpoint format (no metadata)")

# Check training history
history_path = Path(MODEL_PATH).parent / "training_history.npz"
if history_path.exists():
    print(f"\n{'='*70}")
    print("TRAINING HISTORY FOUND")
    print("="*70)
    
    history = np.load(history_path)
    
    print(f"\nTotal epochs trained: {len(history['train_loss'])}")
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Val IoU:    {history['val_iou'][-1]:.4f}")
    print(f"  Val Dice:   {history['val_dice'][-1]:.4f}")
    print(f"  Final LR:   {history['lr'][-1]:.2e}")
    
    print(f"\nBest metrics (across all epochs):")
    best_epoch = np.argmax(history['val_iou'])
    print(f"  Best IoU at epoch {best_epoch+1}: {history['val_iou'][best_epoch]:.4f}")
    print(f"  Val Loss at that epoch: {history['val_loss'][best_epoch]:.4f}")
    
    print(f"\nEpoch-by-epoch summary:")
    print("Epoch | Train Loss | Val Loss | Val IoU | Val Dice | LR")
    print("-"*70)
    for i in range(len(history['train_loss'])):
        print(f"{i+1:5d} | {history['train_loss'][i]:10.4f} | {history['val_loss'][i]:8.4f} | "
              f"{history['val_iou'][i]:7.4f} | {history['val_dice'][i]:8.4f} | {history['lr'][i]:.2e}")
    
    # Check for early stopping pattern
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print("="*70)
    
    epochs_trained = len(history['train_loss'])
    if epochs_trained < 5:
        print("⚠ WARNING: Only trained for 2-3 epochs!")
        print("   Possible causes:")
        print("   1. Script crashed or was interrupted")
        print("   2. Validation loss diverged immediately")
        print("   3. Out of memory error")
    
    # Check if loss is improving
    if history['val_loss'][-1] > history['val_loss'][0]:
        print("⚠ WARNING: Validation loss got WORSE during training")
        print("   This suggests learning rate is too high or data issue")
    
    # Check LR drops
    lr_changes = np.diff(history['lr'])
    num_lr_drops = np.sum(lr_changes < 0)
    if num_lr_drops > 0:
        print(f"ℹ Learning rate was reduced {num_lr_drops} times")
        
else:
    print("\n❌ No training_history.npz found")
    print("   Training might have crashed very early")

print("\n" + "="*70)