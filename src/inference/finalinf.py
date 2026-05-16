"""
3D Model Comparison Viewer — PyVista (TensorFlow/Keras)
=======================================================
Loads a random .npz patch from your validation set, evaluates it sequentially 
across your three models to prevent OOM errors, and visualizes the results 
in a linked 4-pane interactive 3D window.

Dependencies:
    pip install pyvista numpy tensorflow
"""

import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv3D, Lambda, Reshape, Add, Softmax,
                                     MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D,
                                     multiply, concatenate, Activation, BatchNormalization)

# Setup PyVista for interactive 3D
os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")
import pyvista as pv
pv.global_theme.background = "black"
pv.global_theme.font.color = "white"

# ============================================================
# 1. CONFIGURATION
# ============================================================
MIN_SALT_RATIO = 0.20  # Cube must be at least 5% salt
MAX_SALT_RATIO = 0.55  # Cube must be no more than 85% salt

DATA_DIRS = [
    r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\keathley_100i_128x_128z\val',
    r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi_100i_128x_128z\train' 
]

MODELS_TO_COMPARE = {
    "Base Binary": {
        "type": "binary", 
        "path": r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\model\best_model_our_outBT.h5',
        "color": "#1e90ff" # Blue
    },
    "Multiclass Finetuned": {
        "type": "multiclass", 
        "path": r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\model\best_model_multiclass_finetuneNew.h5',
        "color": "#00cc44" # Green
    },
    "Multiclass Scratch": {
        "type": "multiclass", 
        "path": r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\model\best_model_multiclass_NOfinetune.h5',
        "color": "#ffaa00" # Orange
    }
}

# ============================================================
# 2. MODEL ARCHITECTURE BLOCKS
# ============================================================
def SK(inputs, m=2, r=2, L=32, kernel=4):
    d = max(int(kernel * r), L)
    x1 = Conv3D(kernel, 3, strides=1, padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    _x1 = GlobalAveragePooling3D()(x1)
    x2 = Conv3D(kernel, 5, strides=1, padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    _x2 = GlobalAveragePooling3D()(x2)
    U = Add()([_x1, _x2])
    z = Dense(d, activation='relu')(U)
    z = Dense(kernel * 2)(z)
    z = Reshape([1, 1, kernel, m])(z)
    scale = Softmax()(z)
    x = Lambda(lambda x: tf.stack(x, axis=-1))([x1, x2])
    r = multiply([scale, x])
    r = Lambda(lambda x: K.sum(x, axis=-1))(r)
    return r

def DenseNet(inp, layers, filters):
    for i in range(layers):
        x = Conv3D(filters, 3, padding='same', strides=1)(inp if i == 0 else x4)
        x1 = BatchNormalization()(x)
        x2 = Activation('relu')(x1)
        x3 = x2
        if i == 0:
            x4 = concatenate([x3, inp])
            x5 = x4
        else:
            x4 = concatenate([x3, x4])
        if (i > 0) and (i < layers - 1):
            x5 = concatenate([x5, x4])
    return x5

def TD(inp, filters, U):
    x = Conv3D(filters, 1, padding='same', strides=1)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(U)(x)
    return x

def TU(inp, filters, U):
    x = Conv3DTranspose(filters, 3, padding='same', strides=U)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_model(model_type="binary", num_classes=4):
    D, layer = 2, 1
    inpt_img = Input(shape=(128, 128, 100, 1))
    
    C1 = Activation('relu')(BatchNormalization()(Conv3D(D, 3, padding='same', strides=1)(inpt_img)))
    DB1 = SK(DenseNet(C1, layer, D * 2), m=2, r=2, L=8, kernel=D * 2)
    TD1 = TD(DB1, D * 2, 2)
    DB2 = SK(DenseNet(TD1, layer, D * 4), m=2, r=2, L=8, kernel=D * 4)
    TD2 = TD(DB2, D * 4, 2)
    DB3 = SK(DenseNet(TD2, layer, D * 8), m=2, r=2, L=8, kernel=D * 8)
    TD3 = TD(DB3, D * 8, (2, 2, 5))
    DB5 = SK(DenseNet(TD3, layer, D * 16), m=2, r=2, L=8, kernel=D * 16)

    # Reconstruction Branch
    TU2 = TU(DB5, D * 8, (2, 2, 5))
    DF2S = SK(DenseNet(TU2, layer, D * 8), m=2, r=2, L=8, kernel=D * 8)
    TU3 = TU(DF2S, D * 4, 2)
    DF3S = SK(DenseNet(TU3, layer, D * 4), m=2, r=2, L=8, kernel=D * 4)
    TU4 = TU(DF3S, D * 2, 2)
    DF4S = SK(DenseNet(TU4, layer, D * 2), m=2, r=2, L=8, kernel=D * 2)
    DF4F = Activation('relu')(BatchNormalization()(Conv3D(D, 3, padding='same', strides=1)(DF4S)))
    out_rec = Conv3D(1, 3, padding='same', activation='linear', name='out_rec')(DF4F)

    # Label Branch
    lab = TU(DB5, D * 8, (2, 2, 5))
    lab = SK(DenseNet(lab, layer, D * 8), m=2, r=2, L=8, kernel=D * 8)
    lab = TU(lab, D * 4, 2)
    lab = SK(DenseNet(lab, layer, D * 4), m=2, r=2, L=8, kernel=D * 4)
    lab = TU(lab, D * 2, 2)
    lab = SK(DenseNet(lab, layer, D * 2), m=2, r=2, L=8, kernel=D * 2)

    if model_type == "binary":
        out_seg = Conv3D(1, 1, padding='same', strides=1, activation='sigmoid', name='outlab')(lab)
    else:
        out_seg = Conv3D(num_classes, 1, padding='same', strides=1, activation='softmax', name='outlab')(lab)

    return Model(inpt_img, outputs=[out_rec, out_seg])

# ============================================================
# 3. PYVISTA HELPERS
# ============================================================
def volume_to_pyvista(arr):
    """Wrap a 3D numpy array into a PyVista ImageData grid."""
    grid = pv.ImageData()
    grid.dimensions = np.array(arr.shape) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["values"] = arr.flatten(order="F")
    return grid

def iso_surface(binary_vol, smooth_iter=30):
    """Marching-cubes surface from a binary volume with smoothing."""
    padded = np.pad(binary_vol.astype(np.float32), pad_width=1, mode="constant", constant_values=0)
    grid = volume_to_pyvista(padded).cell_data_to_point_data()
    surf = grid.contour([0.5], scalars="values")
    if surf.n_points > 0 and smooth_iter > 0:
        surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.1)
    return surf

def calculate_iou(pred, true):
    inter = np.sum((pred == 1) & (true == 1))
    union = np.sum((pred == 1) | (true == 1))
    return float(inter / union) if union > 0 else 0.0

# ============================================================
# 4. MAIN EXECUTION
# ============================================================
def run():
    # Allow VRAM growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Find and load a random patch
    all_files = []
    for d in DATA_DIRS:
        all_files.extend(glob.glob(os.path.join(d, "*.npz")))
    
    if not all_files:
        raise ValueError("No .npz files found. Check paths.")

    print(f"Searching {len(all_files)} files for a patch with good salt geometry...")
    random.shuffle(all_files)
    
    filepath = None
    seismic = None
    label_vol = None
    
    # Loop through shuffled files until we find one with a good salt ratio
    for f in all_files:
        data = np.load(f)
        temp_lbl = data['label'].astype(np.uint8)
        salt_ratio = np.mean(temp_lbl == 1)
        
        if MIN_SALT_RATIO <= salt_ratio <= MAX_SALT_RATIO:
            filepath = f
            seismic = data['seismic'].astype(np.float32)
            label_vol = temp_lbl
            print(f"Loaded patch: {os.path.basename(filepath)} (Salt ratio: {salt_ratio:.1%})")
            break
            
    # Fallback just in case
    if filepath is None:
        print("Warning: No patch found matching ratio filter. Using a random one.")
        filepath = random.choice(all_files)
        data = np.load(filepath)
        seismic = data['seismic'].astype(np.float32)
        label_vol = data['label'].astype(np.uint8)
        
    # Normalize seismic
    max_abs = np.abs(seismic).max()
    if max_abs > 0:
        seismic /= max_abs

    # Isolate ground truth salt
    gt_salt = (label_vol == 1).astype(np.uint8)
    
    # Run predictions sequentially
    x_input = seismic[np.newaxis, ..., np.newaxis]
    predictions = {}
    ious = {}

    for name, info in MODELS_TO_COMPARE.items():
        if not os.path.exists(info["path"]):
            print(f"Skipping {name} (weights not found)")
            continue
            
        print(f"\nProcessing {name}...")
        model = build_model(model_type=info["type"])
        model.load_weights(info["path"], by_name=True, skip_mismatch=True)
        
        pred = model.predict(x_input, verbose=0)[1][0]
        
        if info["type"] == "binary":
            pred_mask = (pred[..., 0] > 0.5).astype(np.uint8)
        else:
            pred_classes = np.argmax(pred, axis=-1)
            pred_mask = (pred_classes == 1).astype(np.uint8)
            
        predictions[name] = pred_mask
        ious[name] = calculate_iou(pred_mask, gt_salt)
        
        K.clear_session()
        del model

    print("\nGenerating 3D Surfaces...")
    seis_grid = volume_to_pyvista(seismic).cell_data_to_point_data()
    gt_surf = iso_surface(gt_salt)
    
    pred_surfs = {}
    for name, mask in predictions.items():
        if mask.sum() > 50: # Only build surface if salt exists
            pred_surfs[name] = iso_surface(mask)

    # ============================================================
    # 5. VISUALIZATION (4-PANE LINKED GRID)
    # ============================================================
    pl = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))
    pl.link_views() # Links cameras so rotating one rotates all of them!

    # Subplot 0,0: Ground Truth
    pl.subplot(0, 0)
    pl.add_mesh(seis_grid.outline(), color="white")
    if gt_surf.n_points > 0:
        pl.add_mesh(gt_surf, color="white", opacity=0.8, smooth_shading=True)
    pl.add_text("Ground Truth (Salt)", font_size=12, color="white")

    # Subplot 0,1: Base Binary
    pl.subplot(0, 1)
    pl.add_mesh(seis_grid.outline(), color="white")
    name = "Base Binary"
    if name in pred_surfs:
        pl.add_mesh(pred_surfs[name], color=MODELS_TO_COMPARE[name]["color"], opacity=0.8, smooth_shading=True)
    pl.add_text(f"{name}\nIoU: {ious.get(name, 0):.3f}", font_size=12, color=MODELS_TO_COMPARE[name]["color"])

    # Subplot 1,0: Multiclass Finetuned
    pl.subplot(1, 0)
    pl.add_mesh(seis_grid.outline(), color="white")
    name = "Multiclass Finetuned"
    if name in pred_surfs:
        pl.add_mesh(pred_surfs[name], color=MODELS_TO_COMPARE[name]["color"], opacity=0.8, smooth_shading=True)
    pl.add_text(f"{name}\nIoU: {ious.get(name, 0):.3f}", font_size=12, color=MODELS_TO_COMPARE[name]["color"])

    # Subplot 1,1: Multiclass Scratch
    pl.subplot(1, 1)
    pl.add_mesh(seis_grid.outline(), color="white")
    name = "Multiclass Scratch"
    if name in pred_surfs:
        pl.add_mesh(pred_surfs[name], color=MODELS_TO_COMPARE[name]["color"], opacity=0.8, smooth_shading=True)
    pl.add_text(f"{name}\nIoU: {ious.get(name, 0):.3f}", font_size=12, color=MODELS_TO_COMPARE[name]["color"])

    print("\nOpening Interactive 3D Window. Close the window to exit.")
    pl.camera_position = "iso"
    pl.show()

if __name__ == "__main__":
    run()