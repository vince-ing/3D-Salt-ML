import numpy as np
import glob
import os
import random
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv3D, Lambda, Reshape, Add, Softmax, 
                                     Dropout, MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D, 
                                     multiply, concatenate, Activation, BatchNormalization)

# ==========================================
# 1. CUSTOM MODEL FUNCTIONS
# ==========================================
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
    z = Dense(kernel*2)(z)

    z = Reshape([1, 1, kernel, m])(z)
    scale = Softmax()(z)

    x = Lambda(lambda x: tf.stack(x, axis=-1))([x1, x2])
    r = multiply([scale, x])
    r = Lambda(lambda x: K.sum(x, axis=-1))(r)
    return r

def DenseNet(inp, layers, filters):
    for i in range(layers):
        if i == 0:
            x = Conv3D(filters, 3, padding='same', strides=1)(inp)
        else:
            x = Conv3D(filters, 3, padding='same', strides=1)(x4)
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
    x1 = BatchNormalization()(x)
    x2 = Activation('relu')(x1)
    x3 = x2
    x4 = MaxPooling3D(U)(x3)
    return x4

def TU(inp, filters, U):
    x = Conv3DTranspose(filters, 3, padding='same', strides=U)(inp)
    x1 = BatchNormalization()(x)
    x2 = Activation('relu')(x1)
    return x2

# ==========================================
# 2. BUILD THE ARCHITECTURE & LOAD WEIGHTS
# ==========================================
print("Building model architecture...")
D = 2
layer = 1
layers_num = 1 # Renamed slightly to avoid conflict with keras.layers
w1, w2, w3 = 128, 128, 100
input_dims = (w1, w2, w3, 1)
inpt_img = Input(shape=input_dims)

C1 = Conv3D(D, 3, padding='same', strides=1)(inpt_img)
C1 = BatchNormalization()(C1)
C1 = Activation('relu')(C1)

DB1 = DenseNet(C1, layer, D * 2)
DB1 = SK(DB1, m=2, r=2, L=8, kernel=D * 2)
TD1 = TD(DB1, D * 2, 2)

DB2 = DenseNet(TD1, layer, D * 4)
DB2 = SK(DB2, m=2, r=2, L=8, kernel=D * 4)
TD2 = TD(DB2, D * 4, 2)

DB3 = DenseNet(TD2, layer, D * 8)
DB3 = SK(DB3, m=2, r=2, L=8, kernel=D * 8)
TD3 = TD(DB3, D * 8, (2, 2, 5))

DB5 = DenseNet(TD3, layer, D * 16)
DB5 = SK(DB5, m=2, r=2, L=8, kernel=D * 16)

# Reconstruction (Encoder1)
TU2 = TU(DB5, D*8, (2, 2, 5))
DF2 = DenseNet(TU2, layer, D*8)
DF2S= SK(DF2, m=2, r=2, L=8, kernel=D*8)

TU3 = TU(DF2S, D*4, 2)
DF3 = DenseNet(TU3, layer, D*4)
DF3S= SK(DF3, m=2, r=2, L=8, kernel=D*4)

TU4= TU(DF3S, D*2, 2)
DF4 = DenseNet(TU4, layer, D*2)
DF4S= SK(DF4, m=2, r=2, L=8, kernel=D*2)

DF4F = Conv3D(D, 3, padding='same', strides=1)(DF4S)
DF4F = BatchNormalization()(DF4F)
DF4F = Activation('relu')(DF4F)
out_seg = Conv3D(1, 3, padding='same', activation='linear', name='out_rec')(DF4F)

# Label Reconstruction (Encoder2)
lab = TU(DB5, D*8, (2, 2, 5))
lab = DenseNet(lab, layers_num, D*8)
lab = SK(lab, m=2, r=2, L=8, kernel=D*8)

lab = TU(lab, D*4, 2)
lab = DenseNet(lab, layers_num, D*4)
lab = SK(lab, m=2, r=2, L=8, kernel=D*4)

lab = TU(lab, D * 2, 2)
lab = DenseNet(lab, layers_num, D * 2)
lab = SK(lab, m=2, r=2, L=8, kernel=D * 2)

lab = Conv3D(1, 1, padding='same', strides=1, activation='sigmoid', name='outlab')(lab)

model = Model(inpt_img, outputs=[out_seg, lab])

# Load Weights
model_path = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\model\best_model_our_outBT.h5'
model.load_weights(model_path)
print("Model weights loaded successfully!")

# ==========================================
# 3. LOAD & PREPROCESS INDIVIDUAL .NPZ FILES
# ==========================================
print("Finding test patches...")
test_dir = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\mississippi_100i_128x_128z\train"

# Find all .npz files in the test directory
all_test_files = glob.glob(os.path.join(test_dir, "*.npz"))

if not all_test_files:
    raise ValueError(f"No .npz files found in {test_dir}. Check the path!")

num_examples = 4
# Pick random files to test
files_to_view = random.sample(all_test_files, num_examples)

print(f"Loading and predicting {num_examples} patches...")

patch_results = []
for filepath in files_to_view:
    data = np.load(filepath)
    seismic_patch = data['seismic']
    label_patch = data['label']
    
    # Absolute Max Normalization
    max_abs = np.abs(seismic_patch).max()
    if max_abs > 0:
        seismic_patch = seismic_patch / max_abs
        
    label_patch = (label_patch == 1).astype(np.float32)

    # Standard reshape: (1, 128, 128, 100, 1) -> (Batch, Crossline, Depth, Inline, Channel)
    x_sample = np.reshape(seismic_patch, (1, 128, 128, 100, 1))
    y_true = np.reshape(label_patch, (1, 128, 128, 100, 1))

    # Predict directly on the properly shaped data
    predictions = model.predict(x_sample, verbose=0)
    
    # Extract the predicted mask (shape is 128, 128, 100, 1)
    y_pred_mask = (predictions[1][0] > 0.5).astype(np.float32)
    
    # Store for plotting (removing batch and channel dimensions)
    # Stored shape is (128, 128, 100) -> (Crossline, Depth, Inline)
    patch_results.append({
        'filepath': filepath,
        'seismic': x_sample[0, :, :, :, 0],
        'true': y_true[0, :, :, :, 0],
        'pred': y_pred_mask[:, :, :, 0]
    })

# ==========================================
# 4. PREDICT & VISUALIZE
# ==========================================

# 2. Define the 3 slicing geometries for the native (Crossline, Depth, Inline) shape
geometries = [
    # Axis 1 is Depth (128). Middle is 64.
    {'name': 'Map View (Z-Slice Z=64)', 'axis': 1, 'idx': 64},
    # Axis 2 is Inline (100). Middle is 50. 
    {'name': 'Crossline Profile (Inline-Slice I=50)', 'axis': 2, 'idx': 50},
    # Axis 0 is Crossline (128). Middle is 64.
    {'name': 'Inline Profile (Crossline-Slice X=64)', 'axis': 0, 'idx': 64}
]

print("Generating the 3 Geometry Figures...")

# 3. Generate one full 4-row figure for each geometry
for geo in geometries:
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    fig.suptitle(f"Geometry: {geo['name']}", fontsize=16, fontweight='bold')
    
    for i, p_data in enumerate(patch_results):
        ax_row = axes[i]
        
        # Clean up the title
        filename = os.path.basename(p_data['filepath'])
        match = re.search(r'_i(\d+)_x(\d+)_s(\d+)', filename)
        if match:
            i_idx, x_idx, s_idx = match.groups()
            clean_title = f"Seismic\n(IL:{int(i_idx)} XL:{int(x_idx)} Z:{int(s_idx)})"
        else:
            clean_title = f"Seismic\n{filename[:15]}..."
            
        # Extract the correct 2D slice based on the current geometry
        # Remembering our base shape is (Crossline, Depth, Inline)
        if geo['axis'] == 1:
            # Depth Slice. Data -> (XL, IL). 
            # We transpose to (IL, XL) just so Inline is on Y-axis for standard map views.
            s_slice = p_data['seismic'][:, geo['idx'], :].T
            t_slice = p_data['true'][:, geo['idx'], :].T
            p_slice = p_data['pred'][:, geo['idx'], :].T
            
        elif geo['axis'] == 2:
            # Inline Slice (Crossline Profile). Data -> (XL, Depth).
            # Must transpose to (Depth, XL) so Depth is correctly on the Y-axis!
            s_slice = p_data['seismic'][:, :, geo['idx']].T
            t_slice = p_data['true'][:, :, geo['idx']].T
            p_slice = p_data['pred'][:, :, geo['idx']].T
            
        elif geo['axis'] == 0:
            # Crossline Slice (Inline Profile). Data -> (Depth, IL).
            # Depth is ALREADY the first dimension, so it automatically plots on Y-axis! NO TRANSPOSE needed here.
            s_slice = p_data['seismic'][geo['idx'], :, :]
            t_slice = p_data['true'][geo['idx'], :, :]
            p_slice = p_data['pred'][geo['idx'], :, :]

        # Plot Seismic
        im1 = ax_row[0].imshow(s_slice, cmap='gray', aspect='auto')
        ax_row[0].set_title(clean_title, fontsize=10)
        fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)

        # Plot True Label
        im2 = ax_row[1].imshow(t_slice, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax_row[1].set_title('True Salt Label', fontsize=10)
        fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)

        # Plot Predicted Label
        im3 = ax_row[2].imshow(p_slice, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax_row[2].set_title('Predicted Salt Label', fontsize=10)
        fig.colorbar(im3, ax=ax_row[2], fraction=0.046, pad=0.04)
        
        # Turn off axis ticks for a cleaner look
        ax_row[0].axis('off')
        ax_row[1].axis('off')
        ax_row[2].axis('off')

    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.92, bottom=0.05)
    plt.show()