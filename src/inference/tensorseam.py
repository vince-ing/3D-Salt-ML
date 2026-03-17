import numpy as np
import matplotlib.pyplot as plt
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
# 3. LOAD & PREPROCESS DATA (CRITICAL)
# ==========================================
print("Loading and preprocessing data...")
data_path = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\seam_data\samples.bin'
label_path = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\seam_data\labels.bin'

# Use -1 to automatically detect N based on file size
data = np.fromfile(data_path, dtype=np.float32).reshape((-1, 100, 128, 128))
label = np.fromfile(label_path, dtype=np.float32).reshape((-1, 100, 128, 128))

# Transpose
data = np.transpose(data, (0, 2, 3, 1))
label = np.transpose(label, (0, 2, 3, 1))

# Normalize the data cubes exactly like training
for ii in range(data.shape[0]):
    tmp = np.abs(data[ii,:,:,:]).max()
    if tmp > 0: 
        data[ii,:,:,:] = data[ii,:,:,:] / tmp

# Reshape to add channel dimension
X_test = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1))
Y_test = np.reshape(label, (label.shape[0], label.shape[1], label.shape[2], label.shape[3], 1))
print("Data ready for inference!")

# ==========================================
# 4. PREDICT & VISUALIZE
# ==========================================
import random

# How many examples do you want to see?
num_examples = 4

# Pick random samples from your test set
# (Or you can manually set a list like indices_to_view = [0, 5, 10, 15])
total_samples = X_test.shape[0]
indices_to_view = random.sample(range(total_samples), num_examples)

print(f"Running predictions for samples: {indices_to_view}...")

# Set up a large figure with rows = num_examples, and 3 columns
fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
slice_idx = 50 # Middle slice of the 100-depth volume

for i, sample_index in enumerate(indices_to_view):
    # Extract the single sample
    x_sample = X_test[sample_index:sample_index+1] 
    y_true = Y_test[sample_index]

    # Predict
    predictions = model.predict(x_sample, verbose=0) # verbose=0 hides the progress bar per prediction
    
    # Output 1 is the salt label (outlab)
    y_pred = predictions[1][0] 
    y_pred_mask = (y_pred > 0.5).astype(np.float32)

    # Extract 2D slices for plotting and hit 'em with the .T (Transpose)
    # This flips (Crossline, Depth) to (Depth, Crossline) so it plots horizontally!
    seismic_slice = x_sample[0, :, :, slice_idx, 0].T
    true_label_slice = y_true[:, :, slice_idx, 0].T
    pred_label_slice = y_pred_mask[:, :, slice_idx, 0].T

    # Plotting logic for this specific row
    ax_row = axes[i]
    
    im1 = ax_row[0].imshow(seismic_slice, cmap='gray_r', aspect='auto')
    ax_row[0].set_title(f'Sample {sample_index}: Seismic (Slice {slice_idx})')
    fig.colorbar(im1, ax=ax_row[0], fraction=0.046, pad=0.04)

    im2 = ax_row[1].imshow(true_label_slice, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax_row[1].set_title(f'Sample {sample_index}: True Label')
    fig.colorbar(im2, ax=ax_row[1], fraction=0.046, pad=0.04)

    im3 = ax_row[2].imshow(pred_label_slice, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax_row[2].set_title(f'Sample {sample_index}: Predicted Label')
    fig.colorbar(im3, ax=ax_row[2], fraction=0.046, pad=0.04)

plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.95, bottom=0.05)
plt.show()