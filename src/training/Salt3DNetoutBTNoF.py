from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv3D, Lambda, Reshape, Add, Softmax,
                                     MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D,
                                     multiply, concatenate, Activation, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# CONFIGURATION — update paths as needed
# ============================================================
DATA_PATH  = '/home/ig-gbds/saltdata/samples.bin'
LABEL_PATH = '/home/ig-gbds/saltdata/labels.bin'

# Output model path
MODEL_SAVE_PATH = 'model/best_model_multiclass_NOfinetune.h5'

# Number of classes: 0=rock, 1=salt, 2=water, 3=blanked
NUM_CLASSES = 4

# ============================================================
# MODEL BUILDING BLOCKS (same architecture as before)
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

# ============================================================
# BUILD MODEL
# ============================================================
def build_model(num_classes=4):
    D = 2
    layer = 1
    input_dims = (128, 128, 100, 1)
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

    # --- Reconstruction branch (unchanged) ---
    TU2 = TU(DB5, D * 8, (2, 2, 5))
    DF2 = DenseNet(TU2, layer, D * 8)
    DF2S = SK(DF2, m=2, r=2, L=8, kernel=D * 8)

    TU3 = TU(DF2S, D * 4, 2)
    DF3 = DenseNet(TU3, layer, D * 4)
    DF3S = SK(DF3, m=2, r=2, L=8, kernel=D * 4)

    TU4 = TU(DF3S, D * 2, 2)
    DF4 = DenseNet(TU4, layer, D * 2)
    DF4S = SK(DF4, m=2, r=2, L=8, kernel=D * 2)

    DF4F = Conv3D(D, 3, padding='same', strides=1)(DF4S)
    DF4F = BatchNormalization()(DF4F)
    DF4F = Activation('relu')(DF4F)
    out_rec = Conv3D(1, 3, padding='same', activation='linear', name='out_rec')(DF4F)

    # --- Multiclass segmentation branch ---
    lab = TU(DB5, D * 8, (2, 2, 5))
    lab = DenseNet(lab, layer, D * 8)
    lab = SK(lab, m=2, r=2, L=8, kernel=D * 8)

    lab = TU(lab, D * 4, 2)
    lab = DenseNet(lab, layer, D * 4)
    lab = SK(lab, m=2, r=2, L=8, kernel=D * 4)

    lab = TU(lab, D * 2, 2)
    lab = DenseNet(lab, layer, D * 2)
    lab = SK(lab, m=2, r=2, L=8, kernel=D * 2)

    out_seg = Conv3D(num_classes, 1, padding='same', strides=1,
                     activation='softmax', name='outlab')(lab)

    return Model(inpt_img, outputs=[out_rec, out_seg])

# ============================================================
# LOAD & PREPROCESS DATA
# ============================================================
print("Loading data...")
N, IL, XL, Z = 495, 100, 128, 128
data  = np.fromfile(DATA_PATH,  dtype=np.float32).reshape((N, IL, XL, Z))
label = np.fromfile(LABEL_PATH, dtype=np.float32).reshape((N, IL, XL, Z))

print("Transposing...")
data  = np.transpose(data,  (0, 2, 3, 1))   # (N, 128, 128, 100)
label = np.transpose(label, (0, 2, 3, 1))   # (N, 128, 128, 100)

print("Normalizing seismic cubes...")
for ii in range(data.shape[0]):
    tmp = np.abs(data[ii]).max()
    if tmp > 0:
        data[ii] /= tmp

label_int = label.astype(np.int32)[..., np.newaxis]  # (N, 128, 128, 100, 1)

print("Final Data shape:",   data.shape)
print("Final Label shape:",  label_int.shape)

data_ch = data[..., np.newaxis]   # (N, 128, 128, 100, 1)

Xtrain, Xval, Ytrain, Yval = train_test_split(
    data_ch, label_int, test_size=0.2, random_state=42, shuffle=True
)
print(f"Train: {Xtrain.shape[0]} samples  |  Val: {Xval.shape[0]} samples")

# ============================================================
# BUILD MODEL & COMPILE
# ============================================================
print("\nBuilding model...")
model = build_model(num_classes=NUM_CLASSES)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'out_rec': 'mse',
        'outlab':  'sparse_categorical_crossentropy'
    },
    loss_weights={
        'out_rec': 1.0,
        'outlab':  1.0
    },
    metrics={
        'out_rec': 'mse',
        'outlab':  'accuracy'
    }
)

# ============================================================
# CALLBACKS & TRAINING
# ============================================================
os.makedirs('model', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_outlab_accuracy',
        mode='max',
        patience=10,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_outlab_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

print("\nStarting training from scratch on real seismic data (multiclass)...")
model.fit(
    Xtrain,
    {'out_rec': Xtrain, 'outlab': Ytrain},
    batch_size=2,
    epochs=100,
    callbacks=callbacks,
    validation_data=(Xval, {'out_rec': Xval, 'outlab': Yval})
)

print(f"\nDone! Best model saved to: {MODEL_SAVE_PATH}")