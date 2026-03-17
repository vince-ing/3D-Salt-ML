from sklearn.model_selection import train_test_split
import numpy as np
import os

# --- 1. Load Data ---
data_path = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\seam_data/samples.bin'
label_path = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\seam_data/labels.bin'

# Load with the proper geological shape we discovered (N, Inline, Crossline, Depth)
N, IL, XL, Z = 495, 100, 128, 128

print("Loading data...")
data = np.fromfile(data_path, dtype=np.float32).reshape((N, IL, XL, Z))
label = np.fromfile(label_path, dtype=np.float32).reshape((N, IL, XL, Z))

# --- 2. Transpose for the Network ---
# The network's TD3 layer uses a pool size of (2, 2, 5). 
# Therefore, the dimension of size 100 MUST be the 3rd spatial dimension.
# We transpose from (N, 100, 128, 128) -> (N, 128, 128, 100)
print("Transposing data to fit network architecture...")
data = np.transpose(data, (0, 2, 3, 1))
label = np.transpose(label, (0, 2, 3, 1))

# --- 3. Normalization ---
print("Normalizing cubes...")
# Use data.shape[0] instead of hardcoded 360
for ii in range(data.shape[0]):
    tmp = np.abs(data[ii,:,:,:]).max()
    # Avoid division by zero if a cube is completely blank
    if tmp > 0: 
        data[ii,:,:,:] = data[ii,:,:,:] / tmp

print("Final Data shape:", data.shape)    # Should be (495, 128, 128, 100)
print("Final Label shape:", label.shape)  # Should be (495, 128, 128, 100)

# --- 4. Ensure Model Save Directory Exists ---
if not os.path.exists('model'):
    os.makedirs('model')
    print("Created 'model' directory for saving checkpoints.")

from sklearn.model_selection import train_test_split

Xtrain, X_valid, Ytrain,y_valid =  train_test_split(data, label, test_size=0.8, random_state=42,shuffle=True)
Xtrain=np.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2], Xtrain.shape[3],1))
X_valid=np.reshape(X_valid,(X_valid.shape[0],X_valid.shape[1],X_valid.shape[2], X_valid.shape[3],1))
Ytrain=np.reshape(Ytrain,(Ytrain.shape[0],Ytrain.shape[1],Ytrain.shape[2], Ytrain.shape[3],1))
y_valid=np.reshape(y_valid,(y_valid.shape[0],y_valid.shape[1],y_valid.shape[2], y_valid.shape[3],1))



import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape,Add,Softmax
from tensorflow.keras.layers import Dropout, MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D,multiply, concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv3DTranspose
import os
import sys
import random
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def SK(inputs, m=2, r=2, L=32, kernel=4):
    #input_channel = inputs.get_shape().as_list()[-1]
    d = max(int(kernel * r), L)
    # 在这里可以考虑在一段代码用于消除混叠
    #out = Conv3D(kernel, 1, strides=1, padding='same')(inputs)
    #out = BatchNormalization()(out)
    #out = Activation('relu')(out)

    x1 = Conv3D(kernel, 3, strides=1, padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    #x1 = SpatialDropout3D(rate=0.1)(x1)
    _x1 = GlobalAveragePooling3D()(x1)

    x2 = Conv3D(kernel, 5, strides=1, padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    #x2 = SpatialDropout3D(rate=0.1)(x2)
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
    # inp     -  Inputof the DenseNet
    # layers  -  Number of layers
    # filters -  Number of feature maps
    for i in range(layers):
        if i == 0:
            #x = Conv3D(filters, 3, padding='same', strides=1, kernel_initializer="he_uniform")(inp)
            x = Conv3D(filters, 3, padding='same', strides=1)(inp)
        else:
            #x = Conv3D(filters, 3, padding='same', strides=1, kernel_initializer="he_uniform")(x4)
            x = Conv3D(filters, 3, padding='same', strides=1)(x4)
        x1 = BatchNormalization()(x)
        x2 = Activation('relu')(x1)
        x3 = x2
        # x3 = Dropout(0.1)(x2)
        if i == 0:
            x4 = concatenate([x3, inp])
            x5 = x4
        else:
            x4 = concatenate([x3, x4])

        if (i > 0) and (i < layers - 1):
            x5 = concatenate([x5, x4])
    return x5

def TD(inp, filters, U):
    #inp     -  Inputof the TD network
    #filters -  Number of feature maps
    #x = Conv3D(filters, 1, padding = 'same', strides=1, kernel_initializer="he_uniform")(inp)
    x = Conv3D(filters, 1, padding='same', strides=1)(inp)
    x1 = BatchNormalization()(x)
    x2 = Activation('relu')(x1)
    #x3 = Dropout(0.1)(x2)
    x3 = x2
    x4 = MaxPooling3D(U)(x3)
    return x4

def TU(inp,filters,U):
    #inp     -  Inputof the TU network
    #filters -  Number of feature maps
    #x = Conv3DTranspose(filters, 3, padding = 'same', strides=U, kernel_initializer="he_uniform")(inp)
    x = Conv3DTranspose(filters, 3, padding='same', strides=U)(inp)
    x1 = BatchNormalization()(x)
    x2 = Activation('relu')(x1)
    return x2


D = 2
layer = 1
layers = 1
w1 = 128
w2 = 128
w3 = 100
input_dims = (w1, w2, w3, 1)
inpt_img = Input(shape=input_dims)
#C1 = Conv3D(D1, 3, padding='same', strides=1, kernel_initializer="he_uniform")(inpt_img)
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
# -------------------------Data Reconstruction (Encoder1)--------------------------#

    #TU1 = TU(input_data, D*16, (2, 2, 5))
    #DF1 = DenseNet(TU1, layer, D*16)
    ##DF2 = concatenate([DF2, DB3])
    #DF1S= SK(DF1, m=2, r=2, L=8, kernel=D*16)
    #Dro1 = Dropout(0.1)(DF1S)

TU2 = TU(DB5, D*8, (2, 2, 5))
DF2 = DenseNet(TU2, layer, D*8)
    ##DF2 = concatenate([DF2, DB3])
DF2S= SK(DF2, m=2, r=2, L=8, kernel=D*8)
    #Dro2 = Dropout(0.1)(DF2S)

TU3 = TU(DF2S, D*4, 2)
DF3 = DenseNet(TU3, layer, D*4)
    ##DF3 = concatenate([DF3, DB2])
DF3S= SK(DF3, m=2, r=2, L=8, kernel=D*4)
    ##Dro3 = Dropout(0.1)(DF3S)

TU4= TU(DF3S, D*2, 2)
DF4 = DenseNet(TU4, layer, D*2)
    #DF4 = concatenate([DF4, DB1])
#DF4S = SE_Block(DF4, D2)
DF4S= SK(DF4, m=2, r=2, L=8, kernel=D*2)
    #Dro4 = Dropout(0.1)(DF4S)

#DF4F = Conv3D(D1, 3, padding='same', strides=1, kernel_initializer="he_uniform")(DF4S)
DF4F = Conv3D(D, 3, padding='same', strides=1)(DF4S)
DF4F = BatchNormalization()(DF4F)
DF4F = Activation('relu')(DF4F)

#out_seg = Conv3D(1, 3, padding='same', activation='linear', name='out_rec', kernel_initializer="he_uniform")(DF4F)
out_seg = Conv3D(1, 3, padding='same', activation='linear', name='out_rec')(DF4F)
# -------------------------Label Reconstruction (Encoder2)--------------------------#


# lab = TU(input_data, D*16, (2, 2, 5))
# lab = DenseNet(lab, layers, D*16)
##lab = concatenate([lab, DB4])
# lab = SK(lab, m=2, r=2, L=8, kernel=D*16)
##lab = SE_Block(lab, D5)
# lab = Dropout(0.1)(lab)

lab = TU(DB5, D*8, (2, 2, 5))
lab = DenseNet(lab, layers, D*8)
##lab = concatenate([lab, DB3])
lab = SK(lab, m=2, r=2, L=8, kernel=D*8)
##lab = SE_Block(lab, D4)
#  lab = Dropout(0.1)(lab)

lab = TU(lab, D*4, 2)
lab = DenseNet(lab, layers, D*4)
##lab = concatenate([lab, DB2])
lab = SK(lab, m=2, r=2, L=8, kernel=D*4)
##lab = SE_Block(lab, D4)
# lab = Dropout(0.1)(lab)

lab = TU(lab, D * 2, 2)
lab = DenseNet(lab, layers, D * 2)
# lab = concatenate([lab, DB1])
lab = SK(lab, m=2, r=2, L=8, kernel=D * 2)
# lab = SE_Block(lab, D2)
# lab = Dropout(0.1)(lab)

# lab = Conv3D(1, 1, padding='same', strides=1, activation='sigmoid', name='outlab', kernel_initializer="he_uniform")(lab)
lab = Conv3D(1, 1, padding='same', strides=1, activation='sigmoid', name='outlab')(lab)

model = Model(inpt_img, outputs=[out_seg, lab])
model.summary()

model.compile(optimizer='adam', loss=['mse','binary_crossentropy'], metrics=['mse','accuracy'])


callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('model/best_model_our_outBT.h5', monitor='val_outlab_accuracy', mode='max', save_best_only=True)
]
batch = 2
#model.fit([Xtrain],[Xtrain,Ytrain], batch_size=batch, epochs=50, callbacks=callbacks,validation_data=(X_valid, [X_valid,y_valid]))
model.fit([Xtrain],[Xtrain,Ytrain], batch_size=batch, epochs=100, callbacks=callbacks,validation_split=0.2)