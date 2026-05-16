"""
3D Salt Detection Inference — PyVista Visualization (TensorFlow/Keras)
=======================================================================
Loads the fine-tuned multiclass model and runs inference on a random
cube from the processed dataset, then renders results interactively
in 3D using PyVista showing:
  • Seismic volume (gray volume rendering)
  • Ground-truth class labels (coloured surfaces per class)
  • Predicted class labels (coloured surfaces per class)
  • Difference map (TP / FP / FN for salt class)

Classes: 0=Rock, 1=Salt, 2=Water, 3=Blanked

Dependencies (install in PowerShell):
    pip install pyvista scipy
"""

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv3D, Lambda, Reshape, Add, Softmax,
                                     MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D,
                                     multiply, concatenate, Activation, BatchNormalization)
from scipy.ndimage import label as ndlabel

os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")
import pyvista as pv
pv.global_theme.background = "black"
pv.global_theme.font.color  = "white"

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_WEIGHTS   = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\model\best_model_multiclass_finetune.h5'
SAMPLES_BIN     = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\salt3dnetoutstyle\samples.bin'
LABELS_BIN      = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\processed\salt3dnetoutstyle\labels.bin'
N, IL, XL, Z    = 495, 100, 128, 128   # original shape before transpose

SALT_CLASS      = 1          # which class to use for TP/FP/FN difference map
MIN_SALT_RATIO  = 0.05       # only pick cubes with >= 5% salt voxels
MAX_SALT_RATIO  = 0.85
SAVE_SCREENSHOT = True
SCREENSHOT_PATH = r'G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\salt_inference_3d_multiclass.png'

# Only show these classes — Blanked (3) is excluded
SHOW_CLASSES = [0, 1, 2]
CLASS_COLORS = {
    0: "#a0522d",   # Rock  — brown
    1: "#1e90ff",   # Salt  — blue
    2: "#00ced1",   # Water — teal
}
CLASS_NAMES = {0: "Rock", 1: "Salt", 2: "Water"}

# ============================================================
# LIMIT GPU MEMORY (leaves headroom for PyVista)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

# ============================================================
# MODEL ARCHITECTURE  (must match training exactly)
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

def build_model():
    D, layer = 2, 1
    inpt_img = Input(shape=(128, 128, 100, 1))
    C1 = Conv3D(D, 3, padding='same', strides=1)(inpt_img)
    C1 = BatchNormalization()(C1)
    C1 = Activation('relu')(C1)
    DB1 = SK(DenseNet(C1, layer, D*2), m=2, r=2, L=8, kernel=D*2)
    TD1 = TD(DB1, D*2, 2)
    DB2 = SK(DenseNet(TD1, layer, D*4), m=2, r=2, L=8, kernel=D*4)
    TD2 = TD(DB2, D*4, 2)
    DB3 = SK(DenseNet(TD2, layer, D*8), m=2, r=2, L=8, kernel=D*8)
    TD3 = TD(DB3, D*8, (2, 2, 5))
    DB5 = SK(DenseNet(TD3, layer, D*16), m=2, r=2, L=8, kernel=D*16)
    TU2 = TU(DB5, D*8, (2, 2, 5))
    DF2S = SK(DenseNet(TU2, layer, D*8), m=2, r=2, L=8, kernel=D*8)
    TU3 = TU(DF2S, D*4, 2)
    DF3S = SK(DenseNet(TU3, layer, D*4), m=2, r=2, L=8, kernel=D*4)
    TU4 = TU(DF3S, D*2, 2)
    DF4S = SK(DenseNet(TU4, layer, D*2), m=2, r=2, L=8, kernel=D*2)
    DF4F = Conv3D(D, 3, padding='same', strides=1)(DF4S)
    DF4F = BatchNormalization()(DF4F)
    DF4F = Activation('relu')(DF4F)
    out_rec = Conv3D(1, 3, padding='same', activation='linear', name='out_rec')(DF4F)
    lab = TU(DB5, D*8, (2, 2, 5))
    lab = SK(DenseNet(lab, layer, D*8), m=2, r=2, L=8, kernel=D*8)
    lab = TU(lab, D*4, 2)
    lab = SK(DenseNet(lab, layer, D*4), m=2, r=2, L=8, kernel=D*4)
    lab = TU(lab, D*2, 2)
    lab = SK(DenseNet(lab, layer, D*2), m=2, r=2, L=8, kernel=D*2)
    out_seg = Conv3D(4, 1, padding='same', activation='softmax', name='outlab')(lab)
    return Model(inpt_img, outputs=[out_rec, out_seg])

# ============================================================
# PYVISTA HELPERS
# ============================================================
def volume_to_pyvista(arr):
    """Wrap a (X, Y, Z) numpy array into a PyVista ImageData grid."""
    grid = pv.ImageData()
    grid.dimensions = np.array(arr.shape) + 1
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["values"] = arr.flatten(order="F")
    return grid

def iso_surface(binary_vol, smooth_iter=30):
    """Marching-cubes surface from a binary volume with Laplacian smoothing."""
    padded = np.pad(binary_vol.astype(np.float32), pad_width=1,
                    mode="constant", constant_values=0)
    grid = volume_to_pyvista(padded).cell_data_to_point_data()
    surf = grid.contour([0.5], scalars="values")
    if surf.n_points > 0 and smooth_iter > 0:
        surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.1)
    return surf

def compute_salt_metrics(pred_classes, label_int, salt_cls=1):
    pb = (pred_classes == salt_cls).astype(np.float32)
    tb = (label_int    == salt_cls).astype(np.float32)
    inter = (pb * tb).sum()
    union = pb.sum() + tb.sum() - inter
    iou  = float(inter / union) if union else 1.0
    denom = pb.sum() + tb.sum()
    dice = float(2 * inter / denom) if denom else 1.0
    return iou, dice

# ============================================================
# MAIN
# ============================================================
def run():
    # ── 1. Load model ────────────────────────────────────────
    print("Building model architecture...")
    model = build_model()
    print(f"Loading weights from: {MODEL_WEIGHTS}")
    model.load_weights(MODEL_WEIGHTS, by_name=True, skip_mismatch=True)
    print("Model ready!\n")

    # ── 2. Load dataset and pick a suitable cube ─────────────
    print("Loading dataset...")
    data  = np.fromfile(SAMPLES_BIN, dtype=np.float32).reshape((N, IL, XL, Z))
    label = np.fromfile(LABELS_BIN,  dtype=np.float32).reshape((N, IL, XL, Z))
    data  = np.transpose(data,  (0, 2, 3, 1))   # (N, 128, 128, 100)
    label = np.transpose(label, (0, 2, 3, 1))

    # Normalise exactly as in training: divide each cube by its max absolute value
    for ii in range(data.shape[0]):
        tmp = np.abs(data[ii]).max()
        if tmp > 0:
            data[ii] /= tmp

    # Pick a random cube with enough salt
    indices = list(range(N))
    random.shuffle(indices)
    seismic = label_vol = None
    chosen_idx = None
    for idx in indices:
        lbl = label[idx]
        salt_ratio = float((lbl == SALT_CLASS).mean())
        if MIN_SALT_RATIO <= salt_ratio <= MAX_SALT_RATIO:
            seismic   = data[idx]
            label_vol = lbl.astype(np.int32)
            chosen_idx = idx
            print(f"Using cube index {idx}  (salt voxel ratio: {salt_ratio:.1%})")
            break

    if seismic is None:
        print("No cube found matching salt ratio filter — using cube 0.")
        seismic   = data[0]
        label_vol = label[0].astype(np.int32)
        chosen_idx = 0

    # ── 3. Inference ─────────────────────────────────────────
    print("Running inference...")
    X = seismic[np.newaxis, ..., np.newaxis]   # (1, 128, 128, 100, 1)
    _, pred_softmax = model.predict(X, verbose=0)
    pred_classes = np.argmax(pred_softmax[0], axis=-1)   # (128, 128, 100)

    iou, dice = compute_salt_metrics(pred_classes, label_vol, SALT_CLASS)
    print(f"\nSalt class  —  IoU: {iou:.3f}   Dice: {dice:.3f}")

    overall_acc = float(np.mean(pred_classes == label_vol))
    print(f"Overall pixel accuracy: {overall_acc:.1%}\n")

    # ── 4. Build PyVista objects ──────────────────────────────
    # Seismic volume — shape (128,128,100) treated as (X,Y,Z)
    seis_grid = volume_to_pyvista(seismic).cell_data_to_point_data()

    # Per-class binary surfaces — skip blanked (class 3)
    gt_surfs   = {}
    pred_surfs = {}
    for cls in SHOW_CLASSES:
        gt_bin   = (label_vol   == cls).astype(np.uint8)
        pred_bin = (pred_classes == cls).astype(np.uint8)
        if gt_bin.sum() > 100:
            gt_surfs[cls]   = iso_surface(gt_bin,   smooth_iter=40)
        if pred_bin.sum() > 100:
            pred_surfs[cls] = iso_surface(pred_bin, smooth_iter=40)

    # Salt-class difference map
    tp = (pred_classes == SALT_CLASS) & (label_vol == SALT_CLASS)
    fp = (pred_classes == SALT_CLASS) & (label_vol != SALT_CLASS)
    fn = (pred_classes != SALT_CLASS) & (label_vol == SALT_CLASS)
    tp_surf = iso_surface(tp.astype(np.uint8), smooth_iter=30) if tp.sum() > 100 else None
    fp_surf = iso_surface(fp.astype(np.uint8), smooth_iter=30) if fp.sum() > 100 else None
    fn_surf = iso_surface(fn.astype(np.uint8), smooth_iter=30) if fn.sum() > 100 else None

    # ── 5. Plot ───────────────────────────────────────────────
    pl = pv.Plotter(
        shape=(2, 2),
        border=True,
        border_color="white",
        window_size=(1600, 1000),
        title="Multiclass Salt Detection — 3D Inference",
    )

    # Use percentile-based clim for better seismic contrast
    p2, p98 = np.percentile(seismic, 2), np.percentile(seismic, 98)
    clim = [p2, p98]

    def add_outline():
        pl.add_mesh(seis_grid.outline(), color="white", line_width=1.5)

    def set_cam():
        pl.camera_position = "iso"
        pl.reset_camera()

    # ── [0,0] Seismic volume ──────────────────────────────────
    pl.subplot(0, 0)
    pl.add_mesh(
        seis_grid,
        scalars="values",
        cmap="gray_r",
        clim=clim,
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Amplitude", "vertical": True,
                         "color": "white", "fmt": "%.2f",
                         "position_x": 0.85, "position_y": 0.25,
                         "width": 0.08, "height": 0.5},
    )
    add_outline()
    pl.add_text(f"Seismic  (cube #{chosen_idx})", font_size=11,
                color="white", position="upper_left")
    set_cam()
    pl.add_axes(interactive=False)

    # ── [0,1] Ground truth ────────────────────────────────────
    pl.subplot(0, 1)
    add_outline()
    for cls, surf in gt_surfs.items():
        if surf.n_points > 0:
            pl.add_mesh(surf, color=CLASS_COLORS[cls], opacity=0.85,
                        smooth_shading=True)
    # Legend placed bottom-left to avoid cutoff
    pl.add_legend(
        labels=[(CLASS_NAMES[c], CLASS_COLORS[c]) for c in SHOW_CLASSES if c in gt_surfs],
        bcolor=(0.05, 0.05, 0.05), border=True,
        loc="lower left", size=(0.30, 0.18),
    )
    pl.add_text("Ground Truth (Salt)", font_size=11, color="white", position="upper_left")
    set_cam()
    pl.add_axes(interactive=False)

    # ── [1,0] Prediction ─────────────────────────────────────
    pl.subplot(1, 0)
    add_outline()
    for cls, surf in pred_surfs.items():
        if surf.n_points > 0:
            pl.add_mesh(surf, color=CLASS_COLORS[cls], opacity=0.85,
                        smooth_shading=True)
    pl.add_legend(
        labels=[(CLASS_NAMES[c], CLASS_COLORS[c]) for c in SHOW_CLASSES if c in pred_surfs],
        bcolor=(0.05, 0.05, 0.05), border=True,
        loc="lower left", size=(0.30, 0.18),
    )
    pl.add_text(
        f"Prediction   IoU={iou:.3f}  Dice={dice:.3f}  Acc={overall_acc:.1%}",
        font_size=10, color="white", position="upper_left",
    )
    set_cam()
    pl.add_axes(interactive=False)

    # ── [1,1] Salt difference map ─────────────────────────────
    pl.subplot(1, 1)
    add_outline()
    if tp_surf and tp_surf.n_points > 0:
        pl.add_mesh(tp_surf, color="#00cc44", opacity=0.80, smooth_shading=True)
    if fp_surf and fp_surf.n_points > 0:
        pl.add_mesh(fp_surf, color="#ff2222", opacity=0.80, smooth_shading=True)
    if fn_surf and fn_surf.n_points > 0:
        pl.add_mesh(fn_surf, color="#2299ff", opacity=0.80, smooth_shading=True)
    pl.add_legend(
        labels=[
            ("True Positive",  "#00cc44"),
            ("False Positive", "#ff2222"),
            ("False Negative", "#2299ff"),
        ],
        bcolor=(0.05, 0.05, 0.05), border=True,
        loc="lower left", size=(0.32, 0.18),
    )
    pl.add_text("Salt Prediction Error", font_size=11,
                color="white", position="upper_left")
    set_cam()
    pl.add_axes(interactive=False)

    # ── 6. Show ───────────────────────────────────────────────
    print("Interactive window open — use mouse to rotate/zoom.")
    print("Press  Q  or close the window to exit.\n")
    pl.show(auto_close=False)

    if SAVE_SCREENSHOT:
        pl.screenshot(SCREENSHOT_PATH, transparent_background=False)
        print(f"Screenshot saved → {SCREENSHOT_PATH}")

    pl.close()


if __name__ == "__main__":
    run()