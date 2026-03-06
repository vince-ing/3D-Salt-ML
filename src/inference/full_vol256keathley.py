"""
Full-Volume Salt Detection Inference  —  256-cube model, slab-direct edition
=============================================================================
Combines the 256-cube dual-input model with the slab-direct writing strategy
from the no-tempfile script.

How it works
------------
  • Volume is processed as overlapping inline slabs (same as no-tempfile script)
  • Each slab is inferred in full, then only the centre (non-overlap) rows are
    written immediately to the output SEG-Y
  • No blank pre-fill, no sparse sampling, no temp files
  • Ctrl+C leaves a valid partial SEG-Y (complete up to the last finished slab)

256-cube dual input
-------------------
  For every 128³ patch:
    fine    = the 128³ patch itself
    context = a 256³ region centred on the patch, downsampled to 128³

  The slab must therefore be thick enough to supply 256 voxels of context
  on each side of every patch centre. This is handled automatically by
  setting slab_overlap = max(SLAB_OVERLAP, CONTEXT_SOURCE_SIZE // 2).

RAM budget per slab
-------------------
  With slab_thick = 64 + 2*128 = 320 inlines, NX=7076, NS=1001:
    Slab data   : 320 × 7076 × 1001 × 4 bytes ≈ 9 GB
    Prob + wt   : same × 2                     ≈ 18 GB
  Reduce SLAB_STRIDE (e.g. 32) to lower peak RAM at cost of more slabs.

Speed (RTX 2000 Ada)
---------------------
  ~2–4 hours depending on SLAB_STRIDE and MC_PASSES

Dependencies
------------
  pip install torch numpy scipy segyio tqdm
"""

import os
import sys
import signal
import argparse
import threading
import queue
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from itertools import product
from scipy.ndimage import zoom
from tqdm import tqdm

try:
    import segyio
except ImportError:
    sys.exit("segyio not found.  Run:  pip install segyio")

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\experiments\multiscale_256_2026-02-27_2007\best_model.pth"
SGY_IN      = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\raw\raw_seismic_keathley.sgy"
SGY_OUT     = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\full_volume\keathley_salt_prob_256model.sgy"

FINE_SIZE            = 128    # fine patch size — must match training
CONTEXT_SOURCE_SIZE  = 256    # region extracted for context — must match training CUBE_SIZE
CONTEXT_SIZE         = 128    # context after downsampling (zoom 0.5)

XL_STRIDE            = 64     # xline sliding window stride
SLAB_STRIDE          = 64     # how many inlines to advance per slab (write zone width)
SLAB_OVERLAP         = 64     # overlap on each side — automatically raised to 128
                               # to support 256³ context windows

BATCH_SIZE           = 16     # patches per GPU forward pass
USE_FP16             = True
SKIP_EMPTY_THRESHOLD = 0.05
PREFETCH_QUEUE_SIZE  = 4
MC_PASSES            = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_STOP  = threading.Event()


# ============================================================
# SIGNAL HANDLER
# ============================================================
def _handle_sigint(sig, frame):
    print("\n\n  Ctrl+C — stopping cleanly...")
    _STOP.set()

signal.signal(signal.SIGINT, _handle_sigint)


# ============================================================
# MODEL
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), nn.Dropout3d(p),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), nn.Dropout3d(p),
        )
    def forward(self, x): return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False),
                                nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=2,
                                dilation=2, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=4,
                                dilation=4, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv3d(out_ch*3, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(), nn.Dropout3d(0.3))
    def forward(self, x):
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


class SaltModel3D_MultiScale(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        p = dropout_rate
        self.enc1  = ConvBlock(1,  16, p); self.pool1 = nn.MaxPool3d(2)
        self.enc2  = ConvBlock(16, 32, p); self.pool2 = nn.MaxPool3d(2)
        self.enc3  = ConvBlock(32, 64, p); self.pool3 = nn.MaxPool3d(2)
        self.ctx_enc = nn.Sequential(
            ConvBlock(1, 16, p),  nn.MaxPool3d(2),
            ConvBlock(16, 32, p), nn.MaxPool3d(2),
            ConvBlock(32, 64, p), nn.MaxPool3d(2))
        self.aspp = ASPP(128, 64)
        self.up3  = nn.ConvTranspose3d(64, 64, 2, stride=2); self.dec3 = ConvBlock(128, 64, p)
        self.up2  = nn.ConvTranspose3d(64, 32, 2, stride=2); self.dec2 = ConvBlock(64,  32, p)
        self.up1  = nn.ConvTranspose3d(32, 16, 2, stride=2); self.dec1 = ConvBlock(32,  16, p)
        self.final = nn.Conv3d(16, 1, 1)

    def forward(self, xf, xc):
        x1 = self.enc1(xf)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        b  = self.aspp(torch.cat([self.pool3(x3), self.ctx_enc(xc)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(b),  x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        return self.final(d1)


# ============================================================
# HELPERS
# ============================================================
def gaussian_window(size):
    sigma  = size / 4.0
    coords = np.arange(size) - (size - 1) / 2.0
    g      = np.exp(-0.5 * (coords / sigma) ** 2); g /= g.max()
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Matches exactly what training did in processcubes256.py → normalize_cube():
      1. Clip to 1st–99th percentile (removes amplitude spikes)
      2. Z-score normalise using the clipped mean and std
    Applied per-patch at inference, same as per-cube at training time.
    """
    patch = patch.astype(np.float32)
    p1    = np.percentile(patch, 1)
    p99   = np.percentile(patch, 99)
    patch = np.clip(patch, p1, p99)
    mean  = patch.mean()
    std   = patch.std()
    if std < 1e-8:
        return patch - mean   # flat patch, avoid divide-by-zero
    return (patch - mean) / std


def extract_context_256(vol, ci, cx, cs, context_source):
    """
    Extract a context_source³ region centred on (ci, cx, cs) from vol,
    reflect-pad if near edges, then downsample 2× to 128³.
    Matches exactly what training did: zoom(cube256, 0.5, order=0).
    """
    half = context_source // 2
    NI, NX, NS = vol.shape

    i0, i1 = ci - half, ci + half
    x0, x1 = cx - half, cx + half
    s0, s1 = cs - half, cs + half

    pad = [(max(0, -i0), max(0, i1 - NI)),
           (max(0, -x0), max(0, x1 - NX)),
           (max(0, -s0), max(0, s1 - NS))]

    region = vol[max(0,i0):min(NI,i1),
                 max(0,x0):min(NX,x1),
                 max(0,s0):min(NS,s1)]

    if any(p[0] or p[1] for p in pad):
        region = np.pad(region, pad, mode='reflect')

    return zoom(region, zoom=0.5, order=0).astype(np.float32)


def patch_starts(size, patch, stride):
    starts = list(range(0, size - patch, stride))
    if not starts or starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def load_model(model_path, use_fp16):
    model = SaltModel3D_MultiScale().to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
    if use_fp16 and DEVICE.type == "cuda":
        model = model.half()
    return model


# ============================================================
# PREFETCH WORKER
# ============================================================
def prefetch_worker(slab, all_coords, fine_size, context_source,
                    batch_size, skip_std, out_q):
    fine_b, ctx_b, coord_b, skip_b = [], [], [], []
    half_fine = fine_size // 2

    def enqueue():
        if fine_b: out_q.put(("batch", fine_b[:], ctx_b[:], coord_b[:]))
        if skip_b: out_q.put(("skip",  skip_b[:]))
        fine_b.clear(); ctx_b.clear(); coord_b.clear(); skip_b.clear()

    for si, sx, ss in all_coords:
        if _STOP.is_set(): break

        fine = slab[si:si+fine_size,
                    sx:sx+fine_size,
                    ss:ss+fine_size].copy()

        if skip_std > 0 and fine.std() < skip_std:
            skip_b.append((si, sx, ss))
            if len(skip_b) >= batch_size * 4: enqueue()
            continue

        # Centre of fine patch — used as context window centre
        ci = si + half_fine
        cx = sx + half_fine
        cs = ss + half_fine

        ctx = extract_context_256(slab, ci, cx, cs, context_source)

        # Normalise — matches training's normalize_cube(): clip p1/p99 then z-score
        fine = normalize_patch(fine)
        ctx  = normalize_patch(ctx)

        fine_b.append(torch.from_numpy(fine[None]))
        ctx_b .append(torch.from_numpy(ctx [None]))
        coord_b.append((si, sx, ss))

        if len(fine_b) >= batch_size:
            enqueue()
            while out_q.qsize() >= PREFETCH_QUEUE_SIZE and not _STOP.is_set():
                time.sleep(0.005)

    enqueue()
    out_q.put(None)


# ============================================================
# INFER ONE SLAB
# ============================================================
def infer_slab(model, slab, fine_size, context_source,
               stride, batch_size, use_fp16, gauss):
    SI, SX, SS = slab.shape
    dtype = torch.float16 if (use_fp16 and DEVICE.type == "cuda") else torch.float32

    prob_sum   = np.zeros((SI, SX, SS), dtype=np.float32)
    weight_sum = np.zeros((SI, SX, SS), dtype=np.float32)

    all_coords = list(product(
        patch_starts(SI, fine_size, stride),
        patch_starts(SX, fine_size, stride),
        patch_starts(SS, fine_size, stride),
    ))
    skip_std = float(slab.std()) * SKIP_EMPTY_THRESHOLD

    pq     = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE + 2)
    worker = threading.Thread(
        target=prefetch_worker,
        args=(slab, all_coords, fine_size, context_source,
              batch_size, skip_std, pq),
        daemon=True)
    worker.start()

    with tqdm(total=len(all_coords), desc="  patches", unit="p",
              leave=False, ncols=80) as pbar:
        while True:
            item = pq.get()
            if item is None: break
            if _STOP.is_set(): break

            if item[0] == "skip":
                for si, sx, ss in item[1]:
                    weight_sum[si:si+fine_size,
                               sx:sx+fine_size,
                               ss:ss+fine_size] += gauss
                pbar.update(len(item[1]))

            elif item[0] == "batch":
                _, fl, cl, coords = item
                fine_t = torch.stack(fl).to(DEVICE, dtype=dtype)
                ctx_t  = torch.stack(cl).to(DEVICE, dtype=dtype)
                with torch.no_grad():
                    mc = torch.stack([
                        torch.sigmoid(model(fine_t, ctx_t))
                        for _ in range(MC_PASSES)
                    ]).mean(0)
                probs_np = mc[:, 0].cpu().float().numpy()
                for k, (si, sx, ss) in enumerate(coords):
                    p = probs_np[k]
                    prob_sum  [si:si+fine_size,
                               sx:sx+fine_size,
                               ss:ss+fine_size] += p * gauss
                    weight_sum[si:si+fine_size,
                               sx:sx+fine_size,
                               ss:ss+fine_size] += gauss
                pbar.update(len(coords))

    worker.join()
    return prob_sum / np.where(weight_sum < 1e-6, 1e-6, weight_sum)


# ============================================================
# MAIN
# ============================================================
def run_inference(model_path, sgy_in, sgy_out,
                  fine_size=FINE_SIZE,
                  context_source=CONTEXT_SOURCE_SIZE,
                  xl_stride=XL_STRIDE,
                  slab_stride=SLAB_STRIDE,
                  slab_overlap=SLAB_OVERLAP,
                  batch_size=BATCH_SIZE,
                  use_fp16=USE_FP16):

    Path(sgy_out).parent.mkdir(parents=True, exist_ok=True)

    # Context windows need context_source//2 voxels on each side of every patch
    # centre, so the slab overlap must be at least that wide.
    effective_overlap = max(slab_overlap, context_source // 2)
    if effective_overlap > slab_overlap:
        print(f"  slab_overlap raised {slab_overlap} → {effective_overlap} "
              f"to support {context_source}³ context windows")

    print(f"\n  Device : {DEVICE}  |  FP16: {use_fp16}  |  MC passes: {MC_PASSES}")
    print(f"  Fine: {fine_size}³  |  Context source: {context_source}³ → {fine_size}³")

    model = load_model(model_path, use_fp16)
    print("  Model ready (MC dropout active)\n")
    gauss = gaussian_window(fine_size)

    print(f"  Input  : {sgy_in}")
    print(f"  Output : {sgy_out}\n")

    with segyio.open(sgy_in, "r", ignore_geometry=False) as src:
        ilines  = list(src.ilines)
        xlines  = list(src.xlines)
        NI, NX, NS = len(ilines), len(xlines), len(src.samples)

        slab_thick = slab_stride + 2 * effective_overlap
        slab_ram   = slab_thick * NX * NS * 4 / 1e9
        print(f"  Geometry : {NI} × {NX} × {NS}")
        print(f"  Slab     : {slab_thick} inlines thick  (~{slab_ram:.1f} GB data, "
              f"~{slab_ram*3:.1f} GB peak with prob+weight)")
        print()

        # Build slab list — identical logic to no-tempfile script
        slabs = []
        il = 0
        while il < NI:
            il_start    = max(0, il - effective_overlap)
            il_end      = min(NI, il + slab_stride + effective_overlap)
            write_start = il - il_start
            write_end   = write_start + min(slab_stride, NI - il)
            slabs.append((il_start, il_end, write_start, write_end, il))
            il += slab_stride

        n_slabs = len(slabs)
        print(f"  Slabs : {n_slabs}  "
              f"(stride={slab_stride}, overlap={effective_overlap})\n")

        spec = segyio.tools.metadata(src)

        with segyio.create(sgy_out, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin     = src.bin

            for slab_idx, (il_start, il_end, ws, we, il_base) in enumerate(slabs):
                if _STOP.is_set():
                    print(f"\n  Stopped — output is valid up to inline "
                          f"{ilines[il_base-1] if il_base > 0 else 'none'}")
                    break

                n_write = we - ws
                print(f"  Slab {slab_idx+1}/{n_slabs}  "
                      f"inlines {ilines[il_start]}–{ilines[il_end-1]}  "
                      f"(writing {n_write} inlines)")

                # Load slab from SEG-Y
                slab_data = np.zeros((il_end - il_start, NX, NS), dtype=np.float32)
                for i in range(il_end - il_start):
                    trace_base = (il_start + i) * NX
                    for xl in range(NX):
                        slab_data[i, xl, :] = src.trace[trace_base + xl]

                # Pad slab if smaller than fine_size in any dimension
                orig_shape = slab_data.shape
                for axis, pad_axis in enumerate([(0,0),(0,0),(0,0)]):
                    if slab_data.shape[axis] < fine_size:
                        pad = [(0,0),(0,0),(0,0)]
                        pad[axis] = (0, fine_size - slab_data.shape[axis])
                        slab_data = np.pad(slab_data, pad, mode='reflect')

                # Run inference
                prob_slab = infer_slab(model, slab_data, fine_size,
                                       context_source, xl_stride,
                                       batch_size, use_fp16, gauss)
                del slab_data

                if _STOP.is_set(): break

                # Trim padding and write centre zone directly to SEG-Y
                prob_slab = prob_slab[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
                for i in range(ws, we):
                    abs_il     = il_start + i
                    trace_base = abs_il * NX
                    for xl in range(NX):
                        t = trace_base + xl
                        dst.header[t] = src.header[t]
                        dst.trace[t]  = prob_slab[i, xl, :].astype(np.float32)

                del prob_slab

                pct = float(min(il_base + slab_stride, NI)) / NI * 100
                print(f"    Written — {pct:.1f}% complete\n")

    if not _STOP.is_set():
        out_gb = Path(sgy_out).stat().st_size / 1e9
        print(f"\n  Done!  {sgy_out}  ({out_gb:.2f} GB)")
        print("\n  Petrel import:")
        print("    File > Import > Seismic (SEG-Y)")
        print("    Inline byte 189, Xline byte 193")
        print("    Threshold at 0.5 to extract salt body")
    else:
        print("\n  Partial run — rerun to complete.")
        sys.exit(0)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="256-cube salt inference, slab-direct writing")
    p.add_argument("--model",        default=MODEL_PATH)
    p.add_argument("--sgy-in",       default=SGY_IN)
    p.add_argument("--sgy-out",      default=SGY_OUT)
    p.add_argument("--xl-stride",    type=int, default=XL_STRIDE)
    p.add_argument("--slab-stride",  type=int, default=SLAB_STRIDE)
    p.add_argument("--slab-overlap", type=int, default=SLAB_OVERLAP)
    p.add_argument("--batch",        type=int, default=BATCH_SIZE)
    p.add_argument("--no-fp16",      action="store_true")
    a = p.parse_args()

    run_inference(
        model_path   = a.model,
        sgy_in       = a.sgy_in,
        sgy_out      = a.sgy_out,
        xl_stride    = a.xl_stride,
        slab_stride  = a.slab_stride,
        slab_overlap = a.slab_overlap,
        batch_size   = a.batch,
        use_fp16     = not a.no_fp16,
    )