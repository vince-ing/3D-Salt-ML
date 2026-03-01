"""
Full-Volume Salt Detection Inference  —  NO-TEMPFILE edition
=============================================================
Completely rewritten to use ZERO disk temp files.

Previous versions used numpy memmaps which caused unkillable processes
on Windows when interrupted. This version:

  • Holds only one inline SLAB in RAM at a time (~2-4 GB max)
  • Writes results directly to output SEG-Y as each slab completes
  • Ctrl+C works instantly and leaves no locked files behind
  • The only output file written is the final SEG-Y itself

Architecture
------------
  The volume is processed as overlapping inline slabs:

    ┌─────────────────────────────────┐
    │  Slab 0  (inlines 0  → 191)    │  ← 192 inlines thick  (128 + 2*32 overlap)
    │    ┌─────────────────────────┐  │
    │    │  write zone (inlines    │  │  ← only the CENTRE 128 inlines are written
    │    │  32 → 159)              │  │    (overlap discarded at edges)
    │    └─────────────────────────┘  │
    ├─────────────────────────────────┤
    │  Slab 1  (inlines 128 → 319)   │  ← slides by 128 (non-overlap zone width)
    ...

  Within each slab, a standard 3-D sliding window (stride=96) runs over
  the full xline × sample extent. Gaussian blending removes seams within
  the slab. The overlap between slabs removes the inter-slab seam.

RAM budget per slab
-------------------
  Slab data   : 192 × 7076 × 1001 × 4 bytes ≈ 5.5 GB
  Prob volume : same                         ≈ 5.5 GB
  Weight vol  : same                         ≈ 5.5 GB
  Peak total  : ~16 GB   ← fits in system RAM on most workstations
  GPU VRAM    : batch_size × 2 × 128³ × 2 bytes (FP16) ≈ 3 GB at batch=24

  If RAM is tight reduce SLAB_STRIDE (e.g. 64) — smaller write zones,
  more slabs, lower peak RAM.

Speed (RTX 2000 Ada, stride=96)
--------------------------------
  ~1 – 1.5 hours total

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

# Disable torch.compile entirely — Triton is not available on Windows
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\experiments\multi_dataset_multiscale_run_02\best_model.pth"
SGY_IN     = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\raw\raw_seismic_keathley.sgy"
SGY_OUT    = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\full_volume\keathley_salt_prob.sgy"

PATCH_SIZE           = 128
XL_STRIDE            = 64      # tighter stride — more overlap, smoother boundaries
SLAB_STRIDE          = 64      # smaller slab advance — more inter-slab blending
                                # SLAB_OVERLAP is added on each side automatically
SLAB_OVERLAP         = 64      # wider overlap — eliminates inter-slab seams
BATCH_SIZE           = 24      # patches per GPU forward pass (safe for 16 GB + FP16)
USE_FP16             = True
USE_COMPILE          = False   # torch.compile requires Triton which is Linux-only
SKIP_EMPTY_THRESHOLD = 0.05    # skip patches with std < 5% of slab std; 0.0 = disable
PREFETCH_QUEUE_SIZE  = 4
MC_PASSES            = 8   # Monte Carlo dropout passes per batch
                            # higher = smoother boundaries, slower inference
                            # 8 is a good balance; try 4 to go faster

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global flag so Ctrl+C can stop the prefetch thread cleanly
_STOP = threading.Event()


# ============================================================
# SIGNAL HANDLER — makes Ctrl+C work instantly
# ============================================================
def _handle_sigint(sig, frame):
    print("\n\n  Ctrl+C received — stopping cleanly (no temp files to clean up)...")
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
        self.enc1 = ConvBlock(1, 16, p);  self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(16, 32, p); self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(32, 64, p); self.pool3 = nn.MaxPool3d(2)
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
def gaussian_window(size: int) -> np.ndarray:
    sigma  = size / 4.0  # wider blend zone — reduces patch boundary artefacts
    coords = np.arange(size) - (size - 1) / 2.0
    g      = np.exp(-0.5 * (coords / sigma) ** 2); g /= g.max()
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32)


def context_patch(patch: np.ndarray) -> np.ndarray:
    return zoom(np.pad(patch, 64, mode="reflect"), 0.5, order=1).astype(np.float32)


def patch_starts(size: int, patch: int, stride: int) -> list:
    starts = list(range(0, size - patch, stride))
    if not starts or starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


# ============================================================
# PREFETCH WORKER
# ============================================================
def prefetch_worker(slab: np.ndarray, all_coords: list,
                    patch_size: int, batch_size: int,
                    skip_std: float, out_q: queue.Queue):
    """
    Runs in a background thread. Reads patches from the slab (in RAM),
    skips empty ones, and puts batches onto out_q for the GPU thread.
    """
    fine_b, ctx_b, coord_b, skip_b = [], [], [], []

    def enqueue():
        if fine_b:  out_q.put(("batch", fine_b[:], ctx_b[:], coord_b[:]))
        if skip_b:  out_q.put(("skip",  skip_b[:]))
        fine_b.clear(); ctx_b.clear(); coord_b.clear(); skip_b.clear()

    for si, sx, ss in all_coords:
        if _STOP.is_set(): break
        patch = slab[si:si+patch_size, sx:sx+patch_size, ss:ss+patch_size].copy()
        if skip_std > 0 and patch.std() < skip_std:
            skip_b.append((si, sx, ss))
            if len(skip_b) >= batch_size * 4: enqueue()
            continue
        fine_b.append(torch.from_numpy(patch[None]))
        ctx_b .append(torch.from_numpy(context_patch(patch)[None]))
        coord_b.append((si, sx, ss))
        if len(fine_b) >= batch_size:
            enqueue()
            while out_q.qsize() >= PREFETCH_QUEUE_SIZE and not _STOP.is_set():
                time.sleep(0.005)

    enqueue()
    out_q.put(None)  # sentinel


# ============================================================
# INFER ONE SLAB
# ============================================================
def infer_slab(model, slab: np.ndarray, patch_size: int,
               stride: int, batch_size: int,
               use_fp16: bool, gauss: np.ndarray) -> np.ndarray:
    """
    Run sliding-window inference over a slab (NI_slab x NX x NS).
    Returns a probability volume of the same shape, all in RAM.
    No files are written.
    """
    SI, SX, SS = slab.shape
    dtype      = torch.float16 if (use_fp16 and DEVICE.type == "cuda") else torch.float32

    prob_sum   = np.zeros((SI, SX, SS), dtype=np.float32)
    weight_sum = np.zeros((SI, SX, SS), dtype=np.float32)

    # Patch grid over the slab
    all_coords = list(product(
        patch_starts(SI, patch_size, stride),
        patch_starts(SX, patch_size, stride),
        patch_starts(SS, patch_size, stride),
    ))

    skip_std = float(slab.std()) * SKIP_EMPTY_THRESHOLD

    pq     = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE + 2)
    worker = threading.Thread(
        target=prefetch_worker,
        args=(slab, all_coords, patch_size, batch_size, skip_std, pq),
        daemon=True)
    worker.start()

    with tqdm(total=len(all_coords), desc="  patches", unit="p",
              leave=False, ncols=80) as pbar:
        while True:
            item = pq.get()
            if item is None: break
            if _STOP.is_set(): break

            kind = item[0]
            if kind == "skip":
                for si, sx, ss in item[1]:
                    weight_sum[si:si+patch_size,
                               sx:sx+patch_size,
                               ss:ss+patch_size] += gauss
                pbar.update(len(item[1]))

            elif kind == "batch":
                _, fl, cl, coords = item
                fine_t = torch.stack(fl).to(DEVICE, dtype=dtype)
                ctx_t  = torch.stack(cl).to(DEVICE, dtype=dtype)
                with torch.no_grad():
                    # MC dropout: average MC_PASSES stochastic forward passes
                    # Each pass has different dropout masks → averaged result
                    # is a calibrated probability with soft boundaries
                    mc_probs = torch.stack([
                        torch.sigmoid(model(fine_t, ctx_t))
                        for _ in range(MC_PASSES)
                    ]).mean(dim=0)
                probs_np = mc_probs[:, 0].cpu().float().numpy()
                for k, (si, sx, ss) in enumerate(coords):
                    p = probs_np[k]
                    prob_sum  [si:si+patch_size,
                               sx:sx+patch_size,
                               ss:ss+patch_size] += p * gauss
                    weight_sum[si:si+patch_size,
                               sx:sx+patch_size,
                               ss:ss+patch_size] += gauss
                pbar.update(len(coords))

    worker.join()
    w = np.where(weight_sum < 1e-6, 1e-6, weight_sum)
    return prob_sum / w   # shape (SI, SX, SS), values in [0,1]


# ============================================================
# MAIN INFERENCE
# ============================================================
def run_inference(model_path, sgy_in, sgy_out,
                  patch_size=PATCH_SIZE, xl_stride=XL_STRIDE,
                  slab_stride=SLAB_STRIDE, slab_overlap=SLAB_OVERLAP,
                  batch_size=BATCH_SIZE, use_fp16=USE_FP16,
                  use_compile=USE_COMPILE):

    Path(sgy_out).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ───────────────────────────────────────────────────────
    print(f"\n  Device  : {DEVICE}  |  FP16: {use_fp16}  |  compile: disabled (Windows)")
    model = SaltModel3D_MultiScale().to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    # Use Monte Carlo dropout: keep dropout ACTIVE during inference.
    # model.eval() would disable dropout making predictions overconfident
    # and producing hard 0/1 edges. model.train() keeps dropout on so
    # we average N stochastic passes to get calibrated soft probabilities.
    model.train()
    # Disable BatchNorm updates — we want to use the trained running stats,
    # just not update them. Only dropout should remain active.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d,)):
            m.eval()
    if use_fp16 and DEVICE.type == "cuda":
        model = model.half()
    print("  Model ready\n")

    gauss = gaussian_window(patch_size)

    # ── 2. Open SEG-Y and prepare output ────────────────────────────────────
    print(f"  Input  : {sgy_in}")
    print(f"  Output : {sgy_out}\n")

    with segyio.open(sgy_in, "r", ignore_geometry=False) as src:
        ilines  = list(src.ilines)
        xlines  = list(src.xlines)
        samples = src.samples.copy()
        NI, NX, NS = len(ilines), len(xlines), len(samples)
        slab_thick = slab_stride + 2 * slab_overlap   # total inline thickness per slab

        print(f"  Geometry : {NI} inlines x {NX} xlines x {NS} samples")
        slab_ram = slab_thick * NX * NS * 4 / 1e9
        print(f"  Slab RAM : ~{slab_ram:.1f} GB  (slab_thick={slab_thick})")
        print(f"  Peak RAM : ~{slab_ram * 3:.1f} GB  (slab + prob + weight)\n")

        # Build list of slabs
        # Each slab: (il_start, il_end, write_start, write_end)
        # il_start/end   = indices into the full volume (clamped to [0, NI))
        # write_start/end = which rows of the slab result to actually keep
        slabs = []
        il = 0
        while il < NI:
            raw_start = il - slab_overlap
            raw_end   = il + slab_stride + slab_overlap
            # Clamp to volume bounds
            il_start  = max(0, raw_start)
            il_end    = min(NI, raw_end)
            # Write zone within the slab array
            write_start = il - il_start            # offset from slab start
            write_end   = write_start + min(slab_stride, NI - il)
            slabs.append((il_start, il_end, write_start, write_end, il))
            il += slab_stride

        n_slabs = len(slabs)
        print(f"  Slabs : {n_slabs}  (stride={slab_stride}, overlap={slab_overlap})\n")

        # Create output SEG-Y (copy spec from input)
        spec = segyio.tools.metadata(src)

        with segyio.create(sgy_out, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin     = src.bin



            # ── 3. Process slab by slab ──────────────────────────────────
            for slab_idx, (il_start, il_end, ws, we, il_base) in enumerate(slabs):
                if _STOP.is_set():
                    print("\n  Stopped early — partial output SEG-Y is valid up to"
                          f" inline {ilines[il_base - 1] if il_base > 0 else 'none'}")
                    break

                n_write = we - ws
                print(f"  Slab {slab_idx+1}/{n_slabs}  "
                      f"inlines {ilines[il_start]}–{ilines[il_end-1]}  "
                      f"(write {n_write} inlines)")

                # Load slab from SEG-Y into RAM (trace-based, works on all SEG-Y files)
                slab_data = np.zeros((il_end - il_start, NX, NS), dtype=np.float32)
                for i in range(il_end - il_start):
                    trace_base = (il_start + i) * NX
                    for xl in range(NX):
                        slab_data[i, xl, :] = src.trace[trace_base + xl]

                # Run inference on this slab
                prob_slab = infer_slab(model, slab_data, patch_size,
                                       xl_stride, batch_size, use_fp16, gauss)
                del slab_data  # free RAM immediately

                if _STOP.is_set(): break

                # Write only the centre (non-overlap) zone to output SEG-Y
                for i in range(ws, we):
                    abs_il     = il_start + i
                    trace_base = abs_il * NX
                    for xl in range(NX):
                        t = trace_base + xl
                        dst.header[t] = src.header[t]
                        dst.trace[t]  = prob_slab[i, xl, :].astype(np.float32)

                del prob_slab  # free RAM before next slab

                pct = float(min(il_base + slab_stride, NI)) / NI * 100
                print(f"    Written — {pct:.1f}% complete\n")

    if not _STOP.is_set():
        out_gb = Path(sgy_out).stat().st_size / 1e9
        print(f"\n  Done!  Output: {sgy_out}  ({out_gb:.2f} GB)")
        print("\n  Petrel import steps:")
        print("    1. File > Import > Seismic Data (SEG-Y)")
        print("    2. Inline byte 189, Xline byte 193  (standard SEGY rev1)")
        print("    3. Threshold at 0.5 to extract salt body")
    else:
        print("\n  Partial output saved — re-run to complete.")
        sys.exit(0)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-volume salt inference, no temp files")
    p.add_argument("--model",        default=MODEL_PATH)
    p.add_argument("--sgy-in",       default=SGY_IN)
    p.add_argument("--sgy-out",      default=SGY_OUT)
    p.add_argument("--xl-stride",    type=int, default=XL_STRIDE)
    p.add_argument("--slab-stride",  type=int, default=SLAB_STRIDE)
    p.add_argument("--slab-overlap", type=int, default=SLAB_OVERLAP)
    p.add_argument("--batch",        type=int, default=BATCH_SIZE)
    p.add_argument("--no-fp16",      action="store_true")
    p.add_argument("--no-compile",   action="store_true")
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
        use_compile  = not a.no_compile,
    )