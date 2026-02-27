"""
Seismic Survey Diagnostic: Amplitude & Frequency Comparison
============================================================
Compares two SEGY surveys (e.g. before/after equalization, or cross-survey
matching) across amplitude, amplitude difference, and frequency content.

Usage:
    Edit the file paths in the CONFIG section at the bottom, then run:
        python amplitude_spectral.py

Fixes vs previous version:
  - Each survey now uses its OWN sample rate for spectral analysis
    (previously both used the reference dt, giving wrong frequency axes)
  - Wiggle difference plot and amplitude difference now trim to common
    sample length before subtracting (fixes crash when surveys have
    different numbers of samples per trace)
"""

import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_traces(filename, n_samples=2000, seed=42):
    """
    Randomly samples n_samples traces from a SEGY file.
    Returns a 2D numpy array (n_samples, n_samples_per_trace) and the sample
    interval in milliseconds.
    """
    rng = np.random.default_rng(seed)
    print(f"  Sampling {n_samples} traces from: {os.path.basename(filename)}")

    with segyio.open(filename, "r", ignore_geometry=True) as f:
        trace_count = f.tracecount
        dt_us = segyio.dt(f)              # microseconds
        dt_ms = dt_us / 1000.0

        indices = np.sort(rng.choice(trace_count, size=min(n_samples, trace_count), replace=False))
        traces = np.array([f.trace[int(i)] for i in indices], dtype=np.float32)

    print(f"    → {trace_count:,} total traces | dt = {dt_ms:.3f} ms | "
          f"samples/trace = {traces.shape[1]}")
    return traces, dt_ms


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(traces, label=""):
    """Returns a dict of scalar amplitude statistics for a trace array."""
    flat = traces.flatten()
    rms  = float(np.sqrt(np.mean(flat ** 2)))
    stats = {
        "label"        : label,
        "rms"          : rms,
        "mean"         : float(np.mean(flat)),
        "std"          : float(np.std(flat)),
        "median"       : float(np.median(np.abs(flat))),
        "p01"          : float(np.percentile(flat, 1)),
        "p99"          : float(np.percentile(flat, 99)),
        "max_abs"      : float(np.max(np.abs(flat))),
        "per_trace_rms": np.sqrt(np.mean(traces ** 2, axis=1)),
    }
    return stats


def print_stats_table(s1, s2):
    keys   = ["rms", "mean", "std", "median", "p01", "p99", "max_abs"]
    labels = ["RMS", "Mean", "Std Dev", "Median |amp|", "P01", "P99", "Max |amp|"]

    col_w = 18
    sep   = "─" * (12 + col_w * 3)
    hdr   = f"{'Metric':<12}{s1['label']:>{col_w}}{s2['label']:>{col_w}}{'Ratio (Ref/Tgt)':>{col_w}}"

    print("\n" + "═" * len(sep))
    print("  AMPLITUDE STATISTICS")
    print("═" * len(sep))
    print(hdr)
    print(sep)
    for k, lbl in zip(keys, labels):
        v1    = s1[k]
        v2    = s2[k]
        ratio = v1 / v2 if v2 != 0 else float("nan")
        print(f"{lbl:<12}{v1:>{col_w}.4f}{v2:>{col_w}.4f}{ratio:>{col_w}.4f}")
    print(sep)

    ks_stat, ks_p = ks_2samp(s1["per_trace_rms"], s2["per_trace_rms"])
    print(f"\n  KS test on per-trace RMS distributions:")
    print(f"    statistic = {ks_stat:.4f}  |  p-value = {ks_p:.4e}")
    if ks_p > 0.05:
        print("    → Distributions are NOT significantly different (p > 0.05) ✓")
    else:
        print("    → Distributions ARE significantly different (p ≤ 0.05)  ✗")
    print("═" * len(sep) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SPECTRAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_spectrum(traces, dt_ms):
    """
    Returns frequency axis (Hz) and mean power spectrum across all traces.
    Uses Welch's method on each trace then averages.

    IMPORTANT: dt_ms must be the correct sample interval for THIS specific
    survey. Do not share dt between surveys with different sample rates —
    the frequency axis will be wrong.

    Sample rate examples:
        dt = 10 ms  →  fs = 100 Hz  →  Nyquist = 50 Hz
        dt =  4 ms  →  fs = 250 Hz  →  Nyquist = 125 Hz
        dt =  2 ms  →  fs = 500 Hz  →  Nyquist = 250 Hz
    """
    dt_s    = dt_ms / 1000.0
    fs      = 1.0 / dt_s
    n       = traces.shape[1]
    nperseg = min(n, 256)

    psd_list = []
    for tr in traces:
        f, psd = signal.welch(tr, fs=fs, nperseg=nperseg)
        psd_list.append(psd)

    mean_psd = np.mean(psd_list, axis=0)
    return f, mean_psd


def dominant_frequency(freqs, psd):
    """Returns the dominant frequency and the -3 dB bandwidth edges."""
    peak_idx = np.argmax(psd)
    f_dom    = freqs[peak_idx]

    half_power = psd[peak_idx] / 2.0
    above  = np.where(psd >= half_power)[0]
    f_low  = freqs[above[0]]  if len(above) else np.nan
    f_high = freqs[above[-1]] if len(above) else np.nan
    return f_dom, f_low, f_high


def print_spectral_stats(freqs_ref, psd_ref, freqs_tgt, psd_tgt, label1, label2):
    """
    Print spectral statistics. Each survey has its own frequency axis
    because they may have different sample rates.
    """
    f_dom1, fl1, fh1 = dominant_frequency(freqs_ref, psd_ref)
    f_dom2, fl2, fh2 = dominant_frequency(freqs_tgt, psd_tgt)

    print("═" * 55)
    print("  SPECTRAL STATISTICS")
    print("═" * 55)
    print(f"  {'Metric':<28}{label1:>12}{label2:>12}")
    print("─" * 55)
    print(f"  {'Dominant freq (Hz)':<28}{f_dom1:>12.1f}{f_dom2:>12.1f}")
    print(f"  {'Nyquist freq (Hz)':<28}{freqs_ref[-1]:>12.1f}{freqs_tgt[-1]:>12.1f}")
    print(f"  {'-3 dB low edge (Hz)':<28}{fl1:>12.1f}{fl2:>12.1f}")
    print(f"  {'-3 dB high edge (Hz)':<28}{fh1:>12.1f}{fh2:>12.1f}")
    print(f"  {'-3 dB bandwidth (Hz)':<28}{fh1-fl1:>12.1f}{fh2-fl2:>12.1f}")

    peak1    = np.max(psd_ref)
    peak2    = np.max(psd_tgt)
    ratio_db = 10 * np.log10(peak1 / peak2) if peak2 > 0 else np.nan
    print(f"\n  Peak PSD ratio (Ref / Tgt): {ratio_db:+.2f} dB")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"ref": "#2196F3", "tgt": "#FF5722", "diff": "#4CAF50", "neutral": "#9E9E9E"}

def make_diagnostic_figure(traces_ref, traces_tgt, dt_ref, dt_tgt,
                            stats_ref, stats_tgt,
                            label_ref, label_tgt, output_path=None):
    """
    Builds a 4-row diagnostic figure.
    dt_ref and dt_tgt are passed separately so each survey uses its own
    correct sample rate for spectral analysis.

    Traces are trimmed to the shorter sample length before any subtraction
    to handle surveys with different numbers of samples per trace.
    """
    # Trim to common sample length for difference plots
    n_samples_common = min(traces_ref.shape[1], traces_tgt.shape[1])
    traces_ref_c = traces_ref[:, :n_samples_common]
    traces_tgt_c = traces_tgt[:, :n_samples_common]

    fig = plt.figure(figsize=(20, 22), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    title_kw = dict(color="white", fontsize=11, fontweight="bold")
    label_kw = dict(color="#cccccc", fontsize=9)
    tick_kw  = dict(colors="#aaaaaa", labelsize=8)

    def style_ax(ax, title=""):
        ax.set_facecolor("#0f0f23")
        ax.tick_params(axis="both", **tick_kw)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        if title:
            ax.set_title(title, **title_kw, pad=6)
        ax.title.set_color("white")

    fig.suptitle(
        f"Seismic Survey Diagnostic  ·  {label_ref}  vs  {label_tgt}",
        fontsize=14, fontweight="bold", color="white", y=0.99
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.96, bottom=0.04)

    # ── ROW 0: WIGGLE OVERLAY ────────────────────────────────────────────────
    n_wiggle = min(80, traces_ref_c.shape[0], traces_tgt_c.shape[0])
    # Use ref dt for time axis (both trimmed to same sample count)
    t_axis   = np.arange(n_samples_common) * dt_ref

    ax_wig_r = fig.add_subplot(gs[0, 0])
    ax_wig_t = fig.add_subplot(gs[0, 1])
    ax_wig_d = fig.add_subplot(gs[0, 2])

    clip = np.percentile(np.abs(traces_ref_c[:n_wiggle]), 99) * 0.6

    diff_traces = traces_ref_c[:n_wiggle] - traces_tgt_c[:n_wiggle]

    for ax, traces, color, lbl in [
        (ax_wig_r, traces_ref_c[:n_wiggle], COLORS["ref"],  label_ref),
        (ax_wig_t, traces_tgt_c[:n_wiggle], COLORS["tgt"],  label_tgt),
        (ax_wig_d, diff_traces,             COLORS["diff"], "Difference"),
    ]:
        for i, tr in enumerate(traces):
            tr_norm = np.clip(tr / (clip + 1e-9), -1, 1) * 0.5
            ax.plot(tr_norm + i, t_axis, color=color, lw=0.4, alpha=0.7)
        style_ax(ax, f"Wiggle: {lbl} (first {n_wiggle} traces)")
        ax.set_xlabel("Trace index", **label_kw)
        ax.set_ylabel("Time (ms)", **label_kw)
        ax.invert_yaxis()
        ax.set_xlim(-1, n_wiggle + 1)

    # ── ROW 1: HISTOGRAMS + PER-TRACE RMS ───────────────────────────────────
    ax_hist   = fig.add_subplot(gs[1, :2])
    ax_rms_sc = fig.add_subplot(gs[1, 2])

    clip95 = np.percentile(np.abs(np.concatenate([
        traces_ref_c.flatten(), traces_tgt_c.flatten()])), 97)
    bins = np.linspace(-clip95, clip95, 120)

    ax_hist.hist(traces_ref_c.flatten(), bins=bins, color=COLORS["ref"],
                 alpha=0.55, density=True, label=label_ref)
    ax_hist.hist(traces_tgt_c.flatten(), bins=bins, color=COLORS["tgt"],
                 alpha=0.55, density=True, label=label_tgt)
    ax_hist.axvline(0, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
    style_ax(ax_hist, "Amplitude Distribution (normalised density)")
    ax_hist.set_xlabel("Amplitude", **label_kw)
    ax_hist.set_ylabel("Density", **label_kw)
    ax_hist.legend(fontsize=8, facecolor="#111122", labelcolor="white",
                   edgecolor="#333355")

    rms_ref_pt = stats_ref["per_trace_rms"]
    rms_tgt_pt = stats_tgt["per_trace_rms"]
    n_common   = min(len(rms_ref_pt), len(rms_tgt_pt))
    ax_rms_sc.scatter(rms_ref_pt[:n_common], rms_tgt_pt[:n_common],
                      s=4, alpha=0.35, color=COLORS["neutral"], rasterized=True)
    mx = max(rms_ref_pt.max(), rms_tgt_pt.max())
    ax_rms_sc.plot([0, mx], [0, mx], color="white", lw=1, ls="--",
                   alpha=0.7, label="1:1")
    style_ax(ax_rms_sc, "Per-trace RMS scatter")
    ax_rms_sc.set_xlabel(f"RMS  {label_ref}", **label_kw)
    ax_rms_sc.set_ylabel(f"RMS  {label_tgt}", **label_kw)
    ax_rms_sc.legend(fontsize=8, facecolor="#111122", labelcolor="white",
                     edgecolor="#333355")

    # ── ROW 2: AMPLITUDE DIFFERENCE ─────────────────────────────────────────
    ax_diff_hist = fig.add_subplot(gs[2, :2])
    ax_diff_rms  = fig.add_subplot(gs[2, 2])

    n_common_tr = min(traces_ref_c.shape[0], traces_tgt_c.shape[0])
    diff_flat   = (traces_ref_c[:n_common_tr] - traces_tgt_c[:n_common_tr]).flatten()
    diff_rms_pt = np.sqrt(np.mean(
        (traces_ref_c[:n_common_tr] - traces_tgt_c[:n_common_tr]) ** 2, axis=1))

    bins_d = np.linspace(np.percentile(diff_flat, 1),
                         np.percentile(diff_flat, 99), 120)
    ax_diff_hist.hist(diff_flat, bins=bins_d, color=COLORS["diff"],
                      alpha=0.75, density=True)
    ax_diff_hist.axvline(np.mean(diff_flat), color="yellow", lw=1.2,
                         label=f"Mean diff = {np.mean(diff_flat):.3f}")
    ax_diff_hist.axvline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    style_ax(ax_diff_hist, "Amplitude Difference Distribution (Ref − Tgt)")
    ax_diff_hist.set_xlabel("Amplitude difference", **label_kw)
    ax_diff_hist.set_ylabel("Density", **label_kw)
    ax_diff_hist.legend(fontsize=8, facecolor="#111122", labelcolor="white",
                        edgecolor="#333355")

    ax_diff_rms.plot(diff_rms_pt, np.arange(len(diff_rms_pt)),
                     color=COLORS["diff"], lw=0.6, alpha=0.8)
    ax_diff_rms.axvline(np.mean(diff_rms_pt), color="yellow", lw=1.2,
                        ls="--", label=f"Mean = {np.mean(diff_rms_pt):.3f}")
    style_ax(ax_diff_rms, "Per-trace Difference RMS")
    ax_diff_rms.set_xlabel("Difference RMS", **label_kw)
    ax_diff_rms.set_ylabel("Trace index (sampled)", **label_kw)
    ax_diff_rms.invert_yaxis()
    ax_diff_rms.legend(fontsize=8, facecolor="#111122", labelcolor="white",
                       edgecolor="#333355")

    # ── ROW 3: POWER SPECTRA ─────────────────────────────────────────────────
    # Each survey uses its OWN dt so frequency axes are correct
    ax_spec_lin = fig.add_subplot(gs[3, :2])
    ax_spec_db  = fig.add_subplot(gs[3, 2])

    freqs_ref, psd_ref = compute_mean_spectrum(traces_ref, dt_ref)
    freqs_tgt, psd_tgt = compute_mean_spectrum(traces_tgt, dt_tgt)

    psd_ref_n = psd_ref / psd_ref.max()
    psd_tgt_n = psd_tgt / psd_tgt.max()

    ax_spec_lin.fill_between(freqs_ref, psd_ref_n, alpha=0.3, color=COLORS["ref"])
    ax_spec_lin.fill_between(freqs_tgt, psd_tgt_n, alpha=0.3, color=COLORS["tgt"])
    ax_spec_lin.plot(freqs_ref, psd_ref_n, color=COLORS["ref"], lw=1.5, label=label_ref)
    ax_spec_lin.plot(freqs_tgt, psd_tgt_n, color=COLORS["tgt"], lw=1.5, label=label_tgt)
    style_ax(ax_spec_lin, "Mean Power Spectrum (normalised, linear)")
    ax_spec_lin.set_xlabel("Frequency (Hz)", **label_kw)
    ax_spec_lin.set_ylabel("Relative power", **label_kw)
    ax_spec_lin.legend(fontsize=8, facecolor="#111122", labelcolor="white",
                       edgecolor="#333355")
    ax_spec_lin.set_xlim(0, max(freqs_ref[-1], freqs_tgt[-1]))

    eps = 1e-12
    psd_ref_db = 10 * np.log10(psd_ref_n + eps)
    psd_tgt_db = 10 * np.log10(psd_tgt_n + eps)

    ax_spec_db.plot(freqs_ref, psd_ref_db, color=COLORS["ref"], lw=1.5, label=label_ref)
    ax_spec_db.plot(freqs_tgt, psd_tgt_db, color=COLORS["tgt"], lw=1.5, label=label_tgt)
    ax_spec_db.axhline(-3,  color="white",          lw=0.8, ls="--", alpha=0.5, label="-3 dB")
    ax_spec_db.axhline(-10, color=COLORS["neutral"], lw=0.6, ls=":",  alpha=0.4, label="-10 dB")
    style_ax(ax_spec_db, "Power Spectrum (dB, normalised)")
    ax_spec_db.set_xlabel("Frequency (Hz)", **label_kw)
    ax_spec_db.set_ylabel("Power (dB)", **label_kw)
    ax_spec_db.set_ylim(-60, 3)
    ax_spec_db.set_xlim(0, max(freqs_ref[-1], freqs_tgt[-1]))
    ax_spec_db.legend(fontsize=8, facecolor="#111122", labelcolor="white",
                      edgecolor="#333355")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure saved → {output_path}")
    plt.show()

    return freqs_ref, freqs_tgt, psd_ref, psd_tgt


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIG ────────────────────────────────────────────────────────────────
    REF_FILE   = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\filtering\mississippi_filtered_3_37Hz.sgy"
    TGT_FILE   = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\filtering\keathley_filtered_equalized.sgy"

    LABEL_REF  = "Mississippi (ref)"
    LABEL_TGT  = "Keathley (equalized)"

    N_SAMPLES  = 9000
    OUTPUT_PNG = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\outputs\figures\filtering_diagnostic.png"
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  SEISMIC SURVEY DIAGNOSTIC")
    print("=" * 60 + "\n")

    # 1. Load — each survey keeps its own dt
    print("[1/4] Loading trace samples...")
    traces_ref, dt_ref = sample_traces(REF_FILE, n_samples=N_SAMPLES)
    traces_tgt, dt_tgt = sample_traces(TGT_FILE, n_samples=N_SAMPLES)

    if abs(dt_ref - dt_tgt) > 0.01:
        print(f"\n  ⚠  WARNING: sample intervals differ ({dt_ref} ms vs {dt_tgt} ms).")
        print(f"     Nyquist ref = {1000/(2*dt_ref):.1f} Hz | Nyquist tgt = {1000/(2*dt_tgt):.1f} Hz")
        print(f"     Spectral plots will use each survey's own frequency axis.")
        print(f"     Amplitude difference plots trim to the shorter trace length ({min(traces_ref.shape[1], traces_tgt.shape[1])} samples).\n")

    # 2. Statistics
    print("[2/4] Computing statistics...")
    stats_ref = compute_stats(traces_ref, label=LABEL_REF)
    stats_tgt = compute_stats(traces_tgt, label=LABEL_TGT)
    print_stats_table(stats_ref, stats_tgt)

    # 3. Spectral stats — pass each survey's own dt
    print("[3/4] Computing power spectra...")
    freqs_ref, psd_ref = compute_mean_spectrum(traces_ref, dt_ref)
    freqs_tgt, psd_tgt = compute_mean_spectrum(traces_tgt, dt_tgt)
    print_spectral_stats(freqs_ref, psd_ref, freqs_tgt, psd_tgt, LABEL_REF, LABEL_TGT)

    # 4. Equalization quality summary
    rms_ratio = stats_ref["rms"] / stats_tgt["rms"]
    print("═" * 55)
    print("  EQUALIZATION QUALITY SUMMARY")
    print("═" * 55)
    print(f"  RMS ratio (Ref / Tgt)    : {rms_ratio:.4f}  "
          f"({'✓ well matched' if abs(rms_ratio - 1) < 0.05 else '✗ mismatch > 5%'})")

    n_common         = min(traces_ref.shape[0], traces_tgt.shape[0])
    n_samples_common = min(traces_ref.shape[1], traces_tgt.shape[1])
    diff_rms = np.sqrt(np.mean(
        (traces_ref[:n_common, :n_samples_common] -
         traces_tgt[:n_common, :n_samples_common]) ** 2))
    sig_rms = (stats_ref["rms"] + stats_tgt["rms"]) / 2
    nrmd    = diff_rms / sig_rms * 100
    print(f"  Normalised RMS difference: {nrmd:.2f}%  "
          f"({'✓ < 20%' if nrmd < 20 else '⚠ consider trace-by-trace EQ'})")
    print("═" * 55 + "\n")

    # 5. Plot — pass both dt values separately
    print("[4/4] Generating diagnostic figure...")
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    make_diagnostic_figure(
        traces_ref, traces_tgt, dt_ref, dt_tgt,
        stats_ref, stats_tgt,
        LABEL_REF, LABEL_TGT,
        output_path=OUTPUT_PNG,
    )

    print("\nDone.\n")