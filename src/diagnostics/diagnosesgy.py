import segyio
import numpy as np
import matplotlib.pyplot as plt

SEISMIC_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\filtering\mississippi_filtered_3_37Hz.sgy"
LABEL_PATH = r"G:\Working\Students\Undergraduate\For_Vince\Petrel\SaltDetection\data\labels\labelmississippi.sgy"

def diagnose_segy():
    """
    Comprehensive SEG-Y diagnostic to find the issue.
    """
    print("="*70)
    print("SEG-Y DIAGNOSTIC REPORT")
    print("="*70)
    
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f_seis:
        
        # 1. Basic info
        print(f"\n1. BASIC INFO:")
        print(f"   Total traces: {len(f_seis.trace):,}")
        print(f"   Samples per trace: {f_seis.samples.size}")
        print(f"   Sample interval: {segyio.tools.dt(f_seis)/1000:.2f} ms")
        
        # 2. Check if geometry can be inferred
        print(f"\n2. GEOMETRY INFERENCE:")
        try:
            # Try to get shape automatically
            shape = segyio.tools.cube(f_seis).shape
            print(f"   Auto-detected shape: {shape}")
        except:
            print(f"   ⚠️  Could not auto-detect geometry (irregular survey)")
        
        # 3. Sample random traces to see if data exists
        print(f"\n3. TRACE SAMPLING (checking for actual data):")
        
        sample_indices = [0, 100, 1000, 10000, len(f_seis.trace)//2, len(f_seis.trace)-1]
        
        for idx in sample_indices:
            if idx < len(f_seis.trace):
                trace = f_seis.trace[idx]
                print(f"   Trace {idx:8,}: min={trace.min():10.4f}, "
                      f"max={trace.max():10.4f}, "
                      f"mean={trace.mean():10.4f}, "
                      f"std={trace.std():10.4f}")
        
        # 4. Check first and last 10 traces
        print(f"\n4. EDGE TRACES (first 10 and last 10):")
        
        print("   First 10 traces:")
        for i in range(min(10, len(f_seis.trace))):
            trace = f_seis.trace[i]
            is_zero = np.allclose(trace, 0, atol=1e-6)
            print(f"     Trace {i}: {'ZERO/EMPTY' if is_zero else 'HAS DATA'} "
                  f"(std={trace.std():.2e})")
        
        print("   Last 10 traces:")
        for i in range(max(0, len(f_seis.trace)-10), len(f_seis.trace)):
            trace = f_seis.trace[i]
            is_zero = np.allclose(trace, 0, atol=1e-6)
            print(f"     Trace {i}: {'ZERO/EMPTY' if is_zero else 'HAS DATA'} "
                  f"(std={trace.std():.2e})")
        
        # 5. Find first non-zero trace
        print(f"\n5. FINDING FIRST NON-ZERO TRACE:")
        first_valid = None
        for i in range(min(1000, len(f_seis.trace))):
            trace = f_seis.trace[i]
            if not np.allclose(trace, 0, atol=1e-6):
                first_valid = i
                print(f"   ✅ First valid trace: {i}")
                print(f"      Stats: min={trace.min():.4f}, max={trace.max():.4f}, "
                      f"mean={trace.mean():.4f}")
                break
        
        if first_valid is None:
            print(f"   ❌ No valid data found in first 1000 traces!")
            print(f"      This suggests:")
            print(f"      - Survey might be stored in non-standard order")
            print(f"      - Data might be compressed/encoded")
            print(f"      - File might be corrupted")
        
        # 6. Visualize a few traces
        print(f"\n6. GENERATING VISUALIZATION:")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        trace_indices = [0, 100, 1000, 10000, len(f_seis.trace)//2, -1]
        
        for idx, (ax, trace_idx) in enumerate(zip(axes.flat, trace_indices)):
            if trace_idx == -1:
                trace_idx = len(f_seis.trace) - 1
            
            if trace_idx < len(f_seis.trace):
                trace = f_seis.trace[trace_idx]
                ax.plot(trace)
                ax.set_title(f'Trace {trace_idx:,}\n'
                           f'std={trace.std():.2e}, range=[{trace.min():.2f}, {trace.max():.2f}]')
                ax.set_xlabel('Time Sample')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('segy_diagnostic_traces.png', dpi=150)
        print(f"   Saved: segy_diagnostic_traces.png")
        
        # 7. Check headers for inline/xline info
        print(f"\n7. TRACE HEADERS (first trace):")
        try:
            header = f_seis.header[0]
            print(f"   Inline number: {header[segyio.TraceField.INLINE_3D]}")
            print(f"   Crossline number: {header[segyio.TraceField.CROSSLINE_3D]}")
            print(f"   CDP X: {header[segyio.TraceField.CDP_X]}")
            print(f"   CDP Y: {header[segyio.TraceField.CDP_Y]}")
        except:
            print(f"   ⚠️  Could not read trace headers")
    
    # 8. Compare with label file
    print(f"\n8. LABEL FILE COMPARISON:")
    with segyio.open(LABEL_PATH, 'r', ignore_geometry=True) as f_label:
        print(f"   Label file traces: {len(f_label.trace):,}")
        
        # Check if label has data
        label_trace = f_label.trace[0]
        print(f"   First label trace: min={label_trace.min():.4f}, "
              f"max={label_trace.max():.4f}, unique={np.unique(label_trace)}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE - Check output above for issues")
    print("="*70)

if __name__ == "__main__":
    diagnose_segy()