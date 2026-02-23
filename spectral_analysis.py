import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

filename = "raw_seismic_mississippi.sgy"

with segyio.open(filename, "r", ignore_geometry=True) as f:
    traces = segyio.tools.collect(f.trace[13000:14000]) 
    dt = segyio.tools.dt(f) / 1e6  # seconds

# window (e.g., 500–1500 ms)
t0, t1 = 0.5, 1.5
i0, i1 = int(t0/dt), int(t1/dt)

spectra = []
for tr in traces:
    tr_win = tr[i0:i1]
    tr_win = tr_win - np.mean(tr_win)
    tr_win *= windows.hann(len(tr_win))
    
    spec = np.abs(np.fft.rfft(tr_win))
    spectra.append(spec)

avg_spec = np.mean(spectra, axis=0)
freq = np.fft.rfftfreq(len(tr_win), dt)

plt.semilogy(freq, avg_spec)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Average Amplitude Spectrum")
plt.show()

print("dt:", dt)
print("Trace length:", len(traces[0]))

tr = traces[0]
print("Trace min/max:", tr.min(), tr.max())

nt = len(tr)
i0 = int(0.5/dt)
i1 = int(1.5/dt)
i0 = min(i0, nt-1)
i1 = min(i1, nt)

tr_win = tr[i0:i1]
print("Window length:", len(tr_win))
print("Window min/max:", tr_win.min() if len(tr_win)>0 else None,
                        tr_win.max() if len(tr_win)>0 else None)