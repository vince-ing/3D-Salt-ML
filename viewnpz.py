import numpy as np
   
data = np.load('processed_data/train/cube_000000.npz')
print(f"Seismic min/max: {data['seismic'].min():.3f} / {data['seismic'].max():.3f}")
print(f"Label unique values: {np.unique(data['label'])}")
   
