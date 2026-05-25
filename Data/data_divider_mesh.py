import numpy as np
import zipfile
import os
import gc

dt = 0.0005

def process_and_split(npz_file, prefix, step=1):
    steps_per_year = 2000
    idx_train = 3200 * steps_per_year
    idx_test = 4000 * steps_per_year

    with zipfile.ZipFile(npz_file, 'r') as z:
        npy_filename = z.namelist()[0]
        z.extract(npy_filename, path='.')
    
    mmap_data = np.load(npy_filename, mmap_mode='r')
    
    train_data = mmap_data[:idx_train:step].copy()
    np.savez_compressed(f'{prefix}_train_opt.npz', data=train_data)
    del train_data
    gc.collect()
    
    test_data = mmap_data[idx_train:idx_test:step].copy()
    np.savez_compressed(f'{prefix}_test_opt.npz', data=test_data)
    del test_data
    del mmap_data
    gc.collect()
    
    os.remove(npy_filename)

# dt = 0.0005
if dt == 0.0005:
    process_and_split('Data/Z.npz', f'Data/dt_{dt}/Z', step=1)
    process_and_split('Data/Z_dot.npz', f'Data/dt_{dt}/Z_dot', step=1)

# dt = 0.01
elif dt == 0.01:
    process_and_split('Data/Z.npz', f'Data/dt_{dt}/Z', step=20)
    process_and_split('Data/Z_dot.npz', f'Data/dt_{dt}/Z_dot', step=20)