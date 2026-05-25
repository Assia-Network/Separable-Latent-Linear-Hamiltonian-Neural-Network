import numpy as np
import gc

print("Processing Z_train...")
data_train = np.load('Data/Z_train_opt.npz')['data']
max_z_train = np.max(np.abs(data_train), axis=0)
del data_train
gc.collect()

print("Processing Z_test...")
data_test = np.load('Data/Z_test_opt.npz')['data']
max_z_test = np.max(np.abs(data_test), axis=0)
del data_test
gc.collect()

Z_norm_vector = np.maximum(max_z_train, max_z_test)

print("Processing Z_dot_train...")
data_dot_train = np.load('Data/Z_dot_train_opt.npz')['data']
max_z_dot_train = np.max(np.abs(data_dot_train), axis=0)
del data_dot_train
gc.collect()

print("Processing Z_dot_test...")
data_dot_test = np.load('Data/Z_dot_test_opt.npz')['data']
max_z_dot_test = np.max(np.abs(data_dot_test), axis=0)
del data_dot_test
gc.collect()

Z_dot_norm_vector = np.maximum(max_z_dot_train, max_z_dot_test)

print("\n--- Results ---")
print(f"Z_norm_vector shape: {Z_norm_vector.shape}")
print(Z_norm_vector)
print("\n")
print(f"Z_dot_norm_vector shape: {Z_dot_norm_vector.shape}")
print(Z_dot_norm_vector)

np.savez('Data/normalization_vectors.npz', Z_norm=Z_norm_vector, Z_dot_norm=Z_dot_norm_vector)
print("\nNormalization vectors calculated and saved to 'normalization_vectors.npz'!")