import numpy as np

profile = np.load("data/profiles/profile_1.npy")
max_val = np.max(profile)

print("Shape of profile:", profile.shape)
print("Size of profile:", profile.size)
print("Data type:", profile.dtype)
print("Max value:", max_val)
print("Min value:", np.min(profile))
