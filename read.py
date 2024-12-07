import h5py
import numpy as np
import scipy.io  as sio
import time
start = time.time()
mat_path = r'data/muufl/muufl.mat'
mat = sio.loadmat(mat_path)['gt']
# mat = h5py.File('data/Lidar_Trento.mat', 'r+')
# print(mat.keys())
# print(mat.values())
# f = mat['lidar_trento']
# mat_t = np.transpose(f)
print(np.unique(mat))
for i in range(np.unique(mat)):
    print(f"{i} = {np.sum(mat == i)}")
# io.savemat('data/Lidar_Trento.mat', {'lidar_trento': mat_t})
end = time.time()
print(end - start)
# print(mat['GT_houston'].shape)

#
# [1. 2. 3. 4. 5. 6. 7. 8. 9.]
# 1 = 2540
# 2 = 30295
# 3 = 2976
# 4 = 107
# 5 = 49317
# 6 = 42271
# 7 = 58900
# 8 = 3302
# 9 = 10292