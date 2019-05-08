import h5py
import numpy as np

import hdf5_getters

h5 = hdf5_getters.open_h5_file_read('../Databases/MSDsub/A/A/A/TRAAAAW128F429D538.h5')
duration = hdf5_getters.get_num_songs(h5)
h5.close()




# h = h5py.File('../Databases/MSDsub/A/A/A/TRAAAAW128F429D538.h5', 'r+')

# print(list(h.keys()))

# keys = []
# for i in h.keys():
#     keys.append(i)

# for k in keys:
#     for j in h[k]:
#         print(h[k][j])
#         print(h[k][j].value)
# # df1= np.array(X1)

# # for j in (X1):
#     # print(X1[j])
#     # print(X1[j].value)
