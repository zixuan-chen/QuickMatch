import numpy as np
from util import euclidean_dist_matrix

def quickmatch_scale(data, n):
    """
    calculate the $d_{ik}$ for each point
    :param data: [m*n, p] data, m is number of images
                for data[x][p], i = x // n, k = x % n
    :param n: number of points per image
    :return: a $d_{ik}$ array of shape[m*n,]
    """
    npoint = data.shape[0]
    scale = np.zeros((npoint, ), dtype=np.float32)
    for i in range(0, npoint, n):
        dsq_i = euclidean_dist_matrix(data[i:i+n], data[i:i+n])
        for k in range(0, n):
            min_d = 1e5
            for y in range(0, n):
                if k == y:
                    continue
                min_d = min(min_d, dsq_i[k][y])
            scale[i+k] = min_d
    return scale
