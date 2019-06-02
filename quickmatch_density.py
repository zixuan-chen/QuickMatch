import numpy as np 
from util import euclideanDistMatrix

def quickmatch_density(data, scales, rho=1, kernel=None,
                       kernel_support_radius=None,
                       flag_low_memory=False,
                       chunk_size=None,
                       flag_recursive=False):
    """
    for distance
    :param data:
    :param scales:
    :param rho:
    :param kernel:
    :param kernel_support_radius:
    :param flag_low_memory:
    :param chunk_size:
    :param flag_recursive:
    :return:
    """
    sigma = rho * scales
    if (not flag_low_memory) or flag_recursive:
        dsq = data
        dsq = dsq / sigma
        if not kernel:
            p = dsq
        else:
            if kernel_support_radius:
                p = np.zeros(dsq.shape, dtype=np.float32)
                flag = dsq < kernel_support_radius
                p[flag] = kernel(-dsq[flag])
            else:
                p = kernel(-dsq)
        tree_density = np.sum(p, axis=0)
    else:
        # here data is [p, n] raw data
        point_num = data.shape[1]
        tree_density = np.zeros((point_num,), dtype=np.float32)
        for i in range(0, point_num, chunk_size):
            j = min(i+chunk_size, point_num)
            dsq = np.sqrt(euclideanDistMatrix(data[:, i:j], data))

