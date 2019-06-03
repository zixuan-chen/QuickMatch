import numpy as np
from util import euclidean_dist_matrix


def quickmatch_density(data, scales, rho=0.25, kernel=None,
                       kernel_support_radius=None,
                       flag_low_memory=False,
                       chunk_size=None,
                       flag_recursive=False):
    # scales = dik
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
        tree_density = np.sum(p, axis=1)

    else:
        # here data is [p, n] raw data
        npoint = data.shape[0]
        tree_density = np.zeros((npoint,), dtype=np.float32)
        for i in range(0, npoint, chunk_size):
            j = min(i+chunk_size, npoint)
            dsq = euclidean_dist_matrix(data[:, i:j], data)
            tree_density[i:j] = quickmatch_density(dsq, scales=scales, rho=rho,
                                                   kernel=kernel,
                                                   kernel_support_radius=kernel_support_radius,
                                                   flag_recursive=True)

    return tree_density
