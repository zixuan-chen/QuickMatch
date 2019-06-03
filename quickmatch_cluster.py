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


def quickmatch_tree(density, dsq, n):
    """

    :param density:
    :param dsq: distance matrix
    :param n: number of points per image
    :return:
    """
    npoint = density.shape[0]
    tree_parent = np.zeros((npoint, ), dtype=np.int32)
    tree_distance = np.zeros((npoint, ), dtype=np.float32)

    for i in range(npoint):
        parent = i
        for j in range(npoint):
            # parent in the tree is given by the closest point
            # with higher density from another image
            if density[j] <= density[i] or (i // n) == (j // n):
                # density is lower or from the same image
                continue
            if parent == i or dsq[i][j] < dsq[i][parent]:
                parent = j
        tree_parent[i] = parent
        tree_distance[i] = dsq[i][parent]

    return tree_parent, tree_distance


def quickmatch_breaktree_merge(tree_parent, tree_distance, scales, rho_edge=0.5):

    npoint = tree_parent.shape[0]
    clusters_indicator = np.array([i for i in range(npoint)], dtype=np.int32)
    clusters = [[i] for i in range(npoint)]
    match_dis = scales

    idx_sorted = np.argsort(tree_distance, axis=0)
    for i in idx_sorted:
        p = tree_parent[i]
        c1 = clusters_indicator[i]
        c2 = clusters_indicator[p]
        if i != p and c1 != c2:
            match_dis_c1_c2 = min(match_dis[c1], match_dis[c2])
            if tree_distance[i] <= rho_edge * match_dis_c1_c2:
                clusters_indicator[clusters[c2]] = c1
                clusters[c1] = clusters[c1] + clusters[c2]
                clusters[c2] = []
                match_dis[c1] = match_dis_c1_c2

    res = []
    for i in range(len(clusters)):
        if clusters[i]:
            res.append(clusters[i])

    return res


def quickmatch_cluster(data, n, kernel=None, flag_low_memory=False, rho=0.25, rho_edge=0.5):
    """

    :param rho:
    :param rho_edge:
    :param data: [m*n, p]
    :param n: number of points per image
    :param kernel:
    :param flag_low_memory:
    :return:
    """
    scale = quickmatch_scale(data, n)
    dsq = euclidean_dist_matrix(data, data)
    tree_density = quickmatch_density(dsq, scales=scale, rho=rho,
                                      kernel=kernel,
                                      flag_low_memory=flag_low_memory)

    tree_parents, tree_distance = quickmatch_tree(tree_density, dsq, n)
    clusters = quickmatch_breaktree_merge(tree_parents, tree_distance,
                                          scales=scale, rho_edge=rho_edge)
    return clusters

