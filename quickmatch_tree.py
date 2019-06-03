import numpy as np


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
