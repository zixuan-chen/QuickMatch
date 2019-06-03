import numpy as np


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

    for i in range(len(clusters)):
        if not clusters[i]:
            clusters.remove(i)

    return clusters
