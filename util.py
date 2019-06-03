import numpy as np


def euclidean_dist_matrix(a, b):
	"""
	compute Euclidean Distance Matrix
	:param a: [n1, p]
	:param b: [n2, p]
	:return: dsq [n1, n2]
	"""
	n1 = a.shape[0]
	n2 = b.shape[0]
	a_square = np.sum(np.power(a, 2), axis=1, keepdims=True)
	b_square = np.sum(np.power(b, 2), axis=1, keepdims=True)
	dsq = np.dot(a_square, np.ones((1, n2))) +\
		np.dot(np.ones((n1, 1)), b_square.T) - 2*np.dot(a, b.T)

	return np.sqrt(dsq)

