import numpy as np 

def euclideanDistMatrix(A, B):
    # A: [NA, p] B: [NB, p]
	# return [NA, NB]
	NA = A.shape[0]
	NB = B.shape[0]
	A_square = np.sum(np.power(A,2), axis=1, keepdims=True)
	B_square = np.sum(np.power(B,2), axis=1, keepdims=True)
	dsq = np.dot(A_square, np.ones((1, NB))) +\
		  np.dot(np.ones((NA, 1)), B_square.T) - 2*np.dot(A, B.T)
    return dsq
    