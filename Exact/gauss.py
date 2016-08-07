import numpy as np
from numpy.linalg import inv

def prod_gauss(mu1, mu2, cov1, cov2):
	cov = inv(inv(cov1) + inv(cov2))
	mu = np.dot(cov, np.dot(inv(cov1), mu1) + np.dot(inv(cov2), mu2))
	return mu, cov