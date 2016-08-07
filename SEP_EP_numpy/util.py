import autograd.numpy as np
from autograd import grad
from numpy.linalg import inv
import scipy as sp
from scipy.optimize import minimize
import matrix

def generate_data(n, W, sigx, dimx, dimz):
	'''
	generates factor analysis data
	'''
	observed = np.zeros([n, dimx])
	latent = np.zeros([n, dimz])
	
	for i in xrange(n):
		#latent variable
		z = np.random.normal(0,1, size = (dimz,))
		#observed
		mu = np.dot(W,z)
		cov = sigx*np.identity(dimx)
		x = np.random.multivariate_normal(mu, cov)
		observed[i] = x
		latent[i] = z
	
	return observed, latent
 
def stack_params(u, V, R):
	u = u.reshape(-1)
	V = V.reshape(-1)
	R = R.reshape(-1)
	params =  np.hstack((np.hstack((u, V)), R))
	return params

def unstack_params(params):
	f = int(params.shape[0]/8)
	u = params[:f]
	V = params[f:f*f+f].reshape((f,f))
	R = params[f*f+f:].reshape((2, 3))
	return u, V, R

def gradient_descent(objFunc, u, V, R):
	'''
	Very trivial deterministic gradient descent
	'''	
	#print "Starting gradient descent..."
	dfunc = grad(objFunc)
	lrate = 0.1
	sep_lrate = 0.1

	for i in range(1,60):
		params = stack_params(u, V, R)
		gradients = dfunc(params)
		du, dV, dR = unstack_params(gradients)
		u = u - sep_lrate*du
		V = V - sep_lrate*dV
		R = R - lrate*dR
		
		if i % 20 == 0:
			lrate = lrate/10.0
			sep_lrate = sep_lrate/10.0
	
	return u, V, R
	
	
	
def optimizer(objFunc, u, V, R):
	'''
	using scipy optimizer
	'''
	dfunc = grad(objFunc)
	params = stack_params(u, V, R)
	res = minimize(objFunc, params,method='L-BFGS-B', jac = dfunc) #, options = {'maxiter': 1})
	new_params = res.x
	u, V, R = unstack_params(new_params)
	return u, V, R
	
	
	
	
	



	