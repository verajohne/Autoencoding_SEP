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
	
	np.random.seed(12345)
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
	#f = 1
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
	lrate = 0.001
	sep_lrate = 0.001
	old_norm = 100
	norm = 0
	for i in range(1,1000):
		params = stack_params(u, V, R)
		#val, gradients = val_grad(params)
		gradients = dfunc(params)
		du, dV, dR = unstack_params(gradients)
		u = u - sep_lrate*du
		V = V - sep_lrate*dV
		R = R - lrate*dR
		
		norm = np.linalg.norm(gradients)
		if np.linalg.norm(old_norm - norm) < 0.000001:
			break
		old_norm = norm
	
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
	

def adam(objFunc, u, V, R):
	'''
	Adam Stochastic Gradient Descent
	'''

	m1 = 0
	m2 = 0
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	t = 0
	learning_rate = 0.001

	dfunc = grad(objFunc)
	old_norm = 100
	grady = []
	for i in range(1000):
		t+=1
		params = stack_params(u, V, R)
		gradients = dfunc(params)
		
		norm =  np.linalg.norm(gradients)
		grady.append(np.linalg.norm(gradients))
		m1 = beta1 * m1 + (1 - beta1) * gradients
		m2 = beta2 * m2 + (1 - beta2) * gradients**2
 		m1_hat = m1 / (1 - beta1**t)
 		m2_hat = m2 / (1 - beta2**t)

		delta = learning_rate*m1_hat/(np.sqrt(m2_hat) + epsilon)
		du, dV, dR = unstack_params(delta)
		u = u - du
		V = V -	dV
		#R = R -	dR

		if np.linalg.norm(old_norm - norm) < 0.0001:
			break
		old_norm = norm

	grady = np.array(grady)
	np.save('adam_gradients', grady)
	return u, V, R
	
	
	
	



	