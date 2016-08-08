import autograd.numpy as np
from autograd import grad 
from numpy.linalg import inv
import tensorflow as tf

def generate_data(n, W, sigx, dimx, dimz):
	#factor analysis
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
		
def gradient_descent(objFunc, w):
	dfunc = grad(objFunc)
	lrate = 0.000000001
	a = []
	for i in range(8000):
		a.append(objFunc(w))
		improv = lrate * dfunc(w)
		w = w - improv
		if i % 500 == 0:
			lrate = lrate/10.0
		if len(a) > 3 and (np.abs(a[-1] - a[-2]) < 0.00001):
			break
	return w,a


'''
few messy functions for playing with gaussians for testing 
'''

def product_gaussians(mu1, mu2, sig1, sig2):
	#sigma are matrices
	assert mu1.shape == mu2.shape, "Mu dimensions are not the same"
	assert sig1.shape == sig2.shape, "Sigma dimensions are not the same"
	prec1 = inv(sig1)
	prec2 = inv(sig2)
	
	sig = np.linalg.inv( prec1 + prec2 )
	temp = np.dot(prec1, mu1) + np.dot(prec2, mu2)
	mu = np.dot(sig, temp)
	return mu, sig
	
def product_diag_gauss(mu1, mu2, sig1, sig2):
	#sigma are the diagonal values of covariance
	prec1 = 1/sig1
	prec2 = 1/sig2
	
	sig= 1/(prec1 + prec2)
	mu = sig*(prec1*mu1 + prec2*mu2)
	
	return mu, sig
	
	
def exponent_diag_gauss(mu1, sig1, n):
	#sig = (1/float(sig1))*n
	#mu = sig*( (1/float(sig1))*mu1*n )
	sig = (1/sig1)*n
	mu = sig*( (1/sig1)*mu1*n )
	return mu1, sig

def tf_prod_gauss(mu1, mu2, sig1, sig2):
	#assumes precision
	prec1 = tf.matrix_inverse(sig1)
	prec2 = tf.matrix_inverse(sig2)
	prec = tf.add(prec1, prec2)
	sig = tf.matrix_inverse(prec)
	mu = tf.add( tf.matmul(prec1, mu1), tf.matmul(prec2, mu1) ) 

	return mu, sig


	
def tf_exponent_gauss(mu1, prec, n):
	'''
	takes in precision, and returns cov
	'''
	prec = tf.mul(n, prec)

	sig = tf.matrix_inverse(prec)
	mu = tf.matmul(prec, mu1)
	mu = tf.mul(n, mu)

	mu = tf.matmul(sig, mu)
	return mu, sig

def tf_evaluate_gauss(mu, prec, w):
	t = tf.sub(w, mu) #6x1
	tt = tf.transpose(t) #1x6

	val = tf.matmul(tt, prec)
	val = tf.matmul(val, t)
	val =  tf.reshape(val, shape = [])
	val = tf.mul(tf.constant(0.5, dtype = tf.float64), val)
	val = tf.exp(val)

	return val





	
	
	
	
	