import sys
sys.path.append('../')

import util
import gaussian
from autograd import grad

import autograd.scipy as sp
import autograd.numpy as np

from autograd.util import quick_grad_check
from autograd import value_and_grad

np.random.seed(1234)
import re
import profile


class Fa(object):
	def __init__(self, n, dimx, dimz):
		
		'''
		n: size of dataset
		dimx: dimensions of observed variables
		dimz: dimensions of local latent variables
		'''
		'''
		Generative procedure
		global param W and latent variables are not visible
		'''
		self.n = n
		self.sigx = 0.1
		np.random.seed(1234)
		self.W = np.random.normal(0,1, size = (dimx,dimz))
		self.dimz = dimz
		self.dimx = dimx
		#data
		data = util.generate_data(n, self.W, self.sigx, dimx, dimz)
		self.observed = data[0] 
		self.latent = data[1] 
		
		
		'''
		Model Parameters: mean and precision
		'''
		#SEP params
		f = dimx*dimz
		self.SEP_prior_mean = np.zeros(f).reshape((f,1)) #fx1
		self.SEP_prior_prec = np.identity(f)	#fxf

		self.u = np.random.randn(f)
		self.V = 1e-4*np.eye(f)
		
		#recogntion model parameters
		self.R = np.random.randn(dimz,dimx) 
		self.S = self.sigx*np.eye(dimz)
		

	def get_q_dist(self, u, V):
		'''
		returns mean and covariance of q distribution of SEP
		'''
		prec = self.n*V
		prec = np.add(prec, self.SEP_prior_prec)
		qsig = np.linalg.inv(prec)
		qmu = self.n*np.dot(V, u) #TODO: add in prior mean to make general- assumes zero mean prior npw
		qmu = np.dot(qsig, qmu)
			
		return qmu, qsig
		
	def sample_w(self, u, V, no_samples = 1):
		qmu, qsig = self.get_q_dist(self.u, self.V)
		q = gaussian.Gaussian_full(qmu, qsig)
		w = q.sample(no_samples)
		w = w.reshape((self.dimx, self.dimz))
		
		return w
		
	def sample_z(self, x, R):
		mu = np.dot(R, x)
		f = gaussian.Gaussian_full(mu, self.S)
		s = f.sample(1)
		
		return f.sample(1)
	
	def evaluate_joint(self, x, z, w):
		'''
		evaluates joint of x and z
		'''
		pz = gaussian.Gaussian_full(np.zeros(self.dimz), np.eye(self.dimz))
		pz = pz.eval_log(z.reshape((self.dimz,)))
		
		mu = np.dot(w,z).reshape((self.dimx,))
		px = gaussian.Gaussian_full(mu, self.sigx*np.identity(self.dimx))
		px = px.eval_log_properly(x) #----------------------------------------------------------EVAL
		return px+pz
	
	def log_tilted_norm(self, x, R, u, V):
		'''
		returns the log of the normalalisation constant
		'''
		n_samples = 100
		res = 0
		
		gamma = []
		for i in range(n_samples):
			w = self.sample_w(u, V)
			z = self.sample_z(x, R)
			
			gamma.append( self.log_gamma(x, z, w, R, u, V) )
		
		res = sp.misc.logsumexp(np.array(gamma)) - np.log(n_samples)

		return res
			
	def log_gamma(self, x, z, w, R, u, V):
		
		joint = self.evaluate_joint(x, z, w)
	
		#recognition model
		mu = np.dot(R, x)
  
		pz = gaussian.eval_log_prec(mu, self.S, z.reshape((self.dimz,)))

		#sep ---- pq returned as nan
		cov = np.linalg.inv(V)
		pq = gaussian.Gaussian_full(u, cov)
		pq = pq.eval_log(w.reshape(-1))
		#pq = gaussian.eval_log_prec(u, V, w.reshape(-1)) 
		
		#print pq, pz
		res = joint - pz - pq
		
		return res
		
	def get_W(self, x, R, u, V):
		#calculate w and zp
		M = 30
		M_gammas = []
		np.random.seed(1000)
		for i in xrange(M):
			w = self.sample_w(u, V)
			z = self.sample_z(x, R)
			M_gammas.append( self.log_gamma(x, z, w, R, u, V) )
		
		M_gammas = np.array(M_gammas) - np.log(M)
		M_gammas = M_gammas - sp.misc.logsumexp(M_gammas)
		#print M_gammas
		

		return M_gammas
		
		
	def objective(self, x, R, u, V):
		
		M_gammas = self.get_W( x, R, u, V)
		#M_gammas2 = self.get_W( x, R, u, V)
		#M_gammas = M_gammas - (sp.misc.logsumexp(M_gammas2) - np.log(100))
		return np.sum(np.exp(M_gammas)*M_gammas)
		#norm_constant = sp.misc.logsumexp(M_gammas)
		#M_gammas_norm = M_gammas - norm_constant
		#return np.sum(np.exp(M_gammas_norm)*M_gammas) - norm_constant
		
	def objective_wrapper_(self, params, x):
		u, V, R = util.unstack_params(params)
		
		return self.objective(x, R, u, V)
	
	def get_marginal(self, u, V, R, x_test):
		'''
		current metric to test convergence-- log space
		predictive marginal likelihood
		'''
		I = self.sigx*np.identity(self.dimx)
		mu = np.zeros(self.dimx,)
		n_samples = 200
		ll = 0
		test_size = x_test.shape[0]
		
		for i in xrange(test_size):
			x = x_test[i]
			
			mc = 0
			for j in xrange(n_samples):
				w = self.sample_w(u, V)
				var = np.dot(w, np.transpose(w))
				var = np.add(var, I)
				px = gaussian.Gaussian_full(mu, var)
				px = px.eval(x)#eval_log_properly(x)
				mc = mc + px
				
			mc = mc/float(n_samples)
			mc = np.log(mc)
			ll += mc
			
		
		return (ll/float(test_size))

	def true_marg(self, x_test):
		'''
		returns true predictive marginal likelihood
		based on generative model params: W
		'''
		I = self.sigx*np.identity(self.dimx)
		mu = np.zeros(self.dimx,)
		test_size = x_test.shape[0]
		test_size = x_test.shape[0]
		ll = 0
		for i in range(test_size):
			x = x_test[i]
			var = np.dot(self.W, self.W.T)
			var = np.add(var, I)
			px = gaussian.Gaussian_full(mu, var)
			px = px.eval(x)
			ll += np.log(px)


		return (ll/float(test_size))


	 
	def fit(self, n_iter):

		old_params = util.stack_params(self.u, self.V, self.R)
		
		x_test = util.generate_data(5, self.W, self.sigx, self.dimx, self.dimz)[0]
		llh = []

		print "True: ", self.true_marg(x_test)

		def objective_wrapper(params, x):
			u, V, R = util.unstack_params(params)
		
			return self.objective(x, R, u, V)

		for j in xrange(n_iter):
			for i in xrange(self.n):
				x = self.observed[i]
				
				obj_local = lambda param: objective_wrapper(param, x)
				
				#u, V, R = self.gradient_descent(obj_local, self.u, self.V, self.R, x)
				
				u, V, R = util.adam(obj_local, self.u, self.V, self.R)
				#u, V, R = util.optimizer(obj_local, self.u, self.V, self.R)
				#print self.objective(x, R,self.u,self.V)
				#temp
				a = 1/float(self.n)
				self.u = np.dot(self.V, self.u)*(1- a) + a*(np.dot(V,u))
				self.V = self.V + a*(V- self.V)
				self.u = np.dot( np.linalg.inv(self.V), self.u)
				
				#RM update
				a2 = 1/float(self.n)
				self.R = (1-a2)*self.R + a2*R				
				#print "obj: ", self.objective(x, R,self.u,self.V)


			params = util.stack_params(self.u, self.V, self.R)
			diff =  np.linalg.norm(old_params - params)
			old_params = params
				
			ll = self.get_marginal(self.u, self.V, self.R, x_test)
			llh.append(ll)
			
			print "m: ", ll, diff, "iter: ", j
			np.save('DATA50testNr2', np.array(llh))
		

		np.save('DATA50testNr2', np.array(llh))

	



if __name__ == "__main__":

	
	fa = Fa(100,3,2)
	fa.fit(500)
	

	
	
	
	
	
	
	
