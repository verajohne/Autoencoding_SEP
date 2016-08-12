#import numpy as np
import autograd.numpy as np
import matrix



        
class Gaussian_full(object):
	def __init__(self, Mu, Sigma):
		self.Mu = Mu
		self.Sigma = Sigma
		self.size = Sigma.shape[0]
	
	def sample(self, num_samples = 1):
		#np.random.seed(1435454)
		eps = np.random.randn(num_samples, self.size)
		try:
			chol = np.linalg.cholesky(self.Sigma)
		except np.linalg.LinAlgError:
			#Find nearest positive semi-definite matrix
			try:
				chol = np.linalg.cholesky(np.array(matrix.nearPD(self.Sigma)))
				
		output = self.Mu + np.dot(eps, chol.T)  
		return output.T
		
	def eval(self,x):
		#x and mu must have same dimensions
		det = np.linalg.det(self.Sigma)**(-0.5)
		const = (2*np.pi)**(-self.size/2.0)
		const = det*const
		prec = np.linalg.inv(self.Sigma)
		t = np.subtract(x, self.Mu)
		v = np.dot(np.transpose(t), prec)
		v = np.exp(-0.5*np.dot(v, t))
		return const*v

	def eval_log(self, x):
		#det = np.linalg.det(self.Sigma)
		#const = (self.size/2.0)*np.log(2*np.pi)
		#const = -0.5*np.log(det) - const
		prec = np.linalg.inv(self.Sigma)
		t = np.subtract(x, self.Mu)
		v = np.dot(np.transpose(t), prec)
		v = -0.5*np.dot(v, t)
		#return const + v
		return v
	
	def eval_log_properly(self, x):
		det = np.linalg.det(self.Sigma)
		const = (self.size/2.0)*np.log(2*np.pi)
		const = -0.5*np.log(det) - const
		prec = np.linalg.inv(self.Sigma)
		t = np.subtract(x, self.Mu)
		v = np.dot(np.transpose(t), prec)
		v = -0.5*np.dot(v, t)
		return const + v
		#return v

'''
Just for test
'''
def eval_log_prec(Mu, prec, x):
	t = np.subtract(x, Mu)
	v = np.dot(np.transpose(t), prec)
	v = -0.5*np.dot(v, t)
	return v
	








	
	
