import numpy as np
import scipy as sp
from numpy.random import RandomState
import pandas as pd

from scipy.stats import norm
from scipy.linalg import det
import gauss as gauss

import seaborn as sns


RS = RandomState(1213)
np.random.seed(42)

class FAnalysis(object):
	
	def __init__(self,n, dimz = 2, dimx = 3):
		
		self.n = n
		self.W = RS.normal(0,1, size = (dimx,dimz))
		self.sigx = 0.000000000001#RS.normal(0,1)
		self.dimz = dimz
		self.dimx = dimx
		
		data = self.generate_data(n)
		self.observed = data[0]
		self.latent = data[1]
		
		self.prec = (1/self.sigx)*np.dot(self.W.transpose(), self.W)
		self.cov = np.linalg.inv(self.prec)
		
		'''
		values for normalisation computation
		'''
		temp1 = (2*np.pi)**(dimz/2.0)*np.sqrt(det(self.cov))
		temp2 = det(2*np.pi*self.sigx*np.identity(dimz))
		self.pc_norm1 = temp1/temp2
		temp3 = np.linalg.inv(np.dot(self.W.transpose(), self.W))
		self.wtwinv = temp3
		temp3 = np.dot(self.W, temp3)
		self.pc_norm2 = np.dot(temp3, self.W.transpose())
		
		
	def generate_data(self,n):
		observed = np.zeros([n,self.dimx])
		latent = np.zeros([n, self.dimz])
	
		for i in xrange(n):
			#latent variable
			z = np.random.normal(0,1, size = (self.dimz,))

			#observed
			mu = np.dot(self.W,z)
			cov = self.sigx*np.identity(3)
			x = np.random.multivariate_normal(mu, cov)
		
			observed[i] = x
			latent[i] = z

		return observed, latent
	
	def get_normalisation_constant(self, x):
		a = 1/(2*self.sigx)
		temp = - a*np.dot(x.transpose(), x) + a*np.dot(np.dot(x.transpose(), self.pc_norm2), x)
		return self.pc_norm1*np.exp(temp)
		
	def get_mu(self, x):
		W = self.W
		temp = np.dot(self.W.transpose(), W)
		temp = np.linalg.inv(temp)
		temp = np.dot(temp, W.transpose())
		
		return np.dot(temp, x)
		
	def EP(self):
		norm = 1
		mus = np.array([])
		norms = []
		for i in xrange(self.n):
			mu = self.get_mu(self.observed[i])
			nc = self.get_normalisation_constant(self.observed[i])
			norm *= nc
			norms.append(nc)
			mus = np.hstack((mus, mu))
		mus = mus.reshape((10,2))
		sig = np.dot(self.W.transpose(), self.W)
		sig = sig/self.sigx
		sig = np.linalg.inv(sig)
		return sig, mus, norms
		
		
		
	
	
	
	
	
	
	
	
	
	
