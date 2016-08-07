import numpy as np
import tensorflow as tf

        
class Gaussian_diag(object):

    def __init__(self, size, Mu, Sigma):
        self.size = size
        self.Mu = Mu
        self.Sigma = Sigma
        
    def sample(self, num_samples):
        eps = tf.random_normal([num_samples, self.size],dtype=tf.float64)
        output = self.Mu + eps * self.Sigma         
        return output
    
    def update(self, samples):
        # update the parameters by matching empirical moments
        mean, var = tf.nn.moments(samples, axes=[0])
        self.Mu = mean
        self.Sigma = tf.sqrt(var)
        
        
class Gaussian_full(object):

    def __init__(self, size, Mu, Sigma):
        self.Mu = Mu
        self.Sigma = Sigma
        self.size = size
        
    def sample(self, num_samples):
        eps = tf.random_normal([num_samples, self.size], dtype=tf.float64)
        output = tf.add(self.Mu, tf.transpose(tf.matmul(eps, self.Sigma)))
        output = self.Mu + tf.transpose(tf.matmul(eps, self.Sigma))        
        return output