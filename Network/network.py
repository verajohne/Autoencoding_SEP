import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp
from deterministic_layer import Deterministic_Layer, MLP



def construct_mlp(layer_sizes, activation = 'softplus'):
    """
    Construct a deterministic network
    """
    D_layers = []
    L = len(layer_sizes) - 1
    for l in xrange(L):
        D_layers.append(Deterministic_Layer(layer_sizes[l], layer_sizes[l+1], activation))
    
    mlp = MLP(D_layers)
    return mlp        
            
