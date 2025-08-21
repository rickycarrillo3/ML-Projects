import numpy as np
import random
import math

def xavier_normal(in_dim, out_dim):
  return np.random.normal(0, 2 / (in_dim + out_dim))

def xavier_uniform(in_dim, out_dim):
  return random.uniform(-1, 1) * math.sqrt(6 / (in_dim + out_dim))

def kaiming_normal(in_dim, out_dim):
  return np.random.normal(0, math.sqrt(2 / in_dim))

def kaiming_uniform(in_dim, out_dim):
  return random.uniform(-1, 1) * math.sqrt(6 / in_dim)

def uniform(a ,b):
    return random.uniform(a, b)