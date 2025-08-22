import numpy as np
import math

### Activation Functions ###
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return max(0,x)

def tanh(x):
  return math.tanh(x) # [e^(x) - e^(-x)]/ [e^x + e^(-x)]

### Derivatives of Activation Functions ###
def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))

def d_relu(x):
  return 1 if x > 0 else 0

def d_tanh(x):
  return 1 - tanh(x) ** 2
