import numpy as np

### Loss Functions ###
def binary_cross_entropy(a, y):
  return -y * np.log(a) - (1 - y) * np.log(1 - a)

def squared_error(a, y):
    return (a - y) ** 2

### Derivatives of Loss Functions ###
def d_binary_cross_entropy(a, y):
  return (a - y) / (a * (1 - a))

def d_squared_error(a, y):
    return 2 * (a - y)