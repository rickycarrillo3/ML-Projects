import numpy as np

class Node:
    def __init__(self, weights, bias, activation=lambda x:x, d_activation=lambda x:1):
        self.weights = np.array([w for w in weights])
        self.bias = bias
        self.activation = activation
        self.d_activation = d_activation
        # Forward feed parameters
        self.x = None
        self.z = None
        self.a = None
        # Back propagation parameters
        self.dW = None
        self.dB = None

    def forward(self, x):
        """
        Given an input, computes the output of the node.
        """
        # Must save x, z (preactivation), and a(z) (post activation)
        self.x = np.array([x_i for x_i in x])
        self.z = np.dot(self.weights, x) + self.bias
        self.a = self.activation(self.z)
        return self.a

    def back_prop(self, dL_da):
        """
        Given the gradient of the loss function with respect to
        the activation function value a, computes the gradient
        of the loss function with respect to the node's
        weight and bias
        """
        # Notice dL_dz = dL_da * da_dz
        dL_dz = dL_da * self.d_activation(self.z)
        # We need to compute dL_dW and dL_dB
        self.dW = np.array([dL_dz * self.x[i] for i in range(len(self.weights))]) # (dL_dW)_i = dL_dz * x_i
        # Also could do self.dW = np.dot(dL_dz, self.x)
        self.dB = dL_dz
        # Must compute the gradient with respect to the input
        # in order to pass it back to earlier neurons
        dL_dx = np.dot(dL_dz, self.weights) # (dL_dx)_i = dL_dz * w_i
        return dL_dx


    def learning_step(self, learning_rate):
        """
        Perform the learning step of the
        node/neuron
        """
        # print(f"dW: {self.dW}, dB: {self.dB}")
        self.weights = self.weights - learning_rate * self.dW # Can also do [self.weights[i] - learning_rate * self.dW[i] for i in range(len(self.weights))]
        self.bias = self.bias - learning_rate * self.dB
    

    def __str__(self):
      return f"A node/neuron with weights {self.weights} and bias {self.bias}"