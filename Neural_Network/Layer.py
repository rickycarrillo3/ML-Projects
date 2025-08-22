from Node import Node
from Functions.weight_init_functions import kaiming_normal


class Layer:
    def __init__(self, in_dim, out_dim, activation=lambda x:x, d_activation=lambda x:1):
      self.nodes = []
      for _ in range(out_dim):
        # Produce random weights and bias for each node in layer
        weights = [kaiming_normal(in_dim, out_dim)  for _ in range(in_dim)] # in_dim total weights for each node
        bias = 0
        self.nodes.append(Node(weights, bias, activation, d_activation)) # out_dim total nodes
      self.outputs = None

    def forward(self, x):
      self.outputs = [node.forward(x) for node in self.nodes]
      return self.outputs

    def back_prop(self, dL_dA):
        """
        Performs the backpropagation calculation
        for each node in the layer.

        dL_dA is the gradient vector of the loss function
        with respect to the output of the layer.

        For example, with three nodes of activations
        a_1, a_2, and a_3, the dL_dA vector would be
        [dL_da_1, dL_da_2, dL_da_3]
        """
        # dL_dx is a 1 by in_dim vector
        dL_dx = [0] * len(self.nodes[0].weights)
        for i in range(len(self.nodes)):
            dL_dx_node_i = self.nodes[i].back_prop(dL_dA[i]) # This dL_dx_node_i has dims 1 by in_dim
            # Must add all contributions to each input entry
            for j in range(len(dL_dx)):
                dL_dx[j] += dL_dx_node_i[j]
        print(f"dL_dx: {dL_dx}")
        return dL_dx

    def learning_step(self, learning_rate):
        """
        Perform the learning step for the entire layer
        """
        for node in self.nodes:
            node.learning_step(learning_rate)

    def display_weights(self):
        for i, node in enumerate(self.nodes):
            print(f"Node {i}: {node}")