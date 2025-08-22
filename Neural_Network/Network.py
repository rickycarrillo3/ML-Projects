import numpy as np

class Network:
    def __init__(self, layers, loss_fn, dloss_fn):
        self.layers = [layer for layer in layers]
        self.loss_fn = loss_fn
        self.dloss_fn = dloss_fn

    def forward(self, x):
        output = x
        for layer in self.layers:
        # Go through each layer and keep feeding
        # the output of the previous layer
            output = layer.forward(output)
        return output

  
    def back_prop(self, dL_dA_out):
        """
        Performs the backpropagation calculation
        for each node in the layer

        dL_dA_out has dims 1 by output layer size
        """
        back_grad = dL_dA_out
        # Must reverse in order to calculate
        # from last layer to first layer
        for layer in reversed(self.layers):
            back_grad = layer.back_prop(back_grad)
        return back_grad


    def learning_step(self, learning_rate):
        for layer in self.layers:
            layer.learning_step(learning_rate)

    def fit_to_data(self, X, Y, epochs=500, learning_rate=0.1, verbose=True):
        for epoch in range(epochs):
            total_loss_per_epoch = 0
            # Shuffle the indices
            shuffle_indices = np.random.permutation(len(X))
            X = X[shuffle_indices]
            Y = Y[shuffle_indices]
            for x, y in zip(X,Y):
                # Compute output of the network
                out = self.forward(x)
                a_out = out[0] # Because outputs are in a list (This assumes we only have ONE output)
                # Update the loss
                loss = self.loss_fn(a_out, y)
                total_loss_per_epoch += loss
                # Obtain the gradient
                dL_dA_out = self.dloss_fn(a_out, y)
                self.back_prop([dL_dA_out])
                # Update the network weights
                self.learning_step(learning_rate)
            if verbose and (epoch % 100 == 0):
                print(f"Epoch {epoch} | Avg Loss: {total_loss_per_epoch /len(X):.4f}")