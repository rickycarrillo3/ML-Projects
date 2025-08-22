import torch
import torch.nn as nn
import numpy as np
from Network import Network
from Layer import Layer
from Functions.Activations import sigmoid, d_sigmoid, relu, d_relu, tanh, d_tanh
from Functions.Loss_Functions import binary_cross_entropy, d_binary_cross_entropy 

### pyTorch Implementation ###
class XOR_NN(nn.Module):
    def __init__(self):
        """
        Initialize the Neural Network
        """
        super(XOR_NN, self).__init__()
        self.fc1 = nn.Linear(2,3)  # Input layer to hidden layer
        self.hidden1 = nn.Tanh()
        self.fc2 = nn.Linear(3,1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        """
        Forward feed through the network
        """
        x = self.fc1(x)
        x = self.hidden1(x) # Non-linear Activation Function
        x = self.fc2(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    # Create the Torch Datasets
    X_torch = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
    Y_torch = torch.tensor([[1], [1], [0], [0]], dtype=torch.float32)

    # Create Regular Datasets
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    Y = np.array([1, 1, 0, 0])

    # Initialize the Neural Networks
    torch_xor_nn = XOR_NN()
    regular_xor_nn = Network([Layer(2, 3, tanh, d_tanh), Layer(3, 1, sigmoid, d_sigmoid)], 
                             binary_cross_entropy, d_binary_cross_entropy)
    
    # Train both neural networks
    epochs = 500
    learning_rate = 0.1
    # Train the PyTorch model
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(torch_xor_nn.parameters(), lr=learning_rate)
    
    print("PyTorch Implementation\n")

    for epoch in range(epochs):
        for x,y in zip(X_torch, Y_torch):
            # Stochastic Gradient Descent (Picking all samples)
            # Forward feed
            output = torch_xor_nn(x)
            # Compute the loss
            loss = loss_fn(output, y)
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Additional space for readability
    print("\n")

    # Print the PyTorch model's outputs
    for input in X_torch:
        output = torch_xor_nn(input).round()
        print(f"Input: {input.numpy()}| Predicted output: {output.item()}")
    
    print("--------------------------------------")
    print("Our implementation\n")

    # Train our neural network
    regular_xor_nn.fit_to_data(X, Y, epochs, learning_rate)

    # Additional space for readability
    print("\n")
    
    # Print our model's outputs
    for input in X:
        output = regular_xor_nn.forward(input)[0]
        print(f"Input: {input}| Predicted output: {round(output)}")


        
