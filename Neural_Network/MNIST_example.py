import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

if __name__ == "__main__":

    # Define the transformation to convert images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset with the specified transformation
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(type(mnist_train))
    # Create a DataLoader to load the dataset in batches
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=False)

    # Data is 28 by 28 pixels

    class MNISTnn(nn.Module):
        def __init__(self):
            super(MNISTnn, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(784, 256), 
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

        def forward(self, x):
            x = x.view(-1, 784) # Flatten the input
            return self.model(x)
    
    # Initialize the neural network
    model = MNISTnn()

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        for ind, (x ,y) in enumerate(train_loader):
            # Forward pass
            output = model.forward(x)
            loss = loss_fn(output, y)

            # Backward pass
            optimizer.zero_grad() # Reset the gradients to 0
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights
            if ind % 1000 == 0:
                print(f"Epoch {epoch + 1} / {epochs}, Step {ind + 1}, Loss: {loss.item():.3f}")
    
    # Testing loop
    accuracy = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model.forward(x)
            pred = torch.argmax(output, dim=1)
            accuracy += (pred == y).sum().item()
    
    print(f"Accuracy on test set: {accuracy / len(mnist_test) * 100:.3f}%")

