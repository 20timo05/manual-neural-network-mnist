# main.py
from model import Model
from torchvision.datasets import MNIST
import torchvision.transforms as T

if __name__ == "__main__":
    # Load your dataset
    training_data = MNIST(
        root="data", train=True, download=True, transform=T.ToTensor()
    )
    test_data = MNIST(root="data", train=False, download=True, transform=T.ToTensor())

    # Create and train the model
    model = Model()
    model.train(epochs=50, train_data=training_data)
