from torch.utils.data import DataLoader
from layers import LinearLayer
from activation_functions import TanH, softmax
from hyperparameters import BATCH_SIZE, INPUT_NODES, OUTPUT_NODES, LEARNING_RATE
from loss_functions import cross_entropy_loss

class Model:
    def __init__(self):
        self.layers = [
            # (batch_size, 1, 784)
            LinearLayer((INPUT_NODES, 100)),  # (batch_size, 1, hidden_nodes)
            TanH(),
            LinearLayer((100, OUTPUT_NODES)),  # (batch_size, 1, 10)
        ]

        self.parameters = [p for layer in self.layers for p in layer.parameters()]

        for parameter in self.parameters:
            parameter.requires_grad = True
            parameter.retain_grad()

    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        return softmax(output)

    def train(self, epochs, train_data):
        print("------- START TRAINING -------\n")
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

        for epoch in range(epochs):
            batch_losses = []

            for batch in train_loader:
                images, labels = batch

                # flatten imgs so that they can be passed to NN
                flattened_imgs = images.flatten(
                    -2
                )  # (batch_size, 1, 28, 28) => (batch_size, 1, 784)

                # get predictions from NN
                preds = flattened_imgs
                for layer in self.layers:
                    preds = layer(preds)

                # get loss for each output node
                loss, dlogits = cross_entropy_loss(preds.squeeze(1), labels)
                dlogits.retain_grad()

                # manual backprop through layers of model
                doutput = dlogits.unsqueeze(1)
                for layer in reversed(self.layers):
                    doutput = layer.backward(doutput)

                # loss.backward()

                # update weights based on gradients
                for layer in self.layers:
                    for parameter in layer.parameters():
                        parameter.data = parameter.data - parameter.grad * LEARNING_RATE
                        parameter.grad.zero_()

                batch_losses.append(loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            print(f"Epoch {epoch + 1} / {epochs}: tr_loss: {epoch_loss}")
