import torch


class LinearLayer:
    def __init__(self, shape):
        # initialize weights with xavier initialization
        self.weights = torch.randn(shape) * torch.sqrt(
            torch.tensor(2 / (shape[0] + shape[1]))
        )
        self.bias = torch.randn(shape[1])

    def __call__(self, input):
        self.input = input
        out = input @ self.weights + self.bias
        return out

    def backward(self, doutput):
        # --- comments are an example for linear layer with 100 input, 10 output nodes & BATCH_SIZE = 32 ---

        # doutput.shape [32, 1, 10]
        # input.shape [32, 1, 100]  => .T [100, 1, 32]
        # weights.shape [100, 10]
        self.weights.grad = self.input.squeeze(1).T @ doutput.squeeze(
            1
        )  # [100, 32] @ [32, 10] = [100, 10]
        self.bias.grad = doutput.sum(0).squeeze(
            0
        )  # doutput.shape [32, 1, 10] => .sum(0).squeeze(0) [10]

        dinput = doutput @ self.weights.T  # [32, 1, 10] @ [10, 100] = [32, 1, 100]
        return dinput

    def parameters(self):
        return [self.weights, self.bias]
