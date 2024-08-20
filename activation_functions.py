import torch


class TanH:
    def __call__(self, input):
        self.output = (torch.exp(input) - torch.exp(-input)) / (
            torch.exp(input) + torch.exp(-input)
        )
        return self.output

    def backward(self, doutput):
        # d/dx (tanh(x)) = 1 - tanh²(x)
        dinput = (1 - self.output**2) * doutput
        return dinput

    def parameters(self):
        return []


def softmax(logits):
    # Apply softmax to the predictions
    norm_logits = (
        logits - torch.max(logits, dim=1, keepdim=True)[0]
    )  # subtract max for numerical stability
    soft_preds = norm_logits.exp() / norm_logits.exp().sum(
        1, keepdim=True
    )  # softmax formula

    return soft_preds
