
import torch
from smg.model.pytorch import TorchHandler


def mse(model, data):
    loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():  # is this needed?
        predictions = model(data[:][0].to(TorchHandler.device))
        error = loss(predictions, data[:][1].to(
            TorchHandler.device)).item()
    return error, predictions
