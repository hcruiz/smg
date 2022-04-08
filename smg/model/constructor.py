import torch
import torch.nn as nn
from smg.model.pytorch import TorchHandler


class FCNN(nn.Module):
    """A basic model class to create fully connected neural networks.
    Args:
        configs : dictionary with
                    'in_features' -> input dimension,
                    'hidden_layers' -> a list of size of each hidden layer,
                    'out_features' -> output dimension
        activation : activation function; assumes _get_name() method exist

    """

    def __init__(self, configs, activation=nn.ReLU(), verbose=True):
        super().__init__()
        self.model_info = configs
        self.model_info['activation'] = activation._get_name()
        self._build_model(activation, verbose)

    def _build_model(self, activation_func, verbose):

        hidden_layers = self.model_info['hidden_layers']
        input_layer = nn.Linear(
            self.model_info['in_features'], hidden_layers[0])
        output_layer = nn.Linear(
            hidden_layers[-1], self.model_info['out_features'])
        modules = [input_layer, activation_func]
        for h_1, h_2 in zip(hidden_layers[: -1], hidden_layers[1:]):
            modules.append(nn.Linear(h_1, h_2))
            modules.append(activation_func)

        modules.append(output_layer)
        self.model = nn.Sequential(*modules)

        if TorchHandler.device == torch.device('cuda'):
            self.model.cuda()
            if verbose:
                print("--->> Model sent to CUDA <<---")
        if verbose:
            print('Model built with the following modules: \n', modules)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    configs = {
        "hidden_layers": [
            90,
            90,
            90,
            90,
            90
        ],
        "in_features": 7,
        "out_features": 1
    }

    model = FCNN(configs)
    x = torch.randn((5000, 7), device="cuda", requires_grad=False)
    y = model(x)
    print(y.max(), y.min())
