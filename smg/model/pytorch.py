import torch


class TorchHandler:
    """ A class with utilities to consistently manage torch data. """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod  # change to class method?
    def set_device(device):
        """Sets a specific device for torch usage."""
        TorchHandler.device = torch.device(device)
        print(f"Device is set to {TorchHandler.device}.")

    @staticmethod  # change to class method?
    def make_tensor(data, **kwargs):
        """Creates a torch tensor with the given **kwargs and device set."""
        return torch.tensor(data, **kwargs).to(device=TorchHandler.device)

    @staticmethod
    def torch2numpy(data):
        return data.detach().cpu().numpy()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        print(f'The torch RNG is seeded with {seed}')
        print('WARNING: Possibly, you also need to set the seed in numpy.')
