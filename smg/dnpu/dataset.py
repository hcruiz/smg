import numpy as np
from torch.utils.data import Dataset

from smg.model.data import loadnpy
from smg.model.pytorch import TorchHandler


class dnpuData(Dataset):
    def __init__(self, configs):
        self.path = configs["location"]
        inputs, outputs, self.info = loadnpy(
            self.path, configs["steps"])
        try:
            outputs /= configs['amplification']
        except KeyError:
            _raise = True
            print("----> KeyError occured see information below: \n")
            print("INFO-dictionary has following structure:")
            for key in self.info.keys():
                if isinstance(self.info[key], dict):
                    print(f"In {key}:\n {self.info[key].keys()} \n")
                    if 'amplification' in self.info[key].keys():
                        _raise = False
                        configs['amplification'] = self.info[key]['amplification']
                else:
                    print(f"{key} is {type(self.info[key])}")
            if _raise:
                raise
            else:
                print("Amplification available. Saved in configs.",
                      f"Value: {configs['amplification']}; please check!")

        self.inputs = TorchHandler.make_tensor(inputs)
        if len(outputs.shape) < 2:
            self.outputs = TorchHandler.make_tensor(outputs[:, np.newaxis])
        else:
            self.outputs = TorchHandler.make_tensor(outputs)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


if __name__ == '__main__':

    # %% example of how to load the data for training
    from torch.utils.data import DataLoader
    from smg.utils.plotter import output_hist
    configs = {"location": "tmp/example_data/example_1/processed_data.npz",
               "steps": 3}
    data = dnpuData(configs)
    dataloader = DataLoader(data, batch_size=4,
                            shuffle=True)
    for _, samples in enumerate(dataloader):
        x, y = samples
    outputs = TorchHandler.torch2numpy(data.outputs)
    output_hist(outputs, bins=500, show=True)
