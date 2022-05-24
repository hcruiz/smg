'''
This class is intended as a convenience to create a surrogate model (SM).
This class handles the testset separately from the validation and training sets.

Notes:
 * The output is usually scaled down with an amplification factor used in the 
 measurements. Thus, training and validation errors shown are usually not in the
 corresponding units of the device output.
 * If the mse function in model.performance is used, the result must be scaled 
 with (amplification)**2 to obtain the correct units.
 * The amplification must be available in the corresponding sampler configs
 * If no amplification is found, it is set to 1. In this case, self.predictions
 is equivalent to the network's output. 
 * Besides the subtelty of the amplification requirement, SMG is for general 
 purpose to train neural networks for regression with MSE loss.  
'''
import os
import torch
from torch.utils.data import DataLoader
from collections import deque
import numpy as np
from tqdm import trange


from smg.model.train import Trainer
from smg.utils.plotter import plot_all
from smg.model.pytorch import TorchHandler
from smg.utils.io import create_directory, savejson


class SMG():
    def __init__(self, configs):
        self.configs = configs
        if "amplification" in configs["data"].keys():
            self.ampl = configs["data"]["amplification"]
        else:
            print("No amplification value found. Set value to 1.")
            self.ampl = 1
        self.save_dir = create_directory(configs["rootdir"])
        savejson(self.save_dir, "smg_configs", configs)

    def set_trainer(self, model):
        self.model = model
        self.trainer = Trainer(self.model, torch.nn.MSELoss())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        self.model.parameters()),
                                 lr=self.configs["training"]['learning_rate'])
        if "save_dir" not in self.configs["training"].keys():
            self.configs["training"]["save_dir"] = self.configs["rootdir"]
        self.trainer.set_training(self.configs["training"], optim)

    def set_dataloader(self, dataset, splits=[0.8, 0.2], verbose=True):
        try:
            split = [int(len(dataset)*splits[0])+1,
                     int(len(dataset)*splits[1])]
            trainset, valset = torch.utils.data.random_split(dataset, split)
        except ValueError:
            if verbose:
                print("Split does not match number of samples. Trying alternative split.")
            split = [int(len(dataset)*splits[0]), int(len(dataset)*splits[1])]
            trainset, valset = torch.utils.data.random_split(dataset, split)

        if verbose:
            print(f"Splitting data in {split}")
        self.trainset = DataLoader(
            trainset, batch_size=self.configs["data"]["batch_size"], shuffle=True)
        self.valset = DataLoader(
            valset, batch_size=split[-1])

    def generate(self, verbose=True, show=False):
        self.trainer.train(
            self.trainset, validation_data=self.valset, verbose=verbose)
        self.viz_performance(self.valset.dataset,
                             save_dir=self.configs["training"]["save_dir"],
                             show=show)

    def predictions(self, inputs):
        return TorchHandler.torch2numpy(self.model(inputs))*self.ampl

    def viz_performance(self, data, save_dir=None, show=True):
        self.trainer.plot_costs(show=show)
        targets = TorchHandler.torch2numpy(data[:][1])*self.ampl
        predictions = self.predictions(data[:][0])
        plot_all(targets, predictions,
                 save_dir=save_dir, show=show)


def explorer(trials, module, dataset, configs):
    '''Performs several training runs for a given model class and configs.
    Args: 
        trials: Number of runs to perform.
        module: Model class to construct the model.
        configs: Configuration dictionary to build the model.

    Returns:
        Dictionary with two (key,value) pairs. 
        'params': Initialization parameters of the model as numpy arrays.
        'costs': Val cost profile per trial; numpy array, size (trials,epochs)

    TODO: Allow for saving all runs in root directory.
    '''

    smg = SMG(configs)
    smg.set_dataloader(dataset)
    costs = np.zeros((trials, configs["training"]["nr_epochs"]))
    params = deque(maxlen=trials)

    print(f"Saving runs in {configs['rootdir']}")
    looper = trange(trials, desc='Running first trial')

    for run in looper:  # range(trials):
        configs["training"]["save_dir"] = os.path.join(
            configs["rootdir"], f"trial_{run}")
        model = module(configs["model"])
        smg.set_trainer(model)
        smg.generate(verbose=False)
        costs[run, :] = smg.trainer.val_costs
        looper.set_description(
            f' Trial: {run} | Val. Error:{costs[run,-1]:3.6f} ')

    return {"costs": costs, "params": params}


if __name__ == '__main__':
    from smg.utils.io import load_configs
    from smg.model.constructor import FCNN
    from smg.dnpu.dataset import dnpuData

    # configs = load_configs("configs/smg_configs_fcnn.json")
    configs = load_configs("tmp/DUMP/testing_smg/smg_configs.json")
    configs["rootdir"] = "tmp/DUMP/testing_dnpugen"

    # A single training run
    model = FCNN(configs["model"])
    smg = SMG(configs)
    dataset = dnpuData(smg.configs["data"])
    smg.set_trainer(model)
    smg.set_dataloader(dataset)
    smg.generate(show=True)

    # # You can make several runs to explore training
    # configs["rootdir"] = "tmp/DUMP/testing_explorer"
    # final_dict = explorer(3, FCNN, dataset, configs)
