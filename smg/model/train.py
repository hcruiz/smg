"""
@author: hruiz
"""

import torch
import os

import numpy as np
from collections import deque
from tqdm import trange
import matplotlib.pyplot as plt
from smg.model.pytorch import TorchHandler
from smg.utils.io import savemodel, create_directory, savejson

plt.rcParams['svg.fonttype'] = 'none'


class Trainer():
    '''This is a convenience class intended to train a general neural network.
    Args: 
        network : The network to be trained
        loss : the loss defining the task

    Notes:
        * For training a surrogate model, the outputs must be scaled by the
        amplification. Hence, the output of the model and the errors are  NOT in nA.
        --> To get the errors in nA, scale by the amplification**2.
        * The DNPU surrogate model outputs the prediction in nA using the method dnpu.outputs(inputs).
    '''

    def __init__(self, network, loss):
        self.network = network
        self.loss = loss.to(TorchHandler.device)

    def get_validation(self, data):
        self.network.eval()
        with torch.no_grad():
            for inputs, targets in data:
                predictions = self.network(inputs.to(TorchHandler.device))
                cost = self.loss(predictions, targets.to(
                                 TorchHandler.device)).item()
                self.val_costs.append(cost)

    def set_training(self, configs, optimiser):

        # set optimiser
        self.optim = optimiser
        configs["optimiser"] = optimiser.__module__
        # if 'params_groups' in dir(self.network):
        #     self.optim = optimiser(self.network.params_groups(
        #         configs['params_groups']), **kwargs)
        # else:
        #     self.optim = optimiser(
        #         filter(lambda p: p.requires_grad,
        #                self.network.parameters()), **kwargs)
        # set hyperparams
        self.nr_epochs = configs["nr_epochs"]
        if 'save_interval' in configs.keys():
            self.save_interval = configs['save_interval']
        else:
            self.save_interval = False
        # set data directories
        self.save_dir = create_directory(configs["save_dir"])
        savejson(self.save_dir, "training_configs", configs)

    def train(self, training_data, validation_data=None, verbose=True):
        '''
        Method implementing the training loop given a PyTorch Dataset.
        ---------------
        Arguments
        training_data : training set (inputs,targets) as PyTorch Dataset

        ---------------
        Returns:
        train_costs, val_costs : training and validation costs as deque objects
        '''
        # set data containers
        self.nr_samples = len(training_data.dataset)
        self.nr_trainbatches = len(training_data)
        # *self.nr_trainbatches)
        self.train_costs = deque(maxlen=self.nr_epochs)
        if validation_data:
            self.nr_valbatches = len(validation_data)
            # *self.nr_valbatches)
            self.val_costs = deque(maxlen=self.nr_epochs)
        else:
            self.val_costs = validation_data

        if verbose:
            print('------- TRAINING ---------')
            looper = trange(self.nr_epochs, desc='First epoch')
        else:
            looper = range(self.nr_epochs)
        for epoch in looper:

            self.train_costs.append(self._batch_training(training_data))

            if validation_data:
                self.get_validation(validation_data)

            self._checkpoint(epoch)
            if verbose:
                looper.set_description(
                    f'Train. loss:{self.train_costs[-1]:3.5f} | Val. loss:{self.val_costs[-1]:3.4f}')

        savemodel(self.save_dir, 'final_model', self.network)
        np.savez(os.path.join(self.save_dir, 'training_history'),
                 train_costs=self.train_costs, val_costs=self.val_costs)
        if verbose:
            print('------------DONE-------------')

    def _batch_training(self, data):
        self.network.train()
        batch_costs = deque(maxlen=self.nr_trainbatches)
        for inputs, y_targets in data:

            y_pred = self.network(inputs.to(TorchHandler.device))
            self._test_nan(y_pred)
            loss = self.loss(y_pred, y_targets.to(TorchHandler.device))
            self._test_nan(loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            batch_costs.append(len(y_targets)*TorchHandler.torch2numpy(loss))

        return sum(batch_costs)/self.nr_samples

    def plot_costs(self, show=True):
        plt.figure()
        # plt.suptitle("Loss profile")
        # plt.subplot(211)
        # # np.arange(len(self.train_costs))/self.nr_trainbatches
        # timeline = range(1, self.nr_epochs+1)
        # plt.plot(timeline, self.train_costs)
        # plt.xlabel('Update steps (unit is epoch nr.)')
        # plt.ylabel("Training cost")
        # plt.subplot(212)
        # if self.val_costs:
        #     # timeline = np.arange(1, len(self.val_costs)+1)/self.nr_valbatches
        #     plt.plot(timeline, self.val_costs, '-o')
        # plt.xlabel('Batch (unit is epoch nr.)')
        # plt.ylabel("Validation cost")
        plt.suptitle("Loss profile")
        plt.subplot(111)
        # np.arange(len(self.train_costs))/self.nr_trainbatches
        timeline = range(1, len(self.train_costs)+1)
        plt.plot(timeline, self.train_costs, 'b-o', label="Training")

        if self.val_costs:
            # timeline = np.arange(1, len(self.val_costs)+1)/self.nr_valbatches
            timeline = range(1, len(self.val_costs)+1)
            plt.plot(timeline, self.val_costs, 'r-*', label="Validation")
        plt.xlabel('Epochs')
        plt.ylabel("Loss")
        plt.legend()
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss_profiles"), format='svg')
        if show:
            plt.show()
        else:
            plt.close()

    def _checkpoint(self, epoch):
        if self.save_interval and epoch > 0 and epoch % self.save_interval == 0:
            savemodel(self.save_dir, f'checkpoint_{epoch}', self.network)

    def _test_nan(self, x):
        if torch.isnan(x).any():
            raise ValueError("NaN value(s) found!")


if __name__ == '__main__':
    import torch.nn as nn
    from smg.model.constructor import FCNN
    from smg.dnpu.dataset import dnpuData

    # Load datasets
    configs_data = {"location": "tmp/example_data/example_1/processed_data.npz",
                    "steps": 3,
                    "batch_size": 512}
    dataset = dnpuData(configs_data)
    split = [int(len(dataset)*0.85)+1, int(len(dataset)*0.15)]
    print(f"Splitting data in {split}")
    trainset, valset = torch.utils.data.random_split(dataset, split)
    trainset = torch.utils.data.DataLoader(
        trainset, batch_size=configs_data["batch_size"], shuffle=True)
    valset = torch.utils.data.DataLoader(
        valset, batch_size=split[-1], shuffle=True)

    # Generate model
    MODEL_CONFIG = {
        "hidden_layers": [90, 90, 90],
        "in_features": 7,
        "out_features": 1
    }

    model = FCNN(MODEL_CONFIG)

    # Define configs
    CONFIG = {
        "data": configs_data,
        "model": MODEL_CONFIG,
        "training": {}
    }
    CONFIG["training"]['nr_epochs'] = 3
    CONFIG["training"]['learning_rate'] = 3e-3
    CONFIG["training"]["save_dir"] = 'tmp/DUMP/testing_trainer'

    # Set experiment
    testing = Trainer(model, nn.MSELoss())
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=CONFIG["training"]['learning_rate'])

    # Train model
    testing.set_training(CONFIG["training"], optim)
    testing.train(trainset, validation_data=valset)
    testing.plot_costs()
