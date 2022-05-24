
import os
import time
import torch
import numpy as np
from collections import deque
from tqdm import trange
from smg.generator import SMG
from smg.dnpu.dataset import dnpuData
from smg.model.pytorch import TorchHandler


def mse(model, data):
    loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():  # is this needed?
        predictions = model(data[:][0].to(TorchHandler.device))
        error = loss(predictions, data[:][1].to(
            TorchHandler.device)).item()
    return error, predictions


def datasize_swipe(fractions, module, configs):
    '''Performs several training runs for different sizes of the data partitions
    and a given model class and configs.
    Args: 
        fractions: fractions of original data used
        module: Model class to construct the model.
        configs: Configuration dictionary to generate models.

    Returns:
        Dictionary with (key,value) pairs. 
        'costs': Val cost profile per trial; numpy array, size (trials,epochs)
        'models': list with trained models
        'size_data': size of the data used from training
    '''
    trials = len(fractions)
    smg = SMG(configs)
    dataset = dnpuData(smg.configs["data"], verbose=False)
    pop_size = len(dataset)
    print(f"Total data size: {pop_size}")
    costs = deque(maxlen=trials)
    model_lst = deque(maxlen=trials)
    size_lst = deque(maxlen=trials)
    traintime = []

    print(f"Saving runs in {configs['rootdir']}")
    looper = trange(trials, desc='Running first trial')

    for run in looper:
        cut = int(np.ceil(fractions[run]*pop_size))
        # print(f"Training with {cut} samples")
        indices = np.random.permutation(pop_size)[:cut]
        subset = torch.utils.data.Subset(dataset, indices)
        smg.set_dataloader(subset, verbose=False)
        dsize = len(smg.trainset.dataset)
        nrbatches = len(smg.trainset)
#       Set epochs s.t. nr of updates are similar across different runs
        if run == 0:
            nr_updates = configs["training"]["nr_epochs"]*nrbatches
            # print(f"Total number of updates: {nr_updates}")
        else:
            configs["training"]["nr_epochs"] = int(
                np.ceil(nr_updates/nrbatches))

        configs["training"]["save_dir"] = os.path.join(
            configs["rootdir"], f"trial_{run}")  # this changes self.configs
        model = module(configs["model"], verbose=False)
        smg.set_trainer(model)
        start = time.time()
        smg.generate(verbose=False)
        traintime.append(time.time()-start)
        costs.append(smg.trainer.val_costs)
        model_lst.append(model)
        size_lst.append(dsize)
        looper.set_description(
            f'Val. Err:{costs[-1][-1]:3.4f} | #smpl: {dsize} #epochs: {configs["training"]["nr_epochs"]}')

    return {"costs": costs, "models": model_lst, "size_data": size_lst, "time": traintime}


def batchsize_swipe(bsize, module, configs):
    '''Performs several training runs with with different values of the batch size
    for a given model class and configs.
    Args: 
        bsize: a list containing the size of each batch.
        module: Model class to construct the model.
        configs: Configuration dictionary to build the model.

    Returns:
        Dictionary with (key,value) pairs. 
        'costs': Val cost profile per trial; numpy array, size (trials,epochs)
        'models': list with trained models
        'training_time': Time spent in training each case.

    '''
    trials = len(bsize)
    smg = SMG(configs)
    dataset = dnpuData(smg.configs["data"], verbose=False)
    pop_size = len(dataset)
    print(f"Total data size: {pop_size}")
    costs = deque(maxlen=trials)
    model_lst = deque(maxlen=trials)
    training_time = np.zeros(trials)

    print(f"Saving runs in {configs['rootdir']}")
    looper = trange(trials, desc='Running first trial')

    for run in looper:
        configs["data"]["batch_size"] = bsize[run]  # this changes self.configs
        smg.set_dataloader(dataset, verbose=False)
        nrbatches = len(smg.trainset)
        configs["training"]["save_dir"] = os.path.join(
            configs["rootdir"], "batchsize", f"bs{bsize[run]}")  # this changes self.configs
        model = module(configs["model"], verbose=False)
        smg.set_trainer(model)
        start = time.time()
        smg.generate(verbose=False)
        training_time[run] = time.time() - start
        costs.append(smg.trainer.val_costs)
        model_lst.append(model)
        looper.set_description(
            f"Val. Err.:{costs[-1][-1]:3.4f} | #batches: {nrbatches} #batch size {bsize[run]}")

    return {"costs": costs, "models": model_lst, "training_time": training_time}


def lr_swipe(lr_list, module, configs):
    '''Performs several training runs with different values of the learning rate
    for a given model class and configs.
    Args: 
        lr_list: List with values of learning rate for each run.
        module: Model class to construct the model.
        configs: Configuration dictionary to build the model.

    Returns:
        Dictionary with (key,value) pairs. 
        'costs': Val cost profile per trial; numpy array, size (trials,epochs)
        'models': list with trained models

    '''
    trials = len(lr_list)
    smg = SMG(configs)
    dataset = dnpuData(smg.configs["data"], verbose=False)
    smg.set_dataloader(dataset, verbose=False)

    costs = deque(maxlen=trials)
    model_lst = deque(maxlen=trials)

    print(f"Saving runs in {configs['rootdir']}")
    looper = trange(trials, desc='Running first trial')

    for run in looper:
        configs["training"]["learning_rate"] = lr_list[run]
        configs["training"]["save_dir"] = os.path.join(
            configs["rootdir"], "learning_rate", f"lr_{run}")  # this changes self.configs
        model = module(configs["model"], verbose=False)
        smg.set_trainer(model)
        smg.generate(verbose=False)
        costs.append(smg.trainer.val_costs)
        model_lst.append(model)
        looper.set_description(
            f"Val. Error:{costs[-1][-1]:3.4f} | lr: {configs['training']['learning_rate']}")

    return {"costs": costs, "models": model_lst}


def bslr_swipe(bs_list, lr_list, module, configs, end_epochs=10):
    '''Performs several training runs with different values of the batchsize and
    learning rate for a given model class and configs.
    Args: 
        bs_list: List containing the size of each batch.
        lr_list: List with values of learning rate for each run.
        module: Model class to construct the model.
        configs: Configuration dictionary to build the model.
    Kwargs: 
        end_epochs: the number of epochs at the end of training to estimate the 
        mean val. cost.
    Returns:
        Dictionary with (key,value) pairs. 
        'costs_grid': Mean val. costs of the last window with end_epochs
        'training_time': Time spend in training each case.
        'costs_profiles': Profile of val. costs over epochs for each case; 
        numpy array, size (trials,epochs).

    '''
    lr_trials = len(lr_list)
    bs_trials = len(bs_list)

    cost_profiles = np.zeros(
        (lr_trials, bs_trials, configs["training"]["nr_epochs"]))
    costs_grid = np.zeros((lr_trials, bs_trials))
    training_time = np.zeros((lr_trials, bs_trials))

    smg = SMG(configs)
    dataset = dnpuData(smg.configs["data"], verbose=False)

    print(f"Saving runs in {configs['rootdir']}")
    looper = trange(lr_trials, desc='Running first lr trial')

    for run in looper:
        configs["training"]["learning_rate"] = lr_list[run]

        for bs_indx, bs in enumerate(bs_list):
            configs["data"]["batch_size"] = bs
            smg.set_dataloader(dataset, verbose=False)

            configs["training"]["save_dir"] = os.path.join(
                configs["rootdir"], "lrbs_swipe", f"lr_{lr_list[run]}", f"bs_{bs}")  # this changes self.configs?

            model = module(configs["model"], verbose=False)
            smg.set_trainer(model)
            start = time.time()
            smg.generate(verbose=False)
            training_time[run, bs_indx] = time.time() - start

            cost_profiles[run, bs_indx] = smg.trainer.val_costs
            costs_grid[run, bs_indx] = np.mean(
                cost_profiles[run, bs_indx, -end_epochs:])

        c_str = [f"{c:3.4f}" for c in costs_grid[run]]
        looper.set_description(
            f"Val. Errors:{c_str} | lr: {lr_list[run]}")

    return {"costs_grid": costs_grid, "training_time": training_time, "costs_profiles": cost_profiles}


if __name__ == '__main__':
    from smg.utils.io import load_configs
    from smg.model.constructor import FCNN
    print(os.getcwd())
    configs = load_configs("notebooks/configs_explore_training.json")
    configs["training"]["nr_epochs"] = 3
    configs["data"]["location"] = "tmp/example_data/example_1/processed_data.npz"
    configs["data"]["batch_size"] = 2048
    configs["rootdir"] = "tmp/DUMP/test_performance"

    # fracs = [0.025, 0.01]
    # datasize_swipe(fracs, FCNN, configs)

    batch_sizes = [2048, 1024]
    rates = [0.1, 0.01]

    # batchsize_swipe(batch_sizes, FCNN, configs)
    # lr_swipe(rates, FCNN, configs)
    bslr_swipe(batch_sizes, rates, FCNN, configs)
