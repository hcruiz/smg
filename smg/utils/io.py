'''
Library that handles saving and loading data from smg.
'''
import os
import time
import json
import codecs
import pickle
import shutil
import torch


import numpy as np
from smg.model.pytorch import TorchHandler


def save_npz(data_directory, file_name, inputs, outputs, configs):
    save_to = os.path.join(data_directory, file_name)
    print(f'Data saved to {save_to}')
    np.savez(save_to, inputs=inputs, outputs=outputs, info=configs)


# filename, overwrite=False, timestamp=True, **kwargs):
def save(mode, file_path, **kwargs):
   # if timestamp:
   #     path = create_directory_timestamp(path, 'experiment', overwrite=overwrite)
   # else:
   #     path = create_directory(path, overwrite=overwrite)
   # file_path = os.path.join(path, filename)

    if mode == 'numpy':
        np.savez(file_path, **kwargs)
    elif not kwargs['data']:
        raise ValueError(f"Value dictionary is missing in kwargs.")
    else:
        if mode == 'configs':
            save_configs(kwargs['data'], file_path)
        elif mode == 'pickle':
            save_pickle(kwargs['data'], file_path)
        elif mode == 'torch':
            save_torch(kwargs['data'], file_path)
        else:
            raise NotImplementedError(
                f"Mode {mode} is not recognised. Please choose a value between 'numpy', 'torch', 'pickle' and 'configs'.")
    # return path


def save_pickle(pickle_data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(pickle_data, f)
        f.close()


def save_torch(torch_model, file_path):
    """
        Saves the model in given path, all other attributes are saved under
        the 'info' key as a new dictionary.
    """
    torch_model.eval()
    state_dic = torch_model.state_dict()
    state_dic['info'] = torch_model.info
    torch.save(state_dic, file_path)


def savemodel(path, name, model):
    """
    Saves a torch model on eval mode and in cpu.
    """
    model.eval()
    # model.to('cpu')
    state_dic = model.state_dict()
    file_path = os.path.join(path, name + '.pt')
    torch.save(state_dic, file_path)


def load_model(path, ModelClass, *args, device='cpu', **kwargs):
    TorchHandler.device = torch.device(device)
    print(f"Load model to {TorchHandler.device}")
    model = ModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def load_configs(file):
    object_text = codecs.open(file, 'r', encoding='utf-8').read()
    return json.loads(object_text)


def save_configs(configs, file):
    with open(file, 'w') as f:
        for key in configs:
            if type(configs[key]) is np.ndarray:
                configs[key] = configs[key].tolist()
        json.dump(configs, f, indent=4)
        f.close()


def create_directory(path, overwrite=False):
    '''
    This function checks if there exists a directory filepath+datetime_name.
    If not it will create it and return this path.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    return path


def create_directory_timestamp(path, name, overwrite=False):
    datetime = time.strftime("%Y_%m_%d_%H%M%S")
    path = os.path.join(path, name + '_' + datetime)
    return create_directory(path, overwrite=overwrite)


def savejson(path, name, dict):
    file_path = os.path.join(path, name + '.json')
    with open(file_path, "w") as write_to:
        json.dump(dict, write_to, indent=4)
