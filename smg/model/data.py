import os
import numpy as np

from smg.model.pytorch import TorchHandler
from smg.utils.io import load_configs, save_npz


def loadraw(data_directory):
    config_path = os.path.join(data_directory, 'sampler_configs.json')
    configs = load_configs(config_path)
    data_path = os.path.join(data_directory, 'IO.dat')
    data = np.loadtxt(data_path, dtype='float32')
    inputs = data[:, :configs["input_data"]["input_electrodes"]]
    outputs = data[:, -configs["input_data"]["output_electrodes"]:]
    return inputs, outputs, configs


def loadnpy(path, steps=1):
    print('Data loading from: \n' + path)
    with np.load(path, allow_pickle=True) as data:
        info_dictionary = data['info'].tolist()
        print(f'Metadata :\n {info_dictionary.keys()}')
        inputs = data['inputs'][::steps]
        outputs = data['outputs'][::steps]
        print(f'--> Shape of INPUTS: {inputs.shape}')
        print(f'--> Shape of OUTPUTS: {outputs.shape}')
        assert outputs.shape[0] == inputs.shape[0],\
            ValueError('Input/Output data size mismatch!')
    return inputs, outputs, info_dictionary


def load2tensor(path, steps=1):
    inputs, outputs, info = loadnpy(path, steps)
    inputs = TorchHandler.make_tensor(inputs)  # shape: NxIN
    outputs = TorchHandler.make_tensor(outputs)  # Outputs need dim NxOUT
    print(f"Tensors loaded to {TorchHandler.device}")
    return inputs, outputs, info


def process_data(data_directory, clipping_value=[-np.inf, np.inf]):
    '''Process data and cleans clipping by cropping the output given the clipping_values. 
    Arguments:
        - data_directory: string with path to the data directory; assumed are
        sampler_configs.json and a IO.dat file.
        - clipping_value (kwarg): A lower and upper clipping_value to crop data; default is [-np.inf,np.inf]
    NOTES:
        - The data is saved in data_directory to a .npz file with keyes: inputs, outputs and info
        - info is a dictionary with the configs of the sampling procedure.
        - The inputs are on ALL electrodes in Volts and the output in nA.
        - Data does not undergo any transformation, this is left to the user.
        - Outputs and inputs data are arrays of NxD, where N is #samples and D dimension.
    '''
    inputs, outputs, configs = loadraw(data_directory)

    nr_raw_samples = len(outputs)
    print('Number of raw samples: ', nr_raw_samples)

    print(
        f'Output scales: \n Min={np.min(outputs):.2f}, Max = {np.max(outputs):.2f}')
    print('Input scales [Min, Max]:')
    for mn, mx in zip(np.min(inputs, axis=0), np.max(inputs, axis=0)):
        print(f"             [{mn:.2f}, {mx:.2f}]")
    configs['clipping_value'] = clipping_value
    inputs, outputs = crop_data(inputs, outputs, clipping_value)
    print(
        f"% of points cropped: {(1 - len(outputs) / nr_raw_samples) * 100:.2f}")

    save_npz(data_directory, 'processed_data', inputs, outputs, configs)

    return inputs, outputs, configs


def crop_data(inputs, outputs, clipping_value):
    outputs = np.mean(outputs, axis=1)
    # np.mean gives an array (N,)
    # and summarizes the statistics of supersampling an input value
    if type(clipping_value) is list:
        cropping_mask = (
            outputs < clipping_value[1]) * (outputs > clipping_value[0])
    elif type(clipping_value) is float:
        cropping_mask = np.abs(outputs) < clipping_value
    else:
        TypeError(
            f"Clipping value not recognized! Must be list with lower and upper bound or float, was {type(clipping_value)}")

    outputs = outputs[cropping_mask]
    inputs = inputs[cropping_mask, :]
    return inputs, outputs


# %% MAIN
if __name__ == '__main__':
    import os
    print(f"Working in dir {os.getcwd()}")
    from smg.utils.plotter import output_hist
    np.set_printoptions(precision=2)
# %% example of how to process data to generate processed_data.npz
    data_directory = "tmp/example_data/example_3/test"
    # The process_data function should have a
    # clipping value which is dependent on the amplification
    inputs, outputs, info = process_data(
        data_directory, clipping_value=[-150, 143])
    output_hist(outputs, bins=500, show=True)
