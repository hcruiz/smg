'''
This is an example script showing the explicit usage of the modules in the SMG package.
The intended usage is for general datasets, loss/cost functions, optimisers and models, by modifiying
the script accordingly. This flexibility is intended to aid research in surrgate models for a variety of
nanoelectronic devices.
If the use case is simply to generate an establioshed DNPU surrogate model,
the recommendation is to use the dnpuSMG class and the smg_notebook.
'''
import os
import torch
import matplotlib.pyplot as plt

from smg.utils.io import save_configs, create_directory
from smg.model.pytorch import TorchHandler
from smg.dnpu.dataset import dnpuData
from smg.model.performance import mse
from smg.model.constructor import FCNN
from smg.model.train import Trainer

print(f'CUDA: {torch.cuda.is_available()}')

# Define configs
configs_data = {"location": "tmp/example_data/example_1/processed_data.npz",
                "steps": 3,
                "batch_size": 512}


configs_model = {
    "hidden_layers": [90, 90, 90, 90, 90],
    "in_features": 7,
    "out_features": 1
}


configs_training = {}
configs_training['nr_epochs'] = 3
configs_training['learning_rate'] = 3e-3
configs_training['save_dir'] = 'tmp/DUMP/testing_smg'
configs_training['save_interval'] = False

create_directory(configs_training['save_dir'])

# Load model, trainer  and datasets
model = FCNN(configs_model)
testing = Trainer(model, torch.nn.MSELoss())
optim = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=configs_training['learning_rate'])
dataset = dnpuData(configs_data)
split = [int(len(dataset)*0.70)+1, int(len(dataset)*0.15)+1,
         int(len(dataset)*0.15)]
print(f"Sum of split: {sum(split)}")
print(f"Splitting data in {split}")
trainset, valset, testset = torch.utils.data.random_split(dataset, split)
trainset = torch.utils.data.DataLoader(
    trainset, batch_size=configs_data["batch_size"], shuffle=True)
valset = torch.utils.data.DataLoader(
    valset, batch_size=split[-2])
testing.set_training(configs_training, optim)

save_configs({"data": configs_data,
              "model": configs_model,
              "training": configs_training},
             os.path.join(configs_training['save_dir'], 'smg_configs.json'))
# Train surrogate model
testing.train(trainset, validation_data=valset)
testing.plot_costs()

# Test model with unseen test data
error, predictions = mse(model, testset)
print(f'Test MSE: {error:.5f}')
plt.figure()
plt.plot(TorchHandler.torch2numpy(testset[:][1]),
         TorchHandler.torch2numpy(predictions), '.')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
plt.show()
