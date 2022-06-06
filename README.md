# Surrogate Model Generation (SMG) for Nanoelectronic Devices
Tools to generate surrogate models of nanoelectronic devices such as dopant and nano-particle network devices. The use case treated here is a dopant-network device (DNPU) measured at the University of Twente (NanoElectronics Department).  


## Recommendations for surrogate model (SM) generation of DNPUs
    1. Improve data quality: Randomize phase of input waves after a small number of cycles to have better covering and less correlations in the data. 
    2. Data required for training can be reduced up to 50%. If point 1 is implemented it will reduce the data acquisition time significantly. See [notebook "Exploring Training"](https://github.com/hcruiz/smg/blob/master/notebooks/Exploring%20Training.ipynb). This needs to be studied further to determine the best sampling time.
    3. Besides points 1 & 2, it might be good to spend a bit more time in determining the requirements that the model must satisfy in terms of accuracy and precision of the functionality prediction, see point below. The results will allow adjusting the SM generation to reduce time spend on training. One can fine-tune hyperparameters to obtain "best" models given a certain time-budget, see [notebook "Exploring Training"](smg/notebooks/Exploring Training.ipynb). However, fine-tuning is costly and we don’t know how necessary it is. After the analysis described in point 4, training can be reassessed focusing on the question whether reaching high accuracy is worth the time spend in the model generation or if faster model generation is preferred.  
    4. *Analysis of functionality prediction:* Measure the performance of the off-chip training. What I mean by this is to quantify and analyse the reliability of the functionality prediction. The aim is to understand two main aspects. First, how many false positive and false negatives can be expected with off-chip training. Second, how much accuracy the model must have to obtain a satisfactory performance in the off-chip training. This can be done as follows:
        a) Train a SM as accurate as possible (see point above). 
        b) With a high number of initializations, find “all” solutions (as many distinct solutions as possible) using the SM. Validate the solution in hardware and determine if solution on hardware is found/not found. If NOT FOUND, this is a false positive. 
        c) To estimate the false negatives, find solutions that exist in hardware but are not found in the SM. For this use an extensive search via Evolution-in-Materio or even novelty search. 
        d) Once the error rate is determined, it would be beneficial to know how this error rate increases with the error of the SM’s output current prediction. This is to optimize the training of the SM, as high precision models are more cumbersome to generate because they require much more fine-tuning.
        
## A lightweight model for DNPUs
Use a linear-pyramidal architecture for the DNPU SM if the model size matters. I found a extremely lightweight architecture that reduces the size of the model to around 82% of the original model, which was published [here](https://www.nature.com/articles/s41565-020-00779-y). This lightweight model is not published but **if you want to use it, please cite this repository**.
The model (see [notebook “Lightweight Model”](https://github.com/hcruiz/smg/blob/master/notebooks/Lightweight%20Model.ipynb) is composed of two parts that are multiplied to get the output current prediction, a linear layer that weights the inputs and a pyramidal module that gives the non-linearity. More formally, the network is given by $y(\vec{x})=(\vec{w}\cdot\vec{x})F(\vec{x})$ where $F(\vec{x})$ is a feedforward neural network with four layers of width [70, 50, 30, 15] and ReLU activation, and $\vec{w}$ are parameters of a linear layer mapping $\vec{x}$ to a scalar.

## To Do's:

    1. Implement data acquisition (periodic-wave sampler)
    2. Clean code and merge functions where requuired
    3. Documentation
    4. Change structure for clearer module responsibilities
    5. Add synthetic data generation