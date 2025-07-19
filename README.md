This repository contains the code used in the following study: <https://arxiv.org/abs/2506.15626>.


# Radiomics extraction

The file *./radiomics_ext_params.json* contains the parameters used with Pyradiomics to compute 1560 radiomic features from an MR image.


# Machine learning models

Required modules: numpy 1.26.4, scipy 1.14.0, pandas 2.2.2, scikit-learn 1.5.1, nibabel 5.2.1, tensorflow 2.15.0, and declearn 2.6.0.

## Stochastic gradient descent (SGD)

Scripts in *./machine_learning/brainAge_sgd* can be used to train and apply SGD models in both centralized and federated configurations. The variables requiring definition by user are indicated and explained at the top of each script.

It should be noted that the federated version of training is a simulation and is designed for exection on a single machine.

Command example (you must be at the root of the repository): `python -m machine_learning.brainAge_sgd.predict_centralized`


## Deep learning models

The following scripts can be used and adapted to train and apply CNN age predictions:

  - *./machine_learning/brainAge_deep/predict_centralized.py* -> Used to make predictions based on a model trained in a centralized setting. The variables requiring definition by user are indicated and explained at the top of the script.
  - *./machine_learning/brainAge_deep/predict_federated.py* -> Used to make predictions based on a model trained in a federated setting. The variables requiring definition by user are indicated and explained at the top of the script.
  - *./machine_learning/brainAge_deep/training/main_centralized.py* -> Used for centralized training. The variables requiring definition by user are indicated and explained at the top of the script. A GPU is required.
  - *./machine_learning/brainAge_deep/training/main_federated_server.py* -> Used for initializing a server for federated learning. Two command-line inputs are required: the destination folder for model weights and training traces, and the number of clients to wait for before starting. A GPU is required.
  - *./machine_learning/brainAge_deep/training/main_federated_client.py* -> Used to connect a client to the initialized federated server. The variables requiring definition by user are indicated and explained at the top of the script.
  
Command example (you must be at the root of the repository): `python -m machine_learning.brainAge_deep.predict_centralized`
  
