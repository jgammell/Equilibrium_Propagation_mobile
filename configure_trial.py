import pickle
import os
import torch

settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'settings')
settings_name = 'settings.pickle'

topology = \
{
	'layer sizes':                 [28**2, 500, 500, 500, 10],
	  # Number of neurons in each layer, from input to output.
	'type':                        'smallworld replace',
  	  # Which network topology to use.
	  #   'layered': the topology used in Scellier's paper. Full connectivity
	  #     between adjacent layers, no connectivity within layers and no bypass
	  #     connections. The following two arguments are ignored.
	  #   'smallworld add': full connectivity between adjacent layers, full connectivity
	  #     within layers and random bypass connections made between non-adjacent layers.
	  #   'smallworld replace': base topology created with full connectivity between adjacent
	  #     layers and full connectivity within layers. Existing connections are then replaced
	  #     by random bypass connections.
	'bypass connections':          100000,
	  # How many random bypass connections to create. Ignored if 'type' is 'layered'.
	'bypass magnitude':            .05
	  # Intralayer connections and bypass connections will be drawn from 
	  #   Uniform(-topology['bypass magnitude'],topology['bypass magnitude']). Ignored if type
	  #   is 'layered'.
}
hyperparameters = \
{
	'learning rate':               .01,
	  # Extent to which network will update its weights and biases after each batch.
	  #   Can be a scalar, in which case the entire network will have that learning rate, or a 
	  #   list of scalars of length one less than the number of layers, in which case the parameters
	  #   between layer l and l+1 will have learning rate hyperparameters['learning rate'][l].
	'epsilon':                     .5,
	  # Size of steps in discrete approximation of the network's differential equation of motion.
	'beta':                        1.0,
	  # How strongly network output will be clamped in the weakly-clamped phase.
	'free iterations':             500,
	  # Number of steps with which network will approximate differential equation of motion in free phase.
	'weakly-clamped iterations':   8
	  # Number of steps with which network will approximate differential equation of motion in weakly-clamped phase.
}
configuration = \
{
	'epochs':                      1,
	  # Number of epochs for which to train the network.
	'batch size':                  20,
	  # Number of input-output pairs with which network will train simultaneously during each batch.
	'seed':                        0,
	  # Random seed to initialize numpy and pytorch random number generators.
	'device':                      'cuda:0',
	  # Device on which dataset and network state and weight matrices will be loaded and operated on.
	'dtype':                       torch.float,
	  # Data type with which to represent dataset and network state and weight matrices.
	'training examples':           50000,
	  # Number of training examples to use. Number of training and test examples must be at most the number of examples
	  #   contained in the dataset.
	'test examples':               10000,
	  # Number of test examples to use. Number of training and test examples must be at most the number of examples
	  #   contained in the dataset.
	'enable persistent particles': True,
	  # Whether or not to use persistent particles on both training and test batches.
    'per-layer measurement period': 500
}
dataset = 'MNIST'
# Which dataset to train network on.
#   'MNIST': the MNIST dataset. Requires that the network have 28**2 input neurons and 10 output neurons. 70000 pairs.
#   'CIFAR-10': the CIFAR-10 dataset. Requires that the network have 3*32**2 input neurons and 10 output neurons. 60000 pairs.
#   'Linear': a dataset created by randomly generating a matrix mapping the network's input dimensions to its output dimensions
#     and generating a list of input and output pairs. No restriction on input and output dimensions. Unlimited pairs.

Settings = \
{
	'topology': topology,
	'hyperparameters': hyperparameters,
	'configuration': configuration,
	'dataset': dataset
}

with open(os.path.join(settings_path, settings_name), 'wb') as F:
	pickle.dump(Settings, F)
