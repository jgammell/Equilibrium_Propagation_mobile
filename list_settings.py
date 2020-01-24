import pickle
import torch
import os
import sys

assert len(sys.argv) >= 3
assert sys.argv[1] == '-f'

for settings_name in sys.argv[2:]:
	settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings')
	with open(os.path.join(settings_path, settings_name), 'rb') as F:
		Settings = pickle.load(F)
	
	topology = Settings['topology']
	hyperparameters = Settings['hyperparameters']
	configuration = Settings['configuration']
	dataset_label = Settings['dataset']

	print('Settings at {}'.format(os.path.join(settings_path, settings_name)))
	print('\tTopology:')
	print('\t\tLayer sizes: {}.'.format(topology['layer sizes']))
	print('\t\tTopology type: %s.'%(topology['type']))
	print('\t\tNumber of bypass connections: %d.'%(topology['bypass connections']))
	print('\t\tMagnitude of added connections: %.03f.'%(topology['bypass magnitude']))
	print('\tHyperparameters:')
	print('\t\tLearning rate: %.03f.'%(hyperparameters['learning rate']))
	print('\t\tEpsilon: %.03f.'%(hyperparameters['epsilon']))
	print('\t\tBeta: %.03f.'%(hyperparameters['beta']))
	print('\t\tFree iterations: %d.'%(hyperparameters['free iterations']))
	print('\t\tWeakly-clamped iterations: %d.'%(hyperparameters['weakly-clamped iterations']))
	print('\tConfiguration:')
	print('\t\tEpochs: %d.'%(configuration['epochs']))
	print('\t\tBatch size: %d.'%(configuration['batch size']))
	print('\t\tSeed: %d.'%(configuration['seed']))
	print('\t\tDevice: %s.'%(configuration['device']))
	print('\t\tData type: %s.'%(configuration['dtype']))
	print('\t\tTraining examples: %d.'%(configuration['training examples']))
	print('\t\tTest examples: %d.'%(configuration['test examples']))
	print('\t\tPersistant particles in use: {}.'.format(configuration['enable persistent particles']))
	print('\tDataset: %s.'%(dataset_label))
	print()
