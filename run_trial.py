import pickle
import datetime
import os
import sys
from eqp.network import Network

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
	if dataset_label == 'MNIST':
		from eqp.datasets import MNIST
		dataset = MNIST
	elif dataset_label == 'CIFAR-10':
		from eqp.datasets import CIFAR10
		dataset = CIFAR10
	elif dataset_label == 'Linear':
		from eqp.datasets import Linear
		dataset = Linear
	else:
		assert False
	
	num_epochs = configuration['epochs']
	layer_period = configuration['per-layer measurement period']

	network = Network(topology, hyperparameters, configuration, dataset)
	Results = \
	{
		'training error': [],
		'test error':     [],
		'layer rates':    [[] for pair in range(len(topology['layer sizes'])-1)],
		'settings':       Settings
	}

	for epoch in range(num_epochs):
		print('Beginning epoch %d.'%(epoch+1))
		[training_error, layer_rate] = network.train_epoch(measure_classerror=True, measure_perlayer=True, perlayer_batchperiod=layer_period)
		test_error = network.calculate_test_error()
		Results['training error'].append(training_error)
		Results['test error'].append(test_error)
		for pair in range(len(layer_rate)):
			for correction in layer_rate[pair]:
				Results['layer rates'][pair].append(correction)

	dt = datetime.datetime.now()
	results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
	results_name = 'Results_%d-%d-%d_%d-%d-%d.pickle'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
	with open(os.path.join(results_path, results_name), 'wb') as F:
		pickle.dump(Results, F)
