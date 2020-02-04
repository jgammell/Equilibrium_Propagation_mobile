import datetime
import os
from eqp.network import Network
from eqp.datasets import MNIST
import torch

topology = {
	'layer sizes': [28**2, 500, 500, 500, 10],
	'type': 'smallworld replace',
	'bypass connections': 0,
	'bypass magnitude': .05}
hyperparameters = {
	'learning rate': .02,
	'epsilon': .5,
	'beta': 1.0,
	'free iterations': 500,
	'weakly-clamped iterations': 8}
configuration = {
	'epochs': 1,
	'batch size': 20,
	'seed': 0,
	'device': 'cuda:0',
	'dtype': torch.float,
	'training examples': 50000,
	'test examples': 10000,
	'enable persistent particles': True,
	'per-layer measurement period': 500}

num_conns = [n for n in range(0, 200000, 500)]
results = [[], []] # [[number of connections], [training error after 1 epoch]]
for n in num_conns:
	topology['bypass connections'] = n
	num_epochs = configuration['epochs']
	network = Network(topology, hyperparameters, configuration, MNIST)
	[training_error, _] = network.train_epoch(measure_classerror = True, measure_perlayer = False)
	results[0].append(n)
	results[1].append(training_error)
dt = datetime.datetime.now()
results_path = os.path.join(os.path.dirname,(os.path.realpath(__file__)), 'results')
results_name = 'Sweep_%d-%d-%d:%d-%d-%d.pickle'%(dt.month, dt.day, dt.year, dt.hour, dt.minute, dt.second)
with open(os.path.join(results_path, results_name), 'wb') as F:
	pickle.dump(results, F)
