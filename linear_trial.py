from eqp.network import Network
from eqp.datasets import Linear
import numpy as np
import pickle
import datetime
import networkx as nx

def create_graph_from_mask(Mask):
    Mask = Mask.squeeze().clone()
    G = nx.Graph()
    G.add_nodes_from(range(Mask.shape[0]))
    for row in range(Mask.shape[0]):
        for col in range(Mask.shape[1]):
            if(Mask[row, col]) != 0:
                G.add_edge(row,col)
                G.add_edge(col,row)
    return G

def clustering_coefficient(G):
    C = nx.algorithms.cluster.average_clustering(G)
    return C

def characteristic_path_length(G):
    L = nx.algorithms.shortest_paths.average_shortest_path_length(G)
    return L

topology = {
	'layer sizes': 52*[10],
	'type': 'smallworld replace',
	'bypass connections': 0,
	'bypass magnitude': .05}
hyperparameters = {
	'learning rate': .02,
	'epsilon': .5,
	'beta': 1.0,
	'free iterations': 100,
	'weakly-clamped iterations': 4}
configuration = {
	'epochs': 1,
	'batch size': 20,
	'seed': 0,
	'device': 'cuda:0',
	'dtype': torch.float,
	'training examples': 10000,
	'test examples': 0,
	'enable persistent particles': False,
	'per-layer measurement period': 10}
dataset = Linear

num_epochs = 1
layer_period = 10
num_connections = .5*np.sum(l**2 for l in topology['layer sizes'][1:-1]) + np.sum(l1*l2 for l1, l2 in zip(topology['layer sizes'][:-1], topology['layer sizes'][1:]))
replaced_connections = np.linspace(0, num_connections, 500)
num_networks = 20

Results = {
	'cost function': [list() for i in range(num_networks)],
	'layer rates': [list() for i in range(num_networks)],
	'clustering coefficient': [list() for i in range(num_networks)],
	'characteristic path length': [list() for i in range(num_networks)],
	}

for network in range(num_networks):
	for n in replaced_connections:
		topology['bypass connections'] = n
		network = Network(topology, hyperparameters, configuration, dataset)
		for epoch in range(num_epochs):
			[_, layer_rate] = network.train_epoch(measure_classerror=False, measure_perlayer=True, perlayer_batchperiod=layer_period)
			Results['layer rates'][network].append(layer_rate)
			cost = network.calculate_training_cost_function()
			Results['cost function'][network].append(cost)
		G = create_graph_from_mast(network.W_mask)
		L = characteristic_path_length(G)
		C = clustering_coefficient(G)
		Results['characteristic path length'][network].append(L)
		Results['clustering coefficient'][network].append(C)
	hyperparameters['seed'] += 1

dt = datetime.datetime.now()
results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
results_name = r'Linear_sweep_%d-%d-%d_%d-%d-%d.pickle'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
with open(os.path.join(results_path, results_name), 'wb') as F:
	pickle.dump(Results, F)