# -*- coding: utf-8 -*-
#%%

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from eqp.network import Network
from eqp.datasets import Linear

np.set_printoptions(precision=2, linewidth=150)
torch.set_printoptions(precision=2, linewidth=150)

# Test layered matrix initializations
topology = {
    'layer sizes':               [5, 4, 3, 2],
    'type':                      'layered'}
hyperparameters = {
    'learning rate':             .01,
    'epsilon':                   .5,
    'beta':                      .9,
    'free iterations':           500,
    'weakly-clamped iterations': 8}
configuration = {
    'batch size':                20,
    'seed':                      0,
    'device':                    'cpu',
    'dtype':                     torch.float,
    'training examples':         50000,
    'test examples':             10000,
    'enable persistent particles': True}

print('Beginning testing on layered network.')
network = Network(topology, hyperparameters, configuration, Linear)
print('\tNetwork successfully initialized.')
print('\tMask matrix:')
print(network.W_mask)
print('\tWeight matrix:')
print(network.W,'\n\n')


# Test smallworld matrix initializations when connections are added
topology = {
    'layer sizes':               [5, 4, 3, 2],
    'type':                      'smallworld add',
    'bypass connections':        10,
    'bypass magnitude':          .05}
hyperparameters = {
    'learning rate':             .01,
    'epsilon':                   .5,
    'beta':                      .9,
    'free iterations':           500,
    'weakly-clamped iterations': 8}
configuration = {
    'batch size':                20,
    'seed':                      0,
    'device':                    'cpu',
    'dtype':                     torch.float,
    'training examples':         50000,
    'test examples':             10000,
    'enable persistent particles': True}

print('Beginning testing on small-world network with 10 added bypass connections.')
network = Network(topology, hyperparameters, configuration, Linear)
print('\tNetwork successfully initialized.')
print('\tMask matrix:')
print(network.W_mask)
print('\tWeight matrix:')
print(network.W,'\n\n')

# Test smallworld matrix initializations when connections are replaced
topology = {
    'layer sizes':               [5, 4, 3, 2],
    'type':                      'smallworld replace',
    'bypass connections':        10,
    'bypass magnitude':          .05}
hyperparameters = {
    'learning rate':             .01,
    'epsilon':                   .5,
    'beta':                      .9,
    'free iterations':           500,
    'weakly-clamped iterations': 8}
configuration = {
    'batch size':                20,
    'seed':                      0,
    'device':                    'cpu',
    'dtype':                     torch.float,
    'training examples':         50000,
    'test examples':             10000,
    'enable persistent particles': True}

print('Beginning testing on small-world network with 10 replaced bypass connections.')
network = Network(topology, hyperparameters, configuration, Linear)
print('\tNetwork successfully initialized.')
print('\tMask matrix:')
print(network.W_mask)
print('\tWeight matrix:')
print(network.W)