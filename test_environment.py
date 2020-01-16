try:
	import pickle
	import numpy
	import gzip
	import time
	import os
	import torch
	A = torch.tensor([0]).cuda()
	import eqp.network
	import eqp.datasets
	print('Environment setup successful')
except:
	print('Environment setup unsuccessful')
