#import statements
import numpy as np
import torch
import time
import eqp.datasets

def rho(s):
	return torch.clamp(s, 0, 1)
def rhoprime(s, device):
	rp = torch.zeros(s.shape).to(device)
	rp[(0<=s) & (s<=1)] = 1
	return rp

class Network:
	def __init__(self, topology, hyperparameters, configuration, dataset):
		self.layer_sizes =   topology['layer sizes']
		self.weight_type =   topology['type']
		if self.weight_type != 'layered':
			self.bypass_conns = topology['bypass connections']
			self.bypass_mag =   topology['bypass magnitude']
		self.learning_rate = hyperparameters['learning rate']
		self.eps =           hyperparameters['epsilon']
		self.beta =          hyperparameters['beta']
		self.n_iter =       [hyperparameters['free iterations'],
		                     hyperparameters['weakly-clamped iterations']]
		self.batch_size =    configuration['batch size']
		self.seed =          configuration['seed']
		self.device =        configuration['device']
		self.dtype =         configuration['dtype']
		self.n_train =       configuration['training examples']
		self.n_test =        configuration['test examples']
		self.pparts =        configuration['enable persistent particles']
		
		print('Beginning network initialization.')
		print('Initializing dataset.')
		t_0 = time.time()
		if dataset==eqp.datasets.Linear:
			self.dataset =   dataset(self.batch_size, self.device, self.n_train, self.n_test, self.layer_sizes[0], self.layer_sizes[-1])
		else:
			self.dataset =   dataset(self.batch_size, self.device, self.n_train, self.n_test)
		print('\tUsing %s dataset.'%self.dataset.name)
		print('\tDone. Time taken: %s.'%(self.__format_time(time.time()-t_0)))
		
		assert self.weight_type in ['layered', 'smallworld add', 'smallworld replace']
		assert not(self.n_train%self.batch_size)
		assert not(self.n_test%self.batch_size)
		if type(self.learning_rate) == list:
			assert len(self.learning_rate) == len(self.layer_sizes)-1
		else:
			assert type(self.learning_rate) == float
		if self.dataset.dimensions!=None:
			assert self.layer_sizes[0]==self.dataset.dimensions[0] and self.layer_sizes[-1]==self.dataset.dimensions[1]
			
		print('Hyperparameters:')
		if(type(self.learning_rate) == list):
			message = ''.join(['\tUsing per-layer learning rates: ']+['%f,']*(len(self.learning_rate)-1)+['%f.'])
			print(message%tuple(self.learning_rate))
		else:
			print('\tUsing a single learning rate: %f.'%(self.learning_rate))
		print('\tEpsilon: %f.'%(self.eps))
		print('\tBeta: %f.'%(self.beta))
		print('\tFree phase iterations: %d.'%(self.n_iter[0]))
		print('\tWeakly-clamped phase iterations: %d.'%(self.n_iter[1]))
		print('Network topology:')
		message = ''.join(['\tLayer sizes: ']+['%d-']*(len(self.layer_sizes)-1)+['%d.'])
		print(message%tuple(self.layer_sizes))
		if self.weight_type=='layered':
			print('\tNo intralayer connections.')
			print('\tNo bypass connections.')
		elif self.weight_type=='smallworld add':
			print('\tFully-connected layers.')
			print('\t%d bypass connections have been added.'%(self.bypass_conns))
			print('\tIntralayer and bypass connections drawn from Uniform(-%f, %f).'%(self.bypass_mag, self.bypass_mag))
		else:
			print('\tFully connected layers.')
			print('\t%d connections have been changed into bypass connections.'%(self.bypass_conns))
			print('\tIntralayer and bypass connections drawn from Uniform(-%f, %f).'%(self.bypass_mag, self.bypass_mag))
		print('Configuration settings:')
		print('\tBatch size: %d.'%(self.batch_size))
		print('\tSeed: %d.'%(self.seed))
		print('\tDevice: %s.'%(self.device))
		print('\tData type: %s.'%(self.dtype))
		print('\t%d training examples and %d test examples.'%(self.n_train, self.n_test))
		print('\tTraining examples: %d.'%(self.n_train))
		print('\tTest examples: %d.'%(self.n_test))
		print(''.join(['\tPersistant particles are', (' ' if self.pparts else ' not '), 'in use.']))
			
		self.layer_indices = np.cumsum([0] + self.layer_sizes)
		self.num_neurons = np.sum(self.layer_sizes)
		self.ix = slice(0, self.layer_indices[1])
		self.iy = slice(self.layer_indices[-2], self.layer_indices[-1])
		self.ihy = slice(self.layer_indices[1], self.layer_indices[-1])
		
		if self.seed != None:
			torch.manual_seed(seed=self.seed)
			np.random.seed(seed=self.seed)
		
		self.__initialize_state()
		if self.pparts == True:
			self.__initialize_persistant_particles()
		print('Initializing weight matrix.')
		t_0 = time.time()
		self.__initialize_weight_matrix()
		print('\tDone. Time taken: %s.'%(self.__format_time(time.time()-t_0)))
		self.__initialize_biases()
		
	def __initialize_state(self):
		self.s = torch.zeros(self.batch_size, self.num_neurons).to(self.device)
	
	def __initialize_persistant_particles(self):
		self.persistant_particles = [
		  torch.zeros(self.s[:,self.ihy].shape).to(self.device)
		  for particle in range(int((self.n_train+self.n_test)/self.batch_size))]
	
	def __use_persistant_particle(self, index):
		assert 0<=index<len(self.persistant_particles)
		self.s[:, self.ihy] = self.persistant_particles[index].clone()
	
	def __update_persistant_particle(self, index):
		assert 0<=index<len(self.persistant_particles)
		self.persistant_particles[index] = self.s[:, self.ihy].clone()
	
	def __initialize_weight_matrix(self):
		W = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
		W_mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
		interlayer_connections = []
		for i, j, k in zip(self.layer_indices[:-2], self.layer_indices[1:-1], self.layer_indices[2:]):
			conn = np.zeros(W.shape, dtype=np.bool_)
			conn[j:k, i:j] = True
			interlayer_connections.append(conn)
		for conn in interlayer_connections:
			W_mask[conn] = 1
			
		if self.weight_type=='layered':
			pass
			
		elif self.weight_type=='smallworld add':
			# List out potential locations for bypass connections
			potential_conn_indices = []
			for i in range(2, len(self.layer_indices)-1):
				for row in range(self.layer_indices[i], self.layer_indices[-1]):
					for col in range(self.layer_indices[i-2], self.layer_indices[i-1]):
						potential_conn_indices.append([row, col])
			
			# Create random bypass connections
			assert self.bypass_conns <= len(potential_conn_indices)
			for bypass_conn in range(self.bypass_conns):
				new_location_index = np.random.randint(len(potential_conn_indices))
				new_conn = potential_conn_indices[new_location_index]
				W_mask[new_conn[0], new_conn[1]] = 1
				del potential_conn_indices[new_location_index]
				
			# Make layers fully intraconnected
			for i, j in zip(self.layer_indices[1:-2], self.layer_indices[2:-1]):
				W_mask[i:j, i:j] = 1
			
			# Initialize non-interlayer connection weights
			W = np.random.uniform(low=-self.bypass_mag, high=self.bypass_mag, size=(self.num_neurons, self.num_neurons))
			
		elif self.weight_type=='smallworld replace':
			# List out potential locations for bypass connections
			potential_conn_indices = []
			for i in range(2, len(self.layer_indices)-1):
				for row in range(self.layer_indices[i], self.layer_indices[-1]):
					for col in range(self.layer_indices[i-2], self.layer_indices[i-1]):
						potential_conn_indices.append([row, col])
						
			# Make layers fully intraconnected
			for i, j in zip(self.layer_indices[1:-2], self.layer_indices[2:-1]):
				W_mask[i:j, i:j] = 1
			
			# List out locations of existing connections to be replaced
			existing_conn_indices = []
			for row in range(1, W_mask.shape[0]):
				for col in range(row):
					if W_mask[row, col] != 0:
						existing_conn_indices.append([row, col])
			
			# Create random bypass connections
			assert self.bypass_conns <= len(potential_conn_indices)
			for bypass_conn in range(self.bypass_conns):
				new_location_index = np.random.randint(len(potential_conn_indices))
				new_conn = potential_conn_indices[new_location_index]
				existing_location_index = np.random.randint(len(existing_conn_indices))
				existing_conn = existing_conn_indices[existing_location_index]
				W_mask[new_conn[0], new_conn[1]] = 1
				W_mask[existing_conn[0], existing_conn[1]] = 0
				existing_conn_indices[existing_location_index] = new_conn
				potential_conn_indices[new_location_index] = existing_conn
				
			# Initialize non-interlayer connection weights
			W = np.random.uniform(low=-self.bypass_mag, high=self.bypass_mag, size=(self.num_neurons, self.num_neurons))
				
		#Glorot-Bengio weight initialization
		for conn, n_in, n_out in zip(interlayer_connections, self.layer_sizes[:-1], self.layer_sizes[1:]):
			W[conn] = np.random.uniform(low=-np.sqrt(6./(n_in+n_out)), high=np.sqrt(6./(n_in+n_out)), size=n_in*n_out)
										
		# Zero weight matrix elements where connections do not exist
		W *= W_mask
		
		# Make matrices symmetric
		W = np.tril(W, k=-1) + np.tril(W, k=-1).T
		W_mask = np.tril(W_mask, k=-1) + np.tril(W_mask, k=-1).T
		
		# Convert matrices to pytorch tensors
		self.W = torch.from_numpy(W).float().to(self.device).unsqueeze(0)
		self.W_mask = torch.from_numpy(W_mask).float().to(self.device).unsqueeze(0)
		self.interlayer_connections = [torch.from_numpy(conn).float().to(self.device).unsqueeze(0)
									   for conn in interlayer_connections]
		assert self.W.norm() == (self.W*self.W_mask).norm()
		assert (self.W - (self.W.tril() + self.W.tril().transpose(1,2))).norm() == 0
		assert self.W.norm() != 0
	
	def __initialize_biases(self):
		self.B = torch.zeros(self.s.shape).to(self.device)
	
	def __set_x_state(self, x):
		assert x.shape == self.s[:, self.ix].shape
		self.s[:, self.ix] = x
	
	def __set_y_state(self, y):
		assert y.shape == self.s[:, self.iy].shape
		self.s[:, self.iy] = y
		
	def __step_free(self, y):
		Rs = (rho(self.s)@self.W).squeeze()
		dEds = self.eps*(Rs+self.B-rho(self.s))
		dEds[:, self.ix] = 0
		self.s += dEds
		torch.clamp(self.s, 0, 1, out=self.s)
		
	def __step_weakly_clamped(self, y):
		Rs = (rho(self.s)@self.W).squeeze()
		dEds = self.eps*(Rs+self.B-rho(self.s))
		dEds[:, self.ix] = 0
		self.s += dEds
		dCds = self.eps*self.beta*(y-self.s[:, self.iy])
		self.s[:, self.iy] += 2*dCds
		torch.clamp(self.s, 0, 1, out=self.s)
	
	def __evolve_to_equilibrium(self, phase, y=None):
		if phase=='free':
			iterations = self.n_iter[0]
			step = self.__step_free
		elif phase=='weakly-clamped':
			assert type(y) != type(None)
			iterations = self.n_iter[1]
			step = self.__step_weakly_clamped
		else:
			assert False
		for iteration in range(iterations):
			step(y)
	
	def __calculate_weight_update(self, s_free_phase, s_clamped_phase):
		term1 = torch.unsqueeze(rho(s_clamped_phase), dim=2)@torch.unsqueeze(rho(s_clamped_phase), dim=1)
		term2 = torch.unsqueeze(rho(s_free_phase), dim=2)@torch.unsqueeze(rho(s_free_phase), dim=1)
		dW = (1/self.beta)*(term1-term2)
		dW = torch.mean(dW, dim=0).unsqueeze(0)
		dW *= self.W_mask
		return dW
		
	def __calculate_bias_update(self, s_free_phase, s_clamped_phase):
		dB = (1/self.beta)*(rho(s_clamped_phase)-rho(s_free_phase))
		dB = torch.mean(dB, dim=0).unsqueeze(0)
		return dB
	
	def __train_batch(self, x, y, index=None, measure_classerror=False, measure_perlayer=False):
		n_right = None
		layer_corrections = None
	
		# Evolve network through free and weakly-clamped phases
		if self.pparts:
			assert type(index) == int
			self.__use_persistant_particle(index)
		self.__set_x_state(x)
		self.__evolve_to_equilibrium('free')
		s_free_phase = self.s.clone()
		if self.pparts:
			self.__update_persistant_particle(index)
		if measure_classerror:
			n_right = int(torch.eq(torch.argmax(self.s[:, self.iy], dim=1), torch.argmax(y, dim=1)).sum())
		assert torch.norm(self.s[:, self.ix]-x) == 0
		if np.random.randint(0,2):
			self.beta *= -1
		self.__evolve_to_equilibrium('weakly-clamped', y=y)
		s_clamped_phase = self.s.clone()
		
		# Apply weight update
		dW = self.__calculate_weight_update(s_free_phase, s_clamped_phase)
		if measure_perlayer:
			layer_corrections = []
			for conn in self.interlayer_connections:
				correction = torch.norm((dW*conn)/torch.sqrt(torch.norm(conn, p=1)))
				layer_corrections.append(float(correction.cpu()))
		if type(self.learning_rate) == list:
			for lr, conn in zip(self.learning_rate, self.interlayer_connections):
				dW[conn] *= lr
		else:
			dW *= self.learning_rate
		self.W += dW
		
		# Apply bias update
		dB = self.__calculate_bias_update(s_free_phase, s_clamped_phase)
		dB[:, self.ix] = 0
		if type(self.learning_rate) == list:
			for lr, i, j in zip(self.learning_rate, self.layer_indices[1:-1], self.layer_indices[2:]):
				dB[i:j] *= lr
		else:
			dB *= self.learning_rate
		self.B += dB
		
		return [n_right, layer_corrections]

	def __format_time(self, time):
		if time<1:
			return '%.03fmsec'%(time*1000)
		elif 1<=time<60:
			return '%.03fsec'%(time)
		elif 60<=time<60**2:
			return '%.03fmin'%(time/60)
		elif 60**2<=time:
			return '%.03fhr'%(time/60**2)
		else:
			assert False
		
	def train_epoch(self, measure_classerror=False, measure_perlayer=False, perlayer_batchperiod=0):
		assert perlayer_batchperiod<self.n_train/self.batch_size
		print('Training network for one epoch.')
		t_0 = time.time()
		training_error = 0
		perlayer_corrections = [[] for i in range(len(self.layer_sizes)-1)]
		for batch in range(int(self.n_train/self.batch_size)):
			[x, y], index = self.dataset.get_training_batch()
			[n_right, layer_corrections] = self.__train_batch(x, y, index, measure_classerror, measure_perlayer and not(batch%perlayer_batchperiod))
			if measure_classerror:
				training_error += n_right
			if measure_perlayer and not(batch%perlayer_batchperiod):
				for pair in range(len(layer_corrections)):
					perlayer_corrections[pair].append(layer_corrections[pair])
		assert torch.norm(self.W - (torch.tril(self.W, diagonal=-1) + torch.tril(self.W, diagonal=-1).transpose(1, 2))) == 0
		print('\tDone. Time taken: %s.'%(self.__format_time(time.time()-t_0)))
		if measure_classerror:
			training_error = 1-(training_error/self.n_train)
			print('\tTraining error: %.04f%%.'%(100*training_error))
		return [training_error if measure_classerror else None, perlayer_corrections if measure_perlayer else None]
		
	def calculate_training_error(self):
		print('Evaluating network training error.')
		t_0 = time.time()
		training_error = 0
		for batch in range(int(self.n_train/self.batch_size)):
			[x, y], index = self.dataset.get_training_batch()
			self.__set_x_state(x)
			if self.pparts:
				self.use_persistant_particle(index)
			self.__evolve_to_equilibrium('free')
			training_error += int(torch.eq(torch.argmax(network.s[:, network.iy], dim=1), torch.argmax(y, dim=1)).sum())
		training_error = 1-(training_error/self.n_train)
		print('\tDone. Time taken: %s.'%(self.__format_time(time.time()-t_0)))
		print('\tTraining error: %.04f%%.'%(100*training_error))
		return training_error
		
	def calculate_test_error(self):
		print('Evaluating network test error.')
		t_0 = time.time()
		test_error = 0
		for batch in range(int(self.n_test/self.batch_size)):
			[x, y], index = self.dataset.get_test_batch()
			self.__set_x_state(x)
			if self.pparts:
				self.__use_persistant_particle(index)
			self.__evolve_to_equilibrium('free')
			test_error += int(torch.eq(torch.argmax(self.s[:, self.iy], dim=1), torch.argmax(y, dim=1)).sum())
		test_error = 1-(test_error/self.n_test)
		print('\tDone. Time taken: %s.'%(self.__format_time(time.time()-t_0)))
		print('\tTest error: %.04f%%.'%(100*test_error))
		return test_error
