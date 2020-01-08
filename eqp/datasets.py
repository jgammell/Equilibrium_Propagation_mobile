import torch
import numpy as np
import gzip
import pickle

class MNIST:
    def __init__(self, batch_size, device, n_train=60000, n_test=10000):
        self.name = 'MNIST'
        self.dimensions = [28**2, 10]
        try:
            path = r'/home/qittlab/Desktop/jimmy/equilibrium-propagation/mnist.pkl.gz'
            f = gzip.open(path, 'rb')
        except:
            path = r'C:\Users\jig\Documents\GitHub\equilibrium-propagation\mnist_data\mnist.pkl.gz'
            f = gzip.open(path, 'rb')
        (x_train,y_train), (x_validate,y_validate), (x_test, y_test) = pickle.load(f, encoding='latin1')
        f.close()
        
        x = list(x_train)+list(x_validate)+list(x_test)
        y = list(y_train)+list(y_validate)+list(y_test)
        for i, yy in zip(range(len(y)),y):
            v = np.zeros((1,10))
            v[0][yy] = 1
            y[i] = v
        x = [torch.from_numpy(xx).squeeze().to(device) for xx in x]
        y = [torch.from_numpy(yy).squeeze().to(device) for yy in y]
        
        self.n_batch_train = int(n_train/batch_size)
        self.n_batch_test = int(n_test/batch_size)
        self.training_batches = []
        self.test_batches = []
        self.training_index = 0
        self.test_index = 0
        
        for batch in range(self.n_batch_train):
            self.training_batches.append(
                     [[torch.stack(x[batch_size*batch:batch_size*(batch+1)],dim=0).float(),
                      torch.stack(y[batch_size*batch:batch_size*(batch+1)],dim=0).float()],
                      batch])
        for batch in range(self.n_batch_train,self.n_batch_train+self.n_batch_test):
            self.test_batches.append(
                    [[torch.stack(x[batch_size*batch:batch_size*(batch+1)],dim=0).float(),
                      torch.stack(y[batch_size*batch:batch_size*(batch+1)],dim=0).float()],
                      batch])
        
    def get_training_batch(self):
        rv = self.training_batches[self.training_index]
        self.training_index = (self.training_index+1)%(self.n_batch_train)
        return rv
    def get_test_batch(self):
        rv = self.test_batches[self.test_index]
        self.test_index = (self.test_index+1)%(self.n_batch_test)
        return rv
    
class CIFAR10:
    def __init__(self, batch_size, device, dtype=torch.float, n_train=50000, n_test=10000):
        self.name = 'CIFAR-10'
        self.dimensions = [3*32**2, 10]
        training_data = []
        test_data = []
        
        path = '/home/qittlab/Desktop/jimmy/equilibrium-propagation/cifar10/'
        training_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        data_batch_size = int(10000/batch_size)
        for filename, data_batch in zip(training_filenames, range(5)):
            with open(path+filename, 'rb') as F:
                data_batch = pickle.load(F, encoding='bytes')
                for training_batch in range(data_batch_size):
                    training_data.append([])
                    training_data[-1].append([])
                    training_data[-1][-1].append(torch.tensor(
                            data_batch[b'data'][training_batch*batch_size:(training_batch+1)*batch_size], device=device, dtype=dtype))
                    target = np.zeros((batch_size, 10))
                    for pair in range(batch_size):
                        target[pair, data_batch[b'labels'][training_batch*batch_size+pair]] = 1
                    training_data[-1][-1].append(torch.tensor(target, device=device, dtype=dtype))
        with open(path+'test_batch', 'rb') as F:
            data_batch = pickle.load(F, encoding='bytes')
            for test_batch in range(data_batch_size):
                test_data.append([])
                test_data[-1].append([])
                test_data[-1][-1].append(torch.tensor(
                        data_batch[b'data'][test_batch*batch_size:(test_batch+1)*batch_size], device=device, dtype=dtype))
                target = np.zeros((batch_size, 10))
                for pair in range(batch_size):
                    target[pair, data_batch[b'labels'][test_batch*batch_size+pair]] = 1
                test_data[-1][-1].append(torch.tensor(target, device=device, dtype=dtype))
        for i in range(len(training_data)):
            training_data[i].append(i)
        for i in range(len(test_data)):
            test_data[i].append(i+len(training_data))
        self.n_train = n_train
        self.n_test = n_test
        self.batch_size = batch_size
        self.training_data = training_data
        self.test_data = test_data
        self.training_index = 0
        self.test_index = 0
    
    def get_training_batch(self):
        rv = self.training_data[self.training_index]
        self.training_index = int((self.training_index+1)%(self.n_train/self.batch_size))
        return rv
    
    def get_test_batch(self):
        rv = self.test_data[self.test_index]
        self.test_index = int((self.test_index+1)%(self.n_test/self.batch_size))
        return rv

        
class Linear:
    def __init__(self, batch_size, device, n_train, n_test, dim_in, dim_out):
        self.name = 'Linear'
        self.dimensions = None
        
        self.T = torch.rand((dim_out,dim_in), dtype=torch.float)/7
        self.n_batch_train = int(n_train/batch_size)
        self.n_batch_test = int(n_test/batch_size)
        self.training_data = []
        for i in range(self.n_batch_train):
            inputs = torch.rand(dim_in, batch_size)
            outputs = torch.matmul(self.T, inputs)
            #outputs += (.1*torch.rand(outputs.shape))-.05
            self.training_data.append([[inputs.transpose(0,1), outputs.transpose(0,1)], i])
        self.test_data = []
        for i in range(self.n_batch_test):
            inputs = torch.rand(dim_in, batch_size)
            outputs = torch.matmul(self.T, inputs)
            self.test_data.append([[inputs.transpose(0,1), outputs.transpose(0,1)], i+self.n_batch_train])
        self.training_index = 0
        self.test_index = 0
    def get_training_batch(self):
        rv = self.training_data[self.training_index]
        self.training_index = (self.training_index+1)%self.n_batch_train
        return rv
    def get_test_batch(self):
        rv = self.test_data[self.test_index]
        self.test_index = (self.test_index+1)%self.n_batch_test
        return rv