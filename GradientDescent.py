import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        a = self.l1(x)
        return a

class Algo(object):
	def __init__(
		self,
		input_dim,
		output_dim,
		lr=1e-3,
		device='cuda:0'
	):
		self.total_it = 0
		self.log_freq = 10000
		self.model = Model(input_dim, output_dim)
		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

	def predict(self, x):
		_, pred = torch.max(self.model(x), 1)
		return pred

	def train(self, x, y, writer=None):
		self.total_it += 1

		self.optimizer.zero_grad()
		outputs = self.model(x)
		loss = self.criterion(outputs, y)
		loss.backward()
		self.optimizer.step()

