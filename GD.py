import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Model

class Algo(object):
	def __init__(
		self,
		input_dim,
		output_dim,
		lr=0.2,
		device='cuda:0'
	):
		self.total_it = 0
		self.log_freq = 100
		self.model = Model(input_dim, output_dim)

	def predict(self, x):
		_, pred = torch.max(self.model(x), 1)
		return pred

	def train(self, x, y, writer=None):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)

		batch_size = x.shape[0]
		# calc gradients
		softmax = torch.nn.Softmax(dim=1)
		outputs = self.model(x)
		logits = softmax(outputs)
		selected_logits = logits.gather(1, y.unsqueeze(1))
		loss = -torch.log(selected_logits).mean()
		for i in range(batch_size):
			logits[i][y[i]] -= 1

		grad = logits.transpose(0, 1).mm(x) / batch_size
		self.model.W = self.model.W - 1e-3 * grad

		if log_it:	
			writer.add_scalar('train/loss', loss, self.total_it)