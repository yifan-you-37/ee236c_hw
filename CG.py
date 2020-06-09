import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Model

softmax = torch.nn.Softmax(dim=1)
uncertainty = 0.01
def bisection(f_grad):
	l, h = 0.001, 100
	while(h-l > uncertainty):
		m = (h+l)/2
		if (f_grad(m) > 0):
			h = m
		else:
			l = m
	return m

def l2_squared(x):
	return torch.sum(x**2)

def f_grad(W, x, y, p):
	batch_size = x.shape[0]
	outputs = x.mm(W.transpose(0, 1))
	logits = softmax(outputs)
	for i in range(batch_size):
		logits[i][y[i]] -= 1

	grad = logits.transpose(0, 1).mm(x) / batch_size
	return grad.transpose(0, 1).mm(p).trace()
class Algo(object):
	def __init__(
		self,
		input_dim,
		output_dim,
		lr=0.2,
		device='cuda:0'
	):
		self.total_it = 0
		self.log_freq = 1
		self.model = Model(input_dim, output_dim)

		self.alpha = lr
		self.last_grad = None
		self.last_p = None
	def predict(self, x):
		_, pred = torch.max(self.model(x), 1)
		return pred

	def train(self, x, y, writer=None):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)

		batch_size = x.shape[0]
		# calc gradients
		outputs = self.model(x)
		logits = softmax(outputs)
		selected_logits = logits.gather(1, y.unsqueeze(1))
		loss = -torch.log(selected_logits).mean()
		for i in range(batch_size):
			logits[i][y[i]] -= 1

		grad = logits.transpose(0, 1).mm(x) / batch_size

		if self.total_it == 1:
			p = -grad
		else:
			beta = l2_squared(grad) / l2_squared(self.last_grad)
			p = -grad + beta * self.last_p
		
		# alpha = bisection(lambda alpha : f_grad(self.model.W + alpha * p, x, y, p))
		alpha = 0.01
		# print(alpha)
		self.model.W += alpha * p

		self.last_p = p.clone()
		self.last_grad = grad.clone()
		if log_it:	
			writer.add_scalar('train/loss', loss, self.total_it)
		print(loss)