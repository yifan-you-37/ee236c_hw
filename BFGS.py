
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Model

softmax = torch.nn.Softmax(dim=1)
def secant(grad_func, x, d):
	eps = 1e-5
	x_1 = 0.2
	x0 = 0.1

	def f_grad(alpha):
		return grad_func(x + alpha * d).transpose(0, 1).mm(d).trace()
	i = 0
	while True:
		i += 1
		if (abs(x_1 - x0) < eps or (f_grad(x0) - f_grad(x_1)) < eps):
			break
		x1 = (f_grad(x0) * x_1 - f_grad(x_1) * x0) / (f_grad(x0) - f_grad(x_1))
		if i == 100:
			break
		x_1 = x0
		x0 = x1
	return x0

def l2_squared(x):
	return torch.sum(x**2)

def update_H(H, delta_x, delta_g):
	gtx = delta_g.transpose().dot(delta_x)
	hgx = H.dot(delta_g.dot(delta_x.transpose()))
	ghg = delta_g.transpose().dot(H.dot(delta_g))
	xxt = delta_x.dot(delta_x.transpose())
	xtg = delta_x.transpose().dot(delta_g)
	H_next = H + (1 + ghg / gtx) * xxt / xtg - (hgx + hgx.transpose()) / gtx
	return H_next


def f_grad(W, x, y):
	batch_size = x.shape[0]
	outputs = x.mm(W.transpose(0, 1))
	logits = softmax(outputs)
	for i in range(batch_size):
		logits[i][y[i]] -= 1

	grad = logits.transpose(0, 1).mm(x) / batch_size
	return grad
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

		self.alpha = lr
		self.H = torch.eye(input_dim * output_dim)
		self.input_dim = input_dim
		self.output_dim = output_dim
	def predict(self, x):
		_, pred = torch.max(self.model(x), 1)
		return pred

	def train(self, x, y, writer=None):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)

		def grad(W):
			return f_grad(W, x, y)
		
		d = -self.H.mm(grad(self.model.W).view(-1, 1))
		if (self.total_it + 1) % 6 == 0:
			d = -grad(self.model.W).view(-1, 1)

		d = d.view(self.output_dim, self.input_dim)
		# alpha = secant(grad, self.model.W, d)
		alpha = 0.2
		W_next = self.model.W + alpha * d
		delta_W = alpha * d
		delta_g = grad(W_next) - grad(self.model.W)
		H_next = update_H(self.H.numpy(), delta_W.view(-1).numpy(), delta_g.view(-1).numpy())
		H_next = torch.tensor(H_next)
		self.model.W = W_next
		self.H = H_next

		# if log_it:	
			# writer.add_scalar('train/loss', loss, self.total_it)