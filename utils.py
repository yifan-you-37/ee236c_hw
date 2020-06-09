import torch
import numpy as np 

class Model():
	def __init__(self, input_dim, output_dim, init_W=None):
		if init_W is None:
			self.W = torch.rand(output_dim, input_dim) / np.sqrt(input_dim)
		else:
			self.W = init_W.clone()
		self.W = torch.rand(output_dim, input_dim)

	def __call__(self, x):
		a = x.mm(self.W.transpose(0, 1))
		return a