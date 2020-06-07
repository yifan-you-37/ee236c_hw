import numpy as np
import argparse
import os
import datetime
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, loader):
	correct = 0
	total = 0
	for x, y in loader:
		x = Variable(x.view(-1, 28*28))
		pred = policy.predict(x)
		total += y.size(0)
		correct += (pred == y).sum()
	accuracy = 100 * correct/total
	return accuracy


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="GradientDescent")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument('--batch_size', default=100, type=int)          
	parser.add_argument("--eval_freq", default=5e2, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_iters", default=2e3, type=int) 
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--comment", default="")
	parser.add_argument("--exp_name", default="exp_May_31")
	parser.add_argument("--which_cuda", default=0, type=int)

	args = parser.parse_args()

	device = torch.device('cuda:{}'.format(args.which_cuda))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	file_name = "{}_{}".format(args.policy, args.seed)
	file_name += "_{}".format(args.comment) if args.comment != "" else ""
	folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name
	result_folder = 'runs/{}'.format(folder_name) 
	if args.exp_name is not "":
		result_folder = '{}/{}'.format(args.exp_name, folder_name)
	if args.debug: 
		result_folder = 'debug/{}'.format(folder_name)

	print("---------------------------------------")
	print("Policy: {}, Seed: {}".format(args.policy, args.seed))
	print("---------------------------------------")

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	kwargs = {
		'input_dim': 784,
		'output_dim': 10,
		"device": device,
	}

	# Initialize policy
	Algo = __import__(args.policy)
	policy = Algo.Algo(**kwargs)

	# load data
	train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
	test_dataset = 	datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=int(args.batch_size), shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), shuffle=False)

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, test_loader)]

	writer = SummaryWriter(log_dir=result_folder, comment=file_name)

	#record all parameters value
	with open("{}/parameters.txt".format(result_folder), 'w') as file:
		for key, value in vars(args).items():
			file.write("{} = {}\n".format(key, value))

	t = 0
	for i, (x, y) in enumerate(train_loader):
		x = Variable(x.view(-1, 28 * 28))
		y = Variable(y)
		break
	for epoch in range(int(1e6)):
		for i in range(500):
			# x = Variable(x.view(-1, 28 * 28))
			# y = Variable(y)

			policy.train(x, y, writer)

			t+=1
			if t % int(args.eval_freq)==0:
				# calculate Accuracy
				evaluation = eval_policy(policy, test_loader)
				print("Iteration: {}. Accuracy: {}.3".format(t, evaluation))
				evaluations.append(evaluation)
				writer.add_scalar('test/avg_return', evaluation, t)
				np.save("{}/evaluations".format(result_folder), evaluations)

			if t > int(args.max_iters):
				break 
		if t > int(args.max_iters):
			break 
		
		
