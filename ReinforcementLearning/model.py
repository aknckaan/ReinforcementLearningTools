import torch
from torch import nn
from torch import distributions
import numpy as np
import torch.nn.functional as F
class Linear_layer(nn.Module):

	def __init__(self, in_size, out_size, **kwargs):

		super(Linear_layer, self).__init__()
		

		if 'activation' in kwargs:
			self.lin = nn.Sequential( nn.Linear(in_size, out_size), getattr(nn, kwargs["activation"])())
		else:
			self.lin = nn.Sequential( nn.Linear(in_size, out_size))

	def forward(self, x):
		return self.lin(x)
		


class Conv_layer(nn.Module):

	def __init__(self, input_channels, output_channels, stride, kernel_size, padding,input_shape,name=""):
		super(Conv_layer, self).__init__()
		self.conv =nn.Conv2d(input_channels, output_channels, kernel_size= kernel_size, padding=padding, stride=stride)
		nn.init.xavier_normal_(self.conv.weight)
		self.bn= nn.BatchNorm2d(output_channels)	
		self.relu= nn.ReLU()
		self.output_size=self.conv_output_shape(tuple(input_shape), kernel_size = kernel_size, stride = stride, pad = padding )

	def forward(self, x):
		out = self.relu( self.bn(self.conv(x)))
		return out

	def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):

		if type(h_w) is not tuple:
			h_w = (h_w, h_w)

		if type(kernel_size) is not tuple:
			kernel_size = (kernel_size, kernel_size)

		if type(stride) is not tuple:
			stride = (stride, stride)

		if type(pad) is not tuple:
			pad = (pad, pad)

		h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] -1)) - 1)// stride[0] + 1
		w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] -1)) - 1)// stride[1] + 1
		return h,w

class Conv_net(nn.Module):
	def __init__(self, color_channels, kernel_size, stride,input_shape, padding):

		super().__init__()
		self.conv1 = Conv_layer(color_channels, 3, input_shape=input_shape, kernel_size = 7, padding = 1, stride =3 )
		self.conv2 = Conv_layer(3, 2, input_shape=self.conv1.output_size, kernel_size = 7, padding = 1, stride =3 )

		
		self.output_size=self.conv2.output_size

		self.n_neurons_in_middle_layer = int(np.multiply(*self.conv2.output_size)*2)
		self.estimated_sizes = [self.n_neurons_in_middle_layer, self.conv1.output_size, self.conv2.output_size]

	def forward(self, x):
		out = self.conv2(self.conv1(x))

		return out.view(-1, self.n_neurons_in_middle_layer)


class Variational_layer(nn.Module):
	def __init__(self, in_size, out_size, **kwargs):

		super(Variational_layer, self).__init__()

		self.normal = False

		if 'distribution_type' in kwargs:
			self.dist = getattr(distributions, kwargs["distribution_type"])

			if kwargs["distribution_type"] == "Categorical":
				self.mu = Linear_layer(in_size, out_size, activation = "Softmax")

				

			else:
				self.dist = getattr(distributions, "Normal")
				self.mu = Linear_layer(in_size, out_size, activation = "Tanh")

				self.var = Linear_layer(in_size, out_size, activation = "Softplus")

				self.normal = True

		else:
			self.mu = Linear_layer(in_size, out_size, activation = "Tanh")



	def forward(self, x):

		if self.normal:

			x_mu = self.mu(x) 
			x_var = self.var(x)

			fitted_dist=self.dist(x_mu, x_var)

		else:
			x = self.mu(x)	
			fitted_dist=self.dist(x)


		action = fitted_dist.sample()
		log_prob = fitted_dist.log_prob(action)
		entropy = fitted_dist.entropy()

		return action, log_prob, entropy
			

class A2C(nn.Module):
	def __init__(self, input_size, output_size, kwargs):
		super(A2C, self).__init__()

		self.using_conv=False

		if 'activation' in kwargs:
			activation=kwargs["activation"]

		if  len(input_size)>1:
			self.conv = Conv_net(input_size[-1], input_shape = input_size[:-1], kernel_size = 3, stride =2, padding = 1)
			self.using_conv = True
			input_size = self.conv.n_neurons_in_middle_layer
		else:
			input_size=input_size[0]

		self.affine = Linear_layer(input_size, 100, activation =activation)
		self.critic = Linear_layer(100, 1)
		self.actor = Variational_layer(100, output_size, distribution_type = kwargs["distribution_type"])

		self.to(kwargs["device"])
		self.device=kwargs["device"]

		if kwargs["distribution_type"] == "Categorical":

			self.additional_param_size = [1, 1, 1]
		else:
			self.additional_param_size = [1, output_size, output_size]


		self.param_names=["value","log_prob", "entropy"]
		self.loss_param_names=["Actor Loss","Critic Loss"]

	def forward(self, x ):

		if not isinstance(x, torch.Tensor):
			x = torch.tensor(x.copy()).to(self.device).unsqueeze(0).float()

		if self.using_conv:
			x=x.permute(0,3,1,2)
			x = self.conv(x)

		latent = self.affine(x)
		value = self.critic(latent)
		action, log_probs, entropy = self.actor(latent)
		self.extra =  [value, log_probs, entropy]
		return action ,[value, log_probs, entropy]

	def loss_function(self, score, additional):

		GAMMA = 0.99
		Qval = score[-1]
		Qvals = np.zeros_like(score)

		for t in reversed(range(len(score))):
			Qval = score[t] + GAMMA * Qval
			Qvals[t] = Qval

		Qvals = torch.FloatTensor(Qvals).view(-1, 1)
		values = torch.FloatTensor(additional[0]).view(-1, 1)
		log_probs = torch.FloatTensor(additional[1]).view(-1, 1)
		entropy_term = torch.FloatTensor(additional[2]).view(-1, 1)



		advantage = Qvals - values
		actor_loss = (-1* log_probs *  advantage)
		# critic_loss = 0.5 * advantage.pow(2).mean()
		critic_loss = 0.5*F.mse_loss( values, Qvals)
		ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

		return actor_loss.mean(), [actor_loss, critic_loss]

class ANN(nn.Module):
	def __init__(self, input_size, output_size, kwargs ):
		super(ANN, self).__init__()

		if 'activation' in kwargs:
			activation=kwargs["activation"]

		self.lin1 = linear_layer(input_size, output_size, activation =activation)
		self.device=kwargs["device"]

	def forward(self, x):

		if not instance(x,torch.tensor):
			x = torch.tensor(x).to(self.device)

		return self.lin1(x)

	def loss_function(self, score, additional):

		GAMMA = 0.99
		Qval = score[-1]
		Qvals = np.zeros_like(score)
		
		for t in reversed(range(len(score))):
			Qval = score[t] + GAMMA * Qval
			Qvals[t] = Qval

		Qvals = torch.FloatTensor(Qvals).view(-1, 1)
		actions = torch.mean(torch.FloatTensor(additional[0]), axis = 1).view(-1, 1)

		return ac_loss.mean(), []
