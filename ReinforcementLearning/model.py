import torch
from torch import nn
from torch import distributions
import numpy as np
import torch.nn.functional as F
from ReinforcementLearning.components import Linear_layer
from ReinforcementLearning.components import Conv_layer
from ReinforcementLearning.components import Variational_layer


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
		critic_loss = 0.5 * advantage.pow(2).mean()
		# critic_loss = 0.5*F.mse_loss( values, Qvals)
		ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

		return ac_loss.mean(), [actor_loss, critic_loss]

class Random_Agent(nn.Module):
	def __init__(self, input_size, output_size, kwargs ):
		super(Random_Agent, self).__init__()
		self.output_size = output_size
		self.distribution_type = kwargs["distribution_type"]

	def forward(self,x):

		if self.distribution_type == "Categorical": 
			action = distributions.Categorical(torch.Tensor(self.output_size)).sample()
		else: 
			action = distributions.Normal(torch.Tensor(self.output_size),torch.Tensor(self.output_size)).sample()

			
		return action


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
