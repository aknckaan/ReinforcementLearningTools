import torch
from torch import nn
from torch import distributions
import numpy as np
import torch.nn.functional as F
from ReinforcementLearning.components import Variational_layer
from ReinforcementLearning.components import Conv_layer
from ReinforcementLearning.components import TransConv_layer
from ReinforcementLearning.components import Linear_layer
from enum import Enum

class VAE_Common(Enum):
	strides = [3, 3]
	kernels = [5,5]
	paddings = [3,3]
	color_channels = [8,4]
	output_paddings = [1,1]

	def __str__(self):
	    return str(self.value)


class Encoder(nn.Module):
	def __init__(self, color_channel, input_shape, n_latent_features = None):
		super().__init__()

		strides = VAE_Common.strides.value
		paddings = VAE_Common.paddings.value
		color_channels = VAE_Common.color_channels.value
		kernels = VAE_Common.kernels.value

		self.n_latent_features = n_latent_features

		self.conv1 = Conv_layer(color_channel, color_channels[0], input_shape=input_shape, kernel_size = kernels[0], padding = paddings[0], stride = strides[0])
		self.conv2 = Conv_layer(color_channels[0], color_channels[1], input_shape=self.conv1.output_size, kernel_size = kernels[1], padding = paddings[1], stride = strides[1] )
		self.output_size=self.conv2.output_size

		self.n_neurons_in_middle_layer = int(np.multiply(*self.conv2.output_size)*color_channels[1])
		self.estimated_sizes = [self.n_neurons_in_middle_layer, self.conv1.output_size, self.conv2.output_size]

		if n_latent_features: 
			print(self.n_neurons_in_middle_layer)
			self.l1 = Linear_layer(self.n_neurons_in_middle_layer, n_latent_features)

	def forward(self, x):
		out = self.conv2(self.conv1(x))
		out = out.view(-1, self.n_neurons_in_middle_layer)

		if self.n_latent_features :
			out = self.l1(out)

		return out

class VariationalEncoder(nn.Module):
	def __init__(self, color_channel, input_shape, n_latent_features):
		super().__init__()

		self.encoder = Encoder(color_channel, input_shape)
		self.output_size=self.encoder.output_size

		self.n_neurons_in_middle_layer = int(np.multiply(*self.conv2.output_size)*2)
		self.estimated_sizes = [self.n_neurons_in_middle_layer, self.conv1.output_size, self.conv2.output_size]

		self.mu = Linear_layer(self.n_neurons_in_middle_layer, n_latent_features)
		self.var = Linear_layer(self.n_neurons_in_middle_layer, n_latent_features)


	def forward(self, x):
		out = self.encoder()
		mu = self.mu(out)
		logvar = self.var(out)
		z = self._reparameterize(mu, logvar)

		return z, mu, logvar

	def _reparameterize(self, mu, logvar):
	    sigma=logvar.exp_()
	    eps = torch.randn_like(sigma)
	    std = eps.mul(sigma)
	    z = std.add_(mu)

	    return z

class Decoder(nn.Module):
	def __init__(self, color_channel, decoder_input_size, size_before_flatten):
		super().__init__()

		strides = VAE_Common.strides.value
		paddings = VAE_Common.paddings.value
		output_paddings = VAE_Common.output_paddings.value
		self.color_channels = VAE_Common.color_channels.value
		kernels = VAE_Common.kernels.value

		self.decoder_input_size = decoder_input_size
		self.size_before_flatten = size_before_flatten
		self.lin1 = Linear_layer(decoder_input_size, self.color_channels[-1]*np.prod(size_before_flatten))
		self.convt1 = TransConv_layer(self.color_channels[-1], self.color_channels[-2], input_shape=size_before_flatten, kernel = kernels[-1], padding = paddings[-1], stride = strides[-1], output_padding= output_paddings[-1])
		self.convt2 = TransConv_layer(self.color_channels[-2], color_channel, input_shape=self.convt1.out_size, kernel = kernels[-2], padding = paddings[-2], stride = strides[-2], output_padding = output_paddings[-2] , activation="sigmoid")

	def forward(self, x):

		out = self.lin1(x).view(-1,self.color_channels[-1],*self.size_before_flatten)

		return self.convt2(self.convt1(out))  

class VAE(nn.Module):

	def __init__(self, input_shape, n_latent_features=20, device="cpu"):
		super().__init__()
		
		color_channel=input_shape[0]
		self.input_shape=input_shape[1:]

		self.variational_encoder = self.VariationalEncoder( color_channel, input_shape, n_latent_features = n_latent_features)
		self.decoder = self.Decoder(color_channel, decoder_input_size=n_latent_features, size_before_flatten = self.variational_encoder.output_size)

		self.to(device)

		self.additional_param_size=[1,1,[input_shape[1:],input_shape[0]]]


		self.param_names=["mu","logvar", "reconstructed"]
		self.loss_param_names=["Actor Loss","Critic Loss"]

	def forward(self, x):

		if not isinstance(x, torch.Tensor):
			x = torch.Tensor(x.copy()).unsqueeze(0).float()
			x = x.permute(0,3,1,2)

		z, mu, logvar = self.variational_encoder(x)
		reconstructed = self.decoder(z)
		reconstructed = reconstructed.permute(0,2,3,1)

		return z, [mu, logvar, reconstructed]

	def loss_function_(self, recon_x, x, mu, logsigma):


		BCE = F.mse_loss(recon_x, x, size_average=False)
		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

		KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

		return BCE + KLD

    
	def loss_function(self, x, additional):
        # for i,[recon,x,mu,logvar] in enumerate(zip(self.recon_x,self.x,self.mu,self.logvar)):

		mu = torch.FloatTensor(additional[0]).view(-1, 1)
		logvar = torch.FloatTensor(additional[1]).view(-1, 1)
		reconstructed = torch.FloatTensor(additional[2]).view(-1, 1)

		loss=self.loss_function_(reconstructed, x, mu, logvar)

		return loss.mean(), []

class AE(nn.Module):

	def __init__(self, input_shape, n_latent_features=20, device="cpu"):
		super().__init__()

		color_channel=input_shape[0]
		self.input_shape=input_shape[1:]	
		self.encoder = Encoder( color_channel, self.input_shape, n_latent_features = n_latent_features)
		self.decoder = Decoder(color_channel, decoder_input_size=n_latent_features, size_before_flatten = self.encoder.output_size)

		self.additional_param_size=[[*input_shape[1:],input_shape[0]]]


		self.param_names=["reconstructed"]
		self.loss_param_names=["Reconstruction Loss"]
		self.device = device

	def forward(self, x):
		if not isinstance(x, torch.Tensor):
			x = torch.Tensor(x.copy()).unsqueeze(0).float()
			x = x.permute(0,3,1,2)

		encoded = self.encoder(x)
		reconstructed = self.decoder(encoded)
		reconstructed = reconstructed.permute(0,2,3,1)
		return encoded, [reconstructed.squeeze()]

	def loss_function_(self, recon_x, x):


		BCE = F.mse_loss(recon_x, x)

		return BCE

	def loss_function(self, x, additional):

		reconstructed = additional[0]
		x = torch.FloatTensor(x)
		
		loss=self.loss_function_(reconstructed, x)
		return torch.mean(loss),[]

