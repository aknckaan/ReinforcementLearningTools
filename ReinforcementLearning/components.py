import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F

# Linear Layers

class Linear_layer(nn.Module): ## Linear layer with activation

	def __init__(self, in_size, out_size, **kwargs):

		super(Linear_layer, self).__init__()
		

		if 'activation' in kwargs:
			self.lin = nn.Sequential( nn.Linear(in_size, out_size), getattr(nn, kwargs["activation"])())
		else:
			self.lin = nn.Sequential( nn.Linear(in_size, out_size))

	def forward(self, x):
		return self.lin(x)
		

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


## Convolutional Layers


class Conv_layer(nn.Module): ## conv layer with activation and batch norm

	def __init__(self, input_channels, output_channels, stride, kernel_size, padding,input_shape):
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

	def num2tuple(self,num):
	    return num if isinstance(num, tuple) else (num, num)

class TransConv_layer(nn.Module): ## transpose conv layer with activation and batch norm

	def __init__(self, input_channels, output_channels, kernel, stride, output_padding, padding, input_shape, activation = "relu"):
		super(TransConv_layer, self).__init__()
		self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel ,padding=padding, output_padding=output_padding,  stride=stride)
		nn.init.xavier_normal_(self.convt.weight)
		self.bn= nn.BatchNorm2d(output_channels)
		self.input_shape=input_shape

		if activation == "relu":
		    self.activation = nn.ReLU(inplace=True)
		elif activation == "sigmoid":
		    self.activation = nn.Sigmoid()

		self.output_channels=output_channels
		print(self.input_shape)
		self.out_size=self.convtransp2d_output_shape(tuple(self.input_shape), kernel_size=kernel, stride=stride ,pad=padding, out_pad=output_padding)

	def forward(self, x):
	        out=self.activation(self.bn(self.convt(x)))

	        return out


	def convtransp2d_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
	    h_w, kernel_size, stride, pad, dilation, out_pad = self.num2tuple(h_w), \
	    self.num2tuple(kernel_size), self.num2tuple(stride), self.num2tuple(pad), self.num2tuple(dilation), self.num2tuple(out_pad)
	    pad = self.num2tuple(pad[0]), self.num2tuple(pad[1])
	    
	    h = (h_w[0] - 1)*stride[0] - sum(pad[0]) + dilation[0]*(kernel_size[0]-1) + out_pad[0] + 1
	    w = (h_w[1] - 1)*stride[1] - sum(pad[1]) + dilation[1]*(kernel_size[1]-1) + out_pad[1] + 1
	    
	    return h, w

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

	def num2tuple(self,num):
	    return num if isinstance(num, tuple) else (num, num)