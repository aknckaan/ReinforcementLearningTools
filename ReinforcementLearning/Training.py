
import torch

from torch import optim
import importlib
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import random
import ReinforcementLearning
from ReinforcementLearning.model import *
from ReinforcementLearning.Encoders import *
import atexit
from tqdm import tqdm
class AE_Model:
	def __init__(self,env, **kwargs):
			

		self.env = env

		if "device" in kwargs:
			self.device = kwargs["device"]
		else:
			self.device="cpu"
	
		if 'model_name' in kwargs:
			ae_class = getattr(ReinforcementLearning.Encoders, kwargs["model_name"])

		if 'optim' in kwargs:
			self.optim_class = getattr(torch.optim, kwargs["optim"])

		else:
			self.optim = optim.Adam

		if "latent_features" in kwargs:
			self.latent_features = kwargs["latent_features"]
		else:
			self.latent_features = 20

		if isinstance(env.action_space, gym.spaces.Box):
			kwargs["distribution_type"]= "Normal"
			output_size=env.action_space.shape[0]
		else:
			kwargs["distribution_type"]= "Categorical"
			output_size = env.action_space.n

		print(env.action_space.shape)
		print(env.action_space)
		
		input_size=env.observation_space.shape

		if "agent" in kwargs:
			self.agent = kwargs["agent"]
		else:
			
			self.agent = ReinforcementLearning.model.Random_Agent(input_size, output_size, kwargs)

		self.auto_encoder = ae_class([input_size[-1],*input_size[:-1]], self.latent_features, kwargs).to(self.device)

		self.optim = self.optim(self.auto_encoder.parameters())

		print("Autoencoder training using {} as the algorithm; \nin/latent:{} / {}".format(self.agent.__class__.__name__, input_size, self.latent_features ))

	def train(self,**kwargs):

		if not "record_video" in kwargs:
			kwargs["record_video"] = False

		if not "capture_image" in kwargs:
			kwargs["capture_image"] = False

		if not "log_data" in kwargs:
			kwargs["log_data"] = False

		if not "batch_size" in kwargs:
			kwargs["batch_size"] = 100

		if kwargs["log_data"]: 

			writer = SummaryWriter(comment = "{} Model on {}".format(self.agent.__class__.__name__, self.env.unwrapped.spec.id))

			def clean():
				writer.flush()
				writer.close()

			atexit.register(clean)
		if 'num_episode' in kwargs:
			max_episode = kwargs['num_episode']
		else:
			max_episode=1000

		if 'num_step' in kwargs:
			max_step = kwargs['num_step']

		else:
			max_step=1000

		done = True
		cur_episode = 0
		t = tqdm(total=max_episode) 
		while cur_episode < max_episode:
			
			if done :
				t.update(1)
				self.optim.zero_grad()
				cur_episode+=1
				obs = self.env.reset()
				done = False
				
			
			observation_arr = [obs]
			total_loss = []
			additional_arr = [ torch.zeros(1,*i, requires_grad = False ) for i in self.auto_encoder.additional_param_size]
			score_arr = [0]
			action_arr = []
			score=0
			for step in range(max_step):

				z, additional = self.auto_encoder(obs)


				action = self.agent(obs)
				# additional = self.auto_encoder.extra
				additional_arr = [torch.cat((elem1,elem2.unsqueeze(0)), dim=0) for elem1, elem2 in zip(additional_arr, additional)]

				obs, reward, done, infos = self.env.step(action.clone().detach().to(self.device).squeeze().numpy())
			
				score += reward

				score_arr.append(reward)
				action_arr.append(action)
				observation_arr.append(obs)

				if done or step > kwargs["batch_size"]:
					break

			batch_addition = []

			for addition in additional_arr:
				
				batch_addition.append(addition[ -(kwargs["batch_size"]-1): ])

			batch_score = score_arr[ -(kwargs["batch_size"]-1): ]
			batch_observation = observation_arr[ -(kwargs["batch_size"]-1): ]

			
			
			loss, loss_extra = self.auto_encoder.loss_function( batch_observation, batch_addition)
			loss.backward()
			total_loss.append(loss.item())
			self.optim.step()

			if done:
				if kwargs["log_data"] : 

					writer.add_scalar('Reward', score, cur_episode)
					
					writer.add_scalar('Episode length', len(score_arr), cur_episode)


					if len(additional_arr)>0 :

						additional_params = {}
						for i, params in enumerate(additional_arr):
							additional_params[self.auto_encoder.param_names[i]] = torch.mean(params).item()

						writer.add_scalars("Policy_params", additional_params, cur_episode)

					if len(loss_extra)>0 :
						additional_params = {}
						for i, params in enumerate(loss_extra):
							additional_params[self.auto_encoder.loss_param_names[i]] = torch.mean(params).item()

							

					writer.add_scalar('Policy loss', np.mean(total_loss), cur_episode)
				
				if cur_episode %10==0:
					torch.save(self.auto_encoder.state_dict(), "saved_models/{}_AE_on_{}_epoc{}.pt".format(self.auto_encoder.__class__.__name__, self.env.unwrapped.spec.id, cur_episode%10))

				if cur_episode == 1: 
					self.auto_encoder.eval() 
			
					obs =torch.tensor(obs.copy()).to(self.device).unsqueeze(0).float()
					obs = obs.permute(0,3,1,2)
					writer.add_graph(self.auto_encoder, obs)


		t.close()

class RL_Model:
	def __init__(self, env, **kwargs):

		self.env = env
	
		if not 'device' in kwargs:
			kwargs["device"] = "cpu"

		self.device = kwargs["device"]

		if not 'activation' in kwargs:
			kwargs["activation"] = "ReLU"

		if 'model_name' in kwargs:
			agent_class = getattr(ReinforcementLearning.model, kwargs["model_name"])

		if 'optim' in kwargs:
			self.optim_class = getattr(torch.optim, kwargs["optim"])

		else:
			self.optim = optim.Adam

		print(env.action_space.shape)
		print(env.action_space)
		
		input_size=env.observation_space.shape

		if isinstance(env.action_space, gym.spaces.Box):
			kwargs["distribution_type"]= "Normal"
			output_size=env.action_space.shape[0]
		else:
			kwargs["distribution_type"]= "Categorical"
			output_size = env.action_space.n

		if len(input_size)>1:
			print("Input is a multi-dimentional vector. Adding a convolution layer.")


		self.agent = agent_class(input_size, output_size, kwargs)
		self.optim = self.optim(self.agent.parameters())

		print("Residual rl using {} as the rl algorithm; \nin/out:{} / {} \nactivations: {} \nusing {} distribution.".format(self.agent.__class__.__name__, input_size, output_size, kwargs["activation"], kwargs["distribution_type"] ))

	def train(self,**kwargs):

		if not "record_video" in kwargs:
			kwargs["record_video"] = False

		if not "capture_image" in kwargs:
			kwargs["capture_image"] = False

		if not "log_data" in kwargs:
			kwargs["log_data"] = False

		if not "batch_size" in kwargs:
			kwargs["batch_size"] = 100

		if kwargs["log_data"]: 

			writer = SummaryWriter(comment = "{} Model on {}".format(self.agent.__class__.__name__, self.env.unwrapped.spec.id))

			def clean():
				writer.flush()
				writer.close()

			atexit.register(clean)
		if 'num_episode' in kwargs:
			max_episode = kwargs['num_episode']
		else:
			max_episode=1000

		if 'num_step' in kwargs:
			max_step = kwargs['num_step']

		else:
			max_step=1000

		done = True
		cur_episode = 0
		self.optim.zero_grad()
		t = tqdm(total=max_episode) 
		while cur_episode < max_episode:

			if done :
				t.update(1)
				self.optim.zero_grad()
				cur_episode+=1
				obs = self.env.reset()
				done = False

			observation_arr = []
			observation_arr.append(obs)
			total_loss = []
			additional_arr = [ torch.zeros(1,i, requires_grad = False ) for i in self.agent.additional_param_size]
			score_arr = [0]
			action_arr = []
			score=0
		
			

			for step in range(max_step):

				action,additional = self.agent(obs)
				# additional = self.agent.extra
				additional_arr = [torch.cat((elem1,elem2.view(1,-1)), dim=0) for elem1, elem2 in zip(additional_arr, additional)]

				obs, reward, done, infos = self.env.step(action.clone().detach().to(self.device).squeeze().numpy())
				score += reward

				score_arr.append(reward)
				action_arr.append(action)
				observation_arr.append(obs)

				if done or step > kwargs["batch_size"]:
					break
		
			batch_addition = []
		

			for addition in additional_arr:
				
				batch_addition.append(addition[ -kwargs["batch_size"]: ])

			batch_score = score_arr[ -kwargs["batch_size"]: ]

			loss, loss_extra = self.agent.loss_function( batch_score, batch_addition)

			
			loss.backward()
			self.optim.step()
			self.optim.zero_grad()
			total_loss.append(loss.item())

			if done:
				if kwargs["log_data"] : 
					writer.add_scalar('Reward', score, cur_episode)
					
					writer.add_scalar('Episode length', len(score_arr), cur_episode)


					if len(additional_arr[0].size())>0 :

						additional_params = {}
						for i, params in enumerate(additional_arr):
							additional_params[self.agent.param_names[i]] = torch.mean(params).item()

						writer.add_scalars("Policy_params", additional_params, cur_episode)

					if len(loss_extra)>0 :
						additional_params = {}
						for i, params in enumerate(loss_extra):
							additional_params[self.agent.loss_param_names[i]] = torch.mean(params).item()

						writer.add_scalars("Policy loss", additional_params, cur_episode)
						writer.add_scalar('Policy loss', np.mean(total_loss), cur_episode)
				
				if cur_episode %10==0:
					torch.save(self.agent.state_dict(), "saved_models/{}_agent_on_{}_epoc{}.pt".format(self.agent.__class__.__name__, self.env.unwrapped.spec.id, cur_episode%10))

				if cur_episode == 1: 
					self.agent.eval() 
			
					obs =torch.tensor(obs.copy()).to(self.device).unsqueeze(0).float()
					writer.add_graph(self.agent, obs)
