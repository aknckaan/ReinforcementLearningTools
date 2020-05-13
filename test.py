import ReinforcementLearning
import gym
from ReinforcementLearning import RL_Model
import argparse
def main():

	parser = argparse.ArgumentParser(description='RL algorithms test.')

	parser.add_argument('--env',type=str,  default="LunarLander-v2", help="Target environment.")
	parser.add_argument('--episode', type=int, default=1000, help='Maximum number of episodes to run.')
	parser.add_argument('--step', type=int, default=10000, help='Maximum number of steps in one episode. Default is 10000.')
	parser.add_argument('--model', type=str, default="A2C", help='Model to run at the target environment. Default is A2C')
	parser.add_argument('--log', type=bool, default=True, help='To log the training. Default is True')
	parser.add_argument('--batch', type=int, default=200, help='To set the batch size for the training. Default is 200.')
	args = parser.parse_args()

	env = gym.make(args.env)
	# check_env(env, warn=True)

	print(f"Observation space: {env.observation_space}")
	print(f"Action space: {env.action_space}")
	# print(env.action_space.sample())

	# Save a checkpoint every 1000 steps

	# model = SAC('MlpPolicy', env, verbose=2)
	model=RL_Model(env, model_name=args.model)
	model.train(num_episode=args.episode, num_step=args.step, batch_size=args.batch, log_data=args.log)


0
if __name__ == "__main__":
    main()
