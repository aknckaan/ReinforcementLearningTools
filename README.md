# ReinforcementLearningTools
This is a Pytorch library that I am building during my free time. (Sample runs will be added soon)

## Tools
Currently only implemented algorithm is A2C. The library works on openAI gym environments. 
If the input is an image, aoutomaticly an convolution layer added to the network.

Currently there are two example functions. To train machine learning algorithm please use the following command:
'''
python3 test.py --env=gym_environment_name --model=model_name --step=1000 --episode=1000 --batch=100 --log=True
'''

For encoders, the available options are variational autoencoder and autoencoder. To train the algorithm please use the following command:
'''
python3 VAE_test.py --env=gym_environment_name --model=model_name --step=1000 --episode=1000 --batch=100 --log=True

'''

## Parameters and Usage

### For machine learning algorithms
| Parameter | Function | Applicable |
| --- | --- | --- |
| `--env` | name of the target OpenAI Gym environment. | Tested on cart pole, car racing and lunar lander. |
| `--model` | Name of the model to be used to solve the environment. | Only A2C available now. |
| `--step` | Maximum number of steps in the environment in each episode. | Any int. |
| `--episode` | Number of episodes before terminating the environment. | Any int. |
| `--batch` | Number of samples in the training batch. | Any int. |
| `--log` | Save the stats for training proccess. | Boolean |
| `--record` | Saves a video per 100 epocs. | Boolean |
### For machine learning algorithms

| Parameter | Function | Applicable |
| --- | --- | --- |
| `--env` | name of the target OpenAI Gym environment. | Tested car racing. |
| `--model` | Name of the model to be used to solve the environment. | AE or VAE. |
| `--step` | Maximum number of steps in the environment in each episode. | Any int. |
| `--episode` | Number of episodes before terminating the environment. | Any int. |
| `--batch` | Number of samples in the training batch. | Any int. |
| `--log` | Save the stats for training proccess. | Boolean |

## Future Work

### Algorithms

Algorithms that will be added in the future

| Algorithms |
| --- |
| DQN |
| PPO |
| A3C |
| Soft A2C |
| Residual RL |

### Functionality

Functionalities that will be added in the future.

| Functionality |
| --- |
| More control over network shape or loss function |
| More variaty of layers. |
| Recoding videos and capturing photos during the training. |
| Paralel environments and data pooling. |

### Other Training Methods

| Training methods |
| --- |
| Genetic Algorithms |
| NEAT |
| Evolutionary Strategies. |
