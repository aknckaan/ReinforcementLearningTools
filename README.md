# ReinforcementLearningTools
This is a library that I am building during my free time. Currenlty the training algorithm has some bugs and doesn't work. Will be patched soon. 

## Tools
Currently only implemented algorithm is A2C. The library works on openAI gym environments. 
If the input is an image, aoutomaticly an convolution layer added to the network.

Usage:
'''
python3 test.py --env=gym_environment_name --model=model_name --step=1000 --episode=1000 --batch=100 --log=True
'''

## Parameters and Usage

| Parameter | Function | Applicable |
| --- | --- | --- |
| `--env` | name of the target OpenAI Gym environment. | Tested on cart pole, car racing and lunar lander. |
| `--model` | Name of the model to be used to solve the environment. | Only A2C available now. |
| `--step` | Maximum number of steps in the environment in each episode. | Any int. |
| `--episode` | Number of episodes before terminating the environment. | Any int. |
| `--batch` | Number of samples in the training batch. | Any int. |
| `--log` | Save the stats for training proccess. | Boolean |
