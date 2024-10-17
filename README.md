This is an implementation of the PPO RL algorithm in a gym environment helpful for beginners. In this code implementation, I have assumed continuous state and action space. But it can be very easily modified for discrete state and action space setup. 


## Get Started:

In order to run this code, you need to install the following dependencies:
- torch
- numpy
- gym
- tqdm

## To run the following training of PPO RL:

# Use default config
python main.py

# Use specific config
python main.py --config config/experiments/experiment1.yaml

# Override parameters
python main.py --override "training.learning_rate=0.0001" "model.hidden_sizes=[256,256]"


## References:

- [PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners)
- [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
- [PPO Stack Overflow Explanation](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)
- [An Intuitive Explanation of Policy Gradient](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c)




### Contact
If you have any questions about this project or would like to reach out to me in general, please feel free to email me at azimsaqib10@gmail.com