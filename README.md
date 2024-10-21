# Proximal Policy Optimization (PPO) with PyTorch

## Overview

This repository provides a clean and modular implementation of Proximal Policy Optimization (PPO) using PyTorch, designed to help beginners understand and experiment with reinforcement learning algorithms. It includes both continuous and discrete action spaces, demonstrated on environments from OpenAI Gym. The structure is flexible, allowing easy modifications to work with other custom environments.


Key Features:

- Modular and easy-to-understand code
- Supports both continuous and discrete action spaces
- YAML-based configuration for managing hyperparameters
- Out-of-the-box compatibility with OpenAI Gym environments


## Getting Started

### Installation

Clone the repository
```bash
git clone https://github.com/saqib1707/PPO-PyTorch.git
cd PPO-PyTorch
```

### Dependencies

To run this code, you need the following dependencies:

- torch
- numpy
- gym
- pygame
- box2d
- box2d-py

Create a virtual environment and install the dependencies:

1. Create a virtual environment
```bash
python -m venv /path/to/venv/directory
```

2. Activate the virtual environment
```bash
source /path/to/venv/directory/bin/activate
```

3. Install required dependencies
```bash
pip install -r requirements.txt
```

**Note:** For environments like `LunarLander-v2` and `BipedalWalker`, make sure you have `swig` and `box2d` installed.

1. Install `swig` in Mac or Linux:

For MacOS:
```bash
brew install swig
```

For Linux:
```bash
apt-get install swig
```

2. To install box2d, run
```bash
pip install box2d
pip install box2d-py
```

**Note:** Gymnasium (the successor to OpenAI Gym) supports Python versions up to 3.11. There have been issues reported with installing gym[box2d] on Python 3.8, 3.9, and 3.10.


## Usage

### Environments

Supported environments include: 

- CartPole
- LunarLander
- Walker2d
- HalfCheetah
- BipedalWalker

Configuration files for each environment are located in the `configs/` directory. These can be customized to adjust hyperparameters for each run.

### Training:

Run the training script with the desired environment configuration:

```bash
python launcher.py --config_path="../configs/config_cartpole.yaml"
```

For other environments, simply modify the config path, for example:

```bash
python launcher.py --config_path="../configs/config_lunarlander.yaml"
```

To run experiments with modified hyperparameters, you can override the default settings from the YAML file using the `--override` flag:

```bash
python launcher.py --config_path="../configs/config_cartpole.yaml" --override "mode=test" "hidden_dim=256" "gamma=0.95"
```

## Results

To be updated soon...


## Contributing

Contributions are welcome! If you have suggestions for improving the code or adding new features, feel free to submit a pull request or open an issue.


## Citing

If you use this repository in your research, please consider citing it:

```bibtex
@misc{ppo_pytorch,
    author = {Azim, Saqib},
    title = {Proximal Policy Optimization using PyTorch},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/saqib1707/PPO-PyTorch}},
}
```

## References:

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners)
- [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
- [PPO Stack Overflow Explanation](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)
- [An Intuitive Explanation of Policy Gradient](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c)
- [ICLR PPO Implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [PPO-implementation-details](https://github.com/vwxyzjn/ppo-implementation-details)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)


## Contact

Feel free to reach out with any questions or suggestions:

Email: [azimsaqib10@gmail.com](mailto:azimsaqib10@gmail.com)