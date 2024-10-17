import os
import sys
import argparse
import numpy as np
import gym
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import utils
from config import Config
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                      help='path to config file')
    parser.add_argument('--exp_name', type=str, default=None,
                      help='experiment name')

    # Allow overriding any config parameter from command line
    parser.add_argument('--override', nargs='*', default=[],
                      help='override parameters, format: key=value')

    args = parser.parse_args()
    return args


def main():
    # parse command-line arguments
    args = parse_args()

    # load configuration
    config = Config.load_config(args.config)

    # override with command-line arguments
    overrides = {}
    for override in args.override:
        key, value = override.split('=')
        try:
            value = eval(value)
        except:
            pass
        overrides[key] = value
    config = Config.update_config(config, overrides)

    device = utils.set_device()

    print(config)

    # create environment
    print("Environment name : " + config['env_name'])
    try:
        env = gym.make(config['env_name'])
    except:
        raise ValueError(f"Invalid environment name: {config['env_name']}")

    if config['random_seed']:
        # env.seed(config['random_seed'])    # deprecated, not supported in gym 0.26.2
        utils.set_random_seed(config['random_seed'])

    config['log_dir'] = os.path.join(config['log_dir'], config['env_name'])
    config['ckpt_dir'] = os.path.join(config['ckpt_dir'], config['env_name'])

    # create the logs directory if it doesn't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['ckpt_dir'], exist_ok=True)
    config['logpath'] = os.path.join(config['log_dir'], 'log.txt')
    print(f"Logs saved at: {config['logpath']}")
    with open(config['logpath'], 'w') as f:
        pass

    if config['mode'] == 'train':
        train(env, config, device)
    # elif config.mode == 'test':
    #     test(env, config, device)
    else:
        print("Invalid mode. Mode should be either 'train' or 'test'.")

    env.close()


if __name__ == "__main__":
    main()