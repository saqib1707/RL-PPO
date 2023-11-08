import torch
import gym
import argparse
import numpy as np

from ppo import PPO
# from network import FeedForwardNN


def train(env, args):
    print("Training Mode")

    # initialize PPO model
    model = PPO(
        env=env, 
        seed=args.seed, 
        lr=args.lr,
    )

    # if actor and critic model are already saved, load it
    if args.actor_model != '' and args.critic_model != '':
        print(f"Loading actor and critic models from {args.actor_model} and {args.critic_model}")
        model.actor.load_state_dict(torch.load(args.actor_model))
        model.critic.load_state_dict(torch.load(args.critic_model))
        print(f"Successfully Loaded")
    elif args.actor_model != '' or args.critic_model != '':
        print(f"Error: Either specify both actor/critic models or none at all")
        sys.exit(0)
    else:
        print(f"Training from scratch")

    model.learn(
        total_timesteps=args.total_timesteps, 
        timesteps_per_batch=args.timesteps_per_batch, 
        max_episode_length=args.max_episode_length, 
        num_updates_per_itr=args.num_updates_per_itr, 
        clip=args.clip, 
        save_every=args.save_every,
        gamma=args.gamma, 
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', dest='seed', type=int, default=1)
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--env_name', dest='env_name', type=str, default='Pendulum-v1')
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')
    parser.add_argument('--policy_class', dest='policy_class', type=str, default='FeedForwardNN')
    
    parser.add_argument('--timesteps_per_batch', type=int, default=2048)
    parser.add_argument('--num_updates_per_itr', type=int, default=10)
    parser.add_argument('--total_timesteps', type=int, default=int(2e6))
    parser.add_argument('--max_episode_length', type=int, default=200)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    # create a gym environment
    env = gym.make(args.env_name)

    if args.mode == 'train':
        train(env=env, args=args)
    elif args.mode == 'test':
        test(env=env, args=args)


if __name__ == "__main__":
    args = parse_args()
    main(args)