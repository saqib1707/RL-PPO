import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import gym

from network import FeedForwardNN


class PPO:
    def __init__(self, env, seed=1, lr=1e-3):
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        self._set_seed_everywhere(seed)
        
        # extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.action_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.action_cov_mat = torch.diag(torch.full(size=(self.action_dim,), fill_value=0.5))

        # initialize logger
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rewards': [],
            'actor_losses': [],
        }

    def learn(self, total_timesteps, timesteps_per_batch, max_episode_length, num_updates_per_itr, clip=0.2, save_every=1000, gamma=0.9):
        t_so_far = 0    # timesteps simulated so far
        i_so_far = 0

        while t_so_far < total_timesteps:
            # In each iteration, roll out multiple trajectories
            batch_obs, batch_actions, batch_log_probs, batch_reward_to_go, batch_episode_lengths = self.rollout(
                timesteps_per_batch, 
                max_episode_length, 
                gamma
            )
            # print("stage-1:", batch_obs.shape, batch_actions.shape, batch_log_probs.shape, batch_reward_to_go.shape)

            # calculate how many timesteps collected in this batch
            t_so_far += np.sum(batch_episode_lengths)
            i_so_far += 1

            # logging timesteps and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # calculate value function V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_actions)

            # calculate advantage function A_k
            A_k = batch_reward_to_go - V.detach()

            # normalize advantage function
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(num_updates_per_itr):
                # calculate pi_theta(at | st)
                curr_V, curr_log_probs = self.evaluate(batch_obs, batch_actions)

                # calcuate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calcuate surrogate losses
                surr1 = ratios * A_k

                # clips ratio to make sure we are not stepping too far in any direction during gradient ascent
                surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * A_k

                # calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(curr_V, batch_reward_to_go)

                # calculate gradients and backpropagate for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # calculate gradients and backpropagate for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            
            # print a summary of the training so far
            self._log_summary(total_timesteps)

            if i_so_far % save_every == 0:
                torch.save(self.actor.state_dict(), './ckpts/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ckpts/ppo_critic.pth')

    def evaluate(self, batch_obs, batch_actions):
        value = self.critic(batch_obs).squeeze()
        # print(value.shape)

        # calculate the log probabilities of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        # print("Stage-2", mean.shape, self.action_cov_mat.shape, batch_obs.shape, batch_actions.shape)
        dist = MultivariateNormal(mean, self.action_cov_mat)
        # print("This would be printed", dist)
        log_prob = dist.log_prob(batch_actions)
        # print("This would not be printed", dist)

        return value, log_prob

    def rollout(self, timesteps_per_batch, max_episode_length, gamma):
        batch_obs = []   # [number of timesteps per batch, observation length]
        batch_actions = []  # [number of timesteps per batch, action length]
        batch_log_probs = []    # [number of timesteps per batch]
        batch_rewards = []    # [number of timesteps, episode length]
        batch_episode_lengths = []    # [number of timesteps]

        timestep = 0
        while timestep < timesteps_per_batch:
            obs = self.env.reset()    # reset environment
            done = False

            episode_rewards = []
            for step in range(max_episode_length):
                # collect observation
                batch_obs.append(obs)

                action, log_prob = self.compute_action(obs)
                obs, reward, done, _ = self.env.step(action)
                timestep += 1

                # collect action, log probabilities and reward
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                episode_rewards.append(reward)

                if done:
                    break
            
            # collect episode length and rewards
            batch_rewards.append(episode_rewards)
            batch_episode_lengths.append(step+1)
        
        # reshape numpy data as tensors
        batch_obs = torch.from_numpy(np.array(batch_obs, dtype=np.float32))
        batch_actions = torch.from_numpy(np.array(batch_actions, dtype=np.float32))
        batch_actions = batch_actions.unsqueeze(1)
        batch_log_probs = torch.from_numpy(np.array(batch_log_probs, dtype=np.float32))
        batch_reward_to_go = self.compute_reward_to_go(batch_rewards, gamma)
        # print("Stage-0:", np.array(batch_rewards).shape, batch_reward_to_go.shape)
        # batch_episode_lengths = torch.tensor(batch_episode_lengths, dtype=torch.float32)

        # log the episodic rewards and lengths
        self.logger['batch_rewards'] = batch_rewards
        self.logger['batch_lengths'] = batch_episode_lengths
    
        return batch_obs, batch_actions, batch_log_probs, batch_reward_to_go, batch_episode_lengths
    
    def compute_reward_to_go(self, batch_rewards, gamma):
        batch_reward_to_go = []

        # iterate through each episodic rewards
        for episode_rewards in batch_rewards:
            discounted_reward = 0
            reward_to_go = []

            for reward in reversed(episode_rewards):
                discounted_reward = reward + gamma * discounted_reward
                reward_to_go.append(discounted_reward)
            
            reward_to_go = reward_to_go[::-1]
            batch_reward_to_go.append(reward_to_go)
    
        # convert reward-to-go into tensor
        batch_reward_to_go = np.array(batch_reward_to_go, dtype=np.float32)
        batch_reward_to_go = torch.flatten(torch.from_numpy(batch_reward_to_go))

        return batch_reward_to_go

    def compute_action(self, obs):
        # query the actor network for a mean action
        mean = self.actor(obs)

        # create multivariate normal distribution
        dist = MultivariateNormal(mean, self.action_cov_mat)

        # sample an action from the distribution and get its log probability
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def _set_seed_everywhere(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Successfully set seed everywhere to {seed}")

    def _log_summary(self, total_timesteps):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = round((self.logger['delta_t'] - delta_t) / 1e9, 4)

        avg_episode_lens = np.mean(self.logger['batch_lengths'])
        avg_episode_rewards = round(np.mean([np.sum(ep_rewards) for ep_rewards in self.logger['batch_rewards']]), 4)
        avg_actor_loss = round(np.mean([losses.mean() for losses in self.logger['actor_losses']]), 4)

        print(f"{self.logger['t_so_far']}/{total_timesteps} | Avg Loss: {avg_actor_loss} | Avg Ep Len: {avg_episode_lens} | Avg Ep Reward: {avg_episode_rewards} | Itr {self.logger['i_so_far']} took {delta_t} s")
