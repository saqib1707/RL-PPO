import time
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import gym


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class FeedForwardNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_size=64):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(inp_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_dim)
        self.relu = nn.ReLU()

    
    def forward(self, obs):
        # convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        x = self.relu(self.layer1(obs))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)
        return out


class ProximalPolicyOptimization:
    def __init__(self, env, seed=43, lr=1e-3):
        assert type(env.observation_space) == gym.spaces.Box, "This example only works for envs with continuous state spaces."
        assert type(env.action_space) == gym.spaces.Box, "This example only works for envs with continuous action spaces."
        self._set_seed(seed)

        # extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]    # = ns
        self.act_dim = env.action_space.shape[0]    # = na
        print(f"Observation Dimension: {self.obs_dim} | Action Dimension: {self.act_dim}")

        # initialize actor and critic networks
        self.actor = FeedForwardNN(inp_dim=self.obs_dim, out_dim=self.act_dim)
        self.critic = FeedForwardNN(inp_dim=self.obs_dim, out_dim=1)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999))
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, betas=(0.9, 0.999))

        # initialize action covariance matrix for exploration
        self.act_cov = torch.diag(torch.full(size=(self.act_dim,), fill_value=0.5))    # (na,na)
        # print(self.action_cov_mat)

        # initialize logger
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rewards': [],
            'actor_losses': [],
        }


    def learn(self, total_timesteps, timesteps_per_batch, max_eps_len, num_updates_per_itr, clip_thresh=0.2, save_every=1000, gamma=0.9):
        t_so_far = 0    # timesteps simulated so far
        i_so_far = 0

        while t_so_far < total_timesteps:
            # roll out multiple trajectories
            batch_obs, batch_actions, batch_logprobs, batch_reward_to_go, batch_eps_lens = self.collect_rollouts(
                timesteps_per_batch, 
                max_eps_len, 
                gamma
            )
            print("stage-1:", batch_obs.shape, batch_actions.shape, batch_logprobs.shape, batch_reward_to_go.shape)

            # calculate how many timesteps collected in this batch
            t_so_far += np.sum(batch_eps_lens)
            i_so_far += 1

            # logging timesteps and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # calculate value function V_{phi, k} using critic model
            V, _ = self.evaluate(batch_obs, batch_actions)

            # calculate advantage function A_k
            A_k = batch_reward_to_go - V.detach()

            # normalize advantage function
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(num_updates_per_itr):
                # calculate pi_theta(at | st)
                curr_V, curr_logprobs = self.evaluate(batch_obs, batch_actions)

                # calcuate ratios
                ratios = torch.exp(curr_logprobs - batch_logprobs)

                # calcuate surrogate losses
                surr1 = ratios * A_k

                # clips ratio to make sure we are not stepping too far in any direction during gradient ascent
                surr2 = torch.clamp(ratios, 1 - clip_thresh, 1 + clip_thresh) * A_k

                # calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(curr_V, batch_reward_to_go)

                # calculate gradients and backpropagate for actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # calculate gradients and backpropagate for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            
            # print a summary of the training so far
            self._log_summary(total_timesteps)

            if i_so_far % save_every == 0:
                torch.save(self.actor.state_dict(), './checkpoints/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './checkpoints/ppo_critic.pth')


    def evaluate(self, batch_obs, batch_actions):
        value = self.critic(batch_obs).squeeze()
        # print(value.shape)

        # calculate the log probabilities of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        # print("Stage-2", mean.shape, self.action_cov_mat.shape, batch_obs.shape, batch_actions.shape)
        dist = MultivariateNormal(mean, self.act_cov)
        # print("This would be printed", dist)
        logprob = dist.log_prob(batch_actions)
        # print("This would not be printed", dist)
        return value, logprob


    def collect_rollouts(self, max_timesteps, max_eps_len, gamma):
        observations = []
        actions = []
        logprobs = []
        rewards = []
        eps_lens = []

        t = 0
        while t < max_timesteps:
            # reset environment and get initial observation
            obs, _ = self.env.reset()
            done = False
            # print("Stage-2 after reset:", obs)

            eps_rewards = []
            for step in range(max_eps_len):
                action, logprob = self.select_action(obs)
                next_obs, reward, done, _, _ = self.env.step(action)
                t += 1

                # collect observation, action, log probabilities and reward
                observations.append(obs)
                actions.append(action)
                logprobs.append(logprob)
                eps_rewards.append(reward)

                obs = next_obs
                if done:
                    break
            
            # collect episode length and rewards
            rewards.append(eps_rewards)
            eps_lens.append(step+1)

        # reshape numpy data as tensors
        observations = torch.from_numpy(np.array(observations, dtype=np.float32))    # [max_timesteps, ns]
        actions = torch.from_numpy(np.array(actions, dtype=np.float32))    # [max_timesteps, na]
        actions = actions.unsqueeze(1)
        logprobs = torch.from_numpy(np.array(logprobs, dtype=np.float32))    # [max_timesteps]
        rewards_to_go = self.compute_reward_to_go(rewards, gamma)
        # print("Stage-0:", np.array(batch_rewards).shape, batch_reward_to_go.shape)
        # batch_episode_lengths = torch.tensor(batch_episode_lengths, dtype=torch.float32)

        # log the episodic rewards and lengths
        self.logger['batch_rewards'] = rewards
        self.logger['batch_lengths'] = eps_lens
        return observations, actions, logprobs, rewards_to_go, eps_lens


    def compute_reward_to_go(self, rewards, gamma):
        """
        Compute the discounted reward-to-go for each timestep in each episode
        Args:
            rewards: list of lists, where each inner list contains rewards for an episode
            gamma: discount  for future rewards
        Returns:
            rewards_to_go: list of reward-to-go for each timestep in each episode
        """
        rewards_to_go = []

        # iterate through each episodic rewards
        for eps_rewards in rewards:
            eps_rewards_to_go = []
            reward_sum = 0

            for r in reversed(eps_rewards):
                reward_sum = r + gamma * reward_sum    # discounted reward
                eps_rewards_to_go.append(reward_sum)

            eps_rewards_to_go = eps_rewards_to_go[::-1]
            rewards_to_go.append(eps_rewards_to_go)

        # convert reward-to-go into tensor
        rewards_to_go = np.array(rewards_to_go, dtype=np.float32)
        rewards_to_go = torch.flatten(torch.from_numpy(rewards_to_go))

        return rewards_to_go


    def estimate_action(self, obs):
        print("Stage-3:", obs)
        # query the actor network for mean of the distribution
        mean = self.actor(obs)

        # create multivariate normal distribution
        dist = MultivariateNormal(mean, self.act_cov)

        # sample an action from the distribution and compute its logprob
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.detach().numpy(), logprob.detach()


    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Successfully set seed everywhere: {seed}")


    def _log_summary(self, total_timesteps):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = round((self.logger['delta_t'] - delta_t) / 1e9, 4)

        avg_episode_lens = np.mean(self.logger['batch_lengths'])
        avg_episode_rewards = round(np.mean([np.sum(ep_rewards) for ep_rewards in self.logger['batch_rewards']]), 4)
        avg_actor_loss = round(np.mean([losses.mean() for losses in self.logger['actor_losses']]), 4)

        print(f"{self.logger['t_so_far']}/{total_timesteps} | Avg Loss: {avg_actor_loss} | Avg Ep Len: {avg_episode_lens} | Avg Ep Reward: {avg_episode_rewards} | Itr {self.logger['i_so_far']} took {delta_t} s")
