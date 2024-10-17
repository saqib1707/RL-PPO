import os
import time
import torch

from ppo import PPOAgent


def train(env, config, device):
    start_time = time.time()
    obs_dim = env.observation_space.shape[0]
    if config['continuous_action_space']:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    print(f"Observation Dimension: {obs_dim} | Action Dimension: {action_dim}")

    # initialize a PPO agent
    ppo_agent = PPOAgent(
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        hidden_dim=config['hidden_dim'],
        lr_actor=config['lr_actor'], 
        lr_critic=config['lr_critic'], 
        continuous_action_space=config['continuous_action_space'], 
        num_epochs=config['num_epochs'], 
        clip_thresh=config['clip_thresh'], 
        action_std_init=config['action_std_init'], 
        gamma=config['gamma'],
        entropy_coef=config['entropy_coef'],
        value_loss_coef=config['value_loss_coef'],
        batch_size=config['batch_size'],
        max_grad_norm=config['max_grad_norm'],
        device=device,
    )

    # for logging
    mean_eps_reward = 0
    num_episodes = 0

    # start training loop
    t_so_far = 0
    eps_so_far = 0
    while t_so_far < config['num_train_steps']:
        obs, _ = env.reset(seed=config['random_seed'])
        done = False
        eps_reward = 0

        # start episode
        for _ in range(1, config['max_eps_steps']+1):
            # print("Observation:", obs.shape)
            action, logprob, value = ppo_agent.policy.select_action(obs)
            # print("Action:", action, "Logprob:", logprob, "Value:", value)
            next_obs, reward, done, _, _ = env.step(action.item())

            t_so_far += 1
            eps_reward += reward

            # store transitions in buffer
            ppo_agent.buffer.states.append(obs)
            ppo_agent.buffer.actions.append(action)
            ppo_agent.buffer.logprobs.append(logprob)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.dones.append(done)
            ppo_agent.buffer.state_values.append(value)

            if t_so_far % config['policy_update_interval'] == 0:
                # print("Updating policy")
                ppo_agent.update_policy()

            if t_so_far % config['log_interval'] == 0:
                mean_eps_reward /= num_episodes
                print(f'episode: {eps_so_far} | step: {t_so_far} | reward {mean_eps_reward:.4f}')
                with open(config['logpath'], 'a') as f:
                    f.write(f'episode: {eps_so_far} | step: {t_so_far} | reward {mean_eps_reward:.4f}\n')
                mean_eps_reward = 0
                num_episodes = 0

            if t_so_far % config['save_interval'] == 0:
                checkpoint_path = os.path.join(config['ckpt_dir'], f"{config['env_name']}_step_{t_so_far}.pt")
                torch.save(ppo_agent.policy.state_dict(), checkpoint_path)

            obs = next_obs
            if done:
                break
        
        mean_eps_reward += eps_reward
        num_episodes += 1
        eps_so_far += 1

    print(f"Training time: {(time.time()-start_time) / 60.0:.2f} mins")


if __name__ == "__main__":
    train()
