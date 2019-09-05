import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import torch
#import torch.nn as nn
#import torch.nn.functional as F

# import random

from math import e as nate
def compute_q_values(env, agent):
    """ Computes the Q-values on an equidistant grid. """
    dim = 100    # resolution of the resulting image
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    # Create the states
    states = np.zeros((2, dim ** 2))
    states[0, :] = np.tile(np.linspace(env_low[0], env_high[0], num=dim), dim)
    states[1, :] = np.repeat(np.linspace(env_high[1], env_low[1], num=dim), dim)
    q_values = np.zeros((3, dim ** 2))
    for i in range(dim ** 2):
        q = agent.q_values(states[:, i])
        q_values[:, i] = q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else q
    return q_values

def compute_trajectory(env, agent):
    """ Returns a trajectory (as np.ndarray) in the given env(ironment) executed by the given agent. """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    state = env.reset()
    traj = [(state - env_low) / (env_high - env_low)]
    done = False
    while(not done):
        action = agent.sample(state)
        state, _, done, _ = env.step(action)
        traj.append((state - env_low) / (env_high - env_low))
    return np.stack(traj, axis=1)

def plot_as_image(ax, env, values):
    """ Plots a given np.ndarray of values as a sqare image in the given ax(es). """
    dim = int(np.sqrt(len(values)))
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    cax = ax.imshow(values.reshape((dim, dim)), extent=[0, 1, 0, 1], cmap='jet')
    ax.set_xlabel('Car Position')
    ax.set_xticklabels(["%g" % (env_low[0] + i * (env_high[0] - env_low[0]) / 5) for i in range(6)])
    ax.set_ylabel('Car Velocity')
    ax.set_yticklabels(["%g" % (env_low[1] + i * (env_high[1] - env_low[1]) / 5) for i in range(6)])
    return plt.gcf().colorbar(cax, ax=ax)

def plot_all_results(results, env, plot_std=True):
    """ Plots the performance of all experiments in the result-buffers, as well as the value function/policy 
        of the agent at the end of the last experiment. """
    agents, plot_labels, plot_rewards = zip(*results)
    last_agent = agents[-1]

    colors = ['orange', 'red', 'magenta', 'blue', 'green', 'black', 'c', 'y', 'lime']
    # Generate figure and subplot grid
    gs = matplotlib.gridspec.GridSpec(2, 5)
    ax1 = plt.subplot(gs[:, :3])
    ax2 = plt.subplot(gs[0, 3:])
    ax3 = plt.subplot(gs[1, 3:])
    plt.gcf().set_size_inches([16, 7.5])
    # Plot the performance
    if plot_std:
        # Make a nice plot with mean and standard deviation for every 10 samples
        for i, rewards in enumerate(plot_rewards):
            rew = np.array(rewards)
            rew = rew[:(len(rew) - len(rew) % 10)].reshape(int(len(rewards) / 10), 10)
            m, s = np.mean(rew, axis=1), np.std(rew, axis=1)
            x = np.linspace(5, 10 * len(m) + 5, len(m))
            ax1.fill_between(x=x, y1=m - s, y2=m + s, alpha=0.2, linewidth=0, facecolor=colors[i % len(colors)])
            ax1.plot(x, m, color=colors[i % len(colors)])

        ax1.set_xlabel("Environmental Steps")
        ax1.set_ylabel("Episode Reward (STD)")
        ax1.set_ylim(-1000, 0)
        ax1.set_xlim(0, max(map(len, plot_rewards)))
    else:
        # Use an ugly plot that plots every single measurement
        for rewards in plot_rewards:
            ax1.plot(rewards)
        ax1.set_xlabel("Epsiodes")
        ax1.set_ylabel("Episode Reward")

    ax1.legend(plot_labels, loc='lower right')

    # Generate Q-values and example trajectories
    traj_list = [compute_trajectory(env, last_agent) for _ in range(3)]
    colors = ['darkgrey', 'lightgrey', 'white']
    q_values = compute_q_values(env, last_agent)
    # Plot the value function
    plot_as_image(ax2, env, q_values.max(axis=0))
    for i, traj in enumerate(traj_list):
        ax2.plot(traj[0, :], traj[1, :], color=colors[i % len(colors)])
    ax2.set_xlabel('')
    ax2.set_title('Value Function & Greedy Policy')
    # Plot the policy
    cbar = plot_as_image(ax3, env, q_values.argmax(axis=0))
    for traj in traj_list:
         ax3.plot(traj[0, :], traj[1, :], color=colors[i % len(colors)])
    cbar.set_ticks([0, 1, 2])
    cbar.ax.set_yticklabels(['-acc', '0', '+acc'])
    plt.show()

def run_experiment(env, agent, iter_max=1000):
    # Set seeds (for reproduceability)
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Initialise the result lists
    total_reward_list = []
    total_steps = 0
    print('----- Start Learning with %s Q-learning -----' % agent.name)
    # For iter_max episodes
    for iter in range(iter_max):
        # Reset the episode
        state = env.reset()
        total_reward = 0
        done = False
        # Run the episode until it is done (as signaled by the environment)
        while not done:
            current_state = state

            # Epsilon greedy action selection
            if np.random.uniform(0, 1) < agent.get_epsilon():
                # Random choice
                action = np.random.choice(env.action_space.n)
            else:
                # Greedy choice
                action = agent.sample(current_state)

            # One environmental step
            state, reward, done, _ = env.step(action)
            total_reward += reward
            # Update agent
            agent.update(current_state, action, reward, state, done)
            total_steps += 1

        total_reward_list.append(total_reward)
        agent.set_epsilon(iter)
        # Output for user (every 100 episodes)
        if iter % 100 == 49:
            print('Episode #%d (%u steps) -- Total reward = %g, epsilon=%g.' % 
                  (iter+1, total_steps, total_reward, agent.get_epsilon()))

    # Book keeping for the plotting script
    return agent, agent.name, total_reward_list