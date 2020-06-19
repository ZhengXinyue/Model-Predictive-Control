import random

from gym.envs.classic_control import PendulumEnv
import numpy as np
import gym
import matplotlib.pyplot as plt


"""
Use MPC to control the Pendulum.
In this version an additional real model is needed for simulation.
"""


# simulate and choose the best action
def choose_action(state, rollout, horizon):
    """
    :param rollout: trajectories that are simulated.
    :param horizon: steps lookahead.
    :return:
    """
    best_action = None
    max_trajectory_value = -float('inf')
    for trajectory in range(rollout):
        sim_env.state = state
        trajectory_value = 0
        for h in range(horizon):
            action = np.array([random.uniform(-2, 2)])
            if h == 0:
                first_action = action
            _, reward, _, _ = sim_env.step(action)
            trajectory_value += reward
        # check if this trajectory's value is higher.
        if trajectory_value > max_trajectory_value:
            max_trajectory_value = trajectory_value
            best_action = first_action
    return best_action


def start():
    for rollout, horizon in zip(rollout_list, horizon_list):
        episode_reward_list = []
        for episode in range(max_episodes):
            env.reset()
            episode_reward = 0
            for step in range(max_episode_steps):
                if render:
                    env.render()
                state = env.state
                action = choose_action(state, rollout, horizon)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
            # print('episode %d ends with reward %d' % (episode, episode_reward))
            episode_reward_list.append(episode_reward)
        plt.plot(episode_reward_list, label='rollout=%d horizon=%d' % (rollout, horizon))


if __name__ == '__main__':
    seed = 777777
    max_episodes = 50
    max_episode_steps = 200
    env = gym.make('Pendulum-v0').unwrapped
    sim_env = PendulumEnv()  # additional model
    sim_env.reset()
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    render = True

    # custom the rollout number and horizon number
    rollout_list = [50]
    horizon_list = [10]
    start()

    env.close()
    sim_env.close()
    plt.legend()
    plt.grid()
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title('MPC_Pendulum_v0')
    # plt.savefig('Naive_MPC/MPC_Pendulum_v0.png')
