import random
from collections import deque

import gym

import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss, nn


"""
Use MPC to control the Pendulum.
In this version, user a neural network to learn the system dynamics, no real model is needed for simulation.

if you want to know more about this system(Pendulum-v0), 
please refer to "from gym.envs.classic_control import PendulumEnv" or 
"https://github.com/openai/gym/blob/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/envs/classic_control/pendulum.py"
"""


class MemoryBuffer(object):
    """
    you don't really need the memory buffer to store transitions.
    This is just a trick from reinforcement learning to improve the sample efficiency.
    """
    def __init__(self, size, ctx):
        self.buffer = deque(maxlen=size)
        self.ctx = ctx

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        state_batch = nd.array([data[0] for data in mini_batch], ctx=self.ctx)
        action_batch = nd.array([data[1] for data in mini_batch], ctx=self.ctx)
        next_state_batch = nd.array([data[2] for data in mini_batch], ctx=self.ctx)
        return state_batch, action_batch, next_state_batch

    def store_transition(self, state, action, next_state):
        self.buffer.append((state, action, next_state))


class SystemModel(nn.Block):
    """
    this network will simulate the true system dynamics.
    you can cumtom your own neural network.
    """
    def __init__(self, observation_dim):
        super(SystemModel, self).__init__()
        self.observation_dim = observation_dim
        self.dense0 = nn.Dense(500, activation='relu')
        self.dense1 = nn.Dense(500, activation='relu')
        self.dense2 = nn.Dense(self.observation_dim)

    def forward(self, state, action):
        _ = nd.concat(state, action, dim=1)
        predict_state = self.dense2(self.dense1(self.dense0(_)))
        return predict_state


class MPCAgent(object):
    def __init__(self,
                 gamma,
                 action_dim,
                 observation_dim,
                 buffer_size,
                 batch_size,
                 ctx):
        self.gamma = gamma
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ctx = ctx

        self.buffer = MemoryBuffer(self.buffer_size, self.ctx)
        self.system_model = SystemModel(self.observation_dim)
        self.system_model.initialize(init=init.Xavier(), ctx=self.ctx)
        self.system_model_optimizer = gluon.Trainer(self.system_model.collect_params(), 'adam')

    # use the learned system model to simulate the transitions.
    def choose_action(self, state, rollout, horizon):
        def angle_normalize(x):
            return ((x + np.pi) % (2 * np.pi)) - np.pi
        initial_state = nd.array([state], ctx=self.ctx)
        best_action = None
        max_trajectory_value = -float('inf')
        for trajectory in range(rollout):
            state = initial_state
            trajectory_value = 0
            for h in range(horizon):
                action = np.array([random.uniform(-2, 2)])
                if h == 0:
                    first_action = action
                action = nd.array([action], ctx=self.ctx)
                next_state = self.system_model(state, action)  # here is the simulation process.
                theta = next_state[0][0].asnumpy()[0]
                theta_dot = next_state[0][1].asnumpy()[0]
                # compute the reward, I don't multiply the discount factor.
                costs = angle_normalize(theta) ** 2 + .1 * theta_dot ** 2 + .001 * (action.squeeze().asnumpy()[0] ** 2)
                trajectory_value += -costs
                state = next_state
            # check if this trajectory's value is higher.
            if trajectory_value > max_trajectory_value:
                max_trajectory_value = trajectory_value
                best_action = first_action
        return best_action

    # update the model to better simulate the true system dynamics.
    def update(self):
        states, actions, next_states = self.buffer.sample(self.batch_size)
        with autograd.record():
            predict_states = self.system_model(states, actions)
            # print('predict: ', predict_states[0])
            # print('true: ', next_states[0])
            # print('-----------------')
            l2_loss = gloss.L2Loss()
            loss = l2_loss(predict_states, next_states)
        self.system_model.collect_params().zero_grad()
        loss.backward()
        self.system_model_optimizer.step(batch_size=self.batch_size)

    def save(self):
        self.system_model.save_parameters('system_model.params')

    def load(self):
        self.system_model.load_parameters('system_model.params')


def train_system_model(train_episodes, train_episode_steps):
    """
    these two parameters are just to control how long you want to train the model.
    """
    for i in range(train_episodes):
        env.reset()
        for j in range(train_episode_steps):
            action = np.array([random.uniform(-2, 2)])   # random sample to train the model
            state = env.state
            _, reward, done, info = env.step(action)
            next_state = env.state
            if len(agent.buffer) > 200:
                agent.update()
            agent.buffer.store_transition(state, action, next_state)
    print('training finished.')
    agent.save()


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
                action = agent.choose_action(state, rollout, horizon)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
            print('rollout: %d, horizon: %d, episode: %d, reward: %d' % (rollout, horizon, episode, episode_reward))
            episode_reward_list.append(episode_reward)
        plt.plot(episode_reward_list, label='rollout=%d horizon=%d' % (rollout, horizon))


if __name__ == '__main__':
    env = gym.make('Pendulum-v0').unwrapped
    seed = 7777777
    env.seed(seed)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ctx = mx.cpu()
    agent = MPCAgent(gamma=0.99,  # reward discount factor
                     action_dim=1,
                     observation_dim=2,  # use internal state, theta and theta_dot
                     buffer_size=2000,
                     batch_size=64,
                     ctx=ctx)

    # train_system_model(50, 400)  # train the system model or load parameters.
    agent.load()
    render = True

    # due to computation cost. I only do some simple simulation, but that's good enough.
    # the principle is that: more rollouts and longer horizon will make better performance.
    rollout_list = [20]
    horizon_list = [10]
    max_episodes = 20
    max_episode_steps = 400
    start()

    env.close()
    plt.legend()
    plt.grid()
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title('Learning_MPC_Pendulum_v0')
    plt.savefig('Learning_MPC_Pendulum_v0.png')
