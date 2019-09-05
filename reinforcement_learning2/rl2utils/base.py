
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import torch
#import torch.nn as nn
#import torch.nn.functional as F

# import random

from math import e as nate
class QLearner:
    """ Specifies a RL agent that can compute Q-values. """
    gamma = 0.9    # discount factor
    name = None    # the name of this agent (for the legend in plots)
    epsilon = 0.1  # the exploration parameter for epsilon-greedy
    
    def q_values(self, state):
        """ Returns the estimated Q-values (as a np.ndarray) of the given state. """
        raise NotImplementedError("Abstract class must be inherited from to get Q-values.")

    def sample(self, state):
        """ Returns a greedily sampled action according to the estimated Q-values of the given state. """
        raise NotImplementedError("Abstract class must be inherited from to sample.")

    def update(self, state, action, reward, next_state, done):
        """ Updates the Q-value estimate after observing a transition from 'state', using 'action',
            receiving 'reward' and ending up in 'next_state'. The Boolean 'done' indicates whether
            or not the episode has ended with this transition. Returns nothing. """
        raise NotImplementedError("Abstract class must be inherited from to update.")
    
    def get_epsilon(self):
        """ Returns the exploration parameter of epsilon-greedy. """
        return self.epsilon
    
    def set_epsilon(self, iter):
        """ Can be overwritten to change the exploration parameter during training. """
        pass


class BasisFunctions:
    """ Abstract class that specifies basis functions. """
    name = None                  # the name of the basis function
    num_features = None          # the number of basis functions
    _env_low = None
    _env_high = None
    _env_dx = None

    def __init__(self, env):
        self._env_low = env.observation_space.low
        self._env_high = env.observation_space.high
        self._env_dx = self._env_high - self._env_low

    def __call__(self, state):
        """ Returns the basis function outputs of the given state as a vector.  """
        assert False, "Abstract class must be inherited from to call."

    def new_weights(self):
        """ Returns a newly initialized fitting weight vector. """
        return np.zeros(self.num_features)
