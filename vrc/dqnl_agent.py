from gym.spaces import Discrete
import numpy as np
from tensorflow.keras.optimizers import Adam

from vrc.deep_q_network import DeepQNetwork
from vrc.replay_memory import ReplayMemory


class DQNLAgent:
    def __init__(self, env, **kwargs):
        assert isinstance(env.action_space, Discrete)
        self._env = env

        optimizer = Adam(lr=5e-5, epsilon=1e-8)
        self._dqn = DeepQNetwork(env, optimizer)

        self._replay_memory = ReplayMemory(self._dqn, discount=0.99, lambd=kwargs['lambd'])

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

        # Ensure that the cache gets refreshed before training starts
        assert self._prepopulate % self._target_update_freq == 0

        # Compute number of minibatches to conduct per "epoch" (i.e. target net update)
        assert self._target_update_freq % self._train_freq == 0
        self._batches_per_epoch = self._target_update_freq // self._train_freq

    def policy(self, t, state):
        assert t > 0, "timestep must start at 1"
        epsilon = self._epsilon_schedule(t)
        # With probability epsilon, take a random action
        if np.random.rand() < epsilon:
            return self._env.action_space.sample()
        # Else, take the predicted best action (greedy)
        Q = self._dqn.predict(state[None])[0]
        return np.argmax(Q)

    def _epsilon_schedule(self, t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def update(self, t, state, action, reward, done):
        assert t > 0, "timestep must start at 1"
        self._replay_memory.save(state, action, reward, done)

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % self._target_update_freq == 1:
            self._replay_memory.refresh_cache()
            for minibatch in self._replay_memory.iterate_cache(self._batches_per_epoch, self._batch_size):
                self._dqn.train(*minibatch)
