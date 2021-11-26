import numpy as np


class ReplayMemoryDQNL:
    def __init__(self, dqn, discount, lambd, capacity=1_000_000, cache_size=80_000, block_size=100):
        self._dqn = dqn
        self._discount = discount
        self._lambd = lambd

        assert cache_size <= capacity, "cache size cannot be larger than memory capacity"
        assert block_size >= 1, "block size must be a positive integer"
        assert cache_size % block_size == 0, "blocks must evenly divide cache"
        self._capacity = capacity
        self._cache_size = cache_size
        self._block_size = block_size

        self._observations = None
        self._actions = np.empty(capacity, dtype=np.uint8)
        self._rewards = np.empty(capacity, dtype=np.float64)
        self._dones = np.empty(capacity, dtype=np.bool)

        self._write_ptr = 0   # Points to the next experience to be overwritten
        self._population = 0  # Tracks the number of samples in the replay memory

    def _allocate_cache(self, state):
        # Allocate memory for the cached states/actions/returns
        self._cached_states = np.empty(shape=[self._cache_size, *state.shape], dtype=state.dtype)
        self._cached_actions = np.empty_like(self._actions[:self._cache_size])
        self._cached_returns = np.empty_like(self._rewards[:self._cache_size])

    def save(self, state, action, reward, done):
        observation = state[..., -1, None]

        if self._observations is None:
            self._history_len = state.shape[-1]
            self._observations = np.empty(shape=[self._capacity, *observation.shape], dtype=observation.dtype)
            self._allocate_cache(state)

        p = self._write_ptr
        self._observations[p], self._actions[p], self._rewards[p], self._dones[p] = (
            observation, action, reward, done)
        self._write_ptr = (self._write_ptr + 1) % self._capacity
        self._population = min(self._population + 1, self._capacity)

    def refresh_cache(self):
        # Sample blocks and compute returns until we fill up the cache
        for k in range(self._cache_size // self._block_size):
            # Sample a random block
            start = np.random.randint(self._population - self._block_size)
            end = start + self._block_size

            # Add all transitions for the block to the cache
            block_indices = np.arange(start, end + 1)  # Include an extra sample for bootstrapping
            rmem_indices = self._absolute_index(block_indices)

            # Get Q-values from the DQN
            states = self._get_states(block_indices)
            qvalues = self._dqn.predict(states).numpy()

            # Compute returns
            rewards = self._rewards[rmem_indices]
            dones = self._dones[rmem_indices]
            returns = compute_pengs_qlambda_returns(rewards, qvalues, dones, self._discount, self._lambd)

            # Slice off the extra sample that was used for bootstrapping
            states, rmem_indices, returns = states[:-1], rmem_indices[:-1], returns[:-1]

            # Store states/actions/returns for minibatch sampling later
            sl = slice(k * self._block_size, (k + 1) * self._block_size)
            self._cached_states[sl] = states
            self._cached_actions[sl] = self._actions[rmem_indices]
            self._cached_returns[sl] = returns

    def iterate_cache(self, for_n_batches, batch_size):
        for j in self._iterate_cache_indices(for_n_batches, batch_size):
            yield self._create_minibatch(j)

    def _iterate_cache_indices(self, for_n_batches, batch_size):
        # We must be able to sample at least one minibatch
        assert batch_size <= self._cache_size

        # Yield minibatches of indices without replacement
        cache_indices = np.arange(self._cache_size)
        np.random.shuffle(cache_indices)

        start = 0
        for _ in range(for_n_batches):
            end = start + batch_size

            if end > self._cache_size:
                # There aren't enough samples for the requested number of minibatches;
                # re-shuffle and start another pass
                np.random.shuffle(cache_indices)
                start, end = 0, batch_size

            assert len(cache_indices[start:end]) == batch_size
            yield cache_indices[start:end]
            start += batch_size

    def _create_minibatch(self, cache_indices):
        j = cache_indices
        return (self._cached_states[j], self._cached_actions[j], self._cached_returns[j])

    def _get_states(self, indices):
        states = []
        for j in reversed(range(self._history_len)):
            x = self._absolute_index(indices - j)
            states.append(self._observations[x])

        mask = np.ones_like(states[0])
        for j in range(1, self._history_len):
            i = indices - j
            x = self._absolute_index(i)
            mask[self._dones[x]] = 0.0
            mask[np.where(i < 0)] = 0.0
            states[-1 - j] *= mask

        states = np.concatenate(states, axis=-1)
        assert states.shape[0] == len(indices)
        assert (states.shape[-1] % self._history_len) == 0
        return states

    def _absolute_index(self, i):
        assert (i < self._population).all()
        assert (i > -self._history_len).all()
        return (self._write_ptr + i) % self._population


def compute_pengs_qlambda_returns(rewards, qvalues, dones, discount, lambd):
    assert 0.0 <= discount <= 1.0, "discount must be in the interval [0,1]"
    assert 0.0 <= lambd <= 1.0, "lambda-value must be in the interval [0,1]"

    # All returns start with the reward
    returns = rewards.copy()
    # Set up the bootstrap for the last state
    returns[-1] = 0.0 if dones[-1] else qvalues[-1].max()

    # For all timesteps except the last, compute the returns
    T = len(rewards)
    for t in reversed(range(T-1)):
        if not dones[t]:
            # Add the discounted value of the next state
            returns[t] += discount * qvalues[t+1].max()

            # Recursion: Propagate the discounted next complex TD error
            next_td_error = returns[t+1] - qvalues[t+1].max()
            returns[t] += discount * lambd * next_td_error

    return returns
