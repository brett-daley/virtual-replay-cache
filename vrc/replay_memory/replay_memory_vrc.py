import numpy as np

from vrc.replay_memory import ReplayMemoryDQNL


class ReplayMemoryVRC(ReplayMemoryDQNL):
    def _allocate_cache(self, state):
        # Allocate memory for the cached indices/returns
        self._cached_indices = np.empty(self._cache_size, dtype=np.int32)
        self._cached_returns = np.empty_like(self._rewards[:self._cache_size])

    def _store_block(self, cache_slice, rmem_indices, _, returns):
        self._cached_indices[cache_slice] = rmem_indices
        self._cached_returns[cache_slice] = returns

    def _make_minibatch(self, sampled_indices):
        j = self._cached_indices[sampled_indices]
        return (self._get_states(j), self._actions[self._absolute_index(j)], self._cached_returns[sampled_indices])
