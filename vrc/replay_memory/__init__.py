from vrc.replay_memory.replay_memory_dqnl import ReplayMemoryDQNL
from vrc.replay_memory.replay_memory_vrc import ReplayMemoryVRC


def ReplayMemory(*args, virtual_cache=True, **kwargs):
    if virtual_cache:
        return ReplayMemoryVRC(*args, **kwargs)
    return ReplayMemoryDQNL(*args, **kwargs)
