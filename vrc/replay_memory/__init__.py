from vrc.replay_memory.replay_memory_dqnl import ReplayMemoryDQNL
from vrc.replay_memory.replay_memory_vrc import ReplayMemoryVRC


def ReplayMemory(*args, **kwargs):
    # TODO: Need switch for virtual/original
    return ReplayMemoryDQNL(*args, **kwargs)
