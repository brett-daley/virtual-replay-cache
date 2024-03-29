import cv2
import gym
from gym.envs.atari.atari_env import AtariEnv
from gym.spaces import Box
import numpy as np

from vrc.auto_monitor import AutoMonitor
from vrc.image_stacker import ImageStacker


def make(game, interpolation='nearest'):
    env = AtariEnv(game, frameskip=4, obs_type='image')
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    env = NoopResetWrapper(env)

    # 27k timesteps * 4 frames/timestep = 108k frames = 30 minutes of gameplay
    env = TimeLimitWrapper(env, max_timesteps=27_000)

    # To avoid miscounts, monitor must come after no-ops/time limit and before episodic life reset
    env = AutoMonitor(env)

    env = EpisodicLifeWrapper(env)
    env = ClippedRewardWrapper(env)
    env = PreprocessImageWrapper(env, interpolation)
    env = HistoryWrapper(env, history_len=4)
    return env


class ClippedRewardWrapper(gym.RewardWrapper):
    """Clips rewards to be in {-1, 0, +1} based on their signs."""
    def reward(self, reward):
        return np.sign(reward)


class EpisodicLifeWrapper(gym.Wrapper):
    """Signals done when a life is lost, but only resets when the game ends."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        self.observation, reward, done, info = self.env.step(action)
        self.was_real_done = done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            # We lost a life, but force a reset only if it's not game over.
            # Otherwise, the environment just handles it automatically.
            if lives > 0:
                done = True
            # HACK: Force a reset when it's game over too. Temporary workaround to
            # address a bug in the ALE where some games may not terminate correctly.
            # https://github.com/mgbellemare/Arcade-Learning-Environment/issues/434
            if lives == 0:
                self.was_real_done = done = True
        self.lives = lives

        info.update({'real_done': self.was_real_done})
        return self.observation, reward, done, info

    def reset(self):
        if self.was_real_done:
            self.observation = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return self.observation


class FireResetWrapper(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        super().__init__(env)

    def reset(self):
        self.env.reset()
        observation, _, _, _ = self.step(1)
        return observation


class HistoryWrapper(gym.Wrapper):
    """Stacks the previous `history_len` observations along their last axis.
    Pads observations with zeros at the beginning of an episode."""
    def __init__(self, env, history_len=4):
        assert history_len > 1
        super().__init__(env)
        self._image_stacker = ImageStacker(history_len)

        shape = self.observation_space.shape
        self.observation_space.shape = (*shape[:-1], history_len * shape[-1])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._image_stacker.append(observation)
        return self._image_stacker.get_stack(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._image_stacker.append(observation, reset=True)
        return self._image_stacker.get_stack()


class NoopResetWrapper(gym.Wrapper):
    """Sample initial states by taking a random number of no-ops on reset.
    The number is sampled uniformly from [0, `noop_max`]."""
    def __init__(self, env, noop_max=30):
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        super().__init__(env)
        self.noop_max = noop_max
        # Use this for RNG to make instances thread safe
        self.np_random = None

    def reset(self):
        observation = self.env.reset()
        n = self.np_random.randint(self.noop_max + 1)
        for _ in range(n):
            observation, _, _, _ = self.step(0)
        return observation

    def seed(self, seed=None):
        seed_list = self.env.seed(seed)
        self.np_random = np.random.RandomState(seed_list[0])
        return seed_list


class PreprocessImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, interpolation='nearest'):
        super().__init__(env)
        self._shape = (84, 84, 1)
        self.observation_space = Box(low=0, high=255, shape=self._shape, dtype=np.uint8)
        self._interpolation = getattr(cv2, 'INTER_' + interpolation.upper())

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return self._resize(observation).reshape(self._shape)

    def _resize(self, observation):
        return cv2.resize(observation, self._shape[:2][::-1], interpolation=self._interpolation)


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps):
        super().__init__(env)
        assert max_timesteps > 0
        self._max_timesteps = max_timesteps
        self._episode_steps = 0

    def step(self, action):
        self._episode_steps += 1
        observation, reward, done, info = self.env.step(action)
        done = done or (self._episode_steps >= self._max_timesteps)
        return observation, reward, done, info

    def reset(self):
        self._episode_steps = 0
        return self.env.reset()
