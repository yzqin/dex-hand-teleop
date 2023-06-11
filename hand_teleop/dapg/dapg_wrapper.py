import numpy as np
from gym import spaces
from gym.core import Env


class DAPGWrapper(Env):
    def __init__(self, env):
        self.env = env
        self.name = type(env).__name__

        # Gym specific attributes
        self.metadata = None

        # set up observation and action spaces
        obs_dim = self.env.obs_dim
        high = np.inf * np.ones(obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        action_dim = self.env.action_dim
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,))

    def step(self, action: np.ndarray):
        self.env.rl_step(action)
        obs = self.env.get_observation()
        reward = self.env.get_reward(action)
        done = self.env.is_done()
        return obs, reward, done, {}

    def seed(self, seed=None):
        return self.env.seed(seed)

    def set_seed(self, seed=None):
        self.env.seed(seed)

    @property
    def horizon(self):
        return self.env.horizon

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        return self.env.reset

    def render(self, mode="human"):
        self.env.render()

    @property
    def act_repeat(self):
        return self.env.frame_skip

    def get_obs(self):
        return self.env.get_observation()

    def get_env_infos(self):
        return {"env_name": self.name}

    # ===========================================
    # Trajectory optimization related
    # Environments should support these functions in case of trajopt

    def get_env_state(self):
        return self.env.scene.pack()

    def set_env_state(self, state_dict):
        self.env.scene.unpack(state_dict)

    def close(self):
        self.env.scene = None
