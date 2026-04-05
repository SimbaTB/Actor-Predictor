import numpy as np
import torchvision.transforms as T
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction
import ale_py
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from dm_control import suite
import tools

class Grid(gym.Env):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
        # 原点在左上角
        self.move = (
            np.array([0, -1]),      # 上
            np.array([0, 1]),       # 下
            np.array([-1, 0]),      # 左
            np.array([1, 0])        # 右
        )
        self.terminated = False
        self.steps = 0
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        self.rng = np.random.default_rng(seed=seed)
        self.terminated = False
        self.steps = 0
        # 随机选取目标点坐标
        self.goal = self.rng.integers(0, 8, (2,))
        # 随机选取当前点坐标，并且与目标点坐标不同
        self.current = self.rng.integers(0, 8, (2, ))
        while (self.current == self.goal).all():
            self.current = self.rng.integers(0, 8, (2, ))
        
        obs = self.render()
        info = {}
        return obs, info
    
    def step(self, action):
        d1 = np.linalg.norm(self.current - self.goal)
        self.current += self.move[action]
        self.current = np.clip(self.current, 0, 7)
        d2 = np.linalg.norm(self.current - self.goal)
        self.steps += 1

        obs = self.render()
        reward = 0.1 if d2 < d1 else -1.0       # 比上次接近，则奖励为0.1，否则为-1
        self.terminated = (self.current == self.goal).all() or self.terminated or self.steps >= 100
        truncated = False
        info = {}
        return obs, reward, self.terminated, truncated, info

    def render(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[1:63, 1:63] = [255, 255, 255]
        if (self.goal != self.current).any():
            # 绘制目标点（绿色）
            x, y = self.goal * 8
            image[y:y+8, x:x+8] = [0, 255, 0]
            # 绘制当前点（红色）
            x, y = self.current * 8
            image[y:y+8, x:x+8] = [255, 0, 0]
        else:
            # 当前点和目标点重合变为黄色
            x, y = self.goal * 8
            image[y:y+8, x:x+8] = [255, 255, 0]
        return image
    
    def close(self):
        pass

class DMCSuiteEnv(gym.Env):
    def __init__(self, domain_name, task_name, from_pixels=True, height=84, width=84, frame_skip=1, *args, **kwargs):
        self._env = suite.load(domain_name, task_name)
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.frame_skip = frame_skip

        if from_pixels:
            self.observation_space = spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)
        else:
            obs_spec = self._env.observation_spec()
            obs_shape = sum([np.prod(v.shape) for v in obs_spec.values()])
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_shape,), dtype=np.float32)

        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def _get_obs(self):
        if self.from_pixels:
            camera_id = 0  # 默认相机
            obs = self._env.physics.render(self.height, self.width, camera_id=camera_id)
            return obs.astype(np.uint8)
        else:
            timestep = self._env.reset() if not hasattr(self, '_timestep') else self._timestep
            obs = np.concatenate([v.flatten() for v in timestep.observation.values()])
            return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._timestep = self._env.reset()
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        for _ in range(self.frame_skip):
            self._timestep = self._timestep = self._env.step(action)
            reward += self._timestep.reward or 0
            if self._timestep.last():
                break
        terminated = self._timestep.last()
        truncated = False  # DMC 通常不用 truncated
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.from_pixels:
            return self._get_obs()
        else:
            return self._env.physics.render()

    def close(self):
        pass

from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class CartPoleContinuousEnv(CartPoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), )

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        # force = self.force_mag if action == 1 else -self.force_mag
        force = self.force_mag * action[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


gym.register_envs(ale_py)
gym.register(id="Grid", entry_point="envs:Grid")
gym.register(
    id='CartPoleContinuous-v1', 
    entry_point='envs:CartPoleContinuousEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

DMC_tasks = [("ball_in_cup", "catch"), ("cartpole", "balance"), ("cheetah", "run"), ("finger", "spin")]
for domain, task in DMC_tasks:
    gym.register(
        id=f"DMC-{domain}-{task}",
        entry_point="envs:DMCSuiteEnv",
        kwargs=dict(domain_name=domain, task_name=task, from_pixels=True, height=64, width=64)
    )

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.discrete.Discrete)
        self.num_classes = env.observation_space.n
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_classes,),
            dtype=np.float32
        )

    def observation(self, observation):
        # 将离散状态转换为 one-hot 向量
        one_hot = np.eye(self.num_classes)[observation]
        return one_hot

class ImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize(size),
            T.ToTensor()  # 自动转为 (C, H, W) 和 float32 [0,1]
        ])
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1, *size),
            dtype=np.float32
        )

    def observation(self, observation):
        return self.transform(observation).numpy()

class GaussianNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, std=0.1):
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        super().__init__(env)
        self.scale = std
        
    def observation(self, observation):
        noise = np.random.normal(loc=0.0, scale=self.scale, size=observation.shape)
        noisy_obs = observation + noise
        noisy_obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
        return noisy_obs

class PartialObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)
        self.mask = np.array(mask, dtype=np.bool)
        assert self.mask.shape == env.observation_space.shape
        
    def observation(self, observation):
        return self.mask * observation
    
# 将动作空间线性映射到[-1, 1]
class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Only Box action space is supported"
        assert np.all(np.isfinite(env.action_space.low)), "Action space low must be finite"
        assert np.all(np.isfinite(env.action_space.high)), "Action space high must be finite"

        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(self.low),
            high=np.ones_like(self.high),
            dtype=env.action_space.dtype
        )

    def action(self, action):
        action = np.clip(action, -1.0, 1.0)
        return self.low + (action + 1.0) * 0.5 * (self.high - self.low)

    
def make_env(id:str, seed=None):
    # options
    if "-N" in id:
        noise = True
        id = id.replace("-N", "")
    else:
        noise = False

    # create the environment
    if id == "Pendulum-v1-P":
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env = PartialObservationWrapper(env, [1, 1, 0])
    elif id == "MountainCarContinuous-v0-P":
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
        env = PartialObservationWrapper(env, [1, 0])
    elif id == "CartPoleContinuous-v1-P":
        env = gym.make("CartPoleContinuous-v1", render_mode="rgb_array")
        env = PartialObservationWrapper(env, [1, 0, 1, 0])
    else:
        env = gym.make(id, render_mode="rgb_array")

    # wrap the environment
    if "MiniGrid" in id:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    if tools.is_image(env.observation_space):
        env = ImageWrapper(env)
    else:
        if noise:
            env = GaussianNoiseWrapper(env)

    env = NormalizeActionWrapper(env)

    # set the random seed
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed=seed)
    return env
