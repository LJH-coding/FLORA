from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import diambra
import diambra.arena
import gymnasium as gym
import numpy as np
from diambra.arena import EnvironmentSettings, WrappersSettings
from gymnasium.core import RenderFrame

import os
import torch
import torchvision
import datetime
from nnet.structs import AttrDict
import cv2

class DiambraWrapper(gym.Wrapper):
    def __init__(
        self,
        id: str,
        action_space: str = "DISCRETE",
        screen_size: Union[int, Tuple[int, int]] = 64,
        grayscale: bool = False,
        repeat_action: int = 1,
        rank: int = 0,
        diambra_settings: Dict[str, Any] = {},
        diambra_wrappers: Dict[str, Any] = {},
        render_mode: str = "rgb_array",
        log_level: int = 0,
        increase_performance: bool = True,
    ) -> None:
        if isinstance(screen_size, int):
            screen_size = (screen_size,) * 2

        if diambra_settings.pop("frame_shape", None) is not None:
            warnings.warn("The DIAMBRA frame_shape setting is disabled")
        if diambra_settings.pop("n_players", None) is not None:
            warnings.warn("The DIAMBRA n_players setting is disabled")

        role = diambra_settings.pop("role", None)
        if action_space not in {"DISCRETE", "MULTI_DISCRETE"}:
            raise ValueError(
                "The valid values for the `action_space` attribute are "
                f"'DISCRETE' or 'MULTI_DISCRETE', got {action_space}"
            )
        if role is not None and role not in {"P1", "P2"}:
            raise ValueError(f"The valid values for the `role` attribute are 'P1' or 'P2' or None, got {role}")
        self._action_type = action_space.lower()
        settings = EnvironmentSettings(
            **{
                **diambra_settings,
                "game_id": id,
                "action_space": getattr(diambra.arena.SpaceTypes, action_space, diambra.arena.SpaceTypes.DISCRETE),
                "n_players": 1,
                "role": getattr(diambra.arena.Roles, role, diambra.arena.Roles.P1) if role is not None else None,
                "render_mode": render_mode,
            }
        )
        if repeat_action > 1:
            if "step_ratio" not in settings or settings["step_ratio"] > 1:
                warnings.warn(
                    f"step_ratio parameter modified to 1 because the sticky action is active ({repeat_action})"
                )
            settings["step_ratio"] = 1
        if diambra_wrappers.pop("frame_shape", None) is not None:
            warnings.warn("The DIAMBRA frame_shape wrapper is disabled")
        if diambra_wrappers.pop("stack_frames", None) is not None:
            warnings.warn("The DIAMBRA stack_frames wrapper is disabled")
        if diambra_wrappers.pop("dilation", None) is not None:
            warnings.warn("The DIAMBRA dilation wrapper is disabled")
        if diambra_wrappers.pop("flatten", None) is not None:
            warnings.warn("The DIAMBRA flatten wrapper is disabled")
        wrappers = WrappersSettings(
            **{
                **diambra_wrappers,
                "flatten": True,
                "repeat_action": repeat_action,
            }
        )
        if increase_performance:
            settings.frame_shape = tuple(screen_size) + (int(grayscale),)
        else:
            wrappers.frame_shape = tuple(screen_size) + (int(grayscale),)

        env = diambra.arena.make(id, settings, wrappers, rank=rank, render_mode=render_mode, log_level=log_level)
        super().__init__(env)

        # Observation and action space
        self.action_space = self.env.action_space
        obs = {}
        for k in self.env.observation_space.spaces.keys():
            if isinstance(self.env.observation_space[k], gym.spaces.Discrete):
                low = 0
                high = self.env.observation_space[k].n - 1
                shape = (1,)
                dtype = np.int32
            elif isinstance(self.env.observation_space[k], gym.spaces.MultiDiscrete):
                low = np.zeros_like(self.env.observation_space[k].nvec)
                high = self.env.observation_space[k].nvec - 1
                shape = (len(high),)
                dtype = np.int32
            elif not isinstance(self.env.observation_space[k], gym.spaces.Box):
                raise RuntimeError(f"Invalid observation space, got: {type(self.env.observation_space[k])}")
            obs[k] = (
                self.env.observation_space[k]
                if isinstance(self.env.observation_space[k], gym.spaces.Box)
                else gym.spaces.Box(low, high, shape, dtype)
            )
        self.observation_space = gym.spaces.Dict(obs)
        self._render_mode = render_mode

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _convert_obs(self, obs: Dict[str, Union[int, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            k: (np.array(v) if not isinstance(v, np.ndarray) else v).reshape(self.observation_space[k].shape)
            for k, v in obs.items()
        }

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self._action_type == "discrete" and isinstance(action, np.ndarray):
            action = action.squeeze()
            action = action.item()
        obs, reward, terminated, truncated, infos = self.env.step(action)
        infos["env_domain"] = "DIAMBRA"
        return self._convert_obs(obs), reward, terminated or infos.get("env_done", False), truncated, infos

    def render(self, mode: str = "rgb_array", **kwargs) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        infos["env_domain"] = "DIAMBRA"
        return self._convert_obs(obs), infos


class DiambraEnv:
    def obs_space(self):
        if self.grayscale_obs:
            return (["image", (1 * self.history_frames, self.img_size[0], self.img_size[1]), torch.uint8],)
        else:
            return (["image", (3 * self.history_frames, self.img_size[0], self.img_size[1]), torch.uint8],)

    def __init__(
            self,
            game,
            img_size=(64, 64),
            action_repeat=1,
            history_frames=1,
            grayscale_obs=False,
            rank=0,
            action_space="DISCRETE",
            log_level=0,
            increase_performance=True,
            record=False,
            diambra_settings: Dict[str, Any] = {},
            diambra_wrappers: Dict[str, Any] = {}
        ):
        
        # Params
        self.img_size = img_size
        self.grayscale_obs = grayscale_obs
        self.action_repeat = action_repeat
        self.history_frames = history_frames
        self.record = record
        if record == True:
            img_size = (480, 512)
        
        # Create DIAMBRA env
        self.env = DiambraWrapper(
            id=game,
            action_space=action_space,
            screen_size=img_size,
            grayscale=grayscale_obs,
            repeat_action=action_repeat,
            rank=rank,
            render_mode="rgb_array",
            log_level=log_level,
            increase_performance=increase_performance,
            diambra_settings=diambra_settings,
            diambra_wrappers=diambra_wrappers
        )

        assert action_space == "DISCRETE"

        # Action Space
        self.num_actions = self.env.action_space.n

    def sample(self):
        return torch.nn.functional.one_hot(
            torch.randint(low=0, high=self.num_actions, size=()), 
            num_classes=self.num_actions
        ).type(torch.float32)

    def preprocess(self, state, reward, done):
        # Convert state to tensor
        state = state["frame"]
        if self.record == True:
            state = cv2.resize(state, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_AREA,)
        state = torch.tensor(state)

        # (C, H, W)
        if self.grayscale_obs:
            state = state.unsqueeze(dim=0)
        else:
            state = state.permute(2, 0, 1)

        # Convert reward and done to tensor
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        is_last = done

        return state, reward, done, is_last

    def reset(self):
        # Reset environment
        state, _ = self.env.reset()
        state, reward, done, is_last = self.preprocess(state, 0.0, False)

        # Handle history frames
        if self.history_frames > 1:
            self.history = state.repeat(self.history_frames, 1, 1)
        else:
            self.history = state

        # Reset episode score
        self.episode_score = 0.0

        return AttrDict(
            state=self.history,
            reward=torch.tensor(0.0, dtype=torch.float32),
            done=torch.tensor(False, dtype=torch.float32),
            is_first=torch.tensor(True, dtype=torch.float32),
            is_last=torch.tensor(False, dtype=torch.float32)
        )

    def step(self, action):
        
        state, reward, terminated, truncated, info = self.env.step(action.item())
        
        # Update episode score
        self.episode_score += reward

        # Process step results
        state, reward, done, is_last = self.preprocess(state, reward, terminated or truncated)

        # Update history frames
        if self.history_frames > 1:
            if self.grayscale_obs:
                self.history = torch.cat([self.history[1:], state], dim=0)
            else:
                self.history = torch.cat([self.history[3:], state], dim=0)
        else:
            self.history = state

        return AttrDict(
            state=self.history,
            reward=reward,
            done=done,
            is_first=torch.tensor(False, dtype=torch.float32),
            is_last=is_last
        )

    def render(self):
        return self.env.render()