"""Custom feature extractors and policy helpers."""
from __future__ import annotations

from typing import Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SimpleNatureCNN(BaseFeaturesExtractor):
    """Nature CNN variant that avoids sampling from the observation space during init."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(
                "SimpleNatureCNN requires a gymnasium.spaces.Box observation space"
            )
        if not is_image_space(
            observation_space,
            check_channels=False,
            normalized_image=normalized_image,
        ):
            raise ValueError(
                "SimpleNatureCNN can only be used with image observations; "
                f"got {observation_space}"
            )

        super().__init__(observation_space, features_dim)

        n_input_channels, height, width = self._extract_shape(observation_space)

        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        with th.no_grad():
            dummy = th.zeros((1, n_input_channels, height, width), dtype=th.float32)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            th.nn.ReLU(),
        )

    @staticmethod
    def _extract_shape(space: spaces.Box) -> Tuple[int, int, int]:
        if len(space.shape) != 3:
            raise ValueError(
                "Expected 3D observation (C, H, W) for SimpleNatureCNN; "
                f"got shape {space.shape}"
            )
        return int(space.shape[0]), int(space.shape[1]), int(space.shape[2])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))
