"""Custom feature extractors and policy helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .algos import DuelingCnnPolicy


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


class DetectorCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor that merges pixel CNN features with detector vectors."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        detector_hidden_dim: int = 128,
        detector_output_dim: int = 64,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("DetectorCombinedExtractor expects a Dict observation space")

        pixel_space = observation_space.spaces.get("pixels")
        detector_space = observation_space.spaces.get("detector")
        if not isinstance(pixel_space, spaces.Box) or not isinstance(detector_space, spaces.Box):
            raise TypeError("DetectorCombinedExtractor requires Box spaces for both pixels and detector features")

        self.pixel_space = pixel_space
        self.detector_space = detector_space

        features_dim = cnn_output_dim + detector_output_dim
        super().__init__(observation_space, features_dim)

        self.cnn = SimpleNatureCNN(pixel_space, features_dim=cnn_output_dim)
        detector_dim = int(np.prod(detector_space.shape))
        self.detector_net = nn.Sequential(
            nn.Linear(detector_dim, detector_hidden_dim),
            nn.ReLU(),
            nn.Linear(detector_hidden_dim, detector_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:  # type: ignore[override]
        pixels = observations["pixels"].float()
        detector = observations["detector"].float()
        pixel_features = self.cnn(pixels)
        detector_flat = detector.view(detector.size(0), -1)
        detector_features = self.detector_net(detector_flat)
        return th.cat([pixel_features, detector_features], dim=1)


class DetectorAugmentedDuelingCnnPolicy(DuelingCnnPolicy):
    """Dueling CNN policy tailored for detector-augmented observations."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", DetectorCombinedExtractor)
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)
