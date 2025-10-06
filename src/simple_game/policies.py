"""Custom feature extractors and policy helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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


class SlotAttention(nn.Module):
    """Minimal Slot Attention module following Locatello et al. (2020)."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        iterations: int = 3,
        mlp_hidden_dim: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iterations = iterations
        self.eps = eps

        self.scale = slot_dim ** -0.5
        self.slots_mu = nn.Parameter(th.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(th.randn(1, 1, slot_dim))

        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

    def forward(self, inputs: th.Tensor, *, return_attn: bool = False) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """Run slot attention on a set of inputs (B, N, D)."""

        batch_size, num_inputs, _ = inputs.shape

        inputs = self.norm_inputs(inputs)

        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = F.softplus(self.slots_sigma) + self.eps
        slots = mu + sigma * th.randn_like(mu)

        k = self.project_k(inputs)
        v = self.project_v(inputs)

        last_attn: th.Tensor | None = None
        for _ in range(self.iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)

            attn_logits = th.matmul(k, q.transpose(-1, -2)) * self.scale
            attn = F.softmax(attn_logits, dim=1)
            attn = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
            last_attn = attn

            updates = th.matmul(attn.transpose(1, 2), v)

            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))

        if return_attn:
            if last_attn is None:
                raise RuntimeError("slot attention did not compute attention weights")
            return slots, last_attn

        return slots


class SlotAttentionExtractor(BaseFeaturesExtractor):
    """Encode pixels into object-centric slots using Slot Attention."""

    def __init__(
        self,
        observation_space: spaces.Box,
        num_slots: int = 6,
        slot_dim: int = 64,
        iterations: int = 3,
        mlp_hidden_dim: int = 128,
        aux_entropy_weight: float = 0.0,
        aux_slot_variance_weight: float = 0.0,
        aux_reconstruction_weight: float = 0.0,
        aux_reconstruction_downsample: bool = True,
        log_attention_metrics: bool = False,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("SlotAttentionExtractor expects a Box observation space")

        channels, height, width = SimpleNatureCNN._extract_shape(observation_space)

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        features_dim = num_slots * slot_dim
        super().__init__(observation_space, features_dim)

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, slot_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        conv_out = self.encoder_cnn(th.zeros(1, channels, height, width))
        _, _, encoded_h, encoded_w = conv_out.shape
        grid = self._build_position_grid(encoded_h, encoded_w)
        self.register_buffer("position_grid", grid, persistent=False)

        self.input_proj = nn.Linear(slot_dim + 2, slot_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            iterations=iterations,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self.aux_entropy_weight = float(aux_entropy_weight)
        self.aux_slot_variance_weight = float(aux_slot_variance_weight)
        self.aux_reconstruction_weight = float(aux_reconstruction_weight)
        self.aux_reconstruction_downsample = aux_reconstruction_downsample
        self.log_attention_metrics = log_attention_metrics

        self._needs_attention = (
            self.aux_entropy_weight > 0.0
            or self.aux_slot_variance_weight > 0.0
            or self.log_attention_metrics
        )

        self._use_reconstruction = self.aux_reconstruction_weight > 0.0

        if self._use_reconstruction:
            self.decoder = nn.Sequential(
                nn.Linear(features_dim, slot_dim * encoded_h * encoded_w),
                nn.ReLU(),
                nn.Unflatten(1, (slot_dim, encoded_h, encoded_w)),
                nn.ConvTranspose2d(slot_dim, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            )
        else:
            self.decoder = None

        self._cached_aux_loss: th.Tensor | None = None
        self._cached_aux_logs: dict[str, float] = {}

    @staticmethod
    def _build_position_grid(height: int, width: int) -> th.Tensor:
        ys = th.linspace(-1.0, 1.0, height)
        xs = th.linspace(-1.0, 1.0, width)
        grid_y, grid_x = th.meshgrid(ys, xs, indexing="ij")
        grid = th.stack([grid_x, grid_y], dim=-1)
        return grid.reshape(-1, 2)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.encoder_cnn(observations.float())
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).transpose(1, 2)

        pos = self.position_grid.unsqueeze(0).expand(batch_size, -1, -1)
        x = th.cat([x, pos], dim=-1)
        x = self.input_proj(x)

        slots_output = self.slot_attention(x, return_attn=self._needs_attention)
        if self._needs_attention:
            slots, attn = slots_output  # type: ignore[misc]
        else:
            slots = slots_output  # type: ignore[assignment]
            attn = None

        flat_slots = slots.reshape(batch_size, -1)

        self._cached_aux_loss = None
        self._cached_aux_logs = {}

        if self.training:
            aux_losses: list[th.Tensor] = []

            if attn is not None and (self.aux_entropy_weight > 0.0 or self.log_attention_metrics):
                slot_attention = attn.transpose(1, 2)  # (B, num_slots, num_inputs)
                entropy = -(slot_attention * (slot_attention.clamp_min(1e-8).log())).sum(dim=-1)
                entropy_mean = entropy.mean()
                self._cached_aux_logs["slot_attention_entropy"] = float(entropy_mean.detach().cpu())
                if self.aux_entropy_weight > 0.0:
                    aux_losses.append(-self.aux_entropy_weight * entropy_mean)

            if self.aux_slot_variance_weight > 0.0 or self.log_attention_metrics:
                slot_std = slots.std(dim=1).mean()
                self._cached_aux_logs.setdefault("slot_feature_std", float(slot_std.detach().cpu()))
                if self.aux_slot_variance_weight > 0.0:
                    aux_losses.append(-self.aux_slot_variance_weight * slot_std)

            if self._use_reconstruction and self.decoder is not None:
                recon = self.decoder(flat_slots)
                target = observations.float()
                if self.aux_reconstruction_downsample:
                    target = F.interpolate(target, size=recon.shape[-2:], mode="bilinear", align_corners=False)
                recon_loss = F.mse_loss(recon, target)
                self._cached_aux_logs["slot_reconstruction_loss"] = float(recon_loss.detach().cpu())
                aux_losses.append(self.aux_reconstruction_weight * recon_loss)

            if aux_losses:
                total_loss = aux_losses[0]
                for extra in aux_losses[1:]:
                    total_loss = total_loss + extra
                self._cached_aux_loss = total_loss

        return flat_slots

    def pop_auxiliary_loss(self) -> tuple[th.Tensor | None, dict[str, float]]:
        loss = self._cached_aux_loss
        logs = self._cached_aux_logs
        self._cached_aux_loss = None
        self._cached_aux_logs = {}
        return loss, logs


class SlotAttentionDuelingCnnPolicy(DuelingCnnPolicy):
    """Dueling DQN policy that relies on Slot Attention features."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", SlotAttentionExtractor)
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)
