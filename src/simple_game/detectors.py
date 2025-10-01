"""Entity detectors for Breakout observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

# ---------------------------------------------------------------------------


@dataclass
class DetectorSpec:
    """Configuration for detector-derived feature spaces."""

    features: Sequence[str]
    normalization: str = "builtin"
    output_mode: str = "concat"


class BaseDetector:
    """Base interface for detectors that produce structured features per env step."""

    def __init__(self, num_envs: int, spec: DetectorSpec) -> None:
        self.num_envs = num_envs
        self.spec = spec
        self.feature_names = list(spec.features)

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def reset(self, env_indices: Optional[Iterable[int]] = None) -> None:
        raise NotImplementedError

    def extract(self, observations: np.ndarray, env_refs: Sequence[object]) -> np.ndarray:
        """
        Produce detector features for each environment in the vectorized batch.

        :param observations: batched observation array (unused by RAM detector but required by interface).
        :param env_refs: concrete gym env instances (the same order as observations).
        :return: feature matrix shaped (num_envs, feature_dim).
        """

        raise NotImplementedError

    def observation_space(self) -> spaces.Box:
        raise NotImplementedError


class BreakoutRamDetector(BaseDetector):
    """Detector that reads Breakout RAM state to expose entity information."""

    # Atari 2600 Breakout RAM addresses documented via empirical probing
    RAM_ADDRS: Dict[str, int] = {
        "paddle_x": 72,
        "ball_x": 99,
        "ball_y": 101,
    }
    BRICK_RANGE = tuple(range(31, 43))

    def __init__(self, num_envs: int, spec: DetectorSpec) -> None:
        super().__init__(num_envs, spec)
        self._prev_ball = np.full((num_envs, 2), np.nan, dtype=np.float32)
        self._feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        dim = 0
        for feature in self.feature_names:
            if feature == "brick_bitmap":
                dim += len(self.BRICK_RANGE)
            else:
                dim += 1
        return dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)

    def reset(self, env_indices: Optional[Iterable[int]] = None) -> None:
        indices = range(self.num_envs) if env_indices is None else env_indices
        for idx in indices:
            self._prev_ball[idx] = np.array([np.nan, np.nan], dtype=np.float32)

    def extract(self, observations: np.ndarray, env_refs: Sequence[object]) -> np.ndarray:
        batch = np.zeros((self.num_envs, self.feature_dim), dtype=np.float32)
        for env_idx, env in enumerate(env_refs):
            ram = self._read_ram(env)
            features = self._extract_from_ram(env_idx, ram)
            batch[env_idx] = features
        return batch

    @staticmethod
    def _read_ram(env: object) -> np.ndarray:
        base = env
        while hasattr(base, "env"):
            base = base.env
        return np.array(base.unwrapped.ale.getRAM(), copy=False)

    def _extract_from_ram(self, env_idx: int, ram: np.ndarray) -> np.ndarray:
        values: List[float] = []
        ball_x_raw = float(ram[self.RAM_ADDRS["ball_x"]])
        ball_y_raw = float(ram[self.RAM_ADDRS["ball_y"]])
        paddle_raw = float(ram[self.RAM_ADDRS["paddle_x"]])

        # Empirically derived affine transforms mapping RAM to pixel coordinates
        ball_x = (ball_x_raw - 48.0) / 160.0
        ball_y = (ball_y_raw + 10.0) / 210.0
        paddle_x = paddle_raw / 200.0

        prev_ball = self._prev_ball[env_idx]
        if np.isnan(prev_ball[0]):
            vx = 0.0
            vy = 0.0
        else:
            vx = ball_x - prev_ball[0]
            vy = ball_y - prev_ball[1]
        self._prev_ball[env_idx] = np.array([ball_x, ball_y], dtype=np.float32)

        brick_vals = ram[list(self.BRICK_RANGE)].astype(np.float32) / 255.0

        for feature in self.feature_names:
            if feature == "paddle_x":
                values.append(paddle_x)
            elif feature == "ball_x":
                values.append(ball_x)
            elif feature == "ball_y":
                values.append(ball_y)
            elif feature == "ball_vx":
                values.append(vx)
            elif feature == "ball_vy":
                values.append(vy)
            elif feature == "brick_bitmap":
                values.extend(brick_vals)
            elif feature == "ball_valid":
                values.append(0.0 if np.isnan(ball_x_raw) else 1.0)
            else:
                raise ValueError(f"Unsupported RAM detector feature: {feature}")

        return np.asarray(values, dtype=np.float32)


class BreakoutPixelDetector(BaseDetector):
    """Extracts entity features from preprocessed Breakout frames using heuristics."""

    BALL_THRESHOLD = 130.0
    PADDLE_THRESHOLD = 110.0
    BALL_MAX_CLUSTER = 10
    BALL_MAX_Y_FRACTION = 0.9

    def __init__(self, num_envs: int, spec: DetectorSpec) -> None:
        super().__init__(num_envs, spec)
        self._feature_dim = len(self.feature_names)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)

    def reset(self, env_indices: Optional[Iterable[int]] = None) -> None:
        # Stateless detector
        return None

    def extract(self, observations: np.ndarray, env_refs: Sequence[object]) -> np.ndarray:
        features = np.zeros((self.num_envs, self.feature_dim), dtype=np.float32)
        for idx, obs in enumerate(observations):
            frame = self._select_latest_frame(obs)
            parsed = self._extract_from_frame(frame)
            features[idx] = parsed
        return features

    def _select_latest_frame(self, stacked_obs: np.ndarray) -> np.ndarray:
        if stacked_obs.ndim == 3:
            # (C, H, W) – assume last channel is most recent
            return stacked_obs[-1]
        if stacked_obs.ndim == 2:
            return stacked_obs
        raise ValueError(f"Unsupported observation shape for pixel detector: {stacked_obs.shape}")

    def _extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32)
        height, width = frame.shape
        paddle_region = frame[int(height * 0.65) :, :]

        paddle_mask = paddle_region > self.PADDLE_THRESHOLD
        paddle_x = 0.0
        if paddle_mask.any():
            ys, xs = np.where(paddle_mask)
            paddle_x = xs.mean() / width

        ball_mask = frame > self.BALL_THRESHOLD
        ball_x = 0.0
        ball_y = 0.0
        if ball_mask.any():
            cluster = self._select_ball_cluster(ball_mask)
            if cluster is not None:
                ys = cluster[:, 0]
                xs = cluster[:, 1]
                ball_x = xs.mean() / width
                ball_y = ys.mean() / height

        values: List[float] = []
        for feature in self.feature_names:
            if feature == "paddle_x":
                values.append(float(paddle_x))
            elif feature == "ball_x":
                values.append(float(ball_x))
            elif feature == "ball_y":
                values.append(float(ball_y))
            else:
                values.append(0.0)
        return np.asarray(values, dtype=np.float32)

    def _select_ball_cluster(self, mask: np.ndarray) -> Optional[np.ndarray]:
        coords = np.argwhere(mask)
        visited = np.zeros(mask.shape, dtype=bool)
        clusters: List[np.ndarray] = []
        for y, x in coords:
            if visited[y, x]:
                continue
            stack = [(y, x)]
            current: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                if visited[cy, cx] or not mask[cy, cx]:
                    continue
                visited[cy, cx] = True
                current.append((cy, cx))
                for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                    for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            stack.append((ny, nx))
            clusters.append(np.array(current, dtype=np.int32))

        if not clusters:
            return None

        height = mask.shape[0]
        filtered = [
            c
            for c in clusters
            if len(c) <= self.BALL_MAX_CLUSTER and c[:, 0].max() < height * self.BALL_MAX_Y_FRACTION
        ]
        if not filtered:
            return None
        filtered.sort(key=lambda c: c[:, 0].mean())
        return filtered[0]


def build_detector(num_envs: int, detector_cfg: Dict[str, object]) -> BaseDetector:
    """Factory for detector instances based on config dictionaries."""

    spec = DetectorSpec(
        features=detector_cfg.get("features", []),
        normalization=str(detector_cfg.get("normalization", "builtin")),
        output_mode=str(detector_cfg.get("output_mode", "concat")),
    )

    detector_type = detector_cfg.get("type", "ram_tap")
    if detector_type == "ram_tap":
        return BreakoutRamDetector(num_envs=num_envs, spec=spec)
    if detector_type == "pixel":
        return BreakoutPixelDetector(num_envs=num_envs, spec=spec)
    raise ValueError(f"Unsupported detector type: {detector_type}")


class DetectorAugmentedVecEnv(VecEnvWrapper):
    """VecEnv wrapper that augments observations with detector-derived features."""

    def __init__(self, venv, detector: BaseDetector, key_pixels: str = "pixels", key_features: str = "detector") -> None:
        super().__init__(venv)
        self.detector = detector
        self.key_pixels = key_pixels
        self.key_features = key_features

        self.detector.reset()

        pixels_space = venv.observation_space
        if isinstance(pixels_space, spaces.Dict):
            raise TypeError("DetectorAugmentedVecEnv expects base observation space to be a Box")

        features_space = detector.observation_space()
        self.observation_space = spaces.Dict(
            {
                self.key_pixels: pixels_space,
                self.key_features: features_space,
            }
        )

    def reset(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        obs = self.venv.reset()
        self.detector.reset()
        features = self.detector.extract(obs, self._env_refs())
        return {self.key_pixels: obs, self.key_features: features}

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:  # type: ignore[override]
        obs, rewards, dones, infos = self.venv.step_wait()
        features = self.detector.extract(obs, self._env_refs())
        for idx, done in enumerate(dones):
            if done:
                self.detector.reset([idx])
        return {self.key_pixels: obs, self.key_features: features}, rewards, dones, infos

    def _env_refs(self) -> Sequence[object]:
        # VecEnvWrapper maintains envs attribute for Dummy/SubprocVecEnv
        return getattr(self.venv, "envs", [])
