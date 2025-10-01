import numpy as np
import pytest

from src.simple_game.detectors import (
    BreakoutPixelDetector,
    BreakoutRamDetector,
    DetectorSpec,
)


class DummyAle:
    def __init__(self, ram):
        self._ram = np.asarray(ram, dtype=np.uint8)

    def getRAM(self):
        return self._ram


class DummyEnv:
    def __init__(self, ram):
        self.unwrapped = self
        self.ale = DummyAle(ram)


def test_ram_detector_basic_features():
    ram = np.zeros(128, dtype=np.uint8)
    ram[72] = 100  # paddle
    ram[99] = 80   # ball x raw
    ram[101] = 120  # ball y raw
    ram[31:43] = 255

    spec = DetectorSpec(features=["paddle_x", "ball_x", "ball_y", "ball_vx", "ball_vy", "brick_bitmap"])
    detector = BreakoutRamDetector(num_envs=1, spec=spec)

    obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
    features = detector.extract(obs, [DummyEnv(ram)])

    assert features.shape == (1, detector.feature_dim)
    paddle_x, ball_x, ball_y = features[0, :3]
    assert 0.0 <= paddle_x <= 1.0
    assert 0.0 <= ball_x <= 1.0
    assert 0.0 <= ball_y <= 1.0
    # velocities zero on first step
    assert features[0, 3] == pytest.approx(0.0)
    assert features[0, 4] == pytest.approx(0.0)
    # brick bitmap normalized
    bricks = features[0, 5:17]
    assert np.allclose(bricks, 1.0)

    # second extraction should yield velocity
    ram[99] = 90
    ram[101] = 130
    features = detector.extract(obs, [DummyEnv(ram)])
    assert features[0, 3] != 0.0
    assert features[0, 4] != 0.0


def test_pixel_detector_thresholding():
    spec = DetectorSpec(features=["paddle_x", "ball_x", "ball_y"])
    detector = BreakoutPixelDetector(num_envs=1, spec=spec)

    frame = np.zeros((84, 84), dtype=np.float32)
    frame[70:74, 30:54] = 140  # paddle region
    frame[40:42, 20:22] = 145  # ball
    stacked = np.stack([frame] * 4)
    obs = np.expand_dims(stacked, axis=0)

    features = detector.extract(obs, [None])
    paddle_x, ball_x, ball_y = features[0]
    assert 0.3 <= paddle_x <= 0.7
    assert 0.1 <= ball_x <= 0.5
    assert 0.3 <= ball_y <= 0.6
