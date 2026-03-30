import numpy as np
import torch

from ai_ml_methodology.data import ballistic_state, range_doppler_snapshot


def test_ballistic_state_output_shape():
    t = np.array([0.0, 1.0, 2.0])
    x, y, vx, vy = ballistic_state(t, v0=100.0, theta_deg=45.0)

    assert x.shape == t.shape
    assert y.shape == t.shape
    assert vx.shape == t.shape
    assert vy.shape == t.shape
    assert np.isclose(x[0], 0.0)


def test_range_doppler_snapshot_expected_dimensions():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.5, 1.0])
    vx = np.array([100.0, 100.0, 100.0])
    vy = np.array([100.0, 90.0, 80.0])
    snap = range_doppler_snapshot(x, y, vx, vy)

    assert snap.shape == (3, 4)
    assert torch.is_tensor(torch.from_numpy(snap))
