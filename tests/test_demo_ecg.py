import numpy as np


def test_generate_demo_ecg_returns_valid_signal():
    from app import generate_demo_ecg

    signal, fs = generate_demo_ecg(duration=30, fs=360)

    assert fs == 360
    assert signal.shape == (30 * 360,)
    assert np.isfinite(signal).all()
    assert np.abs(np.mean(signal)) < 1.0
