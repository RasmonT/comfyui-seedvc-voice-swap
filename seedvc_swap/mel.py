"""Mel filterbank helpers that replace the two librosa functions Seed-VC needs.

Written from the published Slaney / HTK mel formulas so the pack does not have
to install librosa. `tests/test_units.py` checks the output against librosa.
"""
import numpy as np


def _hz_to_mel(freq, htk):
    freq = np.asanyarray(freq, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    safe = np.maximum(freq, 1e-10)
    return np.where(log_region, min_log_mel + np.log(safe / min_log_hz) / logstep, mels)


def _mel_to_hz(mels, htk):
    mels = np.asanyarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Triangular mel filters with Slaney area normalisation, shape (n_mels, 1 + n_fft // 2), float32."""
    if fmax is None:
        fmax = float(sr) / 2
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, float(sr) / 2, n_freqs)
    mel_points = np.linspace(_hz_to_mel(fmin, htk), _hz_to_mel(fmax, htk), n_mels + 2)
    hz_points = _mel_to_hz(mel_points, htk)
    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def pad_center(data, *, size):
    """Zero-pad a 1-D array symmetrically to `size` samples."""
    data = np.asarray(data)
    n = data.shape[-1]
    if n > size:
        raise ValueError(f"Target size ({size}) must be at least input size ({n})")
    left = (size - n) // 2
    return np.pad(data, (left, size - n - left), mode="constant")
