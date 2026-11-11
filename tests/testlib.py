"""Measurement helpers for the test scripts (not used by the node itself)."""
import math
import shutil
import subprocess
import threading
import time

import numpy as np
import torch
import torchaudio


# ----------------------------------------------------------------------------- signals
def synth_music(seconds, sample_rate, seed=7):
    """Deterministic stereo backing track (chords, bass, kick, hi-hat) used when no real music file is given.

    It is only a stand-in with a known waveform; a real music file gives more realistic numbers.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sample_rate))
    t = np.arange(n) / sample_rate
    bpm = 100.0
    beat = 60.0 / bpm
    chords = [(220.0, 277.18, 329.63), (174.61, 220.0, 261.63), (130.81, 164.81, 196.0), (196.0, 246.94, 293.66)]
    left = np.zeros(n)
    right = np.zeros(n)
    bar = beat * 4
    for index in range(int(math.ceil(seconds / bar))):
        start = int(index * bar * sample_rate)
        stop = min(n, int((index + 1) * bar * sample_rate))
        if start >= n:
            break
        tt = t[start:stop] - t[start]
        env = np.minimum(1.0, tt / 0.05) * np.exp(-tt / 2.5)
        chord = chords[index % len(chords)]
        for k, freq in enumerate(chord):
            tone = sum(np.sin(2 * np.pi * freq * h * tt + 0.3 * k) / h for h in (1, 2, 3, 4, 5))
            pan = 0.3 + 0.2 * k
            left[start:stop] += tone * env * (1 - pan) * 0.12
            right[start:stop] += tone * env * pan * 0.12
        bass = np.sin(2 * np.pi * (chord[0] / 2) * tt) * env * 0.25
        left[start:stop] += bass
        right[start:stop] += bass
    for index in range(int(seconds / beat) + 1):
        start = int(index * beat * sample_rate)
        length = min(n - start, int(0.25 * sample_rate))
        if length <= 0:
            break
        tt = np.arange(length) / sample_rate
        kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 14) * 0.5
        left[start:start + length] += kick
        right[start:start + length] += kick
        hat_start = start + int(beat / 2 * sample_rate)
        hat_len = min(n - hat_start, int(0.05 * sample_rate))
        if hat_len > 0:
            hat = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / sample_rate * 90) * 0.12
            left[hat_start:hat_start + hat_len] += hat * 0.6
            right[hat_start:hat_start + hat_len] += hat
    music = torch.from_numpy(np.stack([left, right])).float()
    return music / music.abs().max() * 0.5


def rms(x):
    return float(torch.sqrt(torch.mean(x.double() ** 2)))


def sdr_db(reference, estimate):
    """10*log10(|ref|^2 / |ref - est|^2): higher means the estimate is closer to the reference."""
    reference = reference.double()
    noise = reference - estimate.double()
    return float(10 * torch.log10(reference.pow(2).sum().clamp_min(1e-20) / noise.pow(2).sum().clamp_min(1e-20)))


def envelope_lag_ms(a, b, sample_rate, max_ms=250.0, hop_ms=2.0):
    """Lag (ms) that best aligns the loudness envelope of b to a. Positive: b is late."""
    hop = max(1, int(sample_rate * hop_ms / 1000))
    def envelope(x):
        usable = (x.numel() // hop) * hop
        e = x[:usable].double().reshape(-1, hop).pow(2).mean(dim=1).sqrt()
        return e - e.mean()
    ea, eb = envelope(a), envelope(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n].numpy(), eb[:n].numpy()
    max_lag = int(max_ms / hop_ms)
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            value = float(np.dot(ea[: n - lag], eb[lag:]))
        else:
            value = float(np.dot(ea[-lag:], eb[: n + lag]))
        if value > best:
            best, best_lag = value, lag
    return best_lag * hop_ms


# ---------------------------------------------------------------- speaker embeddings
def _fbank(mono, sample_rate):
    wave = mono[None].float()
    if sample_rate != 16000:
        wave = torchaudio.functional.resample(wave, sample_rate, 16000)
    feats = torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, dither=0, sample_frequency=16000)
    return feats - feats.mean(dim=0, keepdim=True)


class ERes2NetEmbedder:
    """3D-Speaker ERes2Net (ONNX). Independent of Seed-VC, so it is a fair judge of voice similarity."""
    name = "eres2net"

    def __init__(self, onnx_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def __call__(self, mono, sample_rate):
        feats = _fbank(mono, sample_rate)[None].numpy().astype(np.float32)
        return torch.from_numpy(self.session.run(None, {"x": feats})[0][0])


class CampplusEmbedder:
    """CAM++ as used by Seed-VC itself. Biased (the converter is conditioned on it); reported for reference only."""
    name = "campplus"

    def __init__(self, checkpoint):
        from seedvc_swap.third_party.seed_vc.campplus.DTDNN import CAMPPlus

        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, mono, sample_rate):
        return self.model(_fbank(mono, sample_rate)[None])[0]


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.double()[None], b.double()[None])[0])


# ------------------------------------------------------------------------ GPU memory
class GpuMemoryMonitor:
    """Polls nvidia-smi so memory of other processes (the old bridge runs in its own) is measured the same way."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.available = shutil.which("nvidia-smi") is not None
        self.peak = None
        self.baseline = None
        self._stop = threading.Event()
        self._thread = None

    def _read(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def __enter__(self):
        if self.available:
            self.baseline = self._read()
            self.peak = self.baseline
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            value = self._read()
            if value is not None and (self.peak is None or value > self.peak):
                self.peak = value
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def peak_delta_mb(self):
        if self.peak is None or self.baseline is None:
            return None
        return round(self.peak - self.baseline, 1)
