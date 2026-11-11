"""Audio helpers: decoding voice samples, resampling, length fitting, loudness matching."""
import math

import torch
import torchaudio.functional as AF


class AudioError(RuntimeError):
    """A voice sample or input audio cannot be used."""


def load_audio_file(path):
    """Decode any audio file ffmpeg understands. Returns (float32 tensor [channels, samples], sample_rate)."""
    import av  # PyAV ships with ComfyUI

    try:
        container = av.open(str(path))
    except Exception as exc:  # av raises several unrelated exception types
        raise AudioError(f"Cannot open the audio file: {path}\n  {exc}") from exc
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioError(f"The file has no audio track: {path}")
        sample_rate = int(stream.codec_context.sample_rate or stream.sample_rate)
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(torch.from_numpy(out.to_ndarray().copy()))
        for out in resampler.resample(None):
            chunks.append(torch.from_numpy(out.to_ndarray().copy()))
    finally:
        container.close()
    if not chunks:
        raise AudioError(f"The file contains no audio: {path}")
    wave = torch.cat(chunks, dim=1).float()
    return wave, sample_rate


def resample(wave, src_rate, dst_rate):
    """High quality band-limited resampling (Kaiser windowed sinc); identity when the rates match."""
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate == dst_rate:
        return wave
    return AF.resample(wave, src_rate, dst_rate, lowpass_filter_width=64, rolloff=0.9475937167399596,
                       resampling_method="sinc_interp_kaiser", beta=14.769656459379492)


def fit_length(wave, length):
    """Trim or zero-pad the last dimension to exactly `length` samples."""
    have = wave.shape[-1]
    if have == length:
        return wave
    if have > length:
        return wave[..., :length]
    return torch.nn.functional.pad(wave, (0, length - have))


def to_mono(wave):
    return wave.mean(dim=0) if wave.dim() == 2 else wave


def frame_rms(wave, sample_rate, frame_ms=20.0):
    """RMS per non-overlapping frame of a mono signal."""
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    usable = (wave.shape[-1] // frame) * frame
    if usable == 0:
        return torch.sqrt(torch.mean(wave.double() ** 2, dim=-1, keepdim=True)), wave.shape[-1] or 1
    frames = wave[..., :usable].double().reshape(*wave.shape[:-1], -1, frame)
    return torch.sqrt(torch.mean(frames ** 2, dim=-1)), frame


def speech_activity(mono, sample_rate, frame_ms=20.0, rel_db=-35.0):
    """Boolean mask per frame: True where the signal is within `rel_db` of its loud level."""
    rms, frame = frame_rms(mono, sample_rate, frame_ms)
    if rms.numel() == 0:
        return torch.zeros(0, dtype=torch.bool), frame
    loud = torch.quantile(rms, 0.95)
    if float(loud) <= 1e-7:
        return torch.zeros_like(rms, dtype=torch.bool), frame
    threshold = loud * (10.0 ** (rel_db / 20.0))
    return rms > threshold, frame


def masked_rms(mono, mask, frame):
    """RMS of a mono signal over the frames selected by `mask`."""
    n = min(mask.numel(), mono.shape[-1] // frame)
    if n == 0:
        return 0.0
    frames = mono[: n * frame].double().reshape(n, frame)
    selected = frames[mask[:n]]
    if selected.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(selected ** 2)))


def match_loudness(converted, original, sample_rate, region=None):
    """Scale `converted` so the new voice is as loud as the original speech.

    converted: mono tensor [T]. original: [C, T] (the original speech, same length).
    Loudness is the RMS over the frames where the ORIGINAL speech is active, so
    pauses and noise do not skew the measurement. One gain per channel keeps the
    original left/right balance. `region` (bool or 0..1 tensor [T]) limits the
    measurement to one speaker's part of the clip. Returns ([C, T] tensor, list of gains).
    """
    channels = original.shape[0]
    mono = to_mono(original)
    if region is not None:
        frame = max(1, int(round(sample_rate * 0.02)))
        usable = (mono.shape[-1] // frame) * frame
        inside = region[:usable].float().reshape(-1, frame).mean(dim=1) > 0.5
        rms, _ = frame_rms(mono, sample_rate)
        if inside.any() and float(rms[inside].max()) > 1e-7:
            loud = torch.quantile(rms[inside], 0.95)
            mask = (rms > loud * (10.0 ** (-35.0 / 20.0))) & inside
        else:
            mask = torch.zeros_like(inside)
    else:
        mask, frame = speech_activity(mono, sample_rate)
    conv_rms = masked_rms(converted, mask, frame)
    out = []
    gains = []
    for c in range(channels):
        orig_rms = masked_rms(original[c], mask, frame)
        gain = 1.0 if conv_rms <= 1e-7 or orig_rms <= 1e-7 else orig_rms / conv_rms
        gains.append(gain)
        out.append(converted * gain)
    return torch.stack(out, dim=0), gains


def db(value):
    return -math.inf if value <= 0 else 20.0 * math.log10(value)
