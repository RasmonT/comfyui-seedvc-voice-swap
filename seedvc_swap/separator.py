"""Vocal / background separation with BS-Roformer (overlap-add over fixed windows)."""
import torch

from .audio_utils import fit_length, resample

# Settings of the "BS-Roformer-Viperx-1297" checkpoint (model_bs_roformer_ep_317_sdr_12.9755).
SAMPLE_RATE = 44100
CHUNK_SIZE = 352800  # 8 s
NUM_OVERLAP = 4
MODEL_KWARGS = dict(
    dim=512,
    depth=12,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    flash_attn=True,
    dim_freqs_in=1025,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
)


def build_model():
    from .third_party.bs_roformer.bs_roformer import BSRoformer

    return BSRoformer(**MODEL_KWARGS)


def load_model(checkpoint_path, device):
    model = build_model()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _window(size, fade):
    window = torch.ones(size)
    ramp = torch.linspace(0.0, 1.0, fade)
    window[:fade] = ramp
    window[-fade:] = ramp.flip(0)
    return window


@torch.no_grad()
def separate_vocals(model, mix, device, fp16=True, progress=None, check_interrupt=None):
    """mix: float tensor [2, T] at 44.1 kHz. Returns the vocal stem, same shape, on CPU."""
    length = mix.shape[-1]
    step = CHUNK_SIZE // NUM_OVERLAP
    border = CHUNK_SIZE - step
    fade = CHUNK_SIZE // 10
    padded = length > 2 * border
    if padded:
        mix = torch.nn.functional.pad(mix[None], (border, border), mode="reflect")[0]
    total = mix.shape[-1]
    result = torch.zeros_like(mix)
    counter = torch.zeros(total)
    base_window = _window(CHUNK_SIZE, fade)
    use_amp = fp16 and torch.device(device).type == "cuda"
    starts = list(range(0, total, step))
    for index, start in enumerate(starts):
        if check_interrupt:
            check_interrupt()
        part = mix[:, start:start + CHUNK_SIZE]
        part_len = part.shape[-1]
        if part_len < CHUNK_SIZE:
            mode = "reflect" if part_len > CHUNK_SIZE // 2 else "constant"
            part = torch.nn.functional.pad(part[None], (0, CHUNK_SIZE - part_len), mode=mode)[0]
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
            out = model(part[None].to(device))
        out = out.float().cpu()[0]
        if out.dim() == 3:  # [stems, channels, time]
            out = out[0]
        window = base_window.clone()
        if start == 0:
            window[:fade] = 1.0
        if start + step >= total:
            window[-fade:] = 1.0
        result[:, start:start + part_len] += out[:, :part_len] * window[:part_len]
        counter[start:start + part_len] += window[:part_len]
        if progress:
            progress(index + 1, len(starts))
    result = result / counter.clamp_min(1e-8)
    if padded:
        result = result[:, border:-border]
    return torch.nan_to_num(result)


def split_vocals_background(model, audio, sample_rate, device, fp16=True, progress=None, check_interrupt=None):
    """audio: [C, T] (C = 1 or 2) at any sample rate.

    Returns (vocals, background), both [C, T] at the input rate and exactly the input
    length. background = audio - vocals, so vocals + background reproduces the input
    sample for sample and everything that is not voice keeps its original quality.
    """
    channels, length = audio.shape
    stereo = audio if channels == 2 else audio.repeat(2, 1)
    model_in = resample(stereo, sample_rate, SAMPLE_RATE)
    vocals = separate_vocals(model, model_in, device, fp16=fp16, progress=progress, check_interrupt=check_interrupt)
    vocals = fit_length(resample(vocals, SAMPLE_RATE, sample_rate), length)
    if channels == 1:
        vocals = vocals.mean(dim=0, keepdim=True)
    return vocals, audio - vocals
