"""Seed-VC v1 zero-shot voice conversion (speech 22.05 kHz, singing 44.1 kHz with F0).

The inference procedure follows `inference.py` from https://github.com/Plachtaa/seed-vc
(GPL-3.0), rewritten as a reusable class with local model files, a caller-controlled
seed and no file round trips.
"""
import gc
import json

import numpy as np
import torch
import torchaudio

from .audio_utils import AudioError

MAX_REFERENCE_SECONDS = 25.0
MIN_REFERENCE_SECONDS = 1.0
CONTEXT_SECONDS = 30
OVERLAP_FRAMES = 16
WHISPER_WINDOW = 16000 * 30
WHISPER_OVERLAP_SECONDS = 5
PITCH_MODES = ("original", "octave", "match")


def _crossfade(previous_tail, chunk, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    n = min(len(chunk), overlap)
    chunk[:n] = chunk[:n] * fade_in[:n] + (previous_tail[-overlap:] * fade_out)[:n]
    return chunk


class SeedVCEngine:
    def __init__(self, mode, paths, device, fp16=True):
        if mode not in ("speech", "singing"):
            raise ValueError(f"Unknown mode {mode!r}")
        self.mode = mode
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.f0_condition = mode == "singing"
        self._load(paths)

    # ------------------------------------------------------------------ loading
    def _load(self, paths):
        import yaml

        from .third_party.seed_vc.commons import build_model, load_checkpoint, recursive_munch
        from .third_party.seed_vc.campplus.DTDNN import CAMPPlus
        from .third_party.seed_vc.bigvgan.bigvgan import BigVGAN
        from .third_party.seed_vc.bigvgan.env import AttrDict
        from .third_party.seed_vc.audio import mel_spectrogram

        dit_key = "dit_singing" if self.f0_condition else "dit_speech"
        vocoder_key = "bigvgan44" if self.f0_condition else "bigvgan22"
        with open(paths[dit_key + "_config"], "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        spect = config["preprocess_params"]["spect_params"]
        self.sr = int(config["preprocess_params"]["sr"])
        self.hop_length = int(spect["hop_length"])

        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        model = build_model(model_params, stage="DiT")
        model, _, _, _ = load_checkpoint(model, None, paths[dit_key], load_only_params=True,
                                         ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval().to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.model = model
        self.in_channels = int(model_params.DiT.in_channels)

        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(paths["campplus"], map_location="cpu", weights_only=True))
        self.campplus = campplus.eval().to(self.device)

        with open(paths[vocoder_key + "_config"], "r", encoding="utf-8") as fh:
            hparams = AttrDict(json.load(fh))
        vocoder = BigVGAN(hparams, use_cuda_kernel=False)
        state = torch.load(paths[vocoder_key], map_location="cpu", weights_only=True)["generator"]
        vocoder.load_state_dict(state)
        vocoder.remove_weight_norm()
        self.vocoder = vocoder.eval().to(self.device)

        from pathlib import Path

        from transformers import WhisperFeatureExtractor, WhisperModel

        whisper_dir = str(Path(paths["whisper"]).parent)
        whisper = WhisperModel.from_pretrained(whisper_dir, local_files_only=True)
        self.whisper_encoder = whisper.encoder.eval().to(
            self.device, torch.float16 if self.fp16 else torch.float32)
        del whisper
        self.whisper_features = WhisperFeatureExtractor.from_pretrained(whisper_dir, local_files_only=True)

        self.rmvpe = None
        if self.f0_condition:
            from .third_party.seed_vc.rmvpe import RMVPE

            self.rmvpe = RMVPE(paths["rmvpe"], is_half=False, device=self.device)

        fmax = spect.get("fmax", "None")
        self._mel_args = dict(n_fft=spect["n_fft"], win_size=spect["win_length"], hop_size=spect["hop_length"],
                              num_mels=spect["n_mels"], sampling_rate=self.sr, fmin=spect.get("fmin", 0),
                              fmax=None if fmax == "None" else 8000, center=False)
        self._mel_fn = mel_spectrogram

    def unload(self):
        for name in ("model", "campplus", "vocoder", "whisper_encoder", "whisper_features", "rmvpe"):
            if hasattr(self, name):
                setattr(self, name, None)
        from .third_party.seed_vc import audio as _audio

        _audio.mel_basis.clear()  # module-level caches that hold small tensors on the GPU
        _audio.hann_window.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- inference
    def _semantic(self, wave_16k):
        """Whisper encoder features for a [1, T] 16 kHz signal of at most 30 s."""
        inputs = self.whisper_features([wave_16k.squeeze(0).cpu().numpy()], return_tensors="pt",
                                       return_attention_mask=True, sampling_rate=16000)
        features = inputs.input_features.to(self.device, self.whisper_encoder.conv1.weight.dtype)
        hidden = self.whisper_encoder(features).last_hidden_state.to(torch.float32)
        return hidden[:, : wave_16k.size(-1) // 320 + 1]

    def _semantic_long(self, wave_16k):
        if wave_16k.size(-1) <= WHISPER_WINDOW:
            return self._semantic(wave_16k)
        overlap = 16000 * WHISPER_OVERLAP_SECONDS
        pieces = []
        buffer = None
        traversed = 0
        while traversed < wave_16k.size(-1):
            if buffer is None:
                chunk = wave_16k[:, traversed:traversed + WHISPER_WINDOW]
            else:
                chunk = torch.cat([buffer, wave_16k[:, traversed:traversed + WHISPER_WINDOW - overlap]], dim=-1)
            hidden = self._semantic(chunk)
            pieces.append(hidden if traversed == 0 else hidden[:, 50 * WHISPER_OVERLAP_SECONDS:])
            buffer = chunk[:, -overlap:]
            traversed += WHISPER_WINDOW if traversed == 0 else chunk.size(-1) - overlap
        return torch.cat(pieces, dim=1)

    @torch.inference_mode()
    def convert(self, source, reference, steps=25, cfg=0.5, seed=0, pitch_mode="original", pitch_shift=0,
                progress=None, check_interrupt=None):
        """source, reference: mono float tensors [T] already at `self.sr`.

        Returns a mono float32 CPU tensor at `self.sr`. Its length is the source length
        rounded down to a whole number of mel frames (hop_length samples).

        pitch_mode (singing only): "original" keeps the sung pitch, "octave" moves it by whole
        octaves towards the voice sample's range (stays in key), "match" moves it by the exact
        difference of the median pitches (closest to the sample's range, may leave the key).
        """
        if pitch_mode not in PITCH_MODES:
            raise ValueError(f"Unknown pitch_mode {pitch_mode!r}; expected one of {PITCH_MODES}.")
        sr, hop, device = self.sr, self.hop_length, self.device
        if reference.numel() < int(MIN_REFERENCE_SECONDS * sr):
            raise AudioError(f"The voice sample is too short ({reference.numel() / sr:.2f} s). "
                             f"Use at least {MIN_REFERENCE_SECONDS:.0f} s of clean speech; 5-25 s works best.")
        if source.numel() < hop * 4:
            raise AudioError(f"The input audio is too short to convert ({source.numel() / sr:.3f} s).")
        source = source.float().clamp(-1.0, 1.0).unsqueeze(0).to(device)
        reference = reference.float().clamp(-1.0, 1.0)[: int(sr * MAX_REFERENCE_SECONDS)].unsqueeze(0).to(device)

        source_16k = torchaudio.functional.resample(source, sr, 16000)
        reference_16k = torchaudio.functional.resample(reference, sr, 16000)
        content = self._semantic_long(source_16k)
        prompt_content = self._semantic(reference_16k)

        mel = self._mel_fn(source, **self._mel_args)
        prompt_mel = self._mel_fn(reference, **self._mel_args)
        target_lengths = torch.LongTensor([mel.size(2)]).to(device)
        prompt_lengths = torch.LongTensor([prompt_mel.size(2)]).to(device)

        fbank = torchaudio.compliance.kaldi.fbank(reference_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        style = self.campplus(fbank.unsqueeze(0))

        f0_prompt = None
        f0_source = None
        if self.f0_condition:
            f0_ref = torch.from_numpy(self.rmvpe.infer_from_audio(reference_16k[0], thred=0.03)).to(device)[None]
            f0_src = torch.from_numpy(self.rmvpe.infer_from_audio(source_16k[0], thred=0.03)).to(device)[None]
            voiced_ref = f0_ref[f0_ref > 1]
            voiced_src = f0_src[f0_src > 1]
            log_f0_src = torch.log(f0_src + 1e-5)
            shifted = log_f0_src.clone()
            if pitch_mode != "original" and voiced_ref.numel() and voiced_src.numel():
                median_ref = torch.median(torch.log(voiced_ref + 1e-5))
                median_src = torch.median(torch.log(voiced_src + 1e-5))
                offset = median_ref - median_src  # natural log of the frequency ratio
                if pitch_mode == "octave":
                    octave = float(np.log(2.0))
                    offset = torch.round(offset / octave) * octave
                shifted[f0_src > 1] = log_f0_src[f0_src > 1] + offset
            f0_source = torch.exp(shifted)
            if pitch_shift:
                f0_source[f0_src > 1] = f0_source[f0_src > 1] * (2 ** (pitch_shift / 12))
            f0_prompt = f0_ref

        cond = self.model.length_regulator(content, ylens=target_lengths, n_quantizers=3, f0=f0_source)[0]
        prompt_cond = self.model.length_regulator(prompt_content, ylens=prompt_lengths, n_quantizers=3,
                                                  f0=f0_prompt)[0]

        max_context = sr // hop * CONTEXT_SECONDS
        max_source_window = max_context - prompt_mel.size(2)
        overlap_wave = OVERLAP_FRAMES * hop
        total_frames = cond.size(1)
        total_chunks = max(1, -(-(total_frames - OVERLAP_FRAMES) // max(1, max_source_window - OVERLAP_FRAMES)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

        processed = 0
        chunk_index = 0
        pieces = []
        previous_tail = None
        while processed < total_frames:
            if check_interrupt:
                check_interrupt()
            chunk_cond = cond[:, processed:processed + max_source_window]
            is_last = processed + max_source_window >= total_frames
            cat_cond = torch.cat([prompt_cond, chunk_cond], dim=1)
            noise = torch.randn([1, self.in_channels, cat_cond.size(1)], generator=generator, dtype=torch.float32)

            def on_step(step, total, _chunk=chunk_index):
                if progress:
                    progress(_chunk * total + step, total_chunks * total)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=self.fp16):
                target = self.model.cfm.inference(
                    cat_cond, torch.LongTensor([cat_cond.size(1)]).to(device), prompt_mel, style, None,
                    int(steps), inference_cfg_rate=float(cfg), noise=noise, step_callback=on_step)
                target = target[:, :, prompt_mel.size(-1):]
            wave = self.vocoder(target.float()).squeeze()[None, :]
            if processed == 0:
                if is_last:
                    pieces.append(wave[0].float().cpu().numpy())
                    break
                pieces.append(wave[0, :-overlap_wave].float().cpu().numpy())
                previous_tail = wave[0, -overlap_wave:].float().cpu().numpy()
            elif is_last:
                pieces.append(_crossfade(previous_tail, wave[0].float().cpu().numpy(), overlap_wave))
                break
            else:
                pieces.append(_crossfade(previous_tail, wave[0, :-overlap_wave].float().cpu().numpy(), overlap_wave))
                previous_tail = wave[0, -overlap_wave:].float().cpu().numpy()
            processed += target.size(2) - OVERLAP_FRAMES
            chunk_index += 1
        return torch.from_numpy(np.concatenate(pieces)).float()
