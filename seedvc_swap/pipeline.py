"""Voice swap pipeline: (separate) -> (who speaks when) -> convert -> loudness match -> (remix).

The output always has exactly the shape of the input.
"""
import gc
import time

import torch

from . import diarize as dz
from . import separator
from .audio_utils import AudioError, db, fit_length, load_audio_file, match_loudness, resample, to_mono
from .converter import SeedVCEngine

SPEAKER_SEED_STEP = 7919  # speaker 2 uses seed + 7919, so each voice gets its own noise


def _free():
    gc.collect()
    if torch.cuda.is_available():
        try:  # FFT plans created by the STFTs keep GPU memory until the plan cache is cleared
            torch.backends.cuda.cufft_plan_cache.clear()
        except Exception:
            pass
        clear_blas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_blas is not None:  # PyTorch keeps a per-thread cuBLAS workspace allocated otherwise
            try:
                clear_blas()
            except Exception:
                pass
        torch.cuda.empty_cache()


def _one_speaker_notice(reason, targets):
    """Console block for 'two speakers expected, one found': what happened and what to change."""
    if targets[0] is not None and targets[1] is not None:
        outcome = "ALL speech was converted with voice 1. Voice 2 was NOT used."
    elif targets[0] is not None:
        outcome = "All speech was converted with voice 1 (no second speaker to leave untouched)."
    else:
        outcome = "Voice 1 is off, so NOTHING was converted: the audio is unchanged."
    bar = "[SeedVC] " + "=" * 72
    return "\n".join([
        bar,
        "[SeedVC] ONLY ONE SPEAKER FOUND (two were expected)",
        f"[SeedVC]   Why: {reason}.",
        f"[SeedVC]   Result: {outcome}",
        "[SeedVC]   If there really are two speakers, make them easier to tell apart. For generated video,",
        "[SeedVC]   say so in the prompt:",
        "[SeedVC]     - give them clearly different voices (for example 'a calm voice' and 'a deep voice')",
        "[SeedVC]     - first line at least 2-3 s long, every line 1.5 s or longer",
        "[SeedVC]     - a pause between the lines, and 'they never talk at the same time'",
        "[SeedVC]   A prompt that works is in the README, section 'Prompt for two speakers'.",
        bar,
    ])


def load_reference(path):
    wave, rate = load_audio_file(path)
    mono = to_mono(wave)
    if float(mono.abs().max()) < 1e-4:
        raise AudioError(f"The voice sample is silent: {path}")
    return mono, rate


def voice_swap(waveform, sample_rate, reference, reference_rate, model_paths, **options):
    """One voice for all speech (no speaker separation). See `voice_swap_speakers` for the options."""
    return voice_swap_speakers(waveform, sample_rate, [(reference, reference_rate)], model_paths, speakers=1,
                               **options)


def voice_swap_speakers(waveform, sample_rate, references, model_paths, *, speakers=1, mode="speech", steps=25,
                        cfg=0.5, seed=0, keep_background=True, device="cuda", fp16=True, pitch_mode="original",
                        pitch_shift=0, progress=None, check_interrupt=None, log=print, return_stems=False):
    """Replace the voice(s) in `waveform`.

    waveform: float tensor [B, C, T] or [C, T] with C = 1 or 2, any sample rate.
    references: one entry per speaker, in order of first utterance. Each entry is
        (mono float tensor, sample_rate) for the target voice, or None to keep that
        speaker's original voice.
    speakers: 1 = all speech belongs to speaker 1. 2 = find who speaks when (at most two speakers).
    Returns (output, stats). output has exactly the input shape; stats holds timings, gains,
    what the speaker detection found and, with `return_stems`, the stems of the first item.
    """
    if speakers not in (1, 2):
        raise ValueError(f"speakers must be 1 or 2, got {speakers!r}")
    references = list(references) + [None] * (speakers - len(references))
    if all(reference is None for reference in references[:speakers]):
        raise AudioError("No target voice was given: every speaker would keep the original voice.")
    squeeze = waveform.dim() == 2
    batch = waveform[None] if squeeze else waveform
    if batch.dim() != 3:
        raise AudioError(f"Expected audio shaped [batch, channels, samples], got {tuple(waveform.shape)}.")
    count, channels, length = batch.shape
    if channels not in (1, 2):
        raise AudioError(f"Only mono and stereo audio are supported, got {channels} channels. "
                         f"Downmix the audio to stereo first.")
    if length == 0:
        raise AudioError("The input audio is empty.")
    batch = batch.detach().float().cpu()
    stats = {"sample_rate": int(sample_rate), "channels": channels, "length": length, "items": []}
    report = (lambda stage, done, total: progress(stage, done, total)) if progress else None
    timings = {}

    # 1) split voice from background -------------------------------------------------
    voices, backgrounds = [], []
    if keep_background:
        start = time.perf_counter()
        model = separator.load_model(model_paths["bs_roformer"], device)
        timings["separator_load_s"] = time.perf_counter() - start
        start = time.perf_counter()
        for index in range(count):
            vocals, background = separator.split_vocals_background(
                model, batch[index], sample_rate, device, fp16=fp16, check_interrupt=check_interrupt,
                progress=(lambda d, t, i=index: report("separate", i * t + d, count * t)) if report else None)
            voices.append(vocals)
            backgrounds.append(background)
        timings["separate_s"] = time.perf_counter() - start
        del model
        _free()
    else:
        voices = [batch[index] for index in range(count)]
        backgrounds = [None] * count

    # 2) convert the voice(s) --------------------------------------------------------
    start = time.perf_counter()
    engine = SeedVCEngine(mode, model_paths, device, fp16=fp16)
    timings["seedvc_load_s"] = time.perf_counter() - start
    outputs = []
    try:
        targets = [None if reference is None else resample(reference[0], reference[1], engine.sr)
                   for reference in references]
        active = sum(target is not None for target in targets[:speakers])
        timings["diarize_s"] = 0.0
        timings["convert_s"] = 0.0
        for index in range(count):
            voice = voices[index]
            item = {}
            if float(voice.abs().max()) < 1e-5:
                log("[SeedVC] no voice signal in this audio item; passing it through unchanged")
                outputs.append(batch[index].clone())
                item["skipped"] = "silent"
                stats["items"].append(item)
                continue
            mono = to_mono(voice)

            # who speaks when
            if speakers == 2:
                start = time.perf_counter()
                found = dz.diarize(resample(mono, sample_rate, dz.RATE), engine.campplus, engine.device)
                masks = dz.speaker_masks(found, length, sample_rate)
                timings["diarize_s"] += time.perf_counter() - start
                item["speakers_found"] = found.speakers
                item["diarization"] = found.info
                if found.speakers == 1:
                    log(_one_speaker_notice(found.info.get("reason", "no second voice"), targets))
            else:
                masks = torch.ones(1, length)
                item["speakers_found"] = 1

            # convert once per target voice, then keep each conversion only where its speaker talks
            start = time.perf_counter()
            source = resample(mono, sample_rate, engine.sr)
            new_voice = torch.zeros_like(voice)
            item["speakers"] = []
            jobs = [k for k in range(masks.shape[0]) if targets[k] is not None]
            for speaker in range(masks.shape[0]):
                mask = masks[speaker]
                entry = {"speaker": speaker + 1, "seconds": round(float(mask.sum()) / sample_rate, 2)}
                if targets[speaker] is None:
                    new_voice += voice * mask
                    entry["voice"] = "original"
                else:
                    job = jobs.index(speaker)
                    converted = engine.convert(
                        source, targets[speaker], steps=steps, cfg=cfg,
                        seed=int(seed) + index + SPEAKER_SEED_STEP * speaker, pitch_mode=pitch_mode,
                        pitch_shift=pitch_shift, check_interrupt=check_interrupt,
                        progress=(lambda d, t, i=index, j=job, n=len(jobs):
                                  report("convert", (i * n + j) * t + d, count * n * t)) if report else None)
                    entry["converted_model_samples"] = int(converted.numel())
                    converted = fit_length(resample(converted, engine.sr, sample_rate), length)
                    region = None if masks.shape[0] == 1 else mask
                    leveled, gains = match_loudness(converted, voice, sample_rate, region=region)
                    new_voice += leveled * mask
                    entry["voice"] = "converted"
                    entry["loudness_gain_db"] = [round(db(g), 3) for g in gains]
                item["speakers"].append(entry)
            timings["convert_s"] += time.perf_counter() - start
            if not jobs:
                log("[SeedVC] the speaker(s) found have no voice slot switched on; this audio item is unchanged")
                item["skipped"] = "no active voice for the speakers found"

            mixed = new_voice if backgrounds[index] is None else new_voice + backgrounds[index]
            peak = float(mixed.abs().max())
            item["peak_before_limit"] = round(peak, 5)
            if peak > 1.0:
                # keep the background untouched; lower only the new voice, then guard against residual overs
                new_voice = new_voice / peak
                mixed = new_voice if backgrounds[index] is None else new_voice + backgrounds[index]
                mixed = mixed.clamp(-1.0, 1.0)
                item["voice_reduced_db"] = round(db(1.0 / peak), 3)
            outputs.append(mixed.float())
            stats["items"].append(item)
            if return_stems and index == 0:
                stats["stems"] = {"voice_in": voice, "voice_out": new_voice, "background": backgrounds[index],
                                  "masks": masks}
        stats["active_voices"] = active
    finally:
        engine.unload()
        del engine
        _free()

    stats["timings"] = {k: round(v, 3) for k, v in timings.items()}
    result = torch.stack(outputs, dim=0)
    assert result.shape == batch.shape, (result.shape, batch.shape)
    return (result[0] if squeeze else result), stats
