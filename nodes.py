"""ComfyUI nodes for comfyui-seedvc-voice-swap."""
import os
from pathlib import Path

import folder_paths

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac")
NO_SAMPLE = "(none)"
MODELS_SUBDIR = "seedvc"


def _input_dir():
    return Path(folder_paths.get_input_directory())


def _voice_samples():
    base = _input_dir()
    if not base.is_dir():
        return [NO_SAMPLE]
    files = sorted(p.relative_to(base).as_posix() for p in base.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    return [NO_SAMPLE] + files


def _resolve_sample(name, label="voice_sample"):
    if not name or name == NO_SAMPLE:
        raise ValueError(
            f"Seed-VC Voice Swap: no voice sample selected in '{label}'. Put a 5-25 s recording of the target "
            f"voice into ComfyUI/input and select it, or switch 'enabled' off.")
    base = _input_dir().resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Seed-VC Voice Swap: the voice sample must be inside ComfyUI/input: {name}")
    if not path.is_file():
        raise ValueError(
            f"Seed-VC Voice Swap: voice sample not found: {path}\n"
            f"The file was moved or deleted. Pick another file in '{label}' (press R to refresh the list).")
    return path


def models_dir():
    return os.path.join(folder_paths.models_dir, MODELS_SUBDIR)


def _run(audio, references, speakers, keep_background, mode, singing_pitch, steps, cfg, seed, wan2gp_ckpts_path):
    """Shared execution path of both nodes. `references`: one sample path or None per speaker."""
    if audio is None or "waveform" not in audio:
        raise ValueError("Seed-VC Voice Swap: the 'audio' input is empty.")

    import comfy.model_management as mm
    import comfy.utils

    from .seedvc_swap.audio_utils import AudioError
    from .seedvc_swap.download import ModelError, ensure_models
    from .seedvc_swap.pipeline import load_reference, voice_swap_speakers

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    device = mm.get_torch_device()
    try:
        loaded = [None if path is None else load_reference(path) for path in references]
        paths = ensure_models(models_dir(), mode, keep_background, wan2gp_ckpts=wan2gp_ckpts_path)

        # make room: ComfyUI's models out of VRAM before ours go in
        mm.unload_all_models()
        mm.soft_empty_cache()

        bar = comfy.utils.ProgressBar(100)
        split = 30 if keep_background else 0

        def progress(stage, done, total):
            if stage == "separate":
                bar.update_absolute(int(split * done / max(1, total)), 100)
            else:
                bar.update_absolute(split + int((100 - split) * done / max(1, total)), 100)

        output, stats = voice_swap_speakers(
            waveform, sample_rate, loaded, paths, speakers=speakers, mode=mode, steps=steps, cfg=cfg, seed=seed,
            keep_background=keep_background, pitch_mode=singing_pitch, device=device,
            fp16=device.type == "cuda", progress=progress,
            check_interrupt=mm.throw_exception_if_processing_interrupted)
    except (AudioError, ModelError) as exc:
        raise RuntimeError(f"Seed-VC Voice Swap: {exc}") from exc
    finally:
        mm.soft_empty_cache()
    for item in stats["items"]:
        if "speakers" in item:
            print("[SeedVC] speakers: " + ", ".join(
                f"#{entry['speaker']} {entry['seconds']} s -> {entry['voice']}" for entry in item["speakers"]))
    print(f"[SeedVC] done: {stats['timings']}")
    return ({"waveform": output, "sample_rate": sample_rate},)


def _sample_stamp(name):
    try:
        stat = (_input_dir() / name).stat()
        return f"{name}:{stat.st_size}-{stat.st_mtime_ns}"
    except OSError:
        return f"{name}:missing"


_COMMON_INPUTS = {
    "keep_background": ("BOOLEAN", {"default": True, "tooltip":
                        "On: music and noises are separated first, only the voice is converted and the "
                        "background is mixed back. Off: the whole input is treated as voice (faster; "
                        "for clean speech only)."}),
    "mode": (["speech", "singing"], {"default": "speech", "tooltip":
             "speech: 22 kHz model. singing: 44 kHz model that follows the original melody."}),
    "singing_pitch": (["octave", "original", "match"], {"default": "octave", "tooltip":
                      "Singing mode only. octave: move the melody by whole octaves towards the voice "
                      "sample's range (stays in key). original: keep the sung pitch. match: follow the "
                      "sample's pitch exactly (closest voice, can leave the key of the music)."}),
    "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1,
                      "tooltip": "Diffusion steps. More is slower; 25 for speech, 30-50 for singing."}),
    "cfg": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                      "tooltip": "Guidance strength. Higher follows the voice sample more closely."}),
    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                     "tooltip": "Same seed and inputs give the same result."}),
    "wan2gp_ckpts_path": ("STRING", {"default": "", "tooltip":
                          "Optional. Path to an existing Wan2GP 'ckpts' folder: its models are used and "
                          "nothing is downloaded. Empty: models are downloaded to ComfyUI/models/seedvc."}),
}
_SAMPLE_TOOLTIP = ("Recording of the target voice from ComfyUI/input (subfolders included). "
                   "5-25 s of clean speech; only the first 25 s are used.")


class SeedVCVoiceSwap:
    """Replace the voice in an audio clip with the voice from a sample (one voice for all speech)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": "Audio whose voice will be replaced. Mono or stereo."}),
            "enabled": ("BOOLEAN", {"default": True,
                                    "tooltip": "Off: the audio passes through untouched and no model is loaded."}),
            "voice_sample": (_voice_samples(), {"tooltip": _SAMPLE_TOOLTIP}),
            **_COMMON_INPUTS,
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "swap"
    CATEGORY = "audio/voice conversion"
    DESCRIPTION = ("Zero-shot voice conversion with Seed-VC. Output has exactly the same length, sample rate and "
                   "channel count as the input. Use only with voices you have the rights to use.")

    @classmethod
    def VALIDATE_INPUTS(cls, voice_sample=None):
        # accept names missing from the list here so swap() can raise a clear, actionable message
        return True

    @classmethod
    def IS_CHANGED(cls, enabled=True, voice_sample=NO_SAMPLE, **kwargs):
        if not enabled or voice_sample == NO_SAMPLE:
            return ""
        return _sample_stamp(voice_sample)

    def swap(self, audio, enabled, voice_sample, keep_background, mode, singing_pitch, steps, cfg, seed,
             wan2gp_ckpts_path=""):
        if not enabled:
            return (audio,)
        return _run(audio, [_resolve_sample(voice_sample)], 1, keep_background, mode, singing_pitch, steps, cfg,
                    seed, wan2gp_ckpts_path)


class SeedVCVoiceSwap2Speakers:
    """Replace the voices of up to two speakers. Slot 1 is whoever speaks first, slot 2 the other speaker."""

    @classmethod
    def INPUT_TYPES(cls):
        samples = _voice_samples()
        return {"required": {
            "audio": ("AUDIO", {"tooltip": "Audio whose voices will be replaced. Mono or stereo."}),
            "enabled": ("BOOLEAN", {"default": True,
                                    "tooltip": "Off: the audio passes through untouched and no model is loaded."}),
            "voice_1_on": ("BOOLEAN", {"default": True, "tooltip":
                           "Convert speaker 1 (whoever speaks first). Off: speaker 1 keeps the original voice."}),
            "voice_1": (samples, {"tooltip": "Voice for speaker 1. " + _SAMPLE_TOOLTIP}),
            "voice_2_on": ("BOOLEAN", {"default": False, "tooltip":
                           "Convert speaker 2 (the other speaker). Off: speaker 2 keeps the original voice."}),
            "voice_2": (samples, {"tooltip": "Voice for speaker 2. " + _SAMPLE_TOOLTIP}),
            "speakers": (["auto", "1", "2"], {"default": "auto", "tooltip":
                         "How many speakers the audio contains. auto: 1 when only voice 1 is on (all speech gets "
                         "voice 1, nobody is told apart), 2 when voice 2 is on. Set 2 with only voice 1 on to "
                         "convert the first speaker and leave the second one untouched."}),
            **_COMMON_INPUTS,
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "swap"
    CATEGORY = "audio/voice conversion"
    DESCRIPTION = ("Zero-shot voice conversion with Seed-VC for up to two speakers. Speakers are numbered by who "
                   "talks first. Output has exactly the same length, sample rate and channel count as the input. "
                   "Use only with voices you have the rights to use.")

    @classmethod
    def VALIDATE_INPUTS(cls, voice_1=None, voice_2=None):
        return True

    @classmethod
    def IS_CHANGED(cls, enabled=True, voice_1_on=True, voice_1=NO_SAMPLE, voice_2_on=False, voice_2=NO_SAMPLE,
                   **kwargs):
        if not enabled:
            return ""
        return "|".join(_sample_stamp(name) for on, name in ((voice_1_on, voice_1), (voice_2_on, voice_2))
                        if on and name != NO_SAMPLE)

    def swap(self, audio, enabled, voice_1_on, voice_1, voice_2_on, voice_2, speakers, keep_background, mode,
             singing_pitch, steps, cfg, seed, wan2gp_ckpts_path=""):
        if not enabled:
            return (audio,)
        if not voice_1_on and not voice_2_on:
            raise ValueError("Seed-VC Voice Swap: both voice slots are off. Switch on 'voice_1_on' and/or "
                             "'voice_2_on', or switch 'enabled' off to pass the audio through.")
        count = (2 if voice_2_on else 1) if speakers == "auto" else int(speakers)
        if count == 1 and voice_2_on:
            raise ValueError("Seed-VC Voice Swap: 'speakers' is 1 but voice 2 is switched on. With one speaker "
                             "only voice 1 is used: switch voice 2 off, or set 'speakers' to auto or 2.")
        if count == 1 and not voice_1_on:
            raise ValueError("Seed-VC Voice Swap: 'speakers' is 1 but voice 1 is off, so nothing would change.")
        references = [_resolve_sample(voice_1, "voice_1") if voice_1_on else None,
                      _resolve_sample(voice_2, "voice_2") if voice_2_on else None][:count]
        return _run(audio, references, count, keep_background, mode, singing_pitch, steps, cfg, seed,
                    wan2gp_ckpts_path)


NODE_CLASS_MAPPINGS = {"SeedVCVoiceSwap": SeedVCVoiceSwap, "SeedVCVoiceSwap2Speakers": SeedVCVoiceSwap2Speakers}
NODE_DISPLAY_NAME_MAPPINGS = {"SeedVCVoiceSwap": "Seed-VC Voice Swap (1 voice)",
                              "SeedVCVoiceSwap2Speakers": "Seed-VC Voice Swap (2 speakers)"}
