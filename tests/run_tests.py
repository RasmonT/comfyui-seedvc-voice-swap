"""Measurements for comfyui-seedvc-voice-swap (version 1, one voice).

Runs outside the ComfyUI server, with ComfyUI's own Python, and writes everything into
tests/results/<timestamp>/ (results.json, results.md and the audio files for listening).
It installs nothing, downloads nothing (unless --allow-download) and changes no existing file.

What it measures
  1. output length / sample rate / channels versus the input (in samples)
  2. voice similarity to the voice sample before and after (cosine of speaker embeddings)
  3. background preservation (the music stem is known, so this is exact)
  4. processing time and peak VRAM
  5. same seed -> same result
  6. how much the similarity varies from seed to seed
  7. the same inputs through the old Wan2GP bridge, for comparison (optional)
"""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from seedvc_swap.audio_utils import fit_length, load_audio_file, resample, speech_activity, to_mono  # noqa: E402
from seedvc_swap.download import ensure_models  # noqa: E402
from seedvc_swap.pipeline import load_reference, voice_swap  # noqa: E402
import testlib  # noqa: E402


def parse_args():
    local = {}
    local_file = HERE / "local.json"
    if local_file.is_file():
        local = json.loads(local_file.read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy", default=local.get("comfy"), help="ComfyUI folder (the one containing custom_nodes)")
    p.add_argument("--wan2gp", default=local.get("wan2gp"), help="Wan2GP folder (models, old bridge, embedder)")
    p.add_argument("--speaker", default=local.get("speaker"), help="clip with one speaker, relative to ComfyUI/input")
    p.add_argument("--voice", default=local.get("voice"), help="target voice sample, relative to ComfyUI/input")
    p.add_argument("--music", default=local.get("music"), help="optional music file; synthetic backing track if empty")
    p.add_argument("--embedder", default=local.get("embedder"), help="ERes2Net speaker model (.onnx)")
    p.add_argument("--models-dir", default=local.get("models_dir"), help="model folder if Wan2GP ckpts are not used")
    p.add_argument("--allow-download", action="store_true", help="let the pack download missing models")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--seconds", type=float, default=10.0, help="total length of the test clips")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-old", action="store_true", help="do not run the old Wan2GP bridge")
    p.add_argument("--no-background", action="store_true", help="two-speaker tests on clean speech, no separation")
    p.add_argument("--seed-spread", type=int, default=8, help="extra seeds for the similarity spread (0 = skip)")
    p.add_argument("--speaker2", default=local.get("speaker2"), help="second speaker clip (two-speaker tests)")
    p.add_argument("--voice2", default=local.get("voice2"), help="second target voice (two-speaker tests)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def save_wav(path, wave, rate):
    wavfile.write(str(path), int(rate), wave.clamp(-1, 1).T.contiguous().numpy().astype(np.float32))


def read_wav(path):
    rate, data = wavfile.read(str(path))
    if data.dtype.kind in "iu":
        data = data.astype(np.float32) / float(2 ** (data.dtype.itemsize * 8 - 1))
    data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return torch.from_numpy(data.T.copy()), int(rate)


def environment(device):
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    env = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device),
           "packages": {n: version(n) for n in ("torch", "torchaudio", "numpy", "scipy", "transformers", "einops",
                                                "av", "rotary-embedding-torch", "onnxruntime-gpu", "onnxruntime",
                                                "huggingface-hub", "pyyaml")}}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
    return env


def build_clips(args, out):
    """speech-only clip A and speech+music clip B with a known music stem.

    Layout of B: 1 s music only, speech + music, 1 s music only. The voice-free edges give a region
    where the output must equal the music, which can be measured for any system (old bridge included).
    """
    input_dir = Path(args.comfy) / "input" if args.comfy else None
    speaker_path = Path(args.speaker) if os.path.isabs(args.speaker) else input_dir / args.speaker
    speech, rate = load_audio_file(speaker_path)
    if speech.shape[0] > 2:
        speech = speech[:2]
    edge = int(rate * 1.0)
    speech_len = min(speech.shape[-1], int(rate * args.seconds) - 2 * edge)
    speech = speech[:, :speech_len]
    channels = speech.shape[0]
    total = speech_len + 2 * edge
    speech_padded = torch.zeros(channels, total)
    speech_padded[:, edge:edge + speech_len] = speech

    if args.music:
        music_path = Path(args.music) if os.path.isabs(args.music) else input_dir / args.music
        music, music_rate = load_audio_file(music_path)
        music = fit_length(resample(music, music_rate, rate), total)
        music_source = str(music_path)
    else:
        music = testlib.synth_music(total / rate, rate)
        music = fit_length(music, total)
        music_source = "synthetic (tests/testlib.py synth_music)"
    if music.shape[0] != channels:
        music = music.mean(dim=0, keepdim=True) if channels == 1 else music[:1].repeat(2, 1)
    mask, frame = speech_activity(to_mono(speech), rate)
    active = to_mono(speech)[: mask.numel() * frame].reshape(-1, frame)[mask]
    speech_rms = float(active.pow(2).mean().sqrt()) if active.numel() else testlib.rms(speech)
    music = music * (speech_rms * 10 ** (-9 / 20) / max(testlib.rms(music), 1e-9))  # music 9 dB below speech
    mix = speech_padded + music
    peak = float(mix.abs().max())
    if peak > 0.98:
        scale = 0.98 / peak
        speech_padded, music, mix = speech_padded * scale, music * scale, mix * scale

    save_wav(out / "A_speech_only.wav", speech_padded, rate)
    save_wav(out / "B_speech_plus_music.wav", mix, rate)
    save_wav(out / "B_music_stem.wav", music, rate)
    info = {"speaker_clip": str(speaker_path), "music": music_source, "sample_rate": rate, "channels": channels,
            "samples": total, "seconds": round(total / rate, 3), "voice_free_edge_s": 1.0,
            "music_level_rel_speech_db": -9.0}
    return speech_padded, music, mix, rate, edge, info


def cuda_leftovers(limit=12):
    """CUDA tensors still alive after a run (largest first), to pin down anything the pipeline failed to free."""
    import gc

    gc.collect()
    found = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                found.append((obj.numel() * obj.element_size(), tuple(obj.shape), str(obj.dtype)))
        except Exception:
            continue
    found.sort(reverse=True)
    return [{"mb": round(size / 2 ** 20, 3), "shape": list(shape), "dtype": dtype} for size, shape, dtype in found[:limit]]


def run_new(label, clip, rate, reference, reference_rate, paths, args, keep_background, seed, out, save=True):
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with testlib.GpuMemoryMonitor() as monitor:
        start = time.perf_counter()
        result, stats = voice_swap(clip, rate, reference, reference_rate, paths, mode="speech", steps=args.steps,
                                   cfg=args.cfg, seed=seed, keep_background=keep_background, device=args.device,
                                   fp16=args.device.startswith("cuda"), return_stems=True)
        elapsed = time.perf_counter() - start
    record = {"seconds_total": round(elapsed, 3), "timings": stats["timings"], "item": stats["items"][0],
              "in_shape": list(clip.shape), "out_shape": list(result.shape),
              "length_diff_samples": int(result.shape[-1] - clip.shape[-1]),
              "same_channels": result.shape[0] == clip.shape[0],
              "vram_peak_nvidia_smi_delta_mb": monitor.peak_delta_mb}
    if args.device.startswith("cuda"):
        record["vram_peak_torch_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        record["vram_peak_torch_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
        record["vram_left_allocated_after_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
        record["vram_left_tensors"] = cuda_leftovers()
    if save:
        save_wav(out / f"{label}_new.wav", result, rate)
    return result, stats.get("stems", {}), record


def run_old(label, clip_path, voice_path, args, out):
    """Call the old bridge exactly like its node does: Wan2GP's Python + the bridge's worker.py."""
    root = Path(args.wan2gp).resolve()
    python = root / "env_venv" / "Scripts" / "python.exe"
    worker = Path(args.comfy) / "custom_nodes" / "comfyui-wan2gp-seedvc-bridge" / "worker.py"
    if not python.is_file() or not worker.is_file():
        return None, {"skipped": f"old bridge not found ({python} / {worker})"}
    output = out / f"{label}_old_bridge.wav"
    with testlib.GpuMemoryMonitor() as monitor:
        start = time.perf_counter()
        proc = subprocess.run([str(python), str(worker), str(root), str(clip_path), str(output), str(voice_path)],
                              cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        elapsed = time.perf_counter() - start
    (out / f"{label}_old_bridge.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode or not output.is_file():
        return None, {"failed": True, "returncode": proc.returncode, "log": f"{label}_old_bridge.log"}
    wave, rate = read_wav(output)
    return (wave, rate), {"seconds_total": round(elapsed, 3), "vram_peak_nvidia_smi_delta_mb": monitor.peak_delta_mb,
                          "out_shape": list(wave.shape), "out_sample_rate": rate}


def main():
    args = parse_args()
    if not args.speaker or not args.voice:
        sys.exit("Missing --speaker / --voice (or tests/local.json).")
    out = Path(args.out) if args.out else HERE / "results" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = {"environment": environment(args.device),
               "settings": {"steps": args.steps, "cfg": args.cfg, "seed": args.seed, "mode": "speech"}}
    print(json.dumps(results["environment"], indent=1))

    input_dir = Path(args.comfy) / "input" if args.comfy else None
    voice_path = Path(args.voice) if os.path.isabs(args.voice) else input_dir / args.voice
    reference, reference_rate = load_reference(voice_path)
    results["voice_sample"] = {"file": str(voice_path), "seconds": round(reference.numel() / reference_rate, 2),
                               "sample_rate": reference_rate}

    ckpts = str(Path(args.wan2gp) / "ckpts") if args.wan2gp and not args.models_dir else ""
    models_dir = args.models_dir or str(out / "_no_models_here")
    start = time.perf_counter()
    paths = ensure_models(models_dir, "speech", True, wan2gp_ckpts=ckpts, allow_download=args.allow_download)
    results["models"] = {"source": ckpts or models_dir, "verify_seconds": round(time.perf_counter() - start, 2),
                         "files": {k: os.path.getsize(v) for k, v in paths.items()},
                         "total_mb": round(sum(os.path.getsize(v) for v in paths.values()) / 1e6, 1)}

    speech, music, mix, rate, edge, clip_info = build_clips(args, out)
    results["clips"] = clip_info

    embedder_path = args.embedder or (str(Path(args.wan2gp) / "ckpts" / "sherpa" /
                                          "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
                                      if args.wan2gp else None)
    embedders = [testlib.CampplusEmbedder(paths["campplus"])]
    if embedder_path and Path(embedder_path).is_file():
        embedders.insert(0, testlib.ERes2NetEmbedder(embedder_path))
    else:
        print("WARNING: independent speaker model (ERes2Net onnx) not found; only the biased CAM++ score is reported")
    ref_emb = {e.name: e(reference, reference_rate) for e in embedders}

    def similarity(wave):
        mono = to_mono(wave)
        return {e.name: round(testlib.cosine(ref_emb[e.name], e(mono, rate)), 4) for e in embedders}

    results["similarity_before"] = similarity(speech)
    voice_free = torch.zeros(mix.shape[-1], dtype=torch.bool)
    voice_free[:edge] = True
    voice_free[-edge:] = True

    # --- new node, clip B (speech + music), background kept ------------------------------------
    print("\n[1/5] new pipeline, speech + music, keep_background=True (first run, includes model loading)")
    out_b, stems, rec = run_new("B", mix, rate, reference, reference_rate, paths, args, True, args.seed, out)
    rec["similarity_after_voice_stem"] = similarity(stems["voice_out"])
    rec["similarity_after_output_minus_music"] = similarity(out_b - music)
    rec["background_sdr_db_full_clip"] = round(testlib.sdr_db(music, stems["background"]), 2)
    rec["background_sdr_db_voice_free_edges"] = round(testlib.sdr_db(music[:, voice_free], out_b[:, voice_free]), 2)
    rec["separated_voice_sdr_db"] = round(testlib.sdr_db(speech, stems["voice_in"]), 2)
    rec["voice_envelope_lag_ms"] = testlib.envelope_lag_ms(to_mono(speech), to_mono(stems["voice_out"]), rate)
    save_wav(out / "B_new_voice_stem.wav", stems["voice_out"], rate)
    save_wav(out / "B_new_background_stem.wav", stems["background"], rate)
    results["new_B_speech_plus_music"] = rec

    print("[2/5] same again (second run) for timing")
    out_b2, _, rec2 = run_new("B_run2", mix, rate, reference, reference_rate, paths, args, True, args.seed, out,
                              save=False)
    results["new_B_second_run"] = {k: rec2[k] for k in rec2 if k.startswith(("seconds", "timings", "vram"))}
    diff = (out_b - out_b2).abs()
    results["determinism_same_seed"] = {"max_abs_diff": float(diff.max()), "bit_identical": bool(diff.max() == 0),
                                        "snr_db": round(testlib.sdr_db(out_b, out_b2), 2)}

    print("[3/5] different seed")
    out_b3, _, _ = run_new("B_seed2", mix, rate, reference, reference_rate, paths, args, True, args.seed + 1, out,
                           save=False)
    results["different_seed"] = {"max_abs_diff": float((out_b - out_b3).abs().max()),
                                 "snr_db": round(testlib.sdr_db(out_b, out_b3), 2),
                                 "similarity_after_output_minus_music": similarity(out_b3 - music)}

    # --- new node, clip A (speech only), no separation ----------------------------------------
    print("[4/5] new pipeline, speech only, keep_background=False")
    out_a, stems_a, rec_a = run_new("A", speech, rate, reference, reference_rate, paths, args, False, args.seed, out)
    rec_a["similarity_after"] = similarity(out_a)
    rec_a["voice_envelope_lag_ms"] = testlib.envelope_lag_ms(to_mono(speech), to_mono(out_a), rate)
    results["new_A_speech_only"] = rec_a

    # --- how much does the similarity move from seed to seed? ----------------------------------
    if args.seed_spread > 0:
        print(f"[+] similarity spread over {args.seed_spread} seeds (speech only)")
        values = {e.name: [] for e in embedders}
        for k in range(args.seed_spread):
            out_s, _, _ = run_new("A_spread", speech, rate, reference, reference_rate, paths, args, False,
                                  args.seed + 100 + k, out, save=False)
            for name, value in similarity(out_s).items():
                values[name].append(value)
        results["seed_spread_speech_only"] = {
            name: {"n": len(v), "mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4),
                   "min": min(v), "max": max(v), "values": v} for name, v in values.items()}

    # --- old bridge on the same inputs ---------------------------------------------------------
    if args.skip_old or not args.wan2gp or not args.comfy:
        results["old_bridge"] = {"skipped": True}
    else:
        print("[5/5] old Wan2GP bridge on the same two clips")
        old = {}
        for label, clip, clip_file in (("B", mix, "B_speech_plus_music.wav"), ("A", speech, "A_speech_only.wav")):
            got, rec_old = run_old(label, out / clip_file, voice_path, args, out)
            if got is not None:
                wave, old_rate = got
                rec_old["length_diff_samples"] = int(wave.shape[-1] - clip.shape[-1])
                rec_old["same_sample_rate"] = old_rate == rate
                rec_old["same_channels"] = wave.shape[0] == clip.shape[0]
                aligned = fit_length(resample(wave, old_rate, rate), clip.shape[-1])
                if aligned.shape[0] != clip.shape[0]:
                    aligned = aligned.mean(dim=0, keepdim=True) if clip.shape[0] == 1 else aligned[:1].repeat(2, 1)
                if label == "B":
                    rec_old["similarity_after_output_minus_music"] = similarity(aligned - music)
                    rec_old["background_sdr_db_voice_free_edges"] = round(
                        testlib.sdr_db(music[:, voice_free], aligned[:, voice_free]), 2)
                else:
                    rec_old["similarity_after"] = similarity(aligned)
                    rec_old["voice_envelope_lag_ms"] = testlib.envelope_lag_ms(to_mono(speech), to_mono(aligned), rate)
            old[label] = rec_old
        results["old_bridge"] = old

    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n" + json.dumps(results, indent=1))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
