"""Measurements for the two-speaker node of comfyui-seedvc-voice-swap.

Runs outside the ComfyUI server with ComfyUI's own Python and writes everything into
tests/results/<timestamp>-v2/. It installs nothing, downloads nothing and changes no existing file.

A test clip is assembled from two single-speaker recordings (alternating turns, music underneath),
so it is known exactly who speaks when. Measured:
  1. who-speaks-when accuracy of the pack, and of the pyannote-based reference on the same clip
  2. each speaker's similarity to both target voices after conversion (the right voice must win)
  3. output length / channels, processing time, peak VRAM, and how well the music is preserved
     (overall and separately while each speaker talks)
  4. "speaker without a slot keeps the original voice": checked sample by sample
  5. the old Wan2GP bridge in its two-speaker mode on the same clip (optional)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diarization_eval  # noqa: E402
import run_tests as base  # noqa: E402
import testlib  # noqa: E402
from seedvc_swap import diarize as dz  # noqa: E402
from seedvc_swap.audio_utils import fit_length, load_audio_file, resample, to_mono  # noqa: E402
from seedvc_swap.download import ensure_models  # noqa: E402
from seedvc_swap.pipeline import load_reference, voice_swap_speakers  # noqa: E402

TURN_S = 3.0
GAP_S = 0.35
EDGE_S = 1.0


def resolve(args, name):
    return Path(name) if os.path.isabs(name) else Path(args.comfy) / "input" / name


def build_dialogue(args, out):
    """Speaker A and B alternate (A B A B), with a pause between turns and music under everything."""
    first, rate = load_audio_file(resolve(args, args.speaker))
    second, rate2 = load_audio_file(resolve(args, args.speaker2))
    first = to_mono(first)
    second = resample(to_mono(second), rate2, rate)

    def trim(wave):  # drop leading silence so every turn starts with speech
        active, _ = dz.speech_frames(resample(wave, rate, dz.RATE))
        runs = dz._runs(active)
        return wave[int(runs[0][0] * dz.FRAME * rate / dz.RATE):] if runs else wave

    first, second = trim(first), trim(second)
    level = lambda w: w / (w.pow(2).mean().sqrt() + 1e-9) * 0.07  # noqa: E731
    first, second = level(first), level(second)
    turn = int(TURN_S * rate)
    gap = int(GAP_S * rate)
    edge = int(EDGE_S * rate)
    turns_each = max(1, min(2, first.numel() // turn, second.numel() // turn))
    pieces, truth = [torch.zeros(edge)], [np.full(edge, -1, dtype=np.int8)]
    fade = int(0.01 * rate)
    ramp = torch.linspace(0, 1, fade)
    for index in range(turns_each):
        for speaker, source in ((0, first), (1, second)):
            piece = source[index * turn:(index + 1) * turn].clone()
            piece[:fade] *= ramp
            piece[-fade:] *= ramp.flip(0)
            pieces += [piece, torch.zeros(gap)]
            truth += [np.full(piece.numel(), speaker, dtype=np.int8), np.full(gap, -1, dtype=np.int8)]
    pieces[-1] = torch.zeros(edge)
    truth[-1] = np.full(edge, -1, dtype=np.int8)
    speech = torch.cat(pieces)
    truth = np.concatenate(truth)
    stereo = speech.repeat(2, 1)
    music = testlib.synth_music(stereo.shape[-1] / rate, rate) if not args.music else None
    if music is None:
        music, music_rate = load_audio_file(resolve(args, args.music))
        music = resample(music, music_rate, rate)
        music = music if music.shape[0] == 2 else music[:1].repeat(2, 1)
    music = fit_length(music, stereo.shape[-1])
    music = music * (0.07 * 10 ** (-9 / 20) / max(testlib.rms(music), 1e-9))
    mix = stereo + music
    scale = min(1.0, 0.98 / float(mix.abs().max()))
    stereo, music, mix = stereo * scale, music * scale, mix * scale
    base.save_wav(out / "D_two_speakers_speech.wav", stereo, rate)
    base.save_wav(out / "D_two_speakers_plus_music.wav", mix, rate)
    info = {"speaker_1_clip": str(resolve(args, args.speaker)), "speaker_2_clip": str(resolve(args, args.speaker2)),
            "layout": f"{turns_each} x (speaker 1 {TURN_S} s, pause {GAP_S} s, speaker 2 {TURN_S} s), "
                      f"{EDGE_S} s music-only edges", "sample_rate": rate, "samples": int(mix.shape[-1]),
            "seconds": round(mix.shape[-1] / rate, 2), "music": args.music or "synthetic"}
    return stereo, music, mix, rate, truth, info


def background_report(speech, music, background, truth, rate, paths, args, cuda):
    """Background preservation, split by who is talking.

    `raw` compares the background stem with the music alone, so any non-voice sound inside the speech
    recordings themselves (room noise, effects) counts as an error. `corrected` adds that non-voice part
    (found by separating the clean dialogue on its own) to the reference, which isolates what the music
    itself loses. The background stem is computed before any speaker handling, so these numbers are the
    same for the one-voice node on the same audio.
    """
    from seedvc_swap import separator

    model = separator.load_model(paths["bs_roformer"], args.device)
    _, residue = separator.split_vocals_background(model, speech, rate, args.device, fp16=cuda)
    del model
    if cuda:
        torch.cuda.empty_cache()

    def part(selection):
        if selection is None:
            pick = slice(None)
        else:
            pick = torch.from_numpy(selection)
        entry = {"raw_sdr_db": round(testlib.sdr_db(music[:, pick], background[:, pick]), 2),
                 "corrected_sdr_db": round(testlib.sdr_db((music + residue)[:, pick], background[:, pick]), 2)}
        if float(speech[:, pick].abs().max()) > 0:
            entry["non_voice_in_speech_clip_db"] = round(
                -testlib.sdr_db(speech[:, pick], (speech - residue)[:, pick]), 2)
        return entry

    return {"whole_clip": part(None), "while_speaker_1_talks": part(truth == 0),
            "while_speaker_2_talks": part(truth == 1), "in_pauses": part(truth < 0)}


def frame_truth(truth, rate):
    """Sample-level truth -> 10 ms frames at 16 kHz (as the diarizer uses)."""
    frames = int(truth.size / rate * dz.RATE) // dz.FRAME
    positions = np.minimum((np.arange(frames) * dz.FRAME * rate / dz.RATE).astype(np.int64), truth.size - 1)
    labels = truth[positions]
    changes = []
    speech_runs = dz._runs(labels >= 0)
    for (_, left_stop), (right_start, _) in zip(speech_runs[:-1], speech_runs[1:]):
        if labels[left_stop - 1] != labels[right_start]:
            changes.append((left_stop, right_start))
    return labels, changes


def masks_to_result(masks, rate):
    frames = int(masks.shape[-1] / rate * dz.RATE) // dz.FRAME
    positions = np.minimum((np.arange(frames) * dz.FRAME * rate / dz.RATE).astype(np.int64), masks.shape[-1] - 1)
    labels = masks.argmax(dim=0).numpy()[positions].astype(np.int8)
    return dz.Diarization(labels, int(masks.shape[0]), np.ones(frames, dtype=bool), {})


def main():
    args = base.parse_args()
    for needed in ("speaker", "speaker2", "voice", "voice2"):
        if not getattr(args, needed):
            sys.exit(f"Missing --{needed} (or the '{needed}' entry in tests/local.json).")
    out = Path(args.out) if args.out else HERE / "results" / (time.strftime("%Y%m%d-%H%M%S") + "-v2")
    out.mkdir(parents=True, exist_ok=True)
    cuda = args.device.startswith("cuda")
    results = {"environment": base.environment(args.device),
               "settings": {"steps": args.steps, "cfg": args.cfg, "seed": args.seed, "mode": "speech"}}

    voices = [load_reference(resolve(args, args.voice)), load_reference(resolve(args, args.voice2))]
    results["voice_samples"] = [{"file": str(resolve(args, name)), "seconds": round(v[0].numel() / v[1], 2)}
                                for name, v in zip((args.voice, args.voice2), voices)]
    ckpts = str(Path(args.wan2gp) / "ckpts") if args.wan2gp and not args.models_dir else ""
    paths = ensure_models(args.models_dir or str(out / "_no_models_here"), "speech", True, wan2gp_ckpts=ckpts,
                          allow_download=args.allow_download)

    speech, music, mix, rate, truth, info = build_dialogue(args, out)
    results["clip"] = info
    truth_frames, changes = frame_truth(truth, rate)
    active = dz.speech_frames(resample(to_mono(speech), rate, dz.RATE))[0]

    sherpa_dir = Path(args.wan2gp) / "ckpts" / "sherpa" if args.wan2gp else None
    embedder_path = args.embedder or (str(sherpa_dir / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
                                      if sherpa_dir else None)
    embedders = [testlib.CampplusEmbedder(paths["campplus"])]
    if embedder_path and Path(embedder_path).is_file():
        embedders.insert(0, testlib.ERes2NetEmbedder(embedder_path))
    target_embeddings = [{e.name: e(v[0], v[1]) for e in embedders} for v in voices]

    def similarity_matrix(voice_wave):
        """For each true speaker: similarity of that speaker's part of `voice_wave` to target voice 1 and 2."""
        mono = to_mono(voice_wave)
        matrix = {}
        for speaker in (0, 1):
            part = mono[torch.from_numpy(truth == speaker)]
            matrix[f"speaker_{speaker + 1}"] = {
                e.name: [round(testlib.cosine(target_embeddings[t][e.name], e(part, rate)), 4) for t in (0, 1)]
                for e in embedders}
        return matrix

    results["similarity_before"] = similarity_matrix(speech)

    def run(label, references, speakers, clip, keep_background, save=True):
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with testlib.GpuMemoryMonitor() as monitor:
            start = time.perf_counter()
            output, stats = voice_swap_speakers(
                clip, rate, references, paths, speakers=speakers, mode="speech", steps=args.steps, cfg=args.cfg,
                seed=args.seed, keep_background=keep_background, device=args.device, fp16=cuda, return_stems=True)
            elapsed = time.perf_counter() - start
        record = {"seconds_total": round(elapsed, 3), "timings": stats["timings"], "item": stats["items"][0],
                  "length_diff_samples": int(output.shape[-1] - clip.shape[-1]),
                  "same_channels": output.shape[0] == clip.shape[0],
                  "vram_peak_nvidia_smi_delta_mb": monitor.peak_delta_mb}
        if cuda:
            record["vram_peak_torch_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            record["vram_left_allocated_after_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
        if save:
            base.save_wav(out / f"{label}.wav", output, rate)
        return output, stats["stems"], record

    keep = not args.no_background
    if not keep:  # quicker variant: clean dialogue, no separation
        mix, music = speech, torch.zeros_like(speech)

    # 1) both speakers converted, music kept ------------------------------------------------------
    print("\n[1/5] two speakers, both converted, " + ("speech + music" if keep else "clean speech"))
    output, stems, record = run("D_new_both_converted", voices, 2, mix, keep)
    found = masks_to_result(stems["masks"], rate)
    record["who_speaks_when"] = diarization_eval.score(found, truth_frames, changes, 2, active)
    record["similarity_after"] = similarity_matrix(stems["voice_out"])
    if keep:
        record["background_sdr_db_full_clip"] = round(testlib.sdr_db(music, stems["background"]), 2)
        record["background"] = background_report(speech, music, stems["background"], truth, rate, paths, args, cuda)
    base.save_wav(out / "D_new_voice_stem.wav", stems["voice_out"], rate)
    results["new_both_converted"] = record

    print("[2/5] same again (second run) for timing")
    _, _, record2 = run("D_run2", voices, 2, mix, keep, save=False)
    results["new_both_converted_second_run"] = {k: record2[k] for k in record2
                                                if k.startswith(("seconds", "timings", "vram"))}

    # 2) only speaker 1 converted: speaker 2 must stay untouched -------------------------------------
    print("[3/5] two speakers, only voice 1 on: speaker 2 must keep the original voice")
    output_one, stems_one, record_one = run("D_new_only_speaker_1", [voices[0], None], 2, mix, keep)
    kept = stems_one["masks"][1] >= 1.0  # samples that belong entirely to speaker 2
    difference = (stems_one["voice_out"] - stems_one["voice_in"])[:, kept].abs()
    record_one["speaker_2_region_max_abs_diff"] = float(difference.max()) if difference.numel() else None
    record_one["speaker_2_region_seconds"] = round(float(kept.sum()) / rate, 2)
    record_one["similarity_after"] = similarity_matrix(stems_one["voice_out"])
    results["new_only_speaker_1"] = record_one

    # 3) diarization alone on the clean dialogue: pack vs pyannote-based reference -------------------
    print("[4/5] who speaks when on the clean dialogue: pack vs reference")
    comparison = {}
    from seedvc_swap.third_party.seed_vc.campplus.DTDNN import CAMPPlus

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(paths["campplus"], map_location="cpu", weights_only=True))
    campplus = campplus.eval().to(args.device)
    mono16 = resample(to_mono(speech), rate, dz.RATE)
    start = time.perf_counter()
    ours = dz.diarize(mono16, campplus, args.device)
    comparison["pack"] = diarization_eval.score(ours, truth_frames, changes, 2, active)
    comparison["pack"]["seconds"] = round(time.perf_counter() - start, 3)
    del campplus
    python = Path(args.wan2gp) / "env_venv" / "Scripts" / "python.exe" if args.wan2gp else None
    segmentation = sherpa_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx" if sherpa_dir else None
    if python and python.is_file() and segmentation.is_file() and embedder_path:
        class Reference:
            sherpa_python, sherpa_pythonpath = str(python), None
            sherpa_segmentation, sherpa_embedding = str(segmentation), embedder_path
        jobs = [("dialogue", mono16, truth_frames, changes, 2)]
        for clusters, name in ((2, "reference_2_forced_as_wan2gp"), (-1, "reference_auto_count")):
            try:
                comparison[name] = diarization_eval.run_reference(Reference, jobs, clusters)[0][1]
            except SystemExit as exc:
                comparison[name] = {"failed": str(exc)[-1500:]}
    else:
        comparison["reference"] = {"skipped": "Wan2GP python / sherpa models not found"}
    results["who_speaks_when_clean_dialogue"] = comparison

    # 4) old bridge, two-speaker mode ----------------------------------------------------------------
    if args.skip_old or not args.wan2gp:
        results["old_bridge_two_speakers"] = {"skipped": True}
    else:
        print("[5/5] old Wan2GP bridge, two speakers, same clip")
        root = Path(args.wan2gp).resolve()
        worker = Path(args.comfy) / "custom_nodes" / "comfyui-wan2gp-seedvc-bridge" / "worker.py"
        target = out / "D_old_bridge.wav"
        with testlib.GpuMemoryMonitor() as monitor:
            start = time.perf_counter()
            proc = subprocess.run(
                [str(root / "env_venv" / "Scripts" / "python.exe"), str(worker), str(root),
                 str(out / "D_two_speakers_plus_music.wav"), str(target),
                 str(resolve(args, args.voice)), str(resolve(args, args.voice2))],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            elapsed = time.perf_counter() - start
        (out / "D_old_bridge.log").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode or not target.is_file():
            results["old_bridge_two_speakers"] = {"failed": True, "returncode": proc.returncode}
        else:
            wave, old_rate = base.read_wav(target)
            aligned = fit_length(resample(wave, old_rate, rate), mix.shape[-1])
            aligned = aligned if aligned.shape[0] == 2 else aligned[:1].repeat(2, 1)
            results["old_bridge_two_speakers"] = {
                "seconds_total": round(elapsed, 3), "vram_peak_nvidia_smi_delta_mb": monitor.peak_delta_mb,
                "length_diff_samples": int(wave.shape[-1] - mix.shape[-1]) if old_rate == rate else None,
                "out_sample_rate": old_rate, "similarity_after_output_minus_music": similarity_matrix(aligned - music)}
    results["new_both_converted"]["similarity_after_output_minus_music"] = similarity_matrix(output - music)

    (out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("\n" + json.dumps(results, indent=1, default=str))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
