"""Accuracy of the pack's speaker diarization on clips assembled from single-speaker recordings.

Because the clips are assembled here, it is known exactly who speaks when. Reported per clip:
  count_ok        the number of speakers (1 or 2) was detected correctly
  confusion_pct   share of actual speech time given to the wrong speaker (speakers numbered by first utterance)
  boundary_ms     how far each speaker change is from the true change (0 if it lies in the silence around it)

Usage:  python diarization_eval.py --campplus path/to/campplus_cn_common.bin voice1.wav voice2.wav ...
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from seedvc_swap import diarize as dz  # noqa: E402
from seedvc_swap.audio_utils import load_audio_file, resample, to_mono  # noqa: E402


def load_voice(path):
    wave, rate = load_audio_file(path)
    wave = resample(to_mono(wave), rate, dz.RATE)
    speech, _ = dz.speech_frames(wave)
    runs = dz._runs(speech)
    if runs:
        wave = wave[runs[0][0] * dz.FRAME: runs[-1][1] * dz.FRAME]
    return wave / (wave.pow(2).mean().sqrt() + 1e-9) * 0.08


def build_clip(voices, order, rng, turn_range, gap_range, turn_ranges=None):
    """order: list of voice indices per turn; a negative gap makes the next turn start before the last one ends.

    Returns (wave, labels per frame, change spans, speakers). Labels: speaker index, -1 pause, -2 both talking.
    `turn_ranges`: optional per-turn (min, max) seconds overriding `turn_range`.
    """
    cursor = {index: int(rng.integers(0, max(1, voices[index].numel() - dz.RATE * 6))) for index in set(order)}
    first_seen = []
    placed = []
    position = 0
    for turn, voice in enumerate(order):
        if voice not in first_seen:
            first_seen.append(voice)
        low, high = turn_ranges[turn] if turn_ranges else turn_range
        length = int(rng.uniform(low, high) * dz.RATE) // dz.FRAME * dz.FRAME
        source = voices[voice]
        start = cursor[voice] % max(1, source.numel() - length)
        piece = source[start:start + length].clone()
        cursor[voice] = start + length + dz.RATE // 2
        fade = 160
        ramp = torch.linspace(0, 1, fade)
        piece[:fade] *= ramp
        piece[-fade:] *= ramp.flip(0)
        if turn > 0:
            gap = int(rng.uniform(*gap_range) * dz.RATE) // dz.FRAME * dz.FRAME
            position = max(0, position + gap)
        placed.append((position, piece, first_seen.index(voice)))
        position += piece.numel()
    total = max(start + piece.numel() for start, piece, _ in placed)
    wave = torch.zeros(total)
    labels = np.full(total // dz.FRAME, -1, dtype=np.int8)
    changes = []
    previous = None
    for start, piece, speaker in placed:
        wave[start:start + piece.numel()] += piece
        low, high = start // dz.FRAME, (start + piece.numel()) // dz.FRAME
        both = labels[low:high] >= 0
        labels[low:high] = np.where(both, -2, speaker)
        if previous is not None and previous[2] != speaker:
            edge_a, edge_b = previous[1], low  # end of the previous turn, start of this one
            changes.append((min(edge_a, edge_b), max(edge_a, edge_b)))
        previous = (low, high, speaker)
    wave = wave + torch.from_numpy(rng.standard_normal(wave.numel()).astype(np.float32)) * 1e-4
    return wave, labels, changes, len(first_seen)


def score(result, truth, changes, true_speakers, active):
    """`active`: frames that really contain speech (the turns have natural pauses of their own)."""
    frames = min(result.labels.size, truth.size, active.size)
    predicted, truth, active = result.labels[:frames], truth[:frames], active[:frames]
    speech = (truth >= 0) & active
    # a change may sit anywhere in the silence around the true change, not only in the inserted gap
    widened = []
    for low, high in changes:
        while low > 0 and not active[low - 1]:
            low -= 1
        while high < frames - 1 and not active[high]:
            high += 1
        widened.append((low, high))
    changes = widened
    wrong = (predicted != truth) & speech
    record = {"count_ok": result.speakers == true_speakers, "found": int(result.speakers),
              "confusion_pct": round(100.0 * wrong.sum() / max(1, speech.sum()), 2),
              "centroid_similarity": result.info.get("centroid_similarity"), "info": result.info,
              "truth_second_share_pct": round(100.0 * (truth == 1).sum() / max(1, speech.sum()), 2)}
    second = (truth == 1) & active
    if second.any():  # how much of the second speaker's speech was recognised as theirs
        record["second_speaker_found_pct"] = round(100.0 * ((predicted == 1) & second).sum() / second.sum(), 2)
    both = truth == -2
    if both.any():
        record["overlap_seconds"] = round(float(both.sum()) * dz.FRAME / dz.RATE, 2)
    if true_speakers == 2 and result.speakers == 2:
        found = np.flatnonzero(np.diff(predicted) != 0) + 1
        errors = []
        for low, high in changes:
            if found.size == 0:
                errors.append(None)
                continue
            inside = found[(found >= low) & (found <= high)]
            if inside.size:
                errors.append(0.0)
            else:
                distance = np.minimum(np.abs(found - low), np.abs(found - high)).min()
                errors.append(float(distance) * dz.FRAME / dz.RATE * 1000)
        record["boundary_ms"] = errors
        record["extra_changes"] = int(max(0, found.size - len(changes)))
    return record


def summarise(rows):
    confusion = np.array([r["confusion_pct"] for r in rows])
    boundaries = [e for r in rows for e in r.get("boundary_ms", []) if e is not None]
    summary = {"clips": len(rows), "count_correct_pct": round(100 * np.mean([r["count_ok"] for r in rows]), 1),
               "confusion_pct_mean": round(float(confusion.mean()), 2),
               "confusion_pct_median": round(float(np.median(confusion)), 2),
               "confusion_pct_max": round(float(confusion.max()), 2),
               "clips_under_2pct_confusion": int((confusion < 2).sum()),
               "clips_over_30pct_confusion": int((confusion > 30).sum())}
    if boundaries:
        summary["boundaries"] = len(boundaries)
        summary["boundary_inside_pause_pct"] = round(100 * np.mean(np.array(boundaries) == 0), 1)
        summary["boundary_ms_mean"] = round(float(np.mean(boundaries)), 1)
        summary["boundary_ms_p90"] = round(float(np.percentile(boundaries, 90)), 1)
    found_second = [r["second_speaker_found_pct"] for r in rows if "second_speaker_found_pct" in r]
    if found_second:
        summary["second_speaker_found_pct_mean"] = round(float(np.mean(found_second)), 1)
    similarities = [r["centroid_similarity"] for r in rows if r.get("centroid_similarity") is not None]
    if similarities:
        summary["centroid_similarity_min_mean_max"] = [round(float(f(similarities)), 3) for f in (min, np.mean, max)]
    return summary


def run_reference(args, jobs, clusters):
    """Run the sherpa-onnx reference on the very same clips (in another Python) and score it identically."""
    import subprocess
    import tempfile
    import wave as wavelib

    scored = []
    with tempfile.TemporaryDirectory(prefix="diar_eval_") as tmp:
        files = []
        for index, (label, wave, truth, changes, speakers) in enumerate(jobs):
            path = str(Path(tmp) / f"clip_{index:04d}.wav")
            with wavelib.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(dz.RATE)
                handle.writeframes((wave.clamp(-1, 1) * 32767).short().numpy().tobytes())
            files.append(path)
        out = str(Path(tmp) / "reference.json")
        command = [args.sherpa_python, str(HERE / "ref_diarize_sherpa.py"), "--segmentation", args.sherpa_segmentation,
                   "--embedding", args.sherpa_embedding, "--out", out, "--num-clusters", str(clusters),
                   "--dir", tmp]
        env = dict(__import__("os").environ)
        if args.sherpa_pythonpath:
            env["PYTHONPATH"] = args.sherpa_pythonpath
        done = subprocess.run(command, capture_output=True, text=True, env=env)
        if done.returncode:
            raise SystemExit("reference diarization failed:\n" + done.stdout[-2000:] + done.stderr[-4000:])
        found = json.loads(Path(out).read_text(encoding="utf-8"))
        found = {Path(k).name: v for k, v in found.items()}
        for path, (label, wave, truth, changes, speakers) in zip(files, jobs):
            path = Path(path).name
            active = dz.speech_frames(wave)[0]
            frames = active.size
            labels = np.full(frames, -1, dtype=np.int8)
            order = []
            for start, end, speaker in found[path]["segments"]:
                if speaker not in order:
                    order.append(speaker)
                labels[int(start * 100):int(end * 100)] = order.index(speaker)
            covered = labels >= 0
            count = len(order)
            if count == 0:
                labels[:] = 0
                count = 1
            else:
                labels = dz._fill_pauses(labels, covered)
            if count > 2:  # more than two voices found: everything beyond the second counts as wrong
                labels = np.where(labels > 1, 9, labels).astype(np.int8)
            result = dz.Diarization(labels.astype(np.int8), count, covered, {})
            scored.append((label, score(result, truth, changes, speakers, active)))
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("voices", nargs="+")
    parser.add_argument("--campplus", required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", default=None)
    parser.add_argument("--threshold", type=float, default=dz.SAME_SPEAKER_SIMILARITY)
    parser.add_argument("--only", default=None, help="comma-separated words; run only conditions containing one")
    parser.add_argument("--sherpa-python", default=None, help="Python with sherpa_onnx, to run the reference too")
    parser.add_argument("--sherpa-pythonpath", default=None)
    parser.add_argument("--sherpa-segmentation", default=None, help="pyannote segmentation-3.0 model.onnx")
    parser.add_argument("--sherpa-embedding", default=None, help="3D-Speaker eres2net .onnx")
    args = parser.parse_args()

    from seedvc_swap.third_party.seed_vc.campplus.DTDNN import CAMPPlus

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(args.campplus, map_location="cpu", weights_only=True))
    campplus = campplus.eval().to(args.device)
    voices = [load_voice(path) for path in args.voices]
    names = [Path(path).stem for path in args.voices]
    rng = np.random.default_rng(args.seed)
    pairs = list(itertools.permutations(range(len(voices)), 2))

    conditions = {
        "two speakers, turns 1.5-4.5 s": dict(kind=2, turn=(1.5, 4.5), gap=(0.15, 0.6)),
        "two speakers, short turns 0.7-1.5 s": dict(kind=2, turn=(0.7, 1.5), gap=(0.15, 0.6)),
        "two speakers, no pause between turns": dict(kind=2, turn=(1.5, 4.5), gap=(0.0, 0.02)),
        "one speaker": dict(kind=1, turn=(1.5, 4.5), gap=(0.15, 0.6)),
        # harder cases
        "two SIMILAR voices, turns 1.5-4.5 s": dict(kind=2, turn=(1.5, 4.5), gap=(0.15, 0.6), similar=True),
        "interjections 0.4-0.9 s into long turns": dict(kind="interject", turn=(2.5, 4.5), gap=(0.1, 0.3)),
        "turns overlapping by 0.4-1.0 s": dict(kind=2, turn=(2.0, 4.5), gap=(-1.0, -0.4)),
    }
    if args.only:
        conditions = {k: v for k, v in conditions.items() if any(word in k for word in args.only.split(","))}

    # "similar" = the fifth of all voice pairs whose whole-recording CAM++ embeddings are closest
    whole = np.stack([dz.embed_windows(v, [(0, min(v.numel(), dz.RATE * 8) // dz.FRAME)], campplus, args.device)[0]
                      for v in voices])
    closeness = sorted(((float(whole[a] @ whole[b]), a, b) for a, b in pairs), reverse=True)
    similar_pairs = [(a, b) for _, a, b in closeness[: max(2, len(closeness) // 5)]]
    print("most similar pairs:", sorted({tuple(sorted((names[a], names[b]))) for a, b in similar_pairs})[:8],
          "similarity", round(closeness[len(similar_pairs) - 1][0], 2), "to", round(closeness[0][0], 2))
    report = {}
    reference_jobs = []
    for label, condition in conditions.items():
        rows = []
        for _ in range(args.trials):
            turns = int(rng.integers(3, 7))
            turn_ranges = None
            if condition["kind"] == 2:
                pool = similar_pairs if condition.get("similar") else pairs
                a, b = pool[int(rng.integers(len(pool)))]
                order = [(a, b)[i % 2] for i in range(turns)]
            elif condition["kind"] == "interject":
                a, b = pairs[int(rng.integers(len(pairs)))]
                order = [a, b, a, b, a]
                turn_ranges = [condition["turn"], (0.4, 0.9)] * 2 + [condition["turn"]]
            else:
                a = int(rng.integers(len(voices)))
                order = [a] * turns
            wave, truth, changes, speakers = build_clip(voices, order, rng, condition["turn"], condition["gap"],
                                                        turn_ranges)
            result = dz.diarize(wave, campplus, args.device, max_speakers=2,
                                same_speaker_similarity=args.threshold)
            row = score(result, truth, changes, speakers, dz.speech_frames(wave)[0])
            row["voices"] = [names[i] for i in dict.fromkeys(order)]
            row["seconds"] = round(wave.numel() / dz.RATE, 1)
            rows.append(row)
            if args.sherpa_python:
                reference_jobs.append((label, wave, truth, changes, speakers))
        summary = summarise(rows)
        report[label] = {"summary": summary, "rows": rows}
        print(label, json.dumps(summary))
    if args.sherpa_python:
        for clusters, name in ((2, "reference sherpa/pyannote-seg-3.0, 2 speakers forced (as Wan2GP)"),
                               (-1, "reference sherpa/pyannote-seg-3.0, speaker count automatic")):
            reference = run_reference(args, reference_jobs, clusters)
            print("\n" + name)
            for label in conditions:
                rows = [row for job_label, row in reference if job_label == label]
                summary = summarise(rows)
                report[f"{name} | {label}"] = {"summary": summary, "rows": rows}
                print(label, json.dumps(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
