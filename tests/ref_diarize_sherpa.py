"""Reference diarization for comparison only: pyannote segmentation-3.0 (ONNX) + 3D-Speaker embeddings,
run through sherpa-onnx with the settings Wan2GP uses. Not used by the node.

Runs under any Python that has `sherpa_onnx` and `numpy` (for example Wan2GP's env_venv):
  python ref_diarize_sherpa.py --segmentation model.onnx --embedding eres2net.onnx --out result.json --dir clips

Input: 16 kHz mono 16-bit WAV files. Output: {file: {"segments": [[start_s, end_s, speaker], ...]}}.
"""
import argparse
import json
import wave

import numpy as np
import sherpa_onnx


def read_wav(path):
    with wave.open(path, "rb") as handle:
        if handle.getframerate() != 16000 or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 kHz mono 16-bit PCM")
        data = handle.readframes(handle.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--dir", default=None, help="process every .wav in this folder")
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-clusters", type=int, default=2, help="2 as in Wan2GP; -1 lets the threshold decide")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=args.segmentation)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=args.embedding),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=args.num_clusters, threshold=args.threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise SystemExit("invalid sherpa-onnx diarization config (check the model paths)")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    files = list(args.files)
    if args.dir:
        import glob
        import os

        files += sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    results = {}
    for path in files:
        samples = read_wav(path)
        found = diarizer.process(samples).sort_by_start_time()
        results[path] = {"segments": [[float(s.start), float(s.end), int(s.speaker)] for s in found]}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle)


if __name__ == "__main__":
    main()
