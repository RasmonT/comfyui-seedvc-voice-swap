"""Fast checks that need no model weights and no GPU.  Run:  python -m pytest tests/test_units.py -q"""
import hashlib
import http.server
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK))

from seedvc_swap import download, pipeline, separator  # noqa: E402
from seedvc_swap.audio_utils import AudioError, fit_length, match_loudness, resample  # noqa: E402
from seedvc_swap.manifest import FILES, SETS, ModelFile, required_keys  # noqa: E402
from seedvc_swap.mel import mel_filterbank, pad_center  # noqa: E402


# ------------------------------------------------------------------------------- mel
@pytest.mark.parametrize("kwargs", [
    dict(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None),
    dict(sr=44100, n_fft=2048, n_mels=128, fmin=0, fmax=None),
    dict(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True),
])
def test_mel_matches_librosa(kwargs):
    librosa = pytest.importorskip("librosa")
    ours = mel_filterbank(**kwargs)
    theirs = librosa.filters.mel(**kwargs)
    assert ours.shape == theirs.shape and ours.dtype == theirs.dtype
    assert abs(ours - theirs).max() < 1e-7


def test_pad_center():
    assert pad_center(torch.ones(4).numpy(), size=10).tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]


# -------------------------------------------------------------------------- manifest
def test_manifest_is_complete():
    for keys in SETS.values():
        for key in keys:
            spec = FILES[key]
            assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
            assert spec.size > 0 and spec.url.startswith("https://")
    assert "bs_roformer" in required_keys("speech", True)
    assert "bs_roformer" not in required_keys("speech", False)
    assert "rmvpe" in required_keys("singing", False)
    with pytest.raises(ValueError):
        required_keys("opera", True)


# ------------------------------------------------------------------------ downloader
class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    drop_after = None  # close the connection after this many bytes (simulates an interrupted download)
    requests_seen = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = type(self).payload
        start = 0
        header = self.headers.get("Range")
        type(self).requests_seen.append(header)
        if header:
            start = int(header.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).drop_after is not None:
            body = body[: type(self).drop_after]
            type(self).drop_after = None
        self.wfile.write(body)


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _RangeHandler.requests_seen = []
    _RangeHandler.drop_after = None
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.bin"
    httpd.shutdown()


def _spec(url, payload, sha=None):
    return ModelFile("t", "sub/file.bin", url, len(payload), sha or hashlib.sha256(payload).hexdigest(), "test",
                     "sub/file.bin")


def test_download_resumes_and_verifies(server, tmp_path, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    payload = bytes(range(256)) * 4000
    _RangeHandler.payload = payload
    _RangeHandler.drop_after = 300_000
    dest = tmp_path / "sub" / "file.bin"
    download._download(_spec(server, payload), dest, log=lambda m: None)
    assert dest.read_bytes() == payload
    assert not dest.with_name("file.bin.part").exists()
    assert _RangeHandler.requests_seen == [None, "bytes=300000-"]


def test_download_resumes_existing_part_file(server, tmp_path):
    payload = b"abcdefgh" * 50_000
    _RangeHandler.payload = payload
    dest = tmp_path / "sub" / "file.bin"
    dest.parent.mkdir(parents=True)
    dest.with_name("file.bin.part").write_bytes(payload[:123_456])
    download._download(_spec(server, payload), dest, log=lambda m: None)
    assert dest.read_bytes() == payload
    assert _RangeHandler.requests_seen == ["bytes=123456-"]


def test_download_rejects_wrong_hash(server, tmp_path):
    payload = b"x" * 10_000
    _RangeHandler.payload = payload
    dest = tmp_path / "sub" / "file.bin"
    with pytest.raises(download.ModelError, match="SHA256"):
        download._download(_spec(server, payload, sha="0" * 64), dest, log=lambda m: None)
    assert not dest.exists() and not dest.with_name("file.bin.part").exists()


def test_missing_wan2gp_path_is_ignored_with_a_warning(tmp_path, monkeypatch):
    """A workflow from someone else can carry a ckpts path that does not exist here: behave as if it were empty."""
    models = tmp_path / "models"
    wanted = download.required_keys("speech", False)
    payloads = {key: f"content of {key}".encode() for key in wanted}
    fake = {key: ModelFile(key, FILES[key].path, "https://example.invalid/" + key, len(data),
                           hashlib.sha256(data).hexdigest(), "test", FILES[key].wan2gp)
            for key, data in payloads.items()}
    monkeypatch.setattr(download, "FILES", fake)
    for key, data in payloads.items():  # models already present in the normal folder
        target = models / fake[key].path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    messages = []
    missing = str(tmp_path / "somebody" / "elses" / "Wan2GP" / "ckpts")
    resolved = download.ensure_models(models, "speech", False, wan2gp_ckpts=missing, log=messages.append,
                                      allow_download=False)
    assert set(resolved) == set(wanted)
    assert all(Path(path).is_relative_to(models) for path in resolved.values())
    assert any("WARNING" in m and "does not exist" in m and missing in m for m in messages)
    # and with nothing downloaded yet it goes on to the download step instead of failing on the path
    with pytest.raises(download.ModelError, match="downloading is disabled"):
        download.ensure_models(tmp_path / "empty_models", "speech", False, wan2gp_ckpts=missing,
                               log=lambda m: None, allow_download=False)


def test_existing_wan2gp_folder_with_bad_content_is_still_an_error(tmp_path, monkeypatch):
    ckpts = tmp_path / "ckpts"
    wanted = download.required_keys("speech", False)
    payloads = {key: f"content of {key}".encode() for key in wanted}
    fake = {key: ModelFile(key, FILES[key].path, "https://example.invalid/" + key, len(data),
                           hashlib.sha256(data).hexdigest(), "test", FILES[key].wan2gp)
            for key, data in payloads.items()}
    monkeypatch.setattr(download, "FILES", fake)
    for key, data in payloads.items():
        target = ckpts / fake[key].wan2gp
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert set(download.ensure_models(tmp_path / "m", "speech", False, wan2gp_ckpts=str(ckpts),
                                      log=lambda m: None)) == set(wanted)
    corrupt = ckpts / fake["campplus"].wan2gp
    corrupt.write_bytes(payloads["campplus"][:-1] + b"X")  # same size, wrong content
    with pytest.raises(download.ModelError, match="SHA256"):
        download.ensure_models(tmp_path / "m2", "speech", False, wan2gp_ckpts=str(ckpts), log=lambda m: None)
    corrupt.unlink()
    with pytest.raises(download.ModelError, match="missing 1 required"):
        download.ensure_models(tmp_path / "m3", "speech", False, wan2gp_ckpts=str(ckpts), log=lambda m: None)


def test_ensure_models_errors_are_clear(tmp_path):
    empty = tmp_path / "ckpts"
    empty.mkdir()
    with pytest.raises(download.ModelError, match="missing 9 required"):
        download.ensure_models(tmp_path, "speech", True, wan2gp_ckpts=str(empty))
    with pytest.raises(download.ModelError, match="downloading is disabled"):
        download.ensure_models(tmp_path, "speech", False, allow_download=False)


def test_verify_detects_corruption(tmp_path):
    payload = b"model-bytes" * 100
    path = tmp_path / "f.bin"
    path.write_bytes(payload)
    cache = download._VerifiedCache(tmp_path / "verified.json")
    spec = _spec("https://example.invalid/f", payload)
    download._verify(path, spec, cache, log=lambda m: None)
    assert cache.is_verified(path, spec.sha256)
    path.write_bytes(payload[:-1] + b"X")  # same size, different content
    with pytest.raises(download.ModelError, match="SHA256"):
        download._verify(path, spec, download._VerifiedCache(tmp_path / "other.json"), log=lambda m: None)


# ----------------------------------------------------------------------------- audio
def test_fit_length():
    x = torch.arange(10.0)[None]
    assert fit_length(x, 10) is x
    assert fit_length(x, 7).shape[-1] == 7
    padded = fit_length(x, 13)
    assert padded.shape[-1] == 13 and padded[0, 10:].abs().sum() == 0


def test_match_loudness_restores_level_and_balance():
    torch.manual_seed(0)
    rate = 16000
    speech = torch.randn(rate * 2) * 0.1
    speech[:rate // 2] = 0  # a pause must not skew the measurement
    original = torch.stack([speech, speech * 0.5])
    converted = torch.randn(rate * 2) * 0.03
    converted[:rate // 2] = 0
    out, gains = match_loudness(converted, original, rate)
    active = slice(rate // 2, None)
    for channel in range(2):
        want = original[channel, active].pow(2).mean().sqrt()
        got = out[channel, active].pow(2).mean().sqrt()
        assert abs(float(got / want) - 1) < 0.02
    assert abs(gains[0] / gains[1] - 2.0) < 0.05


# ------------------------------------------------------------------------- separator
class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("length", [1000, 44100 * 3, 352800, 352800 + 1, 44100 * 23 + 17])
def test_overlap_add_is_transparent(length):
    """With a model that returns its input, chunked processing must reproduce the signal exactly."""
    torch.manual_seed(1)
    mix = torch.randn(2, length) * 0.1
    out = separator.separate_vocals(_Identity(), mix, "cpu", fp16=False)
    assert out.shape == mix.shape
    assert float((out - mix).abs().max()) < 1e-5


@pytest.mark.parametrize("channels,rate,length", [(1, 48000, 48000 * 5 + 3), (2, 44100, 44100 * 9), (2, 24000, 77777)])
def test_split_keeps_length_and_sums_to_input(channels, rate, length):
    torch.manual_seed(2)
    audio = torch.randn(channels, length) * 0.1
    vocals, background = separator.split_vocals_background(_Identity(), audio, rate, "cpu", fp16=False)
    assert vocals.shape == audio.shape == background.shape
    assert float((vocals + background - audio).abs().max()) < 1e-6


def test_bs_roformer_builds_with_expected_layout():
    model = separator.build_model()
    keys = model.state_dict().keys()
    assert any(k.startswith("band_split.") for k in keys) and any(k.startswith("mask_estimators.0.") for k in keys)


# -------------------------------------------------------------------------- pipeline
class _FakeEngine:
    """Stands in for Seed-VC: returns the source rounded down to whole mel frames, like the real model."""
    sr, hop = 22050, 256

    def __init__(self, *args, **kwargs):
        pass

    def convert(self, source, reference, **kwargs):
        frames = source.numel() // self.hop
        return source[: frames * self.hop].clone() * 0.5

    def unload(self):
        pass


@pytest.mark.parametrize("shape,rate", [((1, 48000 * 4 + 1), 48000), ((2, 44100 * 7 + 123), 44100),
                                         ((2, 2, 24000 * 3), 24000), ((1, 16000 * 2 + 5), 16000)])
@pytest.mark.parametrize("keep_background", [False, True])
def test_output_has_exactly_the_input_shape(shape, rate, keep_background, monkeypatch):
    monkeypatch.setattr(pipeline, "SeedVCEngine", _FakeEngine)
    monkeypatch.setattr(separator, "load_model", lambda path, device: _Identity())
    torch.manual_seed(3)
    audio = torch.randn(*shape) * 0.1
    reference = torch.randn(22050 * 3) * 0.1
    out, stats = pipeline.voice_swap(audio, rate, reference, 22050, {"bs_roformer": "unused"},
                                     keep_background=keep_background, device="cpu", fp16=False, log=lambda m: None)
    assert out.shape == audio.shape and out.dtype == torch.float32
    assert stats["length"] == shape[-1]


def test_unsupported_channel_count_is_reported(monkeypatch):
    monkeypatch.setattr(pipeline, "SeedVCEngine", _FakeEngine)
    with pytest.raises(AudioError, match="mono and stereo"):
        pipeline.voice_swap(torch.zeros(6, 1000), 48000, torch.zeros(22050), 22050, {}, keep_background=False,
                            device="cpu")


def test_silent_input_passes_through(monkeypatch):
    monkeypatch.setattr(pipeline, "SeedVCEngine", _FakeEngine)
    audio = torch.zeros(2, 48000)
    out, stats = pipeline.voice_swap(audio, 48000, torch.randn(22050), 22050, {}, keep_background=False,
                                     device="cpu", log=lambda m: None)
    assert out.shape == audio.shape and float(out.abs().max()) == 0
    assert stats["items"][0]["skipped"] == "silent"


def test_resample_identity_and_length():
    x = torch.randn(1, 48000)
    assert resample(x, 48000, 48000) is x
    assert abs(resample(x, 48000, 22050).shape[-1] - 22050) <= 1


# ----------------------------------------------------------------------- two speakers
from seedvc_swap import diarize as dz  # noqa: E402


def test_smooth_path_follows_strong_evidence_and_ignores_weak_blips():
    speech = np.ones(600, dtype=bool)
    speech[195:205] = False  # a pause where the speakers change
    margin = np.full(600, -0.4)
    margin[200:400] = 0.4  # speaker 2 for two seconds
    margin[80:110] = 0.1  # 0.3 s of weak contrary evidence inside speaker 1's turn
    margin[480:560] = 0.45  # a clear 0.8 s interjection of speaker 2, no pause around it
    margin[~speech] = 0
    labels = dz._smooth_path(margin, speech)
    assert labels[:195].sum() == 0, "weak blip must follow its neighbours"
    assert labels[205:400].all()
    assert labels[490:550].all(), "clear interjection keeps its own turn"
    assert labels[410:470].sum() == 0 and labels[570:].sum() == 0


def test_fill_pauses_partitions_the_timeline():
    labels = np.array([-1, -1, 0, 0, -1, -1, -1, -1, 1, 1, -1], dtype=np.int8)
    speech = labels >= 0
    filled = dz._fill_pauses(labels.copy(), speech)
    assert filled.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_speaker_masks_sum_to_one_and_follow_labels():
    labels = np.array([0] * 100 + [1] * 100 + [0] * 50, dtype=np.int8)
    found = dz.Diarization(labels, 2, np.ones(250, dtype=bool))
    masks = dz.speaker_masks(found, 2.5 * 48000, 48000) if False else dz.speaker_masks(found, 120000, 48000)
    assert masks.shape == (2, 120000)
    assert float((masks.sum(dim=0) - 1).abs().max()) < 1e-6
    assert float(masks[0, 24000]) == 1.0 and float(masks[1, 72000]) == 1.0 and float(masks[0, 110000]) == 1.0
    assert 0.3 < float(masks[1, 48000]) < 0.7, "crossfade is centred on the change"
    single = dz.speaker_masks(dz.Diarization(np.zeros(10, dtype=np.int8), 1, np.ones(10, dtype=bool)), 500, 16000)
    assert single.shape == (1, 500) and float(single.min()) == 1.0


class _VoiceEngine(_FakeEngine):
    """Marks which target voice was used: the output is the source scaled by the reference's first sample."""
    device = "cpu"
    campplus = None

    def convert(self, source, reference, **kwargs):
        frames = source.numel() // self.hop
        return source[: frames * self.hop].clone() * float(reference[0])


def _two_speaker_setup(monkeypatch, length, rate):
    frames = int(length / rate * dz.RATE) // dz.FRAME
    labels = np.zeros(frames, dtype=np.int8)
    labels[frames // 2:] = 1
    monkeypatch.setattr(pipeline, "SeedVCEngine", _VoiceEngine)
    monkeypatch.setattr(dz, "diarize", lambda wave, campplus, device, **kw: dz.Diarization(
        labels, 2, np.ones(frames, dtype=bool), {}))
    monkeypatch.setattr(pipeline, "match_loudness",
                        lambda converted, original, rate, region=None: (converted.repeat(original.shape[0], 1), [1.0]))


def test_two_speakers_each_get_their_own_voice(monkeypatch):
    rate, length = 22050, 22050 * 4
    _two_speaker_setup(monkeypatch, length, rate)
    audio = torch.ones(1, length) * 0.1
    voice_a, voice_b = torch.full((22050,), 2.0), torch.full((22050,), 3.0)
    out, stats = pipeline.voice_swap_speakers(audio, rate, [(voice_a, rate), (voice_b, rate)], {}, speakers=2,
                                              keep_background=False, device="cpu", fp16=False, log=lambda m: None)
    assert out.shape == audio.shape
    assert abs(float(out[0, length // 4]) - 0.2) < 1e-4, "first half converted with voice 1"
    assert abs(float(out[0, 3 * length // 4]) - 0.3) < 1e-4, "second half converted with voice 2"
    assert [e["voice"] for e in stats["items"][0]["speakers"]] == ["converted", "converted"]


def test_speaker_without_a_slot_keeps_the_original_voice(monkeypatch):
    rate, length = 22050, 22050 * 4
    _two_speaker_setup(monkeypatch, length, rate)
    torch.manual_seed(5)
    audio = torch.randn(2, length) * 0.1
    out, stats = pipeline.voice_swap_speakers(audio, rate, [None, (torch.full((22050,), 2.0), rate)], {},
                                              speakers=2, keep_background=False, device="cpu", fp16=False,
                                              log=lambda m: None)
    quarter = length // 4
    assert torch.equal(out[:, :quarter], audio[:, :quarter]), "speaker 1 has no slot: bit-identical to the input"
    assert not torch.allclose(out[:, 3 * quarter:], audio[:, 3 * quarter:])
    assert [e["voice"] for e in stats["items"][0]["speakers"]] == ["original", "converted"]


def test_one_speaker_found_with_only_slot_2_leaves_audio_unchanged(monkeypatch):
    rate, length = 22050, 22050 * 2
    monkeypatch.setattr(pipeline, "SeedVCEngine", _VoiceEngine)
    frames = int(length / rate * dz.RATE) // dz.FRAME
    monkeypatch.setattr(dz, "diarize", lambda wave, campplus, device, **kw: dz.Diarization(
        np.zeros(frames, dtype=np.int8), 1, np.ones(frames, dtype=bool), {"reason": "test"}))
    audio = torch.randn(1, length) * 0.1
    messages = []
    out, stats = pipeline.voice_swap_speakers(audio, rate, [None, (torch.ones(22050), rate)], {}, speakers=2,
                                              keep_background=False, device="cpu", fp16=False, log=messages.append)
    assert torch.equal(out, audio)
    assert stats["items"][0]["speakers_found"] == 1
    notice = "\n".join(messages)
    assert "ONLY ONE SPEAKER FOUND" in notice and "NOTHING was converted" in notice and "prompt" in notice


def test_one_speaker_notice_says_what_happened():
    both = pipeline._one_speaker_notice("too alike (similarity 0.75 > 0.45)", [object(), object()])
    assert "similarity 0.75 > 0.45" in both and "Voice 2 was NOT used" in both
    assert "never talk at the same time" in both and "README" in both
    assert all(line.startswith("[SeedVC]") for line in both.splitlines())


def test_no_target_voice_is_an_error():
    with pytest.raises(AudioError, match="No target voice"):
        pipeline.voice_swap_speakers(torch.zeros(1, 1000), 16000, [None, None], {}, speakers=2, device="cpu")


def test_every_checkpoint_is_loaded_in_safe_mode():
    """No torch.load in the pack may unpickle arbitrary objects."""
    offenders = []
    for path in (PACK / "seedvc_swap").rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "torch.load(" in line and "weights_only=True" not in line:
                offenders.append(f"{path.relative_to(PACK)}:{number}")
    assert not offenders, offenders
