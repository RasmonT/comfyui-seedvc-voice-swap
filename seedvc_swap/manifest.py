"""Model files used by the pack: where they come from, where they live, how they are verified.

Every download URL is pinned to an exact upstream revision and every weight file
has a SHA256 that is checked before the file is ever loaded.

`path` is relative to ComfyUI/models/seedvc. `wan2gp` is the same file's location
inside a Wan2GP `ckpts` folder (used when the user points the node at one).
"""
from dataclasses import dataclass
from typing import Optional

HF = "https://huggingface.co/{repo}/resolve/{rev}/{file}"

_SEEDVC = ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e")
_CAMPPLUS = ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede")
_RVC = ("lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d")
_BIGVGAN22 = ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513")
_BIGVGAN44 = ("nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb")
_WHISPER = ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")


@dataclass(frozen=True)
class ModelFile:
    key: str
    path: str
    url: str
    size: int
    sha256: str
    license: str
    wan2gp: str


def _hf(repo_rev, file):
    return HF.format(repo=repo_rev[0], rev=repo_rev[1], file=file)


FILES = {f.key: f for f in [
    ModelFile("dit_speech", "seed-vc/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
              _hf(_SEEDVC, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"), 440312082,
              "8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720", "GPL-3.0",
              "seed-vc/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
    ModelFile("dit_speech_config", "seed-vc/config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
              _hf(_SEEDVC, "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"), 2289,
              "27a91ac41bdb41d0b6951c63642d791b4a70d60b9fc5550eb645f9ddcb3c94da", "GPL-3.0",
              "seed-vc/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
    ModelFile("dit_singing", "seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
              _hf(_SEEDVC, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"), 820865494,
              "42aef93ffe65857c840d270252fa040f7ba04514945ec460f3ac1ac2a96de684", "GPL-3.0",
              "seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
    ModelFile("dit_singing_config", "seed-vc/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
              _hf(_SEEDVC, "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"), 2408,
              "ff86f343b08c2def405f075022eeb10dafba23c4eda36d151010f4ca6d7a1a0a", "GPL-3.0",
              "seed-vc/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
    ModelFile("campplus", "seed-vc/campplus_cn_common.bin",
              _hf(_CAMPPLUS, "campplus_cn_common.bin"), 28036335,
              "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8", "Apache-2.0",
              "seed-vc/campplus_cn_common.bin"),
    ModelFile("rmvpe", "seed-vc/rmvpe.pt",
              _hf(_RVC, "rmvpe.pt"), 181184272,
              "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193", "MIT",
              "seed-vc/rmvpe.pt"),
    ModelFile("bigvgan22_config", "bigvgan_v2_22khz_80band_256x/config.json",
              _hf(_BIGVGAN22, "config.json"), 1405, "88a1f47acf747db0b21e97a389d838566147f7a5464583ff5c8d819d870f03ee", "MIT",
              "bigvgan_v2_22khz_80band_256x/config.json"),
    ModelFile("bigvgan22", "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt",
              _hf(_BIGVGAN22, "bigvgan_generator.pt"), 449228171,
              "e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced", "MIT",
              "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt"),
    ModelFile("bigvgan44_config", "bigvgan_v2_44khz_128band_512x/config.json",
              _hf(_BIGVGAN44, "config.json"), 1403, "65b7f487bfaf15256056a75f6667ef5f640214c981922a111927873705ea4ddb", "MIT",
              "bigvgan_v2_44khz_128band_512x/config.json"),
    ModelFile("bigvgan44", "bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt",
              _hf(_BIGVGAN44, "bigvgan_generator.pt"), 489041291,
              "d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357", "MIT",
              "bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt"),
    ModelFile("whisper_config", "whisper-small/config.json",
              _hf(_WHISPER, "config.json"), 1967, "e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4", "Apache-2.0",
              "whisper-small/config.json"),
    ModelFile("whisper_preprocessor", "whisper-small/preprocessor_config.json",
              _hf(_WHISPER, "preprocessor_config.json"), 184990, "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666", "Apache-2.0",
              "whisper-small/preprocessor_config.json"),
    ModelFile("whisper", "whisper-small/model.safetensors",
              _hf(_WHISPER, "model.safetensors"), 966995080,
              "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b", "Apache-2.0",
              "whisper-small/model.safetensors"),
    ModelFile("bs_roformer", "roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
              "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
              "model_bs_roformer_ep_317_sdr_12.9755.ckpt", 639331213,
              "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
              "not stated by the author (viperx); distributed with UVR / python-audio-separator",
              "roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
]}

_COMMON = ["campplus", "whisper_config", "whisper_preprocessor", "whisper"]
SETS = {
    "speech": ["dit_speech", "dit_speech_config", "bigvgan22_config", "bigvgan22"] + _COMMON,
    "singing": ["dit_singing", "dit_singing_config", "rmvpe", "bigvgan44_config", "bigvgan44"] + _COMMON,
    "separator": ["bs_roformer"],
}


def required_keys(mode, keep_background):
    if mode not in ("speech", "singing"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'speech' or 'singing'.")
    keys = list(SETS[mode])
    if keep_background:
        keys += SETS["separator"]
    return keys


def total_size(keys):
    return sum(FILES[k].size for k in keys)
