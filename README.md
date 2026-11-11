# comfyui-seedvc-voice-swap

Replace the voice in an audio clip with another voice, inside ComfyUI, using
[Seed-VC](https://github.com/Plachtaa/seed-vc) zero-shot voice conversion.

- Keeps music and background sounds: only the voice is converted.
- Output has **exactly** the same length (in samples), sample rate and channel count as the
  input, so it stays in sync with video.
- One voice for everything, or two speakers each with their own voice.
- Standalone: no Wan2GP needed. Models download automatically, every file is checked against a SHA256.
- Installs no new packages on a normal ComfyUI and never touches torch or numpy.

> **Use it only with voices you have the rights to use.** Cloning or replacing someone's voice
> without their consent can be illegal and is your responsibility, not the authors'.

*Slovenské zhrnutie je [na konci](#slovensky).*

## Install

ComfyUI Manager: search for **Seed-VC Voice Swap**. Or manually:

```
cd ComfyUI/custom_nodes
git clone https://github.com/rasmont/comfyui-seedvc-voice-swap
```

Restart ComfyUI. The nodes are under `audio/voice conversion`; ready-made graphs are in
`example_workflows/`.

`requirements.txt` lists only packages ComfyUI already ships with or commonly has (transformers,
einops, pyyaml, av, rotary-embedding-torch) and pins no versions, so installing it does not change
anything that is already installed.

## Node: Seed-VC Voice Swap (1 voice)

All speech in the clip gets one voice.

| Input | Default | Meaning |
|---|---|---|
| `audio` | | Audio to convert, mono or stereo. |
| `enabled` | on | Off: audio passes through untouched, nothing is loaded. |
| `voice_sample` | `(none)` | Target voice, a file from `ComfyUI/input` (subfolders included). 5-25 s of clean speech; only the first 25 s are used, at least 1 s is required. |
| `keep_background` | on | Separate music and noises, convert only the voice, mix the background back. Off: the whole input is treated as voice (faster, for clean speech). |
| `mode` | `speech` | `speech` (22 kHz model) or `singing` (44 kHz model, follows the melody). |
| `singing_pitch` | `octave` | Singing only. `octave`: shift by whole octaves towards the sample's range (stays in key). `original`: keep the sung pitch. `match`: follow the sample's pitch exactly (may leave the key). |
| `steps` | 25 | Diffusion steps, 1-100. 25 for speech, 30-50 for singing. |
| `cfg` | 0.5 | Guidance strength, 0-1. Higher follows the voice sample more closely. |
| `seed` | 0 | Same seed and inputs give the same result. |
| `wan2gp_ckpts_path` | empty | Optional path to a Wan2GP `ckpts` folder: its models are used and nothing is downloaded. A path that does not exist on this computer (for example in a workflow you got from someone else) is ignored with a console warning, and the node works as if the field were empty. |

Output: `audio`.

## Node: Seed-VC Voice Swap (2 speakers)

Same inputs, with two voice slots and a speaker count instead of `voice_sample`:

| Input | Default | Meaning |
|---|---|---|
| `voice_1_on`, `voice_1` | on, `(none)` | Voice for **speaker 1 = whoever speaks first**. Off: speaker 1 keeps the original voice. |
| `voice_2_on`, `voice_2` | off, `(none)` | Voice for **speaker 2 = the other speaker**. Off: speaker 2 keeps the original voice. |
| `speakers` | `auto` | `auto`: 1 if only voice 1 is on (all speech gets voice 1, nobody is told apart), 2 if voice 2 is on. Set `2` with only voice 1 on to convert the first speaker and leave the second untouched. |

The console prints what was found, for example
`[SeedVC] speakers: #1 6.4 s -> converted, #2 5.9 s -> converted`.

### How speakers are told apart, and where it fails

The separated voice track is cut into 1.5 s windows, each window gets a speaker embedding from the
CAM++ model that Seed-VC needs anyway, and the windows are grouped into two voices. No gated model
and no extra download is involved.

Limits you should know:

- **Turns should be about 1.5 s or longer.** Shorter turns are often given to the wrong speaker.
- **Speakers are numbered by who talks first.** If the very first utterance is attributed wrongly,
  both voices are swapped for the whole clip.
- **A second voice with less than 0.8 s of speech in total is ignored**, and the clip is treated as
  one speaker (everything gets voice 1). A single "yes" or "hm" from the other person is therefore
  not converted separately.
- **Two people talking at the same time cannot be separated.** That stretch gets one of the two voices.
- **Very similar voices** may be judged to be one person (the two voice clusters are compared; above
  a similarity of 0.45 they are merged).
- Short interjections under one second are usually absorbed into the surrounding speaker
  (measured: about three quarters of such speech is not recognised as the second speaker).

Measured accuracy on assembled test clips is in [Measurements](#measurements).

### Prompt for two speakers

When the audio comes from a generated video, the prompt decides whether the two speakers can be
told apart. Video models tend to give both characters nearly the same voice unless told otherwise,
and then the node correctly reports one speaker. What helps:

- **Give the two characters clearly different voices**, for example "in a calm voice" and "in a deep voice".
- **Make the first line at least 2-3 seconds long.** It decides who is speaker 1.
- **Keep every line at 1.5 seconds or longer.**
- **Put a pause between the lines** ("She pauses.", "A short pause.").
- **Say that they never talk at the same time.**
- No music and a quiet room make it easier still.

A prompt that worked in testing (10 s clip, four lines of 1.7-2.5 s with pauses of about 0.4 s, all
four attributed to the right speaker):

> A woman and a man sit side by side facing the camera, static shot, quiet room, no music. The woman
> speaks first in a calm voice: "I honestly thought this place would be much more crowded today." She
> pauses. The man answers in a deep voice: "Give it an hour, everyone shows up right after lunch." A
> short pause. The woman smiles and says: "Then we should order now, before the line gets long." The
> man nods and says: "Good idea, I'll get us two coffees." They never talk at the same time.

If the node still finds one speaker, the console shows an `ONLY ONE SPEAKER FOUND` block with the
reason (for example `similarity 0.75 > 0.45`: the two voices sounded like one person) and what
happened to the audio. Regenerate with more distinct voices; the similarity has to come out below 0.45.

## Models

Downloaded on first use into `ComfyUI/models/seedvc`. Downloads resume after an interruption and
each file is verified against a SHA256 before it is loaded; all weights are loaded in PyTorch's
safe (`weights_only`) mode.

| What | Files | Size | Licence |
|---|---|---|---|
| Seed-VC speech model | `seed-vc/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` + config | 440 MB | GPL-3.0 |
| CAM++ speaker encoder | `seed-vc/campplus_cn_common.bin` | 28 MB | Apache-2.0 |
| BigVGAN vocoder 22 kHz | `bigvgan_v2_22khz_80band_256x/` | 449 MB | MIT |
| Whisper small (content encoder) | `whisper-small/` | 967 MB | Apache-2.0 |
| BS-Roformer vocal separation (only for `keep_background`) | `roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 639 MB | **not stated by its author** |
| Singing only: Seed-VC singing model, RMVPE, BigVGAN 44 kHz | `seed-vc/`, `bigvgan_v2_44khz_128band_512x/` | 1491 MB | GPL-3.0, MIT, MIT |

Speech with background: 2.5 GB. With singing as well: 4.0 GB.

**About the BS-Roformer checkpoint:** it is the widely used "viperx 1297" vocal model distributed
with Ultimate Vocal Remover and python-audio-separator, but its author has not published a licence
for it. It is downloaded only when `keep_background` is on. If that matters to you, switch
`keep_background` off or use the node on audio without music.

Exact sources, pinned revisions and hashes: [`seedvc_swap/manifest.py`](seedvc_swap/manifest.py).
Authors and licences: [NOTICE](NOTICE).

## Measurements

RTX 5090 Laptop (24 GB), Windows 11, ComfyUI 0.44, PyTorch 2.13, 10 s stereo clip at 44.1 kHz,
speech mode, 25 steps. Scripts are in `tests/`.

| | 1 voice, with background | 1 voice, speech only | 2 speakers, with background (15 s clip) |
|---|---|---|---|
| Output length vs input | 0 samples difference | 0 | 0 |
| Time, first run after start | 29-41 s | | 48 s |
| Time, later runs | 6.6 s | 3.5 s | 11.5 s |
| Peak VRAM (PyTorch allocated) | 1.46 GB | 1.46 GB | 1.57 GB |
| VRAM still allocated afterwards | 0 MB | 0 MB | 0 MB |

- Voice similarity to the sample (cosine of ERes2Net speaker embeddings, an independent model):
  0.15 before, 0.84 after (mean of 8 seeds, standard deviation 0.014).
- Same seed twice: bit-identical output. The new voice is about 4 ms early relative to the original
  (a property of Seed-VC).
- Background kept with `keep_background`, as signal-to-distortion ratio of the background stem with
  music 6-9 dB below the speech: 24.5 dB with a synthetic backing track, 17-22 dB with real
  instrumental music (CPU measurements), 40 dB and more in the pauses between words. The figure drops
  when the speech recording itself contains noise or effects, because those end up in the background.
- Two speakers on an assembled dialogue (3 s turns): 0 % of speech given to the wrong speaker;
  each speaker's similarity to their own target voice 0.80 / 0.86, to the other voice 0.17 / 0.22.

Speaker separation on 60 assembled clips per case (13 recordings, exact ground truth; "wrong" =
share of speech time given to the wrong speaker):

| Case | Mean wrong | Median wrong | Clips under 2 % wrong | Clips over 30 % wrong |
|---|---|---|---|---|
| Turns 1.5-4.5 s with pauses | 3.6 % | 0 % | 46 | 2 |
| Short turns 0.7-1.5 s | 17.5 % | 1.3 % | 30 | 12 |
| Turns without a pause between them | 9.6 % | 7.0 % | 8 | 3 |
| Two similar voices, turns 1.5-4.5 s | 8.6 % | 0 % | 39 | 7 |
| Turns overlapping by 0.4-1.0 s (overlap itself not counted) | 10.2 % | 5.2 % | 9 | 5 |
| 0.4-0.9 s interjections into long turns | 11.5 % | 10.1 % | 8 | 4 |

Interjections under one second are the weakest case: only about a quarter of the interjected speech
is recognised as the second speaker, the rest stays with the main speaker. In 63 % of those clips
the second voice is dropped altogether (too little speech, see the 0.8 s rule above).

For comparison, pyannote segmentation-3.0 with 3D-Speaker embeddings (through sherpa-onnx, two speakers
forced) on the very same clips: 24.1 % mean wrong for turns with pauses, 21.0 % for similar voices,
17.3 % for overlapping turns, 22.8 % for interjections. These clips are spliced from read speech and the
thresholds here were tuned on clips of the same kind, so the comparison favours this pack; natural
conversation may behave differently.

A clip with one speaker is recognised as such in 98 % of cases.

## Troubleshooting

| Message | What to do |
|---|---|
| `no voice sample selected` | Pick a file in the voice slot, or switch the slot / `enabled` off. |
| `voice sample not found` | The file was moved or deleted. Pick another one (press R to refresh the list). |
| `The voice sample is too short` | Use at least 1 s; 5-25 s of clean speech works best. |
| console: `WARNING: wan2gp_ckpts_path does not exist` | Not an error: the path is ignored and models are taken from (or downloaded to) `ComfyUI/models/seedvc`. Clear the field to silence it. |
| `The Wan2GP ckpts folder is missing N required file(s)` | The folder exists but is incomplete. Run that mode once in Wan2GP, or clear the field so the node downloads its own models. |
| `failed the SHA256 check` | The file is corrupt. Delete it and run again. |
| `Could not download ...` | Check the connection and run again; the download resumes where it stopped. |
| console block `ONLY ONE SPEAKER FOUND` | The voices are too alike or the second one speaks under 0.8 s. Everything got voice 1. See [Prompt for two speakers](#prompt-for-two-speakers). |
| `Only mono and stereo audio are supported` | Downmix the audio to stereo first. |

## Licence

GPL-3.0, inherited from Seed-VC. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the third-party
code and models.

---

## Slovensky

Node na výmenu hlasu v zvuku cez Seed-VC, bez Wan2GP.

- **Seed-VC Voice Swap (1 voice):** celá reč dostane jeden hlas.
- **Seed-VC Voice Swap (2 speakers):** slot 1 je ten, kto prehovorí prvý, slot 2 druhý rečník.
  Rečník bez zapnutého slotu si nechá pôvodný hlas.
- Výstup má presne rovnakú dĺžku, vzorkovaciu frekvenciu aj počet kanálov ako vstup.
- `keep_background` oddelí hudbu a ruchy, prevedie len hlas a pozadie vráti späť.
- Vzorky hlasov sa vyberajú zo súborov v `ComfyUI/input` (aj z podpriečinkov); ideálne 5 až 25 s čistej reči.
- Pri generovanom videu s dvoma postavami im v prompte dajte zreteľne odlišné hlasy, repliky od 1,5 s
  (prvú aspoň 2 až 3 s), pauzy medzi replikami a „they never talk at the same time"; vzorový prompt je
  v časti „Prompt for two speakers".
- Cesta vo `wan2gp_ckpts_path`, ktorá na počítači neexistuje, sa ignoruje s varovaním v konzole.
- Modely (2,5 GB pre reč, 4,0 GB so spevom) sa stiahnu samy do `ComfyUI/models/seedvc` s kontrolou
  SHA256. Do poľa `wan2gp_ckpts_path` môžete zadať priečinok `ckpts` z Wan2GP a nič sa nesťahuje.
- Obmedzenia pri dvoch rečníkoch: repliky kratšie než asi 1,5 s sa často priradia zle; druhý hlas
  s menej než 0,8 s reči sa ignoruje; súčasne hovoriacich oddeliť nevie.
- **Používajte len s hlasmi, na ktoré máte práva.**
- Licencia GPL-3.0. Model na oddelenie hlasu (BS-Roformer) nemá od autora uvedenú licenciu;
  sťahuje sa len pri zapnutom `keep_background`.
