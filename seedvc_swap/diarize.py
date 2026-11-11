"""Who speaks when, for at most two speakers, on an already isolated voice track.

No gated model is involved: speech is found by energy (the input is a clean voice stem),
speaker identity comes from the CAM++ encoder that Seed-VC needs anyway, and the two
speakers are found by clustering short overlapping windows.

Output is a full partition of the timeline: every 10 ms frame belongs to exactly one
speaker (pauses are split between their neighbours), so nothing of the original voice
leaks through at a missed word edge. Speakers are numbered by their first utterance.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio

RATE = 16000
FRAME = 160  # 10 ms
WINDOW_S = 1.5
HOP_S = 0.25
MIN_WINDOW_S = 0.5
ACTIVITY_REL_DB = -40.0
ACTIVITY_FLOOR = 1e-4
MERGE_GAP_S = 0.15
MIN_SPEECH_S = 0.10
SAME_SPEAKER_SIMILARITY = 0.45  # centroids closer than this are one person (calibrated with tests/diarization_eval.py)
MIN_SPEAKER_S = 0.8  # a "second speaker" with less speech than this is treated as noise
SWITCH_COST_SPEECH = 25.0  # in units of (similarity margin x 10 ms frames)
SWITCH_COST_PAUSE = 8.0
MIN_CLUSTER_SHARE = 0.15
BOUNDARY_SEARCH_S = 0.6
BOUNDARY_CONTEXT_S = 1.0
BOUNDARY_STEP_S = 0.05
REFINE_ROUNDS = 3
DIP_RATIO = 0.12
FINE_SEARCH_FRAMES = 16  # 160 ms
FINE_CONTEXT_FRAMES = 60  # 600 ms
CONFIDENT_SHARE = 0.6


@dataclass
class Diarization:
    labels: np.ndarray  # int8 per 10 ms frame, values 0 .. speakers-1
    speakers: int
    speech: np.ndarray  # bool per frame
    info: dict = field(default_factory=dict)

    def seconds(self, speaker):
        return float(np.sum((self.labels == speaker) & self.speech) * FRAME / RATE)


def _runs(mask):
    """[(start, stop)] of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def speech_frames(wave):
    """Energy based activity on a clean voice stem: bool per 10 ms frame and the frame RMS."""
    frames = wave.numel() // FRAME
    if frames == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    rms = wave[: frames * FRAME].double().reshape(frames, FRAME).pow(2).mean(dim=1).sqrt().numpy()
    loud = np.percentile(rms, 95)
    if loud <= ACTIVITY_FLOOR:
        return np.zeros(frames, dtype=bool), rms
    active = rms > max(loud * 10 ** (ACTIVITY_REL_DB / 20), ACTIVITY_FLOOR)
    for start, stop in _runs(~active):  # bridge short pauses inside a phrase
        if 0 < start and stop < frames and (stop - start) * FRAME / RATE < MERGE_GAP_S:
            active[start:stop] = True
    for start, stop in _runs(active):  # drop clicks
        if (stop - start) * FRAME / RATE < MIN_SPEECH_S:
            active[start:stop] = False
    return active, rms


def _windows(segments):
    """Analysis windows (start, stop in frames) inside the speech segments."""
    window = int(round(WINDOW_S * RATE / FRAME))
    hop = int(round(HOP_S * RATE / FRAME))
    shortest = int(round(MIN_WINDOW_S * RATE / FRAME))
    out = []
    for start, stop in segments:
        length = stop - start
        if length < shortest:
            continue
        if length <= window:
            out.append((start, stop))
            continue
        position = start
        while position + window < stop:
            out.append((position, position + window))
            position += hop
        out.append((stop - window, stop))
    return out


@torch.no_grad()
def embed_windows(wave, windows, campplus, device, batch=64):
    """CAM++ embedding (L2-normalised) for each window of a mono 16 kHz signal."""
    embeddings = []
    by_length = {}
    for index, (start, stop) in enumerate(windows):
        by_length.setdefault(stop - start, []).append(index)
    result = [None] * len(windows)
    for length, indices in by_length.items():
        for offset in range(0, len(indices), batch):
            chunk = indices[offset:offset + batch]
            feats = []
            for index in chunk:
                start, stop = windows[index]
                piece = wave[start * FRAME:stop * FRAME][None].float()
                fbank = torchaudio.compliance.kaldi.fbank(piece, num_mel_bins=80, dither=0, sample_frequency=RATE)
                feats.append(fbank - fbank.mean(dim=0, keepdim=True))
            out = campplus(torch.stack(feats).to(device)).float().cpu()
            for index, vector in zip(chunk, out):
                result[index] = vector
    embeddings = torch.stack(result).double()
    return torch.nn.functional.normalize(embeddings, dim=1).numpy()


def _kmeans2(embeddings, assignment, rounds=20):
    for _ in range(rounds):
        centroids = np.stack([embeddings[assignment == k].mean(axis=0) for k in (0, 1)])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        updated = np.argmax(embeddings @ centroids.T, axis=1)
        if len(set(updated.tolist())) < 2 or np.array_equal(updated, assignment):
            break
        assignment = updated
    centroids = np.stack([embeddings[assignment == k].mean(axis=0) for k in (0, 1)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    objective = float(np.max(embeddings @ centroids.T, axis=1).sum())
    return assignment, centroids, objective


def _two_clusters(embeddings, durations):
    """Split the windows into two groups: average linkage refined by spherical k-means.

    Average linkage sometimes peels off a few odd windows instead of the second voice. When the
    smaller group holds under MIN_CLUSTER_SHARE of the speech, other starts are tried and the
    tightest balanced split wins.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    def share(assignment):
        return min(durations[assignment == k].sum() for k in (0, 1)) / durations.sum()

    candidates = []
    for method, metric in (("average", "cosine"), ("complete", "cosine"), ("ward", "euclidean")):
        start = fcluster(linkage(embeddings, method=method, metric=metric), t=2, criterion="maxclust") - 1
        if len(set(start.tolist())) < 2:
            continue
        candidate = _kmeans2(embeddings, start)
        if len(set(candidate[0].tolist())) < 2:
            continue
        if method == "average" and share(candidate[0]) >= MIN_CLUSTER_SHARE:
            return candidate[0], candidate[1]
        candidates.append(candidate)
    if not candidates:
        return np.zeros(len(embeddings), dtype=int), None
    balanced = [c for c in candidates if share(c[0]) >= MIN_CLUSTER_SHARE] or candidates
    best = max(balanced, key=lambda c: c[2])
    return best[0], best[1]


def _smooth_path(margin, speech):
    """Most likely speaker per frame given the per-frame evidence and a cost for every change of speaker.

    Changing speaker is cheap in a pause and expensive in the middle of continuous speech, so a short
    phrase with weak evidence follows its neighbours while a clear interjection still gets its own turn.
    (Two-state Viterbi; `margin` > 0 favours speaker 2.)
    """
    frames = margin.size
    cost = np.where(speech[1:] & speech[:-1], SWITCH_COST_SPEECH, SWITCH_COST_PAUSE)
    score = np.zeros((frames, 2))
    back = np.zeros((frames, 2), dtype=np.int8)
    score[0] = (-margin[0], margin[0])
    for t in range(1, frames):
        for state, emission in ((0, -margin[t]), (1, margin[t])):
            stay = score[t - 1, state]
            switch = score[t - 1, 1 - state] - cost[t - 1]
            if stay >= switch:
                score[t, state], back[t, state] = stay + emission, state
            else:
                score[t, state], back[t, state] = switch + emission, 1 - state
    labels = np.zeros(frames, dtype=np.int8)
    labels[-1] = int(score[-1, 1] > score[-1, 0])
    for t in range(frames - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def _refine_change(wave, campplus, device, centroids, rms, change, seg_start, seg_stop, before, after):
    """Best frame for a speaker change inside a continuous phrase.

    Each candidate position is judged by how well the speech just left of it matches the earlier
    speaker and the speech just right of it matches the later one; ties go to the quieter moment.
    """
    search = int(round(BOUNDARY_SEARCH_S * RATE / FRAME))
    side = int(round(BOUNDARY_CONTEXT_S * RATE / FRAME))
    step = max(1, int(round(BOUNDARY_STEP_S * RATE / FRAME)))
    shortest = int(round(MIN_WINDOW_S * RATE / FRAME))
    low, high = max(seg_start + 1, change - search), min(seg_stop - 1, change + search)
    if high > low:
        # a clear dip in loudness near the change is a breath between the two speakers: cut there
        quietest = low + int(np.argmin(rms[low:high]))
        if rms[quietest] < DIP_RATIO * np.median(rms[seg_start:seg_stop]):
            return quietest
    direction = centroids[after] - centroids[before]
    best = change
    # coarse pass with a long context, then a fine pass with a short one around the coarse result
    for reach, context, stride in ((search, side, step), (FINE_SEARCH_FRAMES, FINE_CONTEXT_FRAMES, 2)):
        candidates = [c for c in range(best - reach, best + reach + 1, stride)
                      if c - seg_start >= shortest and seg_stop - c >= shortest]
        if len(candidates) < 2:
            break
        spans = []
        for c in candidates:
            spans.append((max(seg_start, c - context), c))
            spans.append((c, min(seg_stop, c + context)))
        vectors = embed_windows(wave, spans, campplus, device)
        quality = vectors[1::2] @ direction - vectors[0::2] @ direction
        quiet = np.array([rms[max(0, c - 2):c + 3].mean() for c in candidates])
        quality = quality - 0.05 * quiet / (quiet.max() + 1e-12)
        best = candidates[int(np.argmax(quality))]
    return best


def _fill_pauses(labels, speech):
    """Give every non-speech frame to the nearer speech neighbour so the timeline is fully partitioned."""
    frames = labels.size
    runs = _runs(speech)
    if not runs:
        labels[:] = 0
        return labels
    labels[: runs[0][0]] = labels[runs[0][0]]
    labels[runs[-1][1]:] = labels[runs[-1][1] - 1]
    for (_, left_stop), (right_start, _) in zip(runs[:-1], runs[1:]):
        middle = (left_stop + right_start) // 2
        labels[left_stop:middle] = labels[left_stop - 1]
        labels[middle:right_start] = labels[right_start]
    return labels[:frames]


def diarize(wave, campplus, device, max_speakers=2, same_speaker_similarity=SAME_SPEAKER_SIMILARITY):
    """wave: mono float tensor at 16 kHz (voice only). Returns a Diarization with 1 or 2 speakers."""
    wave = wave.detach().float().cpu()
    speech, rms = speech_frames(wave)
    frames = speech.size
    labels = np.zeros(frames, dtype=np.int8)
    info = {"speech_seconds": round(float(speech.sum()) * FRAME / RATE, 2)}
    single = Diarization(labels, 1, speech, info)
    if max_speakers < 2 or not speech.any():
        info["reason"] = "one speaker requested" if max_speakers < 2 else "no speech found"
        return single
    segments = _runs(speech)
    windows = _windows(segments)
    info["windows"] = len(windows)
    if len(windows) < 2:
        info["reason"] = "too little speech to compare speakers"
        return single

    embeddings = embed_windows(wave, windows, campplus, device)
    durations = np.array([stop - start for start, stop in windows], dtype=np.float64)
    assignment, centroids = _two_clusters(embeddings, durations)
    if centroids is None:
        info["reason"] = "all windows fall into one cluster"
        return single
    info["centroid_similarity_initial"] = round(float(centroids[0] @ centroids[1]), 3)

    # windows that straddle a speaker change blur the centroids; re-estimate them from the clearest windows
    for _ in range(REFINE_ROUNDS):
        window_margin = np.abs(embeddings @ (centroids[1] - centroids[0]))
        confident = window_margin >= np.quantile(window_margin, 1.0 - CONFIDENT_SHARE)
        side = np.argmax(embeddings @ centroids.T, axis=1)
        if min(np.sum(confident & (side == k)) for k in (0, 1)) < 1:
            break
        centroids = np.stack([embeddings[confident & (side == k)].mean(axis=0) for k in (0, 1)])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    similarity = float(centroids[0] @ centroids[1])
    info["centroid_similarity"] = round(similarity, 3)
    side = np.argmax(embeddings @ centroids.T, axis=1)
    pairwise = embeddings @ embeddings.T
    groups = [np.flatnonzero(side == k) for k in (0, 1)]
    if min(len(g) for g in groups) >= 1:
        cross = float(pairwise[np.ix_(groups[0], groups[1])].mean())
        within = []
        for g in groups:
            if len(g) > 1:
                block = pairwise[np.ix_(g, g)]
                within.append((block.sum() - len(g)) / (len(g) * (len(g) - 1)))
        within_mean = float(np.mean(within)) if within else 1.0
        info["pair_cross"] = round(cross, 3)
        info["pair_within"] = round(within_mean, 3)
        info["pair_gap"] = round(within_mean - cross, 3)
        info["cluster_share_min"] = round(min(len(g) for g in groups) / len(side), 3)
    if similarity > same_speaker_similarity:
        info["reason"] = (f"the two voice clusters are too alike (similarity {similarity:.2f} > "
                          f"{same_speaker_similarity}); treated as one speaker")
        return single

    # per-frame evidence: similarity to each centroid, averaged over the windows covering the frame
    # (weighted towards windows centred on it, which sharpens the position of a change)
    scores = np.zeros((frames, 2))
    weight = np.zeros(frames)
    window_scores = embeddings @ centroids.T
    for (start, stop), score in zip(windows, window_scores):
        taper = np.bartlett(stop - start + 2)[1:-1]
        scores[start:stop] += score[None, :] * taper[:, None]
        weight[start:stop] += taper
    covered = weight > 0
    scores[covered] /= weight[covered, None]
    if not covered.all():  # short bits of speech without a window take the evidence of the nearest covered frame
        indices = np.flatnonzero(covered)
        nearest = indices[np.clip(np.searchsorted(indices, np.arange(frames)), 0, len(indices) - 1)]
        previous = indices[np.clip(np.searchsorted(indices, np.arange(frames)) - 1, 0, len(indices) - 1)]
        choose_previous = np.abs(np.arange(frames) - previous) < np.abs(nearest - np.arange(frames))
        nearest = np.where(choose_previous, previous, nearest)
        scores[~covered] = scores[nearest[~covered]]
    margin = scores[:, 1] - scores[:, 0]
    margin[~speech] = 0.0
    labels = _smooth_path(margin, speech)

    # a change that falls inside continuous speech is moved to the most plausible frame
    for start, stop in segments:
        changes = np.flatnonzero(np.diff(labels[start:stop]) != 0) + start + 1
        for change in changes:
            before, after = int(labels[change - 1]), int(labels[change])
            best = _refine_change(wave, campplus, device, centroids, rms, int(change), start, stop, before, after)
            if best < change:
                labels[best:change] = after
            else:
                labels[change:best] = before

    labels[~speech] = -1
    counts = [float(np.sum(labels == k)) * FRAME / RATE for k in (0, 1)]
    info["speaker_seconds_raw"] = [round(c, 2) for c in counts]
    if min(counts) < MIN_SPEAKER_S:
        info["reason"] = (f"the second voice has only {min(counts):.2f} s of speech "
                          f"(< {MIN_SPEAKER_S} s); treated as one speaker")
        return single
    first = [int(np.flatnonzero(labels == k)[0]) for k in (0, 1)]
    if first[1] < first[0]:  # number the speakers by who talks first
        swapped = labels.copy()
        swapped[labels == 0] = 1
        swapped[labels == 1] = 0
        labels = swapped
        first = first[::-1]
    info["first_speech_s"] = [round(f * FRAME / RATE, 2) for f in first]
    speech_labels = labels.copy()
    labels = _fill_pauses(labels.copy(), speech)
    result = Diarization(labels.astype(np.int8), 2, speech, info)
    info["speaker_seconds"] = [round(float(np.sum(speech_labels == k)) * FRAME / RATE, 2) for k in (0, 1)]
    return result


def speaker_masks(diarization, length, sample_rate, fade_ms=20.0):
    """Float masks [speakers, length] at `sample_rate` that sum to 1, with short crossfades at speaker changes."""
    labels = diarization.labels
    if diarization.speakers == 1 or labels.size == 0:
        return torch.ones(1, length)
    positions = np.minimum((np.arange(length) * (RATE / sample_rate) / FRAME).astype(np.int64), labels.size - 1)
    hard = torch.from_numpy(labels[positions] == 1).float()
    fade = max(1, int(sample_rate * fade_ms / 1000))
    if fade > 1:
        kernel = torch.hann_window(fade * 2 + 1, periodic=False)
        kernel = kernel / kernel.sum()
        padded = torch.nn.functional.pad(hard[None, None], (fade, fade), mode="replicate")
        hard = torch.nn.functional.conv1d(padded, kernel[None, None])[0, 0]
    second = hard.clamp(0.0, 1.0)
    return torch.stack([1.0 - second, second])
