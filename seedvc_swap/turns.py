"""Manual "who speaks when": speaker turns typed by the user instead of automatic detection.

Text format, one turn per line (seconds from the start of the clip, or m:ss):

    0 - 3.5 = 1
    3.5 - 8 = 2
    0:08 - 0:12.4 = 1

`1` is voice slot 1, `2` is voice slot 2. Lines starting with # and empty lines are ignored.
The times do not have to be exact: every change of speaker is moved to the nearest pause in
the speech (within `snap_s`), because a generated clip rarely follows the prompt to the frame.

Only numpy is needed here, so this module can be tested without the models.
"""
import re

import numpy as np

FRAME_S = 0.01  # the labels use the same 10 ms frames as the automatic detection
SNAP_S = 1.0  # a change of speaker may move this far to reach a pause
MIN_PAUSE_S = 0.08

_TIME = r"(?:\d+:)?\d+(?:[.,]\d+)?"
_LINE = re.compile(rf"^\s*({_TIME})\s*(?:-|–|—|to|\.\.)\s*({_TIME})\s*(?:=|:|->|\s)\s*#?\s*(\d+)\s*$", re.IGNORECASE)


class TurnsError(ValueError):
    pass


def _seconds(text):
    text = text.replace(",", ".")
    if ":" in text:
        minutes, seconds = text.split(":")
        return int(minutes) * 60 + float(seconds)
    return float(text)


def parse_turns(text):
    """'start - stop = speaker' lines -> sorted [(start_s, stop_s, speaker index 0 or 1)]."""
    turns = []
    for number, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if not line:
            continue
        match = _LINE.match(line)
        if not match:
            raise TurnsError(f"turns, line {number}: cannot read '{raw.strip()}'. Write it as "
                             f"'start - stop = speaker', for example '0 - 3.5 = 1'.")
        start, stop, speaker = _seconds(match.group(1)), _seconds(match.group(2)), int(match.group(3))
        if speaker not in (1, 2):
            raise TurnsError(f"turns, line {number}: speaker must be 1 or 2, got {speaker}.")
        if stop <= start:
            raise TurnsError(f"turns, line {number}: the end ({stop:g} s) must be after the start ({start:g} s).")
        turns.append((start, stop, speaker - 1))
    turns.sort()
    for (_, stop, _), (start, _, _) in zip(turns[:-1], turns[1:]):
        if start < stop - 1e-6:
            raise TurnsError(f"turns: two turns overlap around {start:g} s. Every moment can belong to one speaker only.")
    return turns


def _pauses(speech):
    """[(start, stop)] frame ranges without speech, inside the clip (not the leading/trailing silence)."""
    if speech.size == 0:
        return []
    padded = np.concatenate([[True], ~speech, [True]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = list(zip(edges[::2], edges[1::2]))  # runs of speech
    return [(left_stop, right_start) for (_, left_stop), (right_start, _) in zip(runs[:-1], runs[1:])
            if (right_start - left_stop) * FRAME_S >= MIN_PAUSE_S]


def labels_from_turns(turns, speech, snap_s=SNAP_S):
    """Per-frame speaker labels (0/1) for the whole clip from user turns.

    turns: output of `parse_turns`. speech: bool per 10 ms frame (True = somebody talks).
    Returns (labels int8 [frames], info dict). The timeline is fully partitioned: before the first
    turn belongs to the first speaker, after the last turn to the last speaker, and each change of
    speaker sits in the middle of the pause nearest to the requested time.
    """
    if not turns:
        raise TurnsError("turns is empty.")
    frames = int(speech.size)
    merged = []
    for start, stop, speaker in turns:  # neighbouring turns of one speaker are a single turn
        if merged and merged[-1][2] == speaker:
            merged[-1] = (merged[-1][0], stop, speaker)
        else:
            merged.append((start, stop, speaker))
    pauses = _pauses(speech)
    labels = np.full(frames, merged[0][2], dtype=np.int8)
    changes = []
    previous = 0
    for index in range(1, len(merged)):
        left, right = merged[index - 1], merged[index]
        wanted = (left[1] + right[0]) / 2.0
        low = max(previous * FRAME_S, (left[0] + left[1]) / 2.0)   # never before the middle of the left turn
        high = (right[0] + right[1]) / 2.0                          # never after the middle of the right turn
        best, distance = None, None
        for p_start, p_stop in pauses:
            centre = (p_start + p_stop) / 2.0 * FRAME_S
            if not (low <= centre <= high):
                continue
            # a pause that contains the requested time needs no moving at all
            gap = 0.0 if p_start * FRAME_S <= wanted <= p_stop * FRAME_S else min(
                abs(p_start * FRAME_S - wanted), abs(p_stop * FRAME_S - wanted))
            if gap <= snap_s and (distance is None or gap < distance):
                best, distance = centre, gap
        used = wanted if best is None else best
        frame = int(round(min(max(used, 0.0), frames * FRAME_S) / FRAME_S))
        frame = max(frame, previous)
        labels[frame:] = right[2]
        changes.append({"asked_s": round(wanted, 2), "used_s": round(frame * FRAME_S, 2),
                        "to_speaker": right[2] + 1, "in_pause": best is not None})
        previous = frame
    info = {"source": "turns", "turns": [(round(a, 2), round(b, 2), s + 1) for a, b, s in merged],
            "changes": changes, "speech_seconds": round(float(speech.sum()) * FRAME_S, 2),
            "speaker_seconds": [round(float(np.sum((labels == k) & speech)) * FRAME_S, 2) for k in (0, 1)]}
    clip_s = frames * FRAME_S
    if merged[-1][0] >= clip_s:
        info["warning"] = f"the last turn starts at {merged[-1][0]:g} s but the clip is only {clip_s:.2f} s long"
    return labels, info


def describe(info):
    """Console lines that show what was asked for and what was done."""
    lines = ["[SeedVC] speaker turns given by hand (automatic detection is off):"]
    for start, stop, speaker in info["turns"]:
        lines.append(f"[SeedVC]   {start:7.2f} - {stop:7.2f} s -> speaker {speaker}")
    for change in info["changes"]:
        where = "in a pause" if change["in_pause"] else "NO PAUSE NEARBY, cut exactly where asked"
        lines.append(f"[SeedVC]   change to speaker {change['to_speaker']} at {change['used_s']:.2f} s "
                     f"(asked {change['asked_s']:.2f} s, {where})")
    if "warning" in info:
        lines.append(f"[SeedVC]   WARNING: {info['warning']}")
    return "\n".join(lines)
