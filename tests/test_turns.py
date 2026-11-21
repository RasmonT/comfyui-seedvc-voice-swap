"""Manual speaker turns ('turns' input). Needs numpy only.  Run:  python -m pytest tests/test_turns.py -q"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# loaded by path so that this file runs without torch (the package __init__ files import nothing heavy,
# but `seedvc_swap.pipeline` does)
_spec = importlib.util.spec_from_file_location(
    "seedvc_turns", Path(__file__).resolve().parent.parent / "seedvc_swap" / "turns.py")
turns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(turns)


def _speech(runs, seconds=16.0):
    speech = np.zeros(int(seconds * 100), dtype=bool)
    for start, stop in runs:
        speech[int(start * 100):int(stop * 100)] = True
    return speech


RUNS = [(0.2, 3.1), (3.9, 7.6), (8.4, 12.0), (12.5, 14.8)]  # speaker 1, 2, 1, 2 with pauses in between


def test_parse_accepts_the_documented_spellings():
    text = "# scene 1\n0 - 3.5 = 1\n3,5-8 : 2\n\n0:08 – 0:12.4 -> 1\n 12.4 to 15 2  # end\n"
    assert turns.parse_turns(text) == [(0.0, 3.5, 0), (3.5, 8.0, 1), (8.0, 12.4, 0), (12.4, 15.0, 1)]
    assert turns.parse_turns("") == [] and turns.parse_turns(None) == [] and turns.parse_turns("  \n# x\n") == []


@pytest.mark.parametrize("text,message", [
    ("0-3 = 3", "speaker must be 1 or 2"), ("5-2 = 1", "must be after the start"),
    ("hello", "cannot read"), ("0-4=1\n3-6=2", "overlap"),
])
def test_parse_rejects_bad_lines_with_a_clear_message(text, message):
    with pytest.raises(turns.TurnsError, match=message):
        turns.parse_turns(text)


def test_changes_land_in_the_pauses():
    labels, info = turns.labels_from_turns(turns.parse_turns("0-3.5=1\n3.5-8=2\n8-12.4=1\n12.4-15=2"), _speech(RUNS))
    assert [c["used_s"] for c in info["changes"]] == [3.5, 8.0, 12.25]
    assert all(c["in_pause"] for c in info["changes"])
    assert info["speaker_seconds"] == [6.5, 6.0]


def test_times_that_are_off_still_keep_every_line_whole():
    # the prompt said 2.7 / 9 / 13 s, the model spoke with pauses at 3.1-3.9, 7.6-8.4 and 12.0-12.5 s
    labels, info = turns.labels_from_turns(turns.parse_turns("0-2.7=1\n2.7-9=2\n9-13=1\n13-16=2"), _speech(RUNS))
    assert [c["used_s"] for c in info["changes"]] == [3.5, 8.0, 12.25]
    for (start, stop), speaker in zip(RUNS, (0, 1, 0, 1)):
        assert set(labels[int(start * 100):int(stop * 100)].tolist()) == {speaker}


def test_without_a_pause_the_cut_is_exactly_where_asked():
    labels, info = turns.labels_from_turns(turns.parse_turns("0-4=2\n4-10=1"), np.ones(1000, dtype=bool))
    assert info["changes"] == [{"asked_s": 4.0, "used_s": 4.0, "to_speaker": 1, "in_pause": False}]
    assert labels[0] == 1 and labels[399] == 1 and labels[400] == 0 and labels[-1] == 0


def test_gaps_single_speaker_and_turns_past_the_end():
    labels, info = turns.labels_from_turns(turns.parse_turns("1-3=1\n5-7=2"), _speech(RUNS))
    assert info["changes"][0]["used_s"] == 3.5 and labels[0] == 0 and labels[-1] == 1  # full partition
    labels, info = turns.labels_from_turns(turns.parse_turns("0-5=2"), _speech(RUNS))
    assert set(labels.tolist()) == {1} and info["changes"] == []
    labels, info = turns.labels_from_turns(turns.parse_turns("0-5=1\n40-50=2"), _speech(RUNS))
    assert set(labels.tolist()) == {0} and "warning" in info
    assert "speaker turns given by hand" in turns.describe(info)
    with pytest.raises(turns.TurnsError):
        turns.labels_from_turns([], _speech(RUNS))
