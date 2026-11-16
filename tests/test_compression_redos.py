"""Compression runs on the event loop: hostile tool output must take linear time."""

import time

import pytest

from meritorder.compression import CompressionConfig, compress_text

CFG = CompressionConfig(enabled=True, min_chars=10, max_chars=10_000_000)


@pytest.mark.parametrize("text", [
    " " * 400_000,                       # trailing-whitespace run with no newline
    "\t \t" * 130_000 + "x",
    "\x1b]" * 200_000,                   # unterminated OSC sequences
    "\x1b[" + "1;" * 200_000,            # unterminated CSI
])
def test_hostile_input_is_fast(text):
    t0 = time.perf_counter()
    compress_text(text, CFG)
    assert time.perf_counter() - t0 < 0.5


def test_still_strips_ansi_and_trailing_whitespace():
    raw = "\x1b[31mred\x1b[0m  \nline\t\t\n\x1b]0;title\x07done \n\x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\\n" + "pad\n" * 5
    out = compress_text(raw, CFG)
    assert out.startswith("red\nline\ndone\nlink\n") and "\x1b" not in out
