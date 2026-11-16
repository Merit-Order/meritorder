"""Server-sent events: parsing upstream streams and encoding client streams."""

from __future__ import annotations

import codecs
import json
from dataclasses import dataclass
from typing import AsyncIterator, Iterable


@dataclass
class SSEEvent:
    data: str
    event: str | None = None

    def json(self) -> dict | None:
        if not self.data or self.data == "[DONE]":
            return None
        try:
            value = json.loads(self.data)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def encode(self) -> bytes:
        out = []
        if self.event:
            out.append(f"event: {self.event}")
        for line in self.data.split("\n"):
            out.append(f"data: {line}")
        return ("\n".join(out) + "\n\n").encode()


def event(data: dict | str, name: str | None = None) -> SSEEvent:
    if isinstance(data, dict):
        data = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return SSEEvent(data=data, event=name)


def _parse_block(block: str) -> SSEEvent | None:
    name = None
    data: list[str] = []
    for line in block.split("\n"):
        if not line or line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
    if not data and name is None:
        return None
    return SSEEvent(data="\n".join(data), event=name)


async def iter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    async for chunk in chunks:
        buf += decoder.decode(chunk)
        buf = buf.replace("\r\n", "\n")
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            ev = _parse_block(block)
            if ev is not None:
                yield ev
    buf += decoder.decode(b"", final=True)
    if buf.strip():
        ev = _parse_block(buf.replace("\r\n", "\n"))
        if ev is not None:
            yield ev


async def from_list(events: Iterable[SSEEvent]) -> AsyncIterator[SSEEvent]:
    for ev in events:
        yield ev
