"""The gate on every SkyPilot replica: the only thing listening on its open port, and its dead-man switch.

It runs on the GPU machine, in vLLM's container, as the SkyPilot task itself (``python3 replica_gate.py -- vllm serve
...``), so it uses nothing but the standard library. It does four things:

- **Starts vLLM** on 127.0.0.1 and waits on it. When vLLM exits, so does the gate, which ends the task: at once when the
  lease ran out or it was asked to stop; after vLLM exited by itself, once the controller has read why
  (``GET /meritorder/status``) or ``MERITORDER_LINGER_SECONDS`` (120) have passed, whichever comes first, so the reason
  isn't deleted with the machine.
- **Lets in only callers with the deployment's key.** Each connection's first request must carry it
  (``Authorization: Bearer <key>``, or ``x-api-key``); the rest of the connection is then passed to vLLM byte for byte.
  The gate knows only the key's SHA-256 (``MERITORDER_KEY_SHA256``), so the key itself never goes through SkyPilot
  (whose API server keeps every launch request, environment and secrets included, in its database). vLLM's own
  ``--api-key`` would cover only paths under /v1, and vLLM listens only on the loopback address.
- **Holds a lease.** ``POST /meritorder/lease`` with the key renews it for ``MERITORDER_LEASE_SECONDS``. When nobody
  renews it in time (the controller died, its machine sleeps, the network dropped, ~/.sky was lost), the gate stops
  vLLM and exits. The task is then over, the cluster has nothing running, and SkyPilot's autostop (``down: true``, set
  at launch) deletes it. The lease starts when the gate does, so a replica whose controller never shows up stops too.
- **Keeps vLLM's last output.** vLLM's stdout and stderr pass through to the task's log (``sky logs``), and the last
  ``TAIL_LINES`` lines are also kept in memory, with anything that looks like a key or token redacted.
  ``GET /meritorder/status`` (with the key) returns them, with vLLM's exit status and uptime, and the NVIDIA driver and
  CUDA versions (``nvidia-smi``, run once at start). When the driver is too old for the image's CUDA, it says so plainly.

It also times its own start (``startup`` in ``/meritorder/status`` and the lease reply, unix seconds on this machine's
clock): when the machine booted and the container started (from ``/proc``), when the gate and vLLM started, when vLLM
said the weights were downloaded and loaded (its own log lines), and when vLLM first took a connection. The controller
adds its side and turns them into phases (skypilot.startup_phases).

Nothing it receives is logged: it logs only what it does (started, lease renewed, lease expired, vLLM exited). vLLM logs
no request content either: request logging stays off (config.py refuses ``--enable-log-requests`` and
``--enable-log-outputs`` in ``vllm_args``), and a request log line that shows up anyway is left out of the kept tail.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import time

LEASE_PATH = "/meritorder/lease"
STATUS_PATH = "/meritorder/status"
MAX_HEAD = 64 * 1024  # a request head bigger than this is refused
HEAD_TIMEOUT = 5  # seconds a connection has to send its first request head (a gateway sends it at once)
# Connections that haven't yet shown the deployment's key, at most: past this a new one is closed at once, so a scan or a
# flood of idle connections can't use up the replica's sockets. Connections that showed the key aren't counted.
MAX_UNVERIFIED = 256
TAIL_LINES = 200  # vLLM output lines kept for /meritorder/status
LINE_MAX = 2000  # characters kept of one line
# The oldest driver each CUDA major version runs on (NVIDIA's CUDA Toolkit release notes, "minor version compatibility").
MIN_DRIVER = {11: "450.80.02", 12: "525.60.13", 13: "580.65.06"}

# What looks like a secret in a log line: a bearer token, values of fields named like secrets, well-known key prefixes,
# and long runs of letters, digits, _, -, + and = (32 or more, with at least one letter and one digit; not across a /).
_SECRETS = [
    (re.compile(r"(?i)\b(bearer)\s+[^\s'\",]+"), r"\1 [redacted]"),
    (re.compile(r"(?i)\b([\w-]*(?:api[_-]?key|access[_-]?key|private[_-]?key|token|secret|password|passwd|credentials?)\b)(['\"]?\s*[=:]\s*['\"]?)[^\s'\",}]+"),
     r"\1\2[redacted]"),
    (re.compile(r"\b(?:sk|pk|rk|hf|ghp|gho|ghs|glpat|xox[abp])[-_][A-Za-z0-9_-]{8,}|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[redacted]"),
    (re.compile(r"(?<![A-Za-z0-9_+=-])(?=[A-Za-z0-9_+=-]*\d)(?=[A-Za-z0-9_+=-]*[A-Za-z])[A-Za-z0-9_+=-]{32,}"), "[redacted]"),
]
# vLLM's request logger (vllm/entrypoints/logger.py), which runs only with --enable-log-requests or --enable-log-outputs.
_REQUEST_LOG = re.compile(r"Received request|Generated response|prompt_token_ids|\bprompt: ")


def log(msg: str) -> None:
    print(f"[meritorder-gate] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def redact(line: str) -> str:
    """A log line with anything that looks like a key or token replaced by [redacted]."""
    for pattern, repl in _SECRETS:
        line = pattern.sub(repl, line)
    return line


def keepable(line: str) -> str:
    """A line as the tail keeps it: a request log line left out, secrets redacted, at most LINE_MAX characters."""
    if _REQUEST_LOG.search(line):
        return "[a request log line, not kept]"
    return redact(line[:LINE_MAX])


def _version(text: str | None) -> tuple[int, ...] | None:
    m = re.match(r"\s*(\d+)(?:\.(\d+))?", text or "")
    return tuple(int(x) for x in m.groups() if x is not None) if m else None


def driver_problem(driver: str | None, driver_cuda: str | None, image_cuda: str | None) -> str | None:
    """Why this driver can't run the image's CUDA, or None when it can (or either isn't known). A driver runs every CUDA
    release of the major version it reports (minor version compatibility), and none of a newer major version."""
    have, need = _version(driver_cuda), _version(image_cuda)
    if not have or not need or need[0] <= have[0]:
        return None
    floor = MIN_DRIVER.get(need[0])
    return (f"the NVIDIA driver {driver or '?'} (CUDA {driver_cuda} at most) is too old for this image's CUDA {image_cuda}"
            + (f": CUDA {need[0]} images need driver {floor} or newer" if floor else "")
            + f". Use an image built for CUDA {have[0]} (vllm/vllm-openai:v0.32.0-cu129 is CUDA 12.9) or GPUs with a newer driver")


def older_driver(driver_cuda: str | None, image_cuda: str | None) -> bool:
    """Whether the driver reports an older CUDA than the image was built with, minor versions included. Minor version
    compatibility runs the CUDA runtime itself, but not everything built against the newer toolkit: on Lambda's driver
    570 (CUDA 12.8), vLLM's CUDA 12.9.1 image failed to load torchcodec's library at startup (15 November 2026)."""
    have, need = _version(driver_cuda), _version(image_cuda)
    return bool(have and need and need[:2] > have[:2])


def compat_dir() -> str | None:
    """NVIDIA's forward-compatibility libraries in the image (the cuda-compat package, which NVIDIA's CUDA base images
    ship in /usr/local/cuda/compat), or None. Put first on LD_LIBRARY_PATH, they let a newer CUDA run on an older
    driver, on data-center GPUs (A10, A100, H100, L4 and the like; not GeForce)."""
    path = os.path.join(os.environ.get("CUDA_HOME") or "/usr/local/cuda", "compat")
    try:
        return path if any(name.startswith("libcuda.so") for name in os.listdir(path)) else None
    except OSError:
        return None


def image_cuda() -> str | None:
    """The CUDA version this container was built with: NVIDIA's base images set CUDA_VERSION and write version.json."""
    if os.environ.get("CUDA_VERSION"):
        return os.environ["CUDA_VERSION"]
    try:
        with open(os.path.join(os.environ.get("CUDA_HOME") or "/usr/local/cuda", "version.json")) as f:
            return json.load(f)["cuda"]["version"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def gpu_info(run=subprocess.run) -> dict:
    """The driver version, the newest CUDA it runs and the GPUs (``nvidia-smi``, once, at start), and the image's CUDA."""
    info: dict = {"driver_version": None, "cuda_version": None, "gpus": [], "image_cuda": image_cuda(), "nvidia_smi_error": None}
    try:
        out = run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        text = out.stdout or ""
        if out.returncode:
            info["nvidia_smi_error"] = redact((out.stderr or text).strip()[-300:]) or f"exit status {out.returncode}"
        if m := re.search(r"Driver Version:\s*([\d.]+)", text):
            info["driver_version"] = m.group(1)
        if m := re.search(r"CUDA Version:\s*([\d.]+)", text):
            info["cuda_version"] = m.group(1)
        if not out.returncode:
            q = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            info["gpus"] = [x.strip() for x in (q.stdout or "").splitlines() if x.strip()] if not q.returncode else []
    except (OSError, subprocess.SubprocessError) as e:
        info["nvidia_smi_error"] = f"{type(e).__name__}: {e}"[:300]
    info["driver_problem"] = driver_problem(info["driver_version"], info["cuda_version"], info["image_cuda"])
    info["cuda_compat"] = compat_dir() if older_driver(info["cuda_version"], info["image_cuda"]) else None
    return info


def machine_times(proc: str = "/proc", now: float | None = None) -> dict:
    """When this machine booted and when this container started, as unix seconds, from /proc; None where unknown.
    /proc/uptime isn't namespaced, so it gives the machine's boot even inside a container; the container's own first
    process (PID 1) started when Docker started the container, after its image was pulled. With the host's PID
    namespace, PID 1 is the machine's init, which says nothing about the container, so it's left out."""
    now = time.time() if now is None else now
    out: dict = {"machine_booted": None, "container_started": None}
    try:
        with open(os.path.join(proc, "uptime")) as f:
            boot = now - float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return out
    out["machine_booted"] = round(boot, 1)
    try:
        with open(os.path.join(proc, "1", "comm")) as f:
            if f.read().strip() in ("systemd", "init"):
                return out
        with open(os.path.join(proc, "1", "stat")) as f:
            ticks = int(f.read().rpartition(")")[2].split()[19])  # field 22, starttime, in clock ticks since boot
        out["container_started"] = round(boot + ticks / os.sysconf("SC_CLK_TCK"), 1)
    except (OSError, ValueError, IndexError):
        pass
    return out


# vLLM's own log lines that mark its start (vLLM 0.32): the weights downloaded (the default loader, when it downloaded
# anything), the weights loaded into the GPU, and the API server about to listen.
MARKS = (
    ("weights_downloaded", re.compile(r"Time spent downloading weights for .*?: ([\d.]+) seconds")),
    ("weights_loaded", re.compile(r"Model loading took [\d.]+ ?GiB(?: memory)? and ([\d.]+) seconds|Loading weights took ([\d.]+) seconds")),
)


def token_of(head: bytes) -> str | None:
    """The key a request head carries: ``Authorization: Bearer <key>`` or ``x-api-key: <key>``."""
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        name, value = name.strip().lower(), value.strip()
        if name == b"authorization" and value[:7].lower() == b"bearer ":
            return value[7:].strip().decode("latin-1")
        if name == b"x-api-key" and value:
            return value.decode("latin-1")
    return None


def allowed(head: bytes, key_sha256: str) -> bool:
    token = token_of(head)
    if not token:
        return False
    return hmac.compare_digest(hashlib.sha256(token.encode("latin-1")).hexdigest(), key_sha256.lower())


def reply(status: str, body: dict) -> bytes:
    data = json.dumps(body).encode()
    return (f"HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {len(data)}\r\n"
            f"connection: close\r\n\r\n").encode() + data


# Optional packages vLLM imports at startup and can do without, which an image can ship broken. vLLM v0.32.0-cu129
# (torch 2.13.0+cu129) bundles torchcodec 0.17.0 built for CUDA 13 (it links libcudart.so.13); loading it raises an
# OSError, which vLLM's guarded import (ImportError, RuntimeError) doesn't catch, so vLLM dies before it serves
# (found live on Lambda, 15 November 2026). Without it, vLLM just has no audio input.
OPTIONAL_PACKAGES = ("torchcodec",)


def drop_broken_optional_packages(run=subprocess.run, python: str = sys.executable) -> list[str]:
    """Uninstall each optional package that's installed but fails to load with an OSError (a broken native library);
    returns what was dropped and why. A package that loads, or isn't installed, is left alone."""
    dropped = []
    for pkg in OPTIONAL_PACKAGES:
        try:
            probe = run([python, "-c", f"import {pkg}"], capture_output=True, text=True, timeout=300)
            if probe.returncode == 0 or "OSError" not in (probe.stderr or ""):
                continue
            why = redact((probe.stderr or "").strip().splitlines()[-1][:300])
            out = run([python, "-m", "pip", "uninstall", "-y", "-q", pkg], capture_output=True, text=True, timeout=300)
            if not out.returncode:
                dropped.append(f"{pkg} ({why})")
        except (OSError, subprocess.SubprocessError, IndexError):
            continue
    return dropped


class Gate:
    def __init__(self, cmd: list[str], key_sha256: str, lease_seconds: float, port: int, upstream_port: int,
                 grace_seconds: float = 30.0, host: str = "0.0.0.0", linger_seconds: float = 120.0, gpu=gpu_info,
                 preflight=drop_broken_optional_packages, times=machine_times):
        self.cmd, self.key_sha256, self.lease_seconds = cmd, key_sha256, lease_seconds
        self.port, self.upstream_port, self.grace, self.host = port, upstream_port, grace_seconds, host
        self.linger, self.gpu_probe, self.preflight = linger_seconds, gpu, preflight
        self.expires = time.monotonic() + lease_seconds  # the lease starts when the gate does
        self.unverified = asyncio.Semaphore(MAX_UNVERIFIED)
        self.renewed = 0
        self.proc: asyncio.subprocess.Process | None = None
        self.stopping: str | None = None
        self.tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self.started: float | None = None
        self.exited: float | None = None
        self.read_after_exit = False  # /meritorder/status was read after vLLM exited: the gate may go
        self.gpu: dict = {}
        # When each step of this replica's start happened (unix seconds), and how long vLLM said its own steps took.
        self.startup: dict = {**times(), "gate_started": round(time.time(), 1), "vllm_started": None, "weights_downloaded": None,
                              "weights_loaded": None, "vllm_listening": None, "download_seconds": None, "load_seconds": None}

    def remaining(self) -> float:
        return max(0.0, self.expires - time.monotonic())

    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    def status(self) -> dict:
        """What /meritorder/status answers: vLLM's state and last output, and the driver and CUDA versions."""
        end = self.exited or time.monotonic()
        return {"vllm": "running" if self.running() else ("exited" if self.proc is not None else "not started"),
                "exit_code": self.proc.returncode if self.proc is not None else None,
                "uptime_seconds": round(end - self.started, 1) if self.started else None, "stopping": self.stopping,
                "expires_in": round(self.remaining()), **self.gpu, "startup": dict(self.startup), "tail": list(self.tail)}

    def mark(self, line: str) -> None:
        """Note when vLLM says it has downloaded or loaded the weights (the first time it says so)."""
        for name, pattern in MARKS:
            if self.startup.get(name) is None and (m := pattern.search(line)):
                self.startup[name] = round(time.time(), 1)
                took = next((float(g) for g in m.groups() if g), None)
                self.startup["download_seconds" if name == "weights_downloaded" else "load_seconds"] = took

    async def watch_listening(self, every: float = 1.0) -> None:
        """Note when vLLM first takes a connection on its port: its API server listens once the engine is up."""
        while self.running() and self.startup["vllm_listening"] is None:
            try:
                _, w = await asyncio.open_connection("127.0.0.1", self.upstream_port)
            except OSError:
                await asyncio.sleep(every)
                continue
            w.close()
            self.startup["vllm_listening"] = round(time.time(), 1)
            log(f"vLLM is listening, {self.startup['vllm_listening'] - self.startup['gate_started']:.0f}s after the gate started")

    async def pump(self, stream: asyncio.StreamReader) -> None:
        """vLLM's output: through to the task's own log, and its last lines into the tail. A progress bar redraws its line
        with a carriage return, so only a line's last drawing is kept."""
        out = getattr(sys.stdout, "buffer", None)
        part = b""
        while True:
            chunk = await stream.read(65536)
            if out is not None and chunk:
                try:
                    out.write(chunk)
                    out.flush()
                except (OSError, ValueError):  # pragma: no cover (the task's log went away)
                    out = None
            lines = (part + chunk).split(b"\n")
            part = lines.pop() if chunk else b""
            for raw in lines:
                text = raw.decode(errors="replace").rstrip("\r").rsplit("\r", 1)[-1].rstrip()
                if text:
                    self.tail.append(keepable(text))
                    self.mark(text)
            part = part[-LINE_MAX * 4:]
            if not chunk:
                return

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if self.unverified.locked():
                return  # too many connections still to show the key: this one is closed (finally) without a read
            async with self.unverified:
                try:
                    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT)
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
                    return
                if not allowed(head, self.key_sha256):
                    writer.write(reply("401 Unauthorized", {"error": "this replica needs its deployment's key"}))
                    await writer.drain()
                    return
            method, _, rest = head.partition(b" ")
            path = rest.partition(b" ")[0].split(b"?")[0].decode("latin-1")
            if path == STATUS_PATH:
                if self.proc is not None and self.proc.returncode is not None:
                    self.read_after_exit = True
                writer.write(reply("200 OK", self.status()))
                await writer.drain()
                return
            if path == LEASE_PATH:
                if method == b"POST":
                    self.expires = time.monotonic() + self.lease_seconds
                    self.renewed += 1
                    if self.renewed == 1 or self.renewed % 60 == 0:
                        log(f"lease renewed ({self.renewed} times so far), {self.lease_seconds:.0f}s left")
                writer.write(reply("200 OK", {"expires_in": round(self.remaining()), "lease_seconds": self.lease_seconds,
                                              "vllm": "running" if self.running() else "exited", "stopping": self.stopping,
                                              "exit_code": self.proc.returncode if self.proc is not None else None,
                                              "startup": dict(self.startup)}))
                await writer.drain()
                return
            await self.forward(head, reader, writer)
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def forward(self, head: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Pass the connection to vLLM: the head already read, then everything else both ways, until either side closes."""
        try:
            up_reader, up_writer = await asyncio.open_connection("127.0.0.1", self.upstream_port)
        except OSError:
            writer.write(reply("503 Service Unavailable", {"error": "vLLM isn't listening yet"}))
            await writer.drain()
            return
        up_writer.write(head)

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await src.read(65536)
                    if not chunk:
                        break
                    dst.write(chunk)
                    await dst.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                try:
                    dst.close()
                except Exception:  # noqa: BLE001
                    pass

        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

    async def linger_for_status(self, asked: asyncio.Future) -> None:
        """After vLLM exited by itself: keep answering /meritorder/status until the controller has read it (and a moment
        more, so the answer gets out), the lease runs out, the gate is asked to stop, or ``linger`` seconds have passed."""
        deadline = time.monotonic() + self.linger
        log(f"waiting up to {self.linger:.0f}s for the controller to read why (GET {STATUS_PATH})")
        while not asked.done() and self.remaining() > 0 and time.monotonic() < deadline:
            if self.read_after_exit:
                await asyncio.sleep(0.5)
                return
            await asyncio.wait({asked}, timeout=0.2)

    async def stop_vllm(self, why: str) -> None:
        self.stopping = self.stopping or why
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        log(f"{why}: stopping vLLM")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), self.grace)
        except asyncio.TimeoutError:
            log(f"vLLM still running after {self.grace:.0f}s: killing it")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    async def run(self) -> int:
        self.gpu = await asyncio.to_thread(self.gpu_probe)
        for dropped in await asyncio.to_thread(self.preflight):
            log(f"uninstalled {dropped}: it's broken in this image, and vLLM runs without it")
        if self.gpu.get("driver_problem"):
            log(f"warning: {self.gpu['driver_problem']}")
        else:
            log(f"NVIDIA driver {self.gpu.get('driver_version')} (CUDA {self.gpu.get('cuda_version')}), image CUDA {self.gpu.get('image_cuda')}")
        env = None
        if compat := self.gpu.get("cuda_compat"):
            log(f"the driver's CUDA {self.gpu.get('cuda_version')} is older than the image's {self.gpu.get('image_cuda')}: "
                f"running vLLM with NVIDIA's forward-compatibility libraries ({compat})")
            # vLLM's images ship cuda-compat but set VLLM_ENABLE_CUDA_COMPATIBILITY=0; the path is set here as well, so
            # libraries loaded before vLLM reads its flag (torchcodec, at import) get them too.
            env = {**os.environ, "LD_LIBRARY_PATH": ":".join(p for p in (compat, os.environ.get("LD_LIBRARY_PATH")) if p),
                   "VLLM_ENABLE_CUDA_COMPATIBILITY": "1"}
        self.proc = await asyncio.create_subprocess_exec(*self.cmd, start_new_session=True, stdout=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.STDOUT, env=env)
        self.started = time.monotonic()
        self.startup["vllm_started"] = round(time.time(), 1)
        pumping = asyncio.ensure_future(self.pump(self.proc.stdout))
        listening = asyncio.ensure_future(self.watch_listening())
        server = await asyncio.start_server(self.handle, self.host, self.port, limit=MAX_HEAD)
        log(f"vLLM started (pid {self.proc.pid}); gate on port {self.port}, lease {self.lease_seconds:.0f}s")
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):  # pragma: no cover (not on the Linux machines this runs on)
                pass
        code = 0
        exited, asked = asyncio.ensure_future(self.proc.wait()), asyncio.ensure_future(stop.wait())
        try:
            while True:
                if exited.done():
                    code = self.proc.returncode or 0
                    self.exited = time.monotonic()
                    log(f"vLLM exited with status {code}")
                    await asyncio.wait({pumping}, timeout=5)  # its last lines, unless a child of it holds the pipe open
                    await self.linger_for_status(asked)
                    break
                if asked.done():
                    await self.stop_vllm("asked to stop")
                    break
                if self.remaining() <= 0:
                    await self.stop_vllm(f"lease not renewed for {self.lease_seconds:.0f}s")
                    break
                await asyncio.wait({exited, asked}, timeout=min(1.0, max(self.remaining(), 0.05)))
        finally:
            asked.cancel()
            server.close()
            await self.stop_vllm("gate exiting")
            pumping.cancel()
            listening.cancel()
        log("done: the task ends here, and SkyPilot's autostop deletes the machine once it has been idle")
        return code


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    key = os.environ.get("MERITORDER_KEY_SHA256", "")
    if not args or len(key) != 64:
        print("usage: MERITORDER_KEY_SHA256=<hex> replica_gate.py -- vllm serve ...", file=sys.stderr)
        return 2
    gate = Gate(args, key, float(os.environ.get("MERITORDER_LEASE_SECONDS", "900")), int(os.environ.get("MERITORDER_PORT", "8000")),
                int(os.environ.get("MERITORDER_UPSTREAM_PORT", "8001")), float(os.environ.get("MERITORDER_GRACE_SECONDS", "30")),
                linger_seconds=float(os.environ.get("MERITORDER_LINGER_SECONDS", "120")))
    return asyncio.run(gate.run())


if __name__ == "__main__":
    sys.exit(main())
