"""Where model-written code runs during an eval.

Model output is untrusted, so evals run it in a throwaway Docker container: no network, read-only filesystem
apart from a small /tmp, no capabilities, an unprivileged user, and limits on memory, CPU, processes and time.
``LocalSandbox`` runs code directly and exists only for checking the eval's own reference solutions in tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

IMAGE = "python:3.12-slim"
RUNNER = Path(__file__).with_name("pyrunner.py")
MAX_OUTPUT = 64_000


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Sandbox(Protocol):
    def run(self, files: dict[str, str], argv: list[str], timeout: float = 30.0) -> SandboxResult:
        """Write ``files`` (relative path -> text) into an empty directory and run ``argv`` there."""
        ...


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = (root / rel).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"file outside the sandbox: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _clip(s: bytes | str | None) -> str:
    if s is None:
        return ""
    text = s.decode("utf-8", "replace") if isinstance(s, bytes) else s
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + "\n[output truncated]"


class LocalSandbox:
    """Runs on this machine with no isolation. Only for trusted code: the eval's own reference solutions.
    ``path`` replaces PATH, so tools installed on this machine (a ``sky`` CLI, say) don't leak into the run."""

    def __init__(self, path: str | None = None):
        self.path = path

    def variant(self, **_: object) -> LocalSandbox:
        return self  # runs with this machine's Python and packages, whatever a task asks for

    def available(self) -> str | None:
        return None

    def run(self, files: dict[str, str], argv: list[str], timeout: float = 30.0) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="mo-eval-") as d:
            _write(Path(d), files)
            cmd = [os.environ.get("PYTHON", shutil.which("python3") or "python3") if argv[0] == "python" else argv[0], *argv[1:]]
            try:
                p = subprocess.run(cmd, cwd=d, capture_output=True, timeout=timeout,
                                   env={"PATH": self.path if self.path is not None else os.environ.get("PATH", "")})
                return SandboxResult(p.returncode, _clip(p.stdout), _clip(p.stderr))
            except subprocess.TimeoutExpired as e:
                return SandboxResult(-1, _clip(e.stdout), _clip(e.stderr), timed_out=True)


def deps_image(requirements: Path, base: str = IMAGE) -> str:
    """The tag of ``base`` with these pinned requirements installed. It names its contents, so a changed lock file
    gets a new image and an old one is never reused by mistake."""
    digest = hashlib.sha256(f"{base}\n{requirements.read_text(encoding='utf-8')}".encode()).hexdigest()[:12]
    return f"meritorder-eval-deps:{digest}"


_BUILD_LOCK = threading.Lock()


class DockerSandbox:
    """A fresh container per run, with no network and tight limits. ``requirements`` (a pip requirements file with
    hashes) runs it in ``image`` with those packages installed, built here the first time it's needed; the
    container still has no network, so the packages come only from that build."""

    def __init__(self, image: str = IMAGE, memory: str = "512m", cpus: str = "1", docker: str = "docker", pids: int = 64,
                 tmp: str = "64m", requirements: Path | None = None, tmp_exec: bool = False):
        self.base, self.requirements = image, requirements
        self.image = deps_image(requirements, image) if requirements else image
        self.memory, self.cpus, self.docker, self.pids, self.tmp, self.tmp_exec = memory, cpus, docker, pids, tmp, tmp_exec

    def variant(self, requirements: Path | None = None, memory: str | None = None, cpus: str | None = None,
                pids: int | None = None, tmp: str | None = None, tmp_exec: bool | None = None) -> DockerSandbox:
        """The same sandbox with more room or extra packages, for tasks that need them (a whole repository's tests).
        ``tmp_exec`` lets programs written to /tmp run (a repository's tests that make a fake CLI there)."""
        return DockerSandbox(self.base, memory or self.memory, cpus or self.cpus, self.docker, pids or self.pids, tmp or self.tmp,
                             requirements or self.requirements, self.tmp_exec if tmp_exec is None else tmp_exec)

    def available(self) -> str | None:
        """None when Docker works and the image is present (pulled, or built from the requirements, if needed);
        otherwise why not."""
        if not shutil.which(self.docker):
            return "Docker isn't installed: code and repository tasks run model-written code only inside a container"
        try:
            if subprocess.run([self.docker, "image", "inspect", self.image], capture_output=True, timeout=30).returncode == 0:
                return None
            if self.requirements is None:
                pulled = subprocess.run([self.docker, "pull", "-q", self.image], capture_output=True, timeout=600)
                return None if pulled.returncode == 0 else f"couldn't pull {self.image}: {_clip(pulled.stderr)[:300]}"
            return self._build()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"Docker isn't working: {e}"

    def _build(self) -> str | None:
        with _BUILD_LOCK:  # tasks start together; one build, the rest wait for it
            if subprocess.run([self.docker, "image", "inspect", self.image], capture_output=True, timeout=30).returncode == 0:
                return None
            assert self.requirements is not None
            with tempfile.TemporaryDirectory(prefix="mo-eval-build-") as d:
                shutil.copy(self.requirements, Path(d) / "requirements.txt")
                (Path(d) / "Dockerfile").write_text(
                    f"FROM {self.base}\nCOPY requirements.txt /tmp/requirements.txt\n"
                    "RUN pip install --no-cache-dir --disable-pip-version-check --require-hashes -r /tmp/requirements.txt "
                    "&& rm /tmp/requirements.txt\n", encoding="utf-8")
                built = subprocess.run([self.docker, "build", "-q", "-t", self.image, d], capture_output=True, timeout=1800)
            return None if built.returncode == 0 else f"couldn't build {self.image}: {_clip(built.stderr)[-500:]}"

    def run(self, files: dict[str, str], argv: list[str], timeout: float = 30.0) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="mo-eval-") as d:
            _write(Path(d), files)
            os.chmod(d, 0o755)
            for p in Path(d).rglob("*"):
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
            name = f"mo-eval-{os.urandom(6).hex()}"
            cmd = [self.docker, "run", "--rm", "--name", name, "--network", "none", "--memory", self.memory, "--memory-swap", self.memory,
                   "--cpus", self.cpus, "--pids-limit", str(self.pids), "--read-only", "--tmpfs", f"/tmp:rw,{'exec,' if self.tmp_exec else ''}size={self.tmp}", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--user", "65534:65534", "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-v", f"{d}:/work:ro", "-w", "/work", self.image, "timeout", "-s", "KILL", str(int(timeout)), *argv]
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=timeout + 30)
                timed_out = p.returncode in (124, 137)
                return SandboxResult(p.returncode, _clip(p.stdout), _clip(p.stderr), timed_out=timed_out)
            except subprocess.TimeoutExpired as e:
                try:
                    subprocess.run([self.docker, "rm", "-f", name], capture_output=True, timeout=60)
                except (OSError, subprocess.TimeoutExpired):
                    pass  # Docker itself is stuck; the container's own time limit ends it
                return SandboxResult(-1, _clip(e.stdout), _clip(e.stderr), timed_out=True)


def run_tests(sandbox: Sandbox, solution: str, tests: str, forbid: list[str] | None = None, timeout: float = 30.0,
              extra_files: dict[str, str] | None = None, per_test: float = 10.0) -> dict:
    """Run a code task's hidden tests against ``solution``. Returns pyrunner's JSON summary. ``per_test`` is each
    test's CPU-time limit, so one runaway loop fails that test rather than the whole run."""
    files = {"solution.py": solution, "tests.py": tests, "_run.py": RUNNER.read_text(encoding="utf-8"), **(extra_files or {})}
    argv = ["python", "-I", "-B", "_run.py"] + (["--forbid", ",".join(forbid)] if forbid else []) + (["--per-test", str(per_test)] if per_test else [])
    r = sandbox.run(files, argv, timeout)
    if r.timed_out:
        return {"total": 0, "passed": 0, "failed": [], "error": f"timed out after {timeout:.0f}s"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"total": 0, "passed": 0, "failed": [], "error": f"the test run crashed (exit {r.exit_code}): {(r.stderr or r.stdout)[-500:]}"}


async def run_tests_async(sandbox: Sandbox, *args, **kwargs) -> dict:
    return await asyncio.to_thread(run_tests, sandbox, *args, **kwargs)
