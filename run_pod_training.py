#!/usr/bin/env python3
"""Start a Runpod pod, sync the project, run a training cycle, then stop the pod.

End-to-end driver for off-box training. Everything the run needs lives here:
  1. start the pod (runpodctl) and wait until SSH is reachable
  2. rsync the local project to the pod (no git; excludes build/run artifacts)
  3. create a --system-site-packages venv (reuses the image's CUDA torch) + pip install -e .
  4. launch the training command DETACHED on the pod, then monitor it
  5. ALWAYS stop the pod afterwards (even on failure / Ctrl-C), unless --no-stop

Client–server / detached model (default):
  The pod runs the job in its own session (`setsid`), writing to a log file plus pid/exit
  marker files — so it survives an SSH drop or the client going away. The client polls
  those markers over SSH, streaming new log bytes, and **auto-reconnects** (re-resolving
  ip/port) whenever the link breaks, finishing only when the remote exit marker appears.
  Ctrl-C just detaches the client (the run keeps going); reconnect later with `--attach`.
  `--foreground` keeps the legacy behavior (stream over a blocking SSH session that dies
  with the link). This is the disconnect-proof model CLAUDE.md flagged as not-yet-done.

The pod is addressed by id; SSH ip/port are resolved fresh via `runpodctl ssh info`
each run, because they change across restarts.

Pod profiles (--cpu):
  * GPU (default): a `runpod/pytorch:*` image — reuse its system CUDA torch.
  * CPU (`--cpu`):  a `runpod/base:*` image — no torch; install the `[train]` extra
    (CPU torch) into an isolated venv. Correct for the CPU-bound trainers, where the
    GPU buys nothing (batch-1 inference + single-threaded Python rollout).

Usage:
  python run_pod_training.py --pod-id h8n0zosgxesksp \
      --train "-m tokenomics.train.smoke_ppo --iters 50 --steps 1024 --seed 0"

  # CPU pod (base image): install CPU torch via the [train] extra
  python run_pod_training.py --pod-id <cpu-pod-id> --cpu \
      --train "-m tokenomics.train.smoke_ppo --iters 50 --steps 1024 --seed 0"

  # leave the pod running for inspection afterwards:
  python run_pod_training.py --pod-id <id> --no-stop

  # pull a checkpoint back when training writes one:
  python run_pod_training.py --pod-id <id> \
      --train "-m tokenomics.train.smoke_ppo --iters 400 --save /workspace/model.pt" \
      --pull /workspace/model.pt:./model.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Monitor-loop collaborators (dependency-injected so the reconnect logic is testable).
PollFn = Callable[[int], str]        # offset -> raw poll output (raises SshDropped on transport loss)
ContentSink = Callable[[str], None]  # consume newly streamed log content
ReconnectFn = Callable[[], None]     # re-establish the SSH endpoint after a drop
SleepFn = Callable[[float], None]    # sleep between polls (injectable for tests)

# Project dir to sync = the directory holding pyproject.toml (the package root).
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "tokenomics"
DEFAULT_REMOTE_DIR = "/workspace/tokenomics"
RSYNC_EXCLUDES = (
    "build/",
    "*.egg-info/",
    "boundary_runs/",
    "parallel_runs/",
    "__pycache__/",
    ".git/",
    ".venv/",
)


@dataclass
class Ssh:
    """Resolved SSH connection details for the pod."""

    ip: str
    port: int
    key: str

    def base_cmd(self) -> list[str]:
        return [
            "ssh",
            "-i", self.key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=15",
            "-p", str(self.port),
            f"root@{self.ip}",
        ]


def log(msg: str) -> None:
    print(f"[run_pod] {msg}", flush=True)


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a local command. Streams to the terminal unless capture=True."""
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def pod_status(pod_id: str) -> str:
    """Current desiredStatus for the pod (RUNNING / EXITED / ...)."""
    out = run(["runpodctl", "pod", "get", pod_id], capture=True).stdout
    return json.loads(out).get("desiredStatus", "UNKNOWN")


def start_pod(pod_id: str, *, retries: int = 0, retry_delay: int = 30) -> None:
    """Start the pod, retrying transient start failures.

    Runpod refuses `pod start` with "not enough free GPUs on the host machine" when the
    pod's host is momentarily full — a transient capacity error, not a config error. We
    retry up to `retries` extra times with a fixed delay so a queued run can grab a GPU
    as soon as one frees, instead of failing the whole off-box cycle."""
    status = pod_status(pod_id)
    log(f"pod {pod_id} status: {status}")
    if status == "RUNNING":
        log("pod already running")
        return
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        log(f"starting pod... (attempt {attempt}/{attempts})")
        res = run(["runpodctl", "pod", "start", pod_id], check=False, capture=True)
        if res.returncode == 0:
            return
        reason = (res.stderr or res.stdout).strip()
        if attempt == attempts:
            raise RuntimeError(f"pod start failed after {attempts} attempt(s): {reason}")
        log(f"start failed ({reason}); retrying in {retry_delay}s")
        time.sleep(retry_delay)


def stop_pod(pod_id: str) -> None:
    log(f"stopping pod {pod_id}...")
    # Don't let a stop failure mask the real outcome; just report it.
    res = run(["runpodctl", "pod", "stop", pod_id], check=False, capture=True)
    if res.returncode != 0:
        log(f"WARNING: stop failed: {res.stderr.strip() or res.stdout.strip()}")
    else:
        log("pod stopped")


def resolve_ssh(pod_id: str, key_override: str | None) -> Ssh:
    """Resolve the pod's direct SSH endpoint via `runpodctl ssh info` (ip/port + key
    path). On this runpodctl that endpoint is already the direct `root@ip -p port`
    target the driver connects to."""
    out = run(["runpodctl", "ssh", "info", pod_id], capture=True).stdout
    info = json.loads(out)
    key = key_override or info["ssh_key"]["path"]
    return Ssh(ip=info["ip"], port=int(info["port"]), key=key)


def wait_for_ssh(pod_id: str, key_override: str | None, timeout_s: int) -> Ssh:
    """Poll until the pod is RUNNING and accepts SSH, returning live connection details."""
    log(f"waiting for pod to become reachable (timeout {timeout_s}s)...")
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            if pod_status(pod_id) == "RUNNING":
                ssh = resolve_ssh(pod_id, key_override)
                probe = subprocess.run(
                    ssh.base_cmd() + ["echo ok"],
                    text=True, capture_output=True,
                )
                if probe.returncode == 0 and "ok" in probe.stdout:
                    log(f"SSH up at {ssh.ip}:{ssh.port}")
                    return ssh
                last = (probe.stderr or probe.stdout).strip()
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
            last = str(e)
        time.sleep(5)
    raise TimeoutError(f"pod not reachable within {timeout_s}s (last: {last})")


def ssh_run(ssh: Ssh, remote_cmd: str, *, check: bool = True) -> int:
    """Run a remote command over SSH, streaming output. Returns exit code."""
    res = subprocess.run(ssh.base_cmd() + [remote_cmd], text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"remote command failed ({res.returncode}): {remote_cmd}")
    return res.returncode


# ssh exit code 255 = transport-level failure (connect refused/reset/timeout), distinct
# from a remote command's own non-zero exit. The poll command itself always exits 0, so a
# 255 (or a local timeout) means the link dropped — recoverable by reconnecting.
SSH_TRANSPORT_FAIL = 255


def ssh_capture(ssh: Ssh, remote_cmd: str, *, timeout_s: int = 30) -> str:
    """Run a remote command capturing stdout; raise `SshDropped` if the link failed.

    Used by the poll loop: a dropped/!reachable pod must surface as `SshDropped` (so the
    monitor reconnects) rather than as a fatal error."""
    try:
        res = subprocess.run(
            ssh.base_cmd() + [remote_cmd],
            text=True, capture_output=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise SshDropped(f"ssh timeout after {timeout_s}s") from e
    if res.returncode == SSH_TRANSPORT_FAIL:
        raise SshDropped((res.stderr or "ssh transport failure").strip())
    if res.returncode != 0:
        raise RuntimeError(f"remote command failed ({res.returncode}): {res.stderr.strip()}")
    return res.stdout


def default_local_log(remote_log: str) -> str:
    """Local mirror path for a detached run, derived from the remote log's basename: the
    `.live.log` suffix the repo .gitignore excludes, in the cwd (so every run is watchable
    on disk by default without curating a tracked file)."""
    return f"{Path(remote_log).stem}.live.log"


def make_content_sink(local_log: str | None) -> Callable[[str], None]:
    """An `on_content` callback that streams the detached run's new log bytes to stdout and,
    when `local_log` is set, mirrors the same bytes to a local file — so a long detached run
    is observable via `tail -f` locally, not only by re-SSHing the pod. The mirror is
    truncated up front so a fresh run never appends to a prior run's stale log; the monitor
    loop's offset never regresses, so per-poll appends never duplicate content.

    The mirror is OPTIONAL and created AFTER the remote job has launched, so any IO failure
    (read-only cwd, unwritable parent, full disk) is degraded to a warning and disables the
    mirror — it must never raise, or `main()`'s except-handler would stop the pod and kill a
    live remote run."""
    state = {"path": local_log}
    if local_log:
        try:
            Path(local_log).parent.mkdir(parents=True, exist_ok=True)
            open(local_log, "w").close()  # truncate any prior run's mirror
        except OSError as e:
            log(f"WARNING: local log mirror disabled (cannot create {local_log}): {e}")
            state["path"] = None

    def sink(content: str) -> None:
        sys.stdout.write(content)
        sys.stdout.flush()
        if state["path"]:
            try:
                with open(state["path"], "a") as f:
                    f.write(content)
            except OSError as e:
                log(f"WARNING: local log mirror disabled (write failed on {state['path']}): {e}")
                state["path"] = None

    return sink


def monitor_remote(
    pod_id: str,
    ssh: Ssh,
    key_override: str | None,
    log_file: str,
    pid_file: str,
    exit_file: str,
    *,
    ready_timeout: int,
    poll_interval: float,
    local_log: str | None = None,
) -> int:
    """Drive `run_monitor_loop` against a real pod: stream the detached run's log to our
    stdout (and mirror it to `local_log` if set) and reconnect (re-resolving ip/port)
    whenever the SSH link drops. Returns the remote training exit code."""
    state = {"ssh": ssh}

    def poll_fn(offset: int) -> str:
        return ssh_capture(state["ssh"], build_poll_cmd(log_file, pid_file, exit_file, offset))

    def reconnect_fn() -> None:
        state["ssh"] = wait_for_ssh(pod_id, key_override, ready_timeout)

    return run_monitor_loop(
        poll_fn=poll_fn,
        on_content=make_content_sink(local_log),
        reconnect_fn=reconnect_fn,
        sleep_fn=time.sleep,
        poll_interval=poll_interval,
    )


def rsync_up(ssh: Ssh, local_dir: Path, remote_dir: str) -> None:
    log(f"syncing {local_dir} -> {ssh.ip}:{remote_dir}")
    ssh_inline = (
        f"ssh -i {ssh.key} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {ssh.port}"
    )
    cmd = ["rsync", "-az", "--no-o", "--no-g", "--info=stats1", "-e", ssh_inline]
    for ex in RSYNC_EXCLUDES:
        cmd += ["--exclude", ex]
    cmd += [f"{local_dir}/", f"root@{ssh.ip}:{remote_dir}/"]
    run(cmd)


def rsync_down(ssh: Ssh, remote_path: str, local_path: str) -> None:
    log(f"pulling {remote_path} -> {local_path}")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    ssh_inline = (
        f"ssh -i {ssh.key} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {ssh.port}"
    )
    run(["rsync", "-az", "-e", ssh_inline, f"root@{ssh.ip}:{remote_path}", local_path])


def setup_env(ssh: Ssh, remote_dir: str, *, cpu: bool = False, python_bin: str = "python3") -> None:
    """Create the venv and install the package editable.

    Two pod profiles:
      * GPU (default): a `runpod/pytorch:*` image ships CUDA torch in system
        site-packages. Make the venv `--system-site-packages` and install the base
        package (`-e .`) so it REUSES that prebuilt CUDA torch — no multi-GB download.
      * CPU (`cpu=True`): a `runpod/base:*` image has no torch. Make an ISOLATED venv
        (no system site-packages, so a stray system torch can't shadow it) and install
        the `[train]` extra so pip pulls CPU torch into the venv. This is the right
        profile for the single-threaded, CPU-bound trainers (smoke_ppo / parallel /
        producer_improve) — the GPU buys nothing for batch-1 inference + a Python
        round-robin rollout, so a cheap CPU pod is the correct off-box target.

    `python_bin` selects the interpreter the venv is built from — base CPU images
    default `python3` to a too-old build (ubuntu 20.04 ships 3.8, below the package's
    `requires-python >= 3.11`), so pass e.g. `--python python3.12` for those."""
    log(f"setting up venv + installing package ({'CPU' if cpu else 'GPU'} profile, {python_bin})...")
    if cpu:
        # CPU profile: install torch from the CPU wheel index FIRST. PyPI's default
        # `torch` is the CUDA build, which drags in multi-GB `nvidia-*` deps that blow a
        # small container disk (5 GB). The CPU wheel is ~190 MB with no CUDA libs.
        # `--no-cache-dir` keeps pip from also stashing the wheel on the same small disk.
        # The second install then sees torch satisfied and adds only the rest of [train].
        install = (
            f"test -x .venv/bin/python || {python_bin} -m venv .venv && "
            ".venv/bin/pip install --no-cache-dir -q torch "
            "--index-url https://download.pytorch.org/whl/cpu && "
            ".venv/bin/pip install --no-cache-dir -q -e .[train]"
        )
    else:
        # GPU profile: reuse the image's system CUDA torch (no torch install).
        install = (
            f"test -x .venv/bin/python || {python_bin} -m venv --system-site-packages .venv && "
            ".venv/bin/pip install -e . -q"
        )
    cmd = (
        f"cd {remote_dir} && {install} && "
        ".venv/bin/python -c 'import tokenomics, torch; "
        "print(\"env ok | torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"
    )
    ssh_run(ssh, cmd)


POLL_SENTINEL = "__RUNPOD_POLL__"


@dataclass
class PollResult:
    """One poll of a detached run: new log bytes + liveness/completion status."""

    content: str
    size: int
    exit_code: int | None
    alive: bool


def parse_poll_output(raw: str) -> PollResult:
    """Split a poll response into streamable log content and the status sentinel.

    The remote poll emits the new log bytes since the client's offset, then a final
    `POLL_SENTINEL size=<N> exit=<code|none> alive=<0|1>` line. `exit=none` means the run
    has not finished; any integer means it completed with that return code."""
    head, _, tail = raw.rpartition(POLL_SENTINEL + " ")
    if not tail:
        # no sentinel (e.g. truncated read) — treat all as content, status unknown/alive
        return PollResult(content=raw, size=0, exit_code=None, alive=True)
    fields = dict(tok.split("=", 1) for tok in tail.split() if "=" in tok)
    exit_raw = fields.get("exit", "none")
    return PollResult(
        content=head,
        size=int(fields.get("size", "0")),
        exit_code=None if exit_raw == "none" else int(exit_raw),
        alive=fields.get("alive") == "1",
    )


class SshDropped(Exception):
    """The SSH transport failed mid-poll (drop / reset / timeout) — retry after reconnect."""


def run_monitor_loop(
    poll_fn: PollFn,
    on_content: ContentSink,
    reconnect_fn: ReconnectFn,
    sleep_fn: SleepFn,
    poll_interval: float,
) -> int:
    """Poll a detached run to completion, surviving SSH drops, and return its exit code.

    Each iteration polls from the current byte offset, emits the new log content, and
    advances the offset by the reported size. A `SshDropped` from `poll_fn` is non-fatal:
    call `reconnect_fn` (re-resolve the endpoint — ip/port can change) and retry from the
    *same* offset, so no log is lost or duplicated across a disconnect. Returns once the
    remote exit marker appears."""
    offset = 0
    while True:
        try:
            res = parse_poll_output(poll_fn(offset))
        except SshDropped as e:
            log(f"connection lost ({e}); reconnecting...")
            reconnect_fn()
            continue
        if res.content:
            on_content(res.content)
        offset = max(offset, res.size)
        if res.exit_code is not None:
            return res.exit_code
        sleep_fn(poll_interval)


def marker_paths(log_file: str) -> tuple[str, str]:
    """Derive the pid/exit marker paths as siblings of the remote log (`.pid` / `.exit`).

    Co-locating them with the log keeps a detached run fully described by one base path,
    so `--attach` can reconnect knowing only `--remote-log`."""
    base = log_file[: -len(suffix)] if (suffix := Path(log_file).suffix) else log_file
    return f"{base}.pid", f"{base}.exit"


def build_poll_cmd(log_file: str, pid_file: str, exit_file: str, offset: int) -> str:
    """Remote command for one poll: stream new log bytes past `offset`, then a sentinel.

    A single SSH round-trip both advances the client's view of the log and reports
    status. `wc -c` gives the current size; `tail -c +<offset+1>` emits only the bytes the
    client hasn't seen (1-indexed, hence +1). The trailing sentinel carries the new size
    (next offset), the exit code if the run has finished (`none` while running), and a
    `kill -0` liveness check on the wrapper pid (distinguishes "still running" from
    "process gone but no exit marker" = a crash)."""
    return (
        f"sz=$(wc -c < {log_file} 2>/dev/null || echo 0); "
        f"if [ \"$sz\" -gt {offset} ]; then tail -c +{offset + 1} {log_file}; fi; "
        f"echo \"{POLL_SENTINEL} size=$sz "
        f"exit=$(cat {exit_file} 2>/dev/null || echo none) "
        f"alive=$(kill -0 $(cat {pid_file} 2>/dev/null) 2>/dev/null && echo 1 || echo 0)\""
    )


def build_launch_cmd(
    remote_dir: str,
    python_bin_path: str,
    train_args: str,
    log_file: str,
    pid_file: str,
    exit_file: str,
) -> str:
    """Remote shell command that launches training *detached* from the SSH session.

    `setsid` puts the run in its own session so closing/dropping the SSH channel does not
    SIGHUP it — the training keeps going while the client is offline. The inner shell
    records its own pid (a liveness handle) before training and the python exit code
    after, so a reconnecting client can tell "running" from "done (rc=N)" purely from
    files on the pod. A stale exit marker is removed first so it can't read as completion
    of the new run. stdin/stdout/stderr of the backgrounded job are detached to /dev/null
    (training output goes to `log_file`) so the launching SSH call returns immediately."""
    inner = (
        f"echo $$ > {pid_file}; "
        f"{python_bin_path} {train_args} > {log_file} 2>&1; "
        f"echo $? > {exit_file}"
    )
    return (
        f"cd {remote_dir} && rm -f {exit_file} && "
        f"setsid sh -c {_shquote(inner)} < /dev/null > /dev/null 2>&1 &"
    )


def _shquote(s: str) -> str:
    """Single-quote a string for safe embedding as one POSIX shell word."""
    return "'" + s.replace("'", "'\\''") + "'"


def train(ssh: Ssh, remote_dir: str, train_args: str, log_file: str) -> None:
    log(f"running training: {train_args}")
    # Run in foreground so output streams and the SSH session blocks until done;
    # tee to a log file on the pod for later inspection.
    cmd = (
        f"cd {remote_dir} && "
        f".venv/bin/python {train_args} 2>&1 | tee {log_file}"
    )
    ssh_run(ssh, cmd)


def _do_pulls(ssh: Ssh, pull_specs: list[str]) -> None:
    """Best-effort rsync of result files back. A missing artifact (e.g. after a failed
    run) is warned, not fatal — it must not mask the real training exit code."""
    for spec in pull_specs:
        remote_path, _, local_path = spec.partition(":")
        if not remote_path or not local_path:
            log(f"WARNING: bad --pull spec (want REMOTE:LOCAL): {spec}")
            continue
        try:
            rsync_down(ssh, remote_path, local_path)
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: pull failed for {spec}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod-id", required=True, help="Runpod pod id")
    ap.add_argument(
        "--train",
        default="-m tokenomics.train.smoke_ppo --iters 50 --steps 1024 --seed 0",
        help="args passed to the pod's `.venv/bin/python` (default: a smoke PPO run)",
    )
    ap.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR, help="local project dir to sync")
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="remote project dir")
    ap.add_argument("--ssh-key", default=None, help="override SSH private key path")
    ap.add_argument("--remote-log", default="/workspace/train.log", help="log path on the pod")
    ap.add_argument("--ready-timeout", type=int, default=600, help="seconds to wait for SSH after start")
    ap.add_argument("--start-retries", type=int, default=0, help="extra pod-start attempts on transient failure (e.g. host out of free GPUs)")
    ap.add_argument("--start-retry-delay", type=int, default=30, help="seconds between pod-start retries")
    ap.add_argument("--poll-interval", type=float, default=15.0, help="seconds between detached-run status polls")
    ap.add_argument("--local-log", default=None,
                    help="local file to mirror the detached run's log into for `tail -f` "
                         "observability (default: <remote-log-basename>.live.log in the cwd, "
                         "which the repo .gitignore excludes)")
    ap.add_argument("--no-local-log", action="store_true",
                    help="disable the local log mirror (stdout streaming only)")
    ap.add_argument(
        "--pull",
        action="append",
        default=[],
        metavar="REMOTE:LOCAL",
        help="file to rsync back after training (repeatable)",
    )
    ap.add_argument("--foreground", action="store_true",
                    help="legacy mode: run training in the foreground over SSH (dies if the link "
                         "drops). Default is detached: the pod runs the job under its own session "
                         "(survives disconnects) and the client polls + auto-reconnects.")
    ap.add_argument("--attach", action="store_true",
                    help="reconnect to an already-launched detached run (same --remote-log): skip "
                         "start/sync/install/launch, just resume monitoring, then pull + stop. Use "
                         "after the client died or you disconnected.")
    ap.add_argument("--no-stop", action="store_true", help="leave the pod running when done")
    ap.add_argument("--no-sync", action="store_true", help="skip rsync (code already on pod)")
    ap.add_argument("--no-install", action="store_true", help="skip venv/install step")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU-pod profile: isolated venv + install the [train] extra (CPU torch) "
                         "instead of reusing a GPU image's system CUDA torch (default: GPU profile)")
    ap.add_argument("--python", default="python3", dest="python_bin",
                    help="interpreter to build the pod venv from (default python3; base CPU "
                         "images need e.g. python3.12, as their python3 is below requires-python)")
    args = ap.parse_args()

    if not args.attach and not args.local_dir.exists():
        log(f"ERROR: local dir not found: {args.local_dir}")
        return 2

    detached = not args.foreground
    pid_file, exit_file = marker_paths(args.remote_log)
    local_log = None if args.no_local_log else (args.local_log or default_local_log(args.remote_log))
    started_ok = False
    detached_interrupt = False
    try:
        start_pod(args.pod_id, retries=args.start_retries, retry_delay=args.start_retry_delay)
        ssh = wait_for_ssh(args.pod_id, args.ssh_key, args.ready_timeout)
        started_ok = True

        if args.foreground:
            # legacy: stream in foreground, blocks until the run finishes.
            if not args.no_sync:
                rsync_up(ssh, args.local_dir, args.remote_dir)
            if not args.no_install:
                setup_env(ssh, args.remote_dir, cpu=args.cpu, python_bin=args.python_bin)
            train(ssh, args.remote_dir, args.train, args.remote_log)
            _do_pulls(ssh, args.pull)
            log("training complete")
            return 0

        # detached client-server flow
        if args.attach:
            log(f"attaching to detached run (log {args.remote_log})...")
        else:
            if not args.no_sync:
                rsync_up(ssh, args.local_dir, args.remote_dir)
            if not args.no_install:
                setup_env(ssh, args.remote_dir, cpu=args.cpu, python_bin=args.python_bin)
            launch = build_launch_cmd(
                args.remote_dir, ".venv/bin/python", args.train,
                args.remote_log, pid_file, exit_file,
            )
            log(f"launching detached: {args.train}")
            ssh_run(ssh, launch)
            log(f"launched (survives disconnect). Reattach any time: "
                f"python run_pod_training.py --pod-id {args.pod_id} --attach "
                f"--remote-log {args.remote_log}{' --cpu' if args.cpu else ''}")

        if local_log:
            log(f"mirroring remote log -> {local_log} (tail -f for live progress)")
        rc = monitor_remote(
            args.pod_id, ssh, args.ssh_key, args.remote_log, pid_file, exit_file,
            ready_timeout=args.ready_timeout, poll_interval=args.poll_interval,
            local_log=local_log,
        )
        log(f"detached run finished (exit {rc}); pulling results")
        ssh = resolve_ssh(args.pod_id, args.ssh_key)  # refresh endpoint for the pull
        _do_pulls(ssh, args.pull)
        return rc
    except KeyboardInterrupt:
        if detached:
            detached_interrupt = True
            log("client detached — the run KEEPS GOING on the pod. "
                f"Reattach: python run_pod_training.py --pod-id {args.pod_id} --attach "
                f"--remote-log {args.remote_log}{' --cpu' if args.cpu else ''}")
            return 130
        log("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001 - report and still stop the pod
        log(f"ERROR: {e}")
        return 1
    finally:
        if args.no_stop:
            log("--no-stop set: leaving pod running")
        elif detached_interrupt:
            log("leaving pod running so the detached run can finish (use --attach to resume)")
        elif started_ok or pod_status(args.pod_id) == "RUNNING":
            stop_pod(args.pod_id)


if __name__ == "__main__":
    sys.exit(main())
