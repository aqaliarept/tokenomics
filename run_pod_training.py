#!/usr/bin/env python3
"""Start a Runpod pod, sync the project, run a training cycle, then stop the pod.

End-to-end driver for off-box training. Everything the run needs lives here:
  1. start the pod (runpodctl) and wait until SSH is reachable
  2. rsync the local project to the pod (no git; excludes build/run artifacts)
  3. create a --system-site-packages venv (reuses the image's CUDA torch) + pip install -e .
  4. run the training command, streaming its output live
  5. ALWAYS stop the pod afterwards (even on failure / Ctrl-C), unless --no-stop

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


def train(ssh: Ssh, remote_dir: str, train_args: str, log_file: str) -> None:
    log(f"running training: {train_args}")
    # Run in foreground so output streams and the SSH session blocks until done;
    # tee to a log file on the pod for later inspection.
    cmd = (
        f"cd {remote_dir} && "
        f".venv/bin/python {train_args} 2>&1 | tee {log_file}"
    )
    ssh_run(ssh, cmd)


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
    ap.add_argument(
        "--pull",
        action="append",
        default=[],
        metavar="REMOTE:LOCAL",
        help="file to rsync back after training (repeatable)",
    )
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

    if not args.local_dir.exists():
        log(f"ERROR: local dir not found: {args.local_dir}")
        return 2

    started_ok = False
    try:
        start_pod(args.pod_id, retries=args.start_retries, retry_delay=args.start_retry_delay)
        ssh = wait_for_ssh(args.pod_id, args.ssh_key, args.ready_timeout)
        started_ok = True

        if not args.no_sync:
            rsync_up(ssh, args.local_dir, args.remote_dir)
        if not args.no_install:
            setup_env(ssh, args.remote_dir, cpu=args.cpu, python_bin=args.python_bin)

        train(ssh, args.remote_dir, args.train, args.remote_log)

        for spec in args.pull:
            remote_path, _, local_path = spec.partition(":")
            if not remote_path or not local_path:
                log(f"WARNING: bad --pull spec (want REMOTE:LOCAL): {spec}")
                continue
            rsync_down(ssh, remote_path, local_path)

        log("training complete")
        return 0
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001 - report and still stop the pod
        log(f"ERROR: {e}")
        return 1
    finally:
        if args.no_stop:
            log("--no-stop set: leaving pod running")
        elif started_ok or pod_status(args.pod_id) == "RUNNING":
            stop_pod(args.pod_id)


if __name__ == "__main__":
    sys.exit(main())
