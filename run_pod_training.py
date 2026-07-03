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

  # create a FRESH pod from an explicit spec, run, then stop + TERMINATE it (for when an
  # existing pod's host is stuck and `pod start` keeps failing). Pick an --image that ships
  # python3.11+ and SSH so no Python is installed on the pod:
  python run_pod_training.py --create-pod --image <img-with-py311+ssh> \
      --vcpu 16 --mem 32 --disk 30 \
      --train "-m tokenomics.eval.curriculum_mining ..." \
      --pull /workspace/tokenomics/mine:./experiments/EXP-NNNN-title

Pod target (exactly one):
  * --pod-id <id>:     run on an existing pod; stopped afterwards, NEVER removed.
  * --create-pod:      enrol a FRESH pod from the explicit --image/--vcpu/--mem/--disk/...
                       flags (default: a 16 vCPU / 32 GB / 30 GB CPU pod), run, then stop
                       AND terminate the created pod. Only a pod created here is ever
                       removed. --keep-pod stops but keeps it for inspection.
  * --clone-from <id>: same lifecycle, but seed the spec from an existing pod via
                       `runpodctl pod get` instead of the explicit flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RUNPOD_REST = "https://rest.runpod.io/v1"

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

    def base_cmd(self, *, no_stdin: bool = False) -> list[str]:
        # `-n` redirects stdin from /dev/null: for the detached launch we never feed stdin,
        # and leaving the channel's stdin open is one way the launching SSH can hang waiting
        # on the remote shell instead of returning once the job is backgrounded.
        return [
            "ssh",
            *(["-n"] if no_stdin else []),
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


def remove_pod(pod_id: str) -> None:
    """Terminate (delete) a pod. ONLY ever called on a pod THIS run created (`--clone-from`)
    — a user-supplied `--pod-id` is never removed, only stopped. Best-effort: a remove
    failure is warned, never fatal, so it can't mask the run's real exit code."""
    log(f"terminating pod {pod_id} (removing)...")
    res = run(["runpodctl", "pod", "delete", pod_id], check=False, capture=True)
    if res.returncode != 0:
        log(f"WARNING: remove failed: {res.stderr.strip() or res.stdout.strip()}")
    else:
        log("pod terminated")


# --- clone-a-pod: enrol a fresh pod from an existing pod's characteristics ------------
# For when an existing pod's host is stuck and `pod start` keeps failing: read the
# template pod's spec (works even while it's stopped), create a NEW pod with the same
# shape on an available host, run, then stop + terminate the created pod.


@dataclass
class PodSpec:
    """The cloneable characteristics of a pod, normalized from `runpodctl pod get`."""

    image: str
    compute_type: str = "GPU"          # "GPU" | "CPU"
    gpu_type: str | None = None
    gpu_count: int = 1
    vcpu: int | None = None
    mem_gb: int | None = None
    container_disk_gb: int = 20
    volume_gb: int = 1
    volume_path: str = "/workspace"
    network_volume_id: str | None = None
    ports: list | str | None = None    # REST returns a JSON list; a string is also accepted
    secure_cloud: bool = True
    data_center_id: str | None = None
    cpu_flavor: str | None = None       # e.g. "cpu5c"; needed to size a CPU pod via REST


def _first(*values, default=None):
    """First value that is not None/empty (tolerant of runpodctl field-name variants)."""
    for v in values:
        if v not in (None, ""):
            return v
    return default


def pod_spec_from_json(raw: dict) -> PodSpec:
    """Extract cloneable characteristics from a `runpodctl pod get` JSON blob.

    Tolerant of field-name / nesting variants across runpodctl versions: GPU type, cloud
    and datacenter live under `machine` on some versions and top-level on others, so both
    are checked. `computeType` is derived from the GPU count when absent — a 0-GPU pod is a
    CPU pod. The derived spec is logged by `fetch_pod_spec`, so a wrong mapping is visible
    (and the create command is printed) before any billable pod is created."""
    m = raw.get("machine") or {}
    gpu_count = int(_first(raw.get("gpuCount"), m.get("gpuCount"), default=0) or 0)
    compute = _first(raw.get("computeType"), default=("GPU" if gpu_count else "CPU"))
    return PodSpec(
        image=_first(raw.get("imageName"), raw.get("image"), default=""),
        compute_type=compute,
        gpu_type=(_first(raw.get("gpuTypeId"), m.get("gpuTypeId"),
                         raw.get("gpuDisplayName"), m.get("gpuDisplayName"))
                  if compute == "GPU" else None),
        gpu_count=gpu_count or 1,
        vcpu=_first(raw.get("vcpuCount"), raw.get("vcpu"), m.get("vcpuCount")),
        mem_gb=_first(raw.get("memoryInGb"), raw.get("mem")),
        container_disk_gb=int(_first(raw.get("containerDiskInGb"),
                                     raw.get("containerDiskSize"), default=20)),
        volume_gb=int(_first(raw.get("volumeInGb"), raw.get("volumeSize"), default=1)),
        volume_path=_first(raw.get("volumeMountPath"), raw.get("volumePath"),
                           default="/workspace"),
        network_volume_id=_first(raw.get("networkVolumeId"),
                                 (raw.get("networkVolume") or {}).get("id")),
        ports=_first(raw.get("ports")),
        secure_cloud=bool(_first(raw.get("secureCloud"), m.get("secureCloud"), default=True)),
        data_center_id=_first(raw.get("dataCenterId"), m.get("dataCenterId")),
        cpu_flavor=_first(raw.get("cpuFlavorId"),
                          (raw.get("cpuFlavorIds") or [None])[0], m.get("cpuFlavorId")),
    )


def build_create_body(spec: PodSpec, name: str) -> dict:
    """The REST `POST /v1/pods` JSON body that reproduces `spec` under a new `name`.

    Pure (no IO) so the exact request is unit-tested before a billable create. We POST to
    the REST API instead of `runpodctl pod create` because the CLI has NO vCPU flag — a CPU
    pod created via CLI gets the smallest (2 vCPU) flavor. REST honours `vcpuCount` +
    `cpuFlavorIds` (verified: 16 vCPU / cpu5c comes up as asked). `22/tcp` is always exposed
    so the driver can reach the pod over SSH (RunPod injects the account's SSH keys). A
    network-volume pod attaches the volume + its datacenter and omits a per-pod
    `volumeInGb`; otherwise a per-pod volume is requested only when sized."""
    # spec.ports may be a REST JSON list (['8888/http', '22/tcp'] — the common cloned shape)
    # or a comma/space string; normalize both to a de-duped list that always exposes 22/tcp.
    if isinstance(spec.ports, (list, tuple)):
        extra = [str(p) for p in spec.ports]
    elif spec.ports:
        extra = re.split(r"[,\s]+", spec.ports)
    else:
        extra = []
    ports = ["22/tcp"] + [p for p in extra if p and p != "22/tcp"]
    body: dict = {
        "name": name,
        "imageName": spec.image,
        "computeType": spec.compute_type,
        "containerDiskInGb": spec.container_disk_gb,
        "volumeMountPath": spec.volume_path,
        "cloudType": "SECURE" if spec.secure_cloud else "COMMUNITY",
        "ports": ports,
    }
    if spec.compute_type == "CPU":
        if spec.vcpu:
            body["vcpuCount"] = spec.vcpu
        if spec.cpu_flavor:
            body["cpuFlavorIds"] = [spec.cpu_flavor]
    else:
        if spec.gpu_type:
            body["gpuTypeIds"] = [spec.gpu_type]
        body["gpuCount"] = spec.gpu_count
    if spec.network_volume_id:
        body["networkVolumeId"] = spec.network_volume_id
    elif spec.volume_gb > 0:
        # per-pod persistent volume only when asked; volume_gb == 0 keeps /workspace on the
        # container disk (fine for an ephemeral pod we terminate after pulling results).
        body["volumeInGb"] = spec.volume_gb
    if spec.data_center_id:
        body["dataCenterIds"] = [spec.data_center_id]
    return body


def _runpod_api_key() -> str:
    """The RunPod API key from env or ~/.runpod/config.toml (never logged)."""
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        m = re.search(r"apikey\s*=\s*['\"]([^'\"]+)['\"]", cfg.read_text())
        if m:
            return m.group(1)
    raise RuntimeError("no RunPod API key (set RUNPOD_API_KEY or run `runpodctl config`)")


def parse_created_pod_id(output: str) -> str:
    """The new pod's id from a `POST /v1/pods` JSON response (`id`), tolerant of a legacy
    `pod "<id>" created ...` human line."""
    output = output.strip()
    try:
        data = json.loads(output)
        if isinstance(data, dict) and data.get("id"):
            return str(data["id"])
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'pod\s+"?([A-Za-z0-9]+)"?', output)
    if m:
        return m.group(1)
    raise RuntimeError(f"could not parse created pod id from: {output!r}")


def pod_spec_from_args(args: argparse.Namespace) -> PodSpec:
    """Build a PodSpec straight from explicit `--image/--vcpu/...` flags (no template).

    The default shape is a CPU pod (16 vCPU / 32 GB / 30 GB container disk). Pick an
    `--image` that already ships python3.11+ and SSH so no Python has to be installed on
    the pod (the base image's python3.8 lacked ensurepip — choosing the image sidesteps the
    whole `--python python3.12` workaround)."""
    ct = args.compute_type
    return PodSpec(
        image=args.image,
        compute_type=ct,
        gpu_type=(args.gpu_type if ct == "GPU" else None),
        gpu_count=args.gpu_count,
        vcpu=(args.vcpu if ct == "CPU" else None),
        mem_gb=(args.mem if ct == "CPU" else None),
        container_disk_gb=args.disk,
        volume_gb=args.volume_size,
        volume_path=args.volume_path,
        network_volume_id=args.network_volume_id,
        data_center_id=args.data_center,
        secure_cloud=not args.community,
        cpu_flavor=(args.cpu_flavor if ct == "CPU" else None),
    )


def fetch_pod_spec(template_id: str) -> PodSpec:
    """Read the template pod's cloneable spec via REST `GET /v1/pods/<id>` (works while it is
    stopped). REST, not `runpodctl pod get`, because only REST returns `cpuFlavorId` — the
    CLI blob omits it, so a CLI-seeded clone would drop the flavor and mis-size vCPU."""
    req = urllib.request.Request(
        f"{RUNPOD_REST}/pods/{template_id}",
        headers={"Authorization": f"Bearer {_runpod_api_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"pod get failed ({e.code}): {e.read().decode().strip()}")
    spec = pod_spec_from_json(raw)
    log(f"cloning spec from {template_id}: {spec}")
    return spec


def create_pod(spec: PodSpec, name: str) -> str:
    """Create a fresh pod from `spec` via REST `POST /v1/pods`; return its id. Prints the
    request body first (never the API key), and surfaces the API error body on failure (so a
    bad spec isn't an opaque HTTP code). REST is used, not `runpodctl pod create`, because
    the CLI cannot set vCPU (a CPU pod would come up at 2 vCPU)."""
    body = build_create_body(spec, name)
    log("creating pod: POST /v1/pods " + json.dumps(body))
    req = urllib.request.Request(
        f"{RUNPOD_REST}/pods", method="POST",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_runpod_api_key()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"pod create failed ({e.code}): {e.read().decode().strip()}")
    pod_id = parse_created_pod_id(out)
    log(f"created pod {pod_id}")
    return pod_id


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


def ssh_run(
    ssh: Ssh,
    remote_cmd: str,
    *,
    check: bool = True,
    timeout: float | None = None,
    no_stdin: bool = False,
) -> int:
    """Run a remote command over SSH, streaming output. Returns exit code.

    `timeout` bounds the call (raises `subprocess.TimeoutExpired`) — used for the detached
    launch, which must return promptly once the job is backgrounded; a hung launch shell must
    not block the client forever (it never reached `monitor_remote`, so nothing streamed).
    `no_stdin` passes `ssh -n` for the same launch (don't hold the channel on stdin)."""
    res = subprocess.run(
        ssh.base_cmd(no_stdin=no_stdin) + [remote_cmd], text=True, timeout=timeout
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"remote command failed ({res.returncode}): {remote_cmd}")
    return res.returncode


# ssh exit code 255 = transport-level failure (connect refused/reset/timeout), distinct
# from a remote command's own non-zero exit. The poll command itself always exits 0, so a
# 255 (or a local timeout) means the link dropped — recoverable by reconnecting.
SSH_TRANSPORT_FAIL = 255

# Bound the detached-launch SSH call. The launch only backgrounds the job and returns, so it
# should take seconds; if it hasn't returned by now the outer shell has hung (EXP-0027) and we
# fall through to monitor_remote, which confirms liveness from the pod's pid/exit files.
LAUNCH_TIMEOUT_S = 60


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


def default_status_path(remote_log: str) -> str:
    """Local heartbeat path (`<remote-basename>.status`) that sits next to the `.live.log`
    mirror — a compact, always-fresh snapshot of the run's phase + last line, so remote
    status is on disk even when nobody is actively tailing."""
    return f"{Path(remote_log).stem}.status"


def init_local_mirror(local_log: str, header: str) -> bool:
    """Truncate the mirror and write a run header at LAUNCH START — before the multi-minute
    rsync+install — so during setup the local file already reflects THIS run (pod id, cmd,
    start time) instead of masquerading as a prior run's stale log. Returns False (never
    raises) on any IO failure, so a broken mirror path can't sink a live remote run."""
    try:
        Path(local_log).parent.mkdir(parents=True, exist_ok=True)
        with open(local_log, "w") as f:
            f.write(header)
        return True
    except OSError as e:
        log(f"WARNING: local log mirror disabled (cannot create {local_log}): {e}")
        return False


_PHASE_RE = re.compile(r"\[phase \d+/\d+\][^\n]*")


def parse_run_status(text: str) -> dict:
    """Compact status from streamed log text: the latest `[phase i/N] ...` marker and the
    last non-empty line (the live progress signal), plus a line count. Pure — the heartbeat
    and any status reader derive from this, no SSH."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    phases = _PHASE_RE.findall(text)
    return {
        "phase": phases[-1].rstrip() if phases else "",
        "last_line": lines[-1] if lines else "",
        "n_lines": len(lines),
    }


def write_run_status(path: str, *, pod_id: str | None, remote_log: str, state: str,
                     started: str, text: str, phase: str | None = None) -> None:
    """Write the compact heartbeat `.status` (JSON) for a detached run. Degrades silently on
    any IO failure — like the mirror, a heartbeat must never raise into `main()` and stop a
    live pod. `phase`, if given, overrides the phase parsed from the bounded `text` tail — the
    caller tracks it stickily so a phase whose marker scrolled past the tail isn't lost."""
    st = parse_run_status(text)
    if phase is not None:
        st["phase"] = phase
    st.update({"pod_id": pod_id, "remote_log": remote_log, "state": state,
               "started": started, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    try:
        Path(path).write_text(json.dumps(st, indent=2) + "\n")
    except OSError as e:
        log(f"WARNING: run status heartbeat disabled (write failed on {path}): {e}")


def make_content_sink(local_log: str | None, *, truncate: bool = True,
                      status: dict | None = None,
                      heartbeat_interval: float = 5.0) -> Callable[[str], None]:
    """An `on_content` callback that streams the detached run's new log bytes to stdout and,
    when `local_log` is set, mirrors the same bytes to a local file — so a long detached run
    is observable via `tail -f` locally, not only by re-SSHing the pod. `truncate=True` wipes
    any prior mirror up front (the attach/reconnect default); pass `truncate=False` when
    `init_local_mirror` already wrote a run header this launch, so the stream APPENDS. The
    monitor loop's offset never regresses, so per-poll appends never duplicate content.

    If `status` (`{path,pod_id,remote_log,started}`) is given, the sink also drops a compact
    heartbeat `.status` (throttled to `heartbeat_interval` seconds) parsed from the streamed
    tail — remote phase + last line on disk without an active tail. The mirror + heartbeat are
    OPTIONAL and degrade to a warning on any IO failure — they must never raise, or `main()`'s
    except-handler would stop the pod and kill a live remote run."""
    state = {"path": local_log, "buf": "", "last_hb": 0.0, "phase": ""}
    if local_log and truncate:
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
        if status:
            # Track the phase stickily off the FULL incoming chunk, not the bounded tail: a
            # long phase can emit >8 KB of same-phase output before the next marker, scrolling
            # its `[phase i/N]` line out of `buf` — reparsing only the tail would then report an
            # empty phase mid-run. Scanning `content` here keeps the last-seen phase pinned.
            found = _PHASE_RE.findall(content)
            if found:
                state["phase"] = found[-1].rstrip()
            # keep a bounded tail so the heartbeat's last-line stays cheap to derive.
            state["buf"] = (state["buf"] + content)[-8192:]
            now = time.time()
            if now - state["last_hb"] >= heartbeat_interval:
                state["last_hb"] = now
                write_run_status(status["path"], pod_id=status.get("pod_id"),
                                 remote_log=status["remote_log"], state="running",
                                 started=status["started"], text=state["buf"],
                                 phase=state["phase"])

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
    truncate: bool = True,
    status: dict | None = None,
) -> int:
    """Drive `run_monitor_loop` against a real pod: stream the detached run's log to our
    stdout (and mirror it to `local_log` if set, plus a heartbeat `.status` if `status` is
    given) and reconnect (re-resolving ip/port) whenever the SSH link drops. `truncate=False`
    appends to a mirror `init_local_mirror` pre-seeded with a header this launch. Returns the
    remote training exit code."""
    state = {"ssh": ssh}

    def poll_fn(offset: int) -> str:
        return ssh_capture(state["ssh"], build_poll_cmd(log_file, pid_file, exit_file, offset))

    def reconnect_fn() -> None:
        state["ssh"] = wait_for_ssh(pod_id, key_override, ready_timeout)

    return run_monitor_loop(
        poll_fn=poll_fn,
        on_content=make_content_sink(local_log, truncate=truncate, status=status),
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
    (training output goes to `log_file`) so the launching SSH call returns immediately.

    The trailing `exit 0` is load-bearing: without it the outer sshd-spawned shell can sit in
    `do_wait` after backgrounding `setsid` (observed EXP-0027), keeping the SSH channel open so
    the client's `ssh_run(launch)` never returns. Forcing the outer shell to exit closes the
    channel the instant the job is detached."""
    inner = (
        f"echo $$ > {pid_file}; "
        f"{python_bin_path} {train_args} > {log_file} 2>&1; "
        f"echo $? > {exit_file}"
    )
    return (
        f"cd {remote_dir} && rm -f {exit_file} && "
        f"setsid sh -c {_shquote(inner)} < /dev/null > /dev/null 2>&1 & "
        f"exit 0"
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


def _do_pulls(ssh: Ssh, pull_specs: list[str]) -> list[str]:
    """rsync result files back; return the list of specs that FAILED. A failure is warned
    (not raised) so it can't mask the real training exit code — but the caller must inspect
    the return: a created pod holding the only copy of the artifacts must NOT be terminated
    when a pull failed, or the results are lost (see the finally in the run driver)."""
    failed: list[str] = []
    for spec in pull_specs:
        remote_path, _, local_path = spec.partition(":")
        if not remote_path or not local_path:
            log(f"WARNING: bad --pull spec (want REMOTE:LOCAL): {spec}")
            failed.append(spec)
            continue
        try:
            rsync_down(ssh, remote_path, local_path)
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: pull failed for {spec}: {e}")
            failed.append(spec)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod-id", default=None, help="existing Runpod pod id to run on (stopped after, never removed)")
    ap.add_argument("--create-pod", action="store_true",
                    help="enrol a FRESH pod from the explicit --image/--vcpu/... spec flags, "
                         "run, then stop + TERMINATE the created pod. Use when an existing "
                         "pod's host is stuck on start.")
    ap.add_argument("--clone-from", default=None, metavar="TEMPLATE_POD_ID",
                    help="like --create-pod but seed the spec from this existing pod via "
                         "`runpodctl pod get` instead of the explicit flags.")
    ap.add_argument("--pod-name", default=None,
                    help="name for the created pod (default: clone-<src>-<timestamp>)")
    ap.add_argument("--keep-pod", action="store_true",
                    help="with --create-pod/--clone-from: stop but do NOT terminate the "
                         "created pod (leave it for inspection).")
    # explicit pod spec (used by --create-pod). Defaults = the CPU shape for the CPU-bound
    # trainers/experiments. Pick an --image with python3.11+ and SSH to skip installing
    # Python on the pod.
    ap.add_argument("--image", default=None,
                    help="container image for --create-pod (needs python3.11+ and SSH; "
                         "e.g. a runpod/pytorch py3.11 image)")
    ap.add_argument("--compute-type", default="CPU", choices=("CPU", "GPU"),
                    help="created-pod compute type (default CPU)")
    ap.add_argument("--vcpu", type=int, default=16, help="created CPU pod: vCPUs (default 16)")
    ap.add_argument("--mem", type=int, default=32, help="created CPU pod: system memory GB (default 32)")
    ap.add_argument("--cpu-flavor", default="cpu5c",
                    help="created CPU pod: RunPod cpu flavor id (default cpu5c = the 16 vCPU shape)")
    ap.add_argument("--disk", type=int, default=30, help="created pod: container disk GB (default 30)")
    ap.add_argument("--volume-size", type=int, default=0,
                    help="created pod: per-pod persistent volume GB (default 0 = none, "
                         "/workspace on container disk; results are pulled before terminate)")
    ap.add_argument("--volume-path", default="/workspace", help="created pod: volume mount path")
    ap.add_argument("--gpu-type", default=None, help="created GPU pod: gpu type id")
    ap.add_argument("--gpu-count", type=int, default=1, help="created GPU pod: gpu count")
    ap.add_argument("--data-center", default=None, help="created pod: datacenter id")
    ap.add_argument("--network-volume-id", default=None, help="created pod: attach this network volume")
    ap.add_argument("--community", action="store_true",
                    help="created pod: use community cloud (default secure cloud)")
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

    # exactly one target: an existing pod (--pod-id) or a freshly created/cloned one.
    create_mode = args.create_pod or bool(args.clone_from)
    if bool(args.pod_id) == create_mode:
        log("ERROR: pass exactly one of --pod-id or --create-pod/--clone-from")
        return 2
    if create_mode and args.attach:
        log("ERROR: --attach reconnects to an existing run; it can't create a pod")
        return 2
    if args.create_pod and not args.clone_from and not args.image:
        log("ERROR: --create-pod needs --image (pick one with python3.11+ and SSH)")
        return 2

    if not args.attach and not args.local_dir.exists():
        log(f"ERROR: local dir not found: {args.local_dir}")
        return 2

    detached = not args.foreground
    pid_file, exit_file = marker_paths(args.remote_log)
    local_log = None if args.no_local_log else (args.local_log or default_local_log(args.remote_log))
    status_path = None if not local_log else default_status_path(args.remote_log)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mirror_headed = False  # True once init_local_mirror wrote a launch header (append, don't wipe)
    started_ok = False
    detached_interrupt = False
    pod_id = args.pod_id  # None in clone mode until the pod is created
    created = False
    # Never terminate a CREATED pod that may still hold un-pulled artifacts. True only once
    # every requested --pull has succeeded (or none were requested); any pull failure OR an
    # error before pulling leaves it False, so the finally keeps the pod for recovery.
    pulls_ok = not args.pull
    try:
        if create_mode:
            # enrol a fresh pod (from a template's spec or explicit flags), then treat it
            # like any pod. Only a pod created HERE is eligible for termination.
            spec = fetch_pod_spec(args.clone_from) if args.clone_from else pod_spec_from_args(args)
            name = args.pod_name or f"clone-{args.clone_from or 'new'}-{int(time.time())}"
            pod_id = create_pod(spec, name)
            created = True
        start_pod(pod_id, retries=args.start_retries, retry_delay=args.start_retry_delay)
        ssh = wait_for_ssh(pod_id, args.ssh_key, args.ready_timeout)
        started_ok = True

        if args.foreground:
            # legacy: stream in foreground, blocks until the run finishes.
            if not args.no_sync:
                rsync_up(ssh, args.local_dir, args.remote_dir)
            if not args.no_install:
                setup_env(ssh, args.remote_dir, cpu=args.cpu, python_bin=args.python_bin)
            train(ssh, args.remote_dir, args.train, args.remote_log)
            pulls_ok = not _do_pulls(ssh, args.pull)
            log("training complete")
            return 0

        # detached client-server flow
        if args.attach:
            log(f"attaching to detached run (log {args.remote_log})...")
        else:
            # Seed the local mirror with a header BEFORE the multi-minute rsync+install, so a
            # `tail -f` during setup shows THIS run (pod/cmd/start), never a prior stale log.
            if local_log:
                header = (f"# run_pod detached mirror\n# pod {pod_id} | remote {args.remote_log}"
                          f" | started {started_at}\n# cmd: {args.train}\n"
                          "# (setting up venv + syncing code; remote output streams below)\n\n")
                mirror_headed = init_local_mirror(local_log, header)
                write_run_status(status_path, pod_id=pod_id, remote_log=args.remote_log,
                                 state="starting", started=started_at, text="")
            if not args.no_sync:
                rsync_up(ssh, args.local_dir, args.remote_dir)
            if not args.no_install:
                setup_env(ssh, args.remote_dir, cpu=args.cpu, python_bin=args.python_bin)
            launch = build_launch_cmd(
                args.remote_dir, ".venv/bin/python", args.train,
                args.remote_log, pid_file, exit_file,
            )
            log(f"launching detached: {args.train}")
            try:
                ssh_run(ssh, launch, timeout=LAUNCH_TIMEOUT_S, no_stdin=True)
            except subprocess.TimeoutExpired:
                # The launch shell didn't return in time (the EXP-0027 hang). The detached
                # job runs in its own session regardless — proceed to monitor, which verifies
                # liveness purely from the pod's pid/exit files.
                log(f"launch ssh did not return within {LAUNCH_TIMEOUT_S}s; the job is "
                    "detached (own session) — proceeding to monitor via pid/exit files")
            log(f"launched (survives disconnect). Reattach any time: "
                f"python run_pod_training.py --pod-id {pod_id} --attach "
                f"--remote-log {args.remote_log}{' --cpu' if args.cpu else ''}")

        status_meta = (None if not status_path else
                       {"path": status_path, "pod_id": pod_id, "remote_log": args.remote_log,
                        "started": started_at})
        if local_log:
            log(f"mirroring remote log -> {local_log} (tail -f); status -> {status_path}")
        rc = monitor_remote(
            pod_id, ssh, args.ssh_key, args.remote_log, pid_file, exit_file,
            ready_timeout=args.ready_timeout, poll_interval=args.poll_interval,
            local_log=local_log, truncate=not mirror_headed, status=status_meta,
        )
        if status_path:  # final heartbeat: full mirror text -> accurate last phase/line + state
            try:
                final_text = Path(local_log).read_text() if local_log else ""
            except OSError:
                final_text = ""
            write_run_status(status_path, pod_id=pod_id, remote_log=args.remote_log,
                             state="finished" if rc == 0 else "failed",
                             started=started_at, text=final_text)
        log(f"detached run finished (exit {rc}); pulling results")
        ssh = resolve_ssh(pod_id, args.ssh_key)  # refresh endpoint for the pull
        pulls_ok = not _do_pulls(ssh, args.pull)
        return rc
    except KeyboardInterrupt:
        if detached:
            detached_interrupt = True
            log("client detached — the run KEEPS GOING on the pod. "
                f"Reattach: python run_pod_training.py --pod-id {pod_id} --attach "
                f"--remote-log {args.remote_log}{' --cpu' if args.cpu else ''}")
            if created:
                log(f"NOTE: cloned pod {pod_id} was NOT terminated (run still going). "
                    f"Remove it yourself when done: runpodctl remove pod {pod_id}")
            return 130
        log("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001 - report and still stop the pod
        log(f"ERROR: {e}")
        return 1
    finally:
        # A cloned pod is stopped AND terminated (the whole point: don't leave a fresh pod
        # billing). A user-supplied --pod-id is only ever stopped, never removed.
        if pod_id is None:
            pass  # clone creation failed before a pod existed — nothing to clean up
        elif args.no_stop:
            log("--no-stop set: leaving pod running")
            if created:
                log(f"NOTE: cloned pod {pod_id} left running — remove with: "
                    f"runpodctl remove pod {pod_id}")
        elif detached_interrupt:
            log("leaving pod running so the detached run can finish (use --attach to resume)")
        elif created and started_ok and not pulls_ok and not args.keep_pod:
            # A created pod that came up holds the ONLY copy of the artifacts and a --pull
            # did not confirm success (failed pull, or an error before pulling) —
            # terminating it would destroy the results. Keep it (stopped) for recovery.
            # (A pod that never came up produced nothing, so it falls through to terminate.)
            stop_pod(pod_id)
            log(f"NOTE: results were NOT confirmed pulled — cloned pod {pod_id} kept "
                f"(stopped), not terminated, so the artifacts survive. Recover them, then "
                f"remove it: runpodctl remove pod {pod_id}")
        elif started_ok or pod_status(pod_id) == "RUNNING":
            stop_pod(pod_id)
            if created and not args.keep_pod:
                remove_pod(pod_id)
        elif created and not args.keep_pod:
            # created but never came up (e.g. SSH never ready) — still terminate it so a
            # half-started clone doesn't linger and bill.
            remove_pod(pod_id)


if __name__ == "__main__":
    sys.exit(main())
