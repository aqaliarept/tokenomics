"""Tests for the detached client-server model in run_pod_training.py.

run_pod_training.py is a repo-root script (outside the package), so load it by path.
The detached model is the disconnect-resilient rewrite: the pod runs training under a
new session (survives SSH drop), and the client polls log/pid/exit markers, reconnecting
on connection loss. These tests cover the pure command-building + parsing helpers that
make that loop testable without a live pod.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "run_pod_training.py"
_spec = importlib.util.spec_from_file_location("run_pod_training", _PATH)
assert _spec and _spec.loader
rpt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rpt
_spec.loader.exec_module(rpt)


def test_build_launch_cmd_detaches_and_records_pid_and_exit() -> None:
    cmd = rpt.build_launch_cmd(
        remote_dir="/workspace/tokenomics",
        python_bin_path=".venv/bin/python",
        train_args="-m tokenomics.train.smoke_ppo --iters 50",
        log_file="/workspace/train.log",
        pid_file="/workspace/train.pid",
        exit_file="/workspace/train.exit",
    )
    # detaches into its own session so closing SSH does not SIGHUP the training
    assert "setsid" in cmd
    # runs in the right dir, with the right interpreter + args
    assert "cd /workspace/tokenomics" in cmd
    assert ".venv/bin/python -m tokenomics.train.smoke_ppo --iters 50" in cmd
    # a stale exit marker from a previous run must not be mistaken for "done"
    assert "rm -f /workspace/train.exit" in cmd
    # records the wrapper pid (liveness) and the final exit code (completion)
    assert "/workspace/train.pid" in cmd
    assert "$? > /workspace/train.exit" in cmd or "$? >/workspace/train.exit" in cmd
    # training output is redirected to the log file (not the dropped SSH channel)
    assert "/workspace/train.log" in cmd


def test_parse_poll_output_separates_new_log_from_status_sentinel() -> None:
    raw = (
        "epoch 1 loss 0.5\n"
        "epoch 2 loss 0.4\n"
        f"{rpt.POLL_SENTINEL} size=4096 exit=none alive=1\n"
    )
    res = rpt.parse_poll_output(raw)
    # streamed log content excludes the sentinel line
    assert res.content == "epoch 1 loss 0.5\nepoch 2 loss 0.4\n"
    assert res.size == 4096
    assert res.exit_code is None  # still running -> no completion
    assert res.alive is True


def test_parse_poll_output_reports_completion_exit_code() -> None:
    raw = f"final line\n{rpt.POLL_SENTINEL} size=999 exit=3 alive=0\n"
    res = rpt.parse_poll_output(raw)
    assert res.content == "final line\n"
    assert res.size == 999
    assert res.exit_code == 3  # completed, non-zero
    assert res.alive is False


def test_build_poll_cmd_emits_new_bytes_and_status_sentinel() -> None:
    cmd = rpt.build_poll_cmd(
        log_file="/workspace/train.log",
        pid_file="/workspace/train.pid",
        exit_file="/workspace/train.exit",
        offset=2048,
    )
    # measures current log size and streams only the bytes past the client's offset
    assert "/workspace/train.log" in cmd
    assert "2049" in cmd or "2048" in cmd  # tail -c +<offset+1> (1-indexed) or arithmetic
    # ends with a parseable sentinel carrying size + exit + liveness
    assert rpt.POLL_SENTINEL in cmd
    assert "/workspace/train.exit" in cmd  # exit code source
    assert "kill -0" in cmd                 # liveness check on the pid
    assert "/workspace/train.pid" in cmd


def test_marker_paths_derive_pid_and_exit_from_log() -> None:
    pid_file, exit_file = rpt.marker_paths("/workspace/train.log")
    assert pid_file == "/workspace/train.pid"
    assert exit_file == "/workspace/train.exit"
    # a log without an extension still gets distinct sibling markers
    pid2, exit2 = rpt.marker_paths("/workspace/runs/exp0006b")
    assert pid2 == "/workspace/runs/exp0006b.pid"
    assert exit2 == "/workspace/runs/exp0006b.exit"


def test_monitor_loop_reconnects_on_drop_and_returns_exit_code() -> None:
    # poll responses in order: running, DROP (raises), running-again, then done(exit=0).
    responses = [
        f"line A\n{rpt.POLL_SENTINEL} size=7 exit=none alive=1\n",
        rpt.SshDropped("connection reset"),
        f"line B\n{rpt.POLL_SENTINEL} size=14 exit=none alive=1\n",
        f"line C\n{rpt.POLL_SENTINEL} size=21 exit=0 alive=0\n",
    ]
    offsets_seen: list[int] = []

    def poll_fn(offset: int) -> str:
        offsets_seen.append(offset)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    reconnects: list[int] = []
    captured: list[str] = []
    rc = rpt.run_monitor_loop(
        poll_fn=poll_fn,
        on_content=captured.append,
        reconnect_fn=lambda: reconnects.append(1),
        sleep_fn=lambda _s: None,
        poll_interval=0,
    )
    assert rc == 0                                  # completion exit code surfaced
    assert reconnects == [1]                        # reconnected exactly once after the drop
    assert "".join(captured) == "line A\nline B\nline C\n"  # no content lost across the drop
    # offset advances with the log size and does NOT regress after a reconnect
    assert offsets_seen == [0, 7, 7, 14]
