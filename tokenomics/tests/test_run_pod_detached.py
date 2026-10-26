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
        train_args="-m tokenomics.eval.signal_check --horizon 50",
        log_file="/workspace/train.log",
        pid_file="/workspace/train.pid",
        exit_file="/workspace/train.exit",
    )
    # detaches into its own session so closing SSH does not SIGHUP the training
    assert "setsid" in cmd
    # runs in the right dir, with the right interpreter + args
    assert "cd /workspace/tokenomics" in cmd
    assert ".venv/bin/python -m tokenomics.eval.signal_check --horizon 50" in cmd
    # a stale exit marker from a previous run must not be mistaken for "done"
    assert "rm -f /workspace/train.exit" in cmd
    # records the wrapper pid (liveness) and the final exit code (completion)
    assert "/workspace/train.pid" in cmd
    assert "$? > /workspace/train.exit" in cmd or "$? >/workspace/train.exit" in cmd
    # training output is redirected to the log file (not the dropped SSH channel)
    assert "/workspace/train.log" in cmd


def test_build_launch_cmd_forces_outer_shell_to_exit() -> None:
    # EXP-0027 hang fix: the outer sshd-spawned shell must exit right after backgrounding the
    # setsid job (else it sits in do_wait, keeping the SSH channel open so ssh_run never returns).
    cmd = rpt.build_launch_cmd(
        remote_dir="/workspace/tokenomics",
        python_bin_path=".venv/bin/python",
        train_args="-m tokenomics.eval.signal_check --horizon 50",
        log_file="/workspace/train.log",
        pid_file="/workspace/train.pid",
        exit_file="/workspace/train.exit",
    )
    assert cmd.rstrip().endswith("exit 0")
    # the exit follows the backgrounded job (order matters: background THEN exit)
    assert cmd.index("setsid") < cmd.index("exit 0")
    assert "&" in cmd.split("exit 0")[0]


def test_base_cmd_no_stdin_passes_dash_n() -> None:
    # the detached launch uses `ssh -n` so the channel isn't held open on stdin.
    ssh = rpt.Ssh(ip="1.2.3.4", port=22, key="/tmp/k")
    assert "-n" in ssh.base_cmd(no_stdin=True)
    # default (poll/capture) must NOT pass -n — those read remote stdout normally.
    assert "-n" not in ssh.base_cmd()


def test_ssh_run_times_out_and_raises(monkeypatch) -> None:
    # a hung launch must surface as TimeoutExpired (the caller then falls through to monitor),
    # not block forever. Bound via the `timeout` arg threaded into subprocess.run.
    import subprocess

    seen: dict[str, object] = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        seen["cmd"] = cmd
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(rpt.subprocess, "run", fake_run)
    ssh = rpt.Ssh(ip="1.2.3.4", port=22, key="/tmp/k")
    try:
        rpt.ssh_run(ssh, "sleep 999", timeout=60, no_stdin=True)
        assert False, "expected TimeoutExpired"
    except subprocess.TimeoutExpired:
        pass
    assert seen["timeout"] == 60
    assert "-n" in seen["cmd"]  # no_stdin threaded through to base_cmd


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


def test_default_local_log_derives_live_log_from_remote_basename() -> None:
    # every detached run gets a local mirror named after the remote log's basename, with
    # the `.live.log` suffix the repo .gitignore already excludes
    assert rpt.default_local_log("/workspace/champion_iterate_r1.log") == "champion_iterate_r1.live.log"
    assert rpt.default_local_log("/workspace/runs/exp0006b") == "exp0006b.live.log"


def test_content_sink_mirrors_to_local_file_and_truncates_stale(tmp_path) -> None:
    # local observability: streamed chunks land on disk so a detached run is watchable via
    # `tail -f` without re-SSHing; a fresh run starts clean, not appended to a stale mirror
    p = tmp_path / "run.live.log"
    p.write_text("STALE FROM A PRIOR RUN\n")
    sink = rpt.make_content_sink(str(p))
    sink("chunk one\n")
    sink("chunk two\n")
    assert p.read_text() == "chunk one\nchunk two\n"


def test_init_local_mirror_truncates_and_writes_header(tmp_path) -> None:
    # the mirror is truncated + given a run header at LAUNCH START (before the ~5 min
    # install), so during setup the local file reflects THIS run, never a prior run's log.
    p = tmp_path / "run.live.log"
    p.write_text("STALE FROM A PRIOR RUN\n")
    ok = rpt.init_local_mirror(str(p), "# run xyz | started now\n")
    assert ok is True
    assert p.read_text() == "# run xyz | started now\n"
    assert "STALE" not in p.read_text()


def test_init_local_mirror_never_raises_on_bad_path(tmp_path) -> None:
    (tmp_path / "afile").write_text("x")
    assert rpt.init_local_mirror(str(tmp_path / "afile" / "run.live.log"), "hdr") is False


def test_content_sink_appends_when_truncate_false(tmp_path) -> None:
    # after init_local_mirror wrote the header, the stream must APPEND (not wipe it).
    p = tmp_path / "run.live.log"
    p.write_text("# header\n")
    sink = rpt.make_content_sink(str(p), truncate=False)
    sink("chunk one\n")
    assert p.read_text() == "# header\nchunk one\n"


def test_parse_run_status_extracts_latest_phase_and_last_line() -> None:
    text = ("[phase 1/7] split\nsome work\n[phase 2/7] producer distill\n"
            "  epoch 400/400 mse 0.002\n")
    st = rpt.parse_run_status(text)
    assert st["phase"] == "[phase 2/7] producer distill"
    assert st["last_line"] == "  epoch 400/400 mse 0.002"
    assert st["n_lines"] == 4


def test_parse_run_status_no_phase_marker() -> None:
    st = rpt.parse_run_status("just logs\nmore logs\n")
    assert st["phase"] == ""
    assert st["last_line"] == "more logs"


def test_write_run_status_roundtrips_json(tmp_path) -> None:
    import json
    p = tmp_path / "run.status"
    rpt.write_run_status(str(p), pod_id="pod1", remote_log="/workspace/train.log",
                         state="running", started="T0", text="[phase 3/7] orderer\nwork\n")
    st = json.loads(p.read_text())
    assert st["pod_id"] == "pod1" and st["state"] == "running"
    assert st["phase"] == "[phase 3/7] orderer" and st["last_line"] == "work"


def test_write_run_status_never_raises_on_bad_path(tmp_path) -> None:
    (tmp_path / "afile").write_text("x")
    # must degrade like the mirror — a heartbeat IO failure can't sink a live run.
    rpt.write_run_status(str(tmp_path / "afile" / "run.status"), pod_id="p",
                         remote_log="r", state="running", started="T0", text="x\n")


def test_content_sink_writes_heartbeat_status(tmp_path) -> None:
    # with a status target + interval 0 (every chunk), the sink drops a compact .status
    # next to the mirror so remote progress is on disk even without actively tailing.
    import json
    log_p = tmp_path / "run.live.log"
    status_p = tmp_path / "run.status"
    sink = rpt.make_content_sink(
        str(log_p), truncate=False,
        status={"path": str(status_p), "pod_id": "pod9", "remote_log": "/workspace/train.log",
                "started": "T0"},
        heartbeat_interval=0.0)
    sink("[phase 4/7] mined validation replay\n")
    sink("  task s99 gap=2.6\n")
    st = json.loads(status_p.read_text())
    assert st["phase"] == "[phase 4/7] mined validation replay"
    assert st["last_line"] == "  task s99 gap=2.6"
    assert st["state"] == "running" and st["pod_id"] == "pod9"


def test_content_sink_keeps_phase_after_marker_scrolls_past_tail(tmp_path) -> None:
    # a long phase can emit >8 KB of same-phase output before the next [phase] marker, pushing
    # the marker out of the bounded tail. The sink must still report that phase (sticky), not "".
    import json
    status_p = tmp_path / "run.status"
    sink = rpt.make_content_sink(
        None, truncate=False,
        status={"path": str(status_p), "pod_id": "p", "remote_log": "/workspace/train.log",
                "started": "T0"},
        heartbeat_interval=0.0)
    sink("[phase 2/7] producer distill\n")
    for i in range(400):  # ~ >8 KB, no new marker -> marker scrolls out of the 8192-char buf
        sink(f"  epoch {i}/400 masked-mse 0.00{i % 10}  padding padding padding padding\n")
    st = json.loads(status_p.read_text())
    assert st["phase"] == "[phase 2/7] producer distill"  # sticky, not lost to the tail
    assert st["last_line"].startswith("  epoch 399/400")


def test_write_run_status_phase_override_wins_over_parsed(tmp_path) -> None:
    import json
    p = tmp_path / "run.status"
    # text tail has no marker, but the caller pins the sticky phase explicitly.
    rpt.write_run_status(str(p), pod_id="p", remote_log="r", state="running", started="T0",
                         text="  epoch 399/400\n", phase="[phase 2/7] producer distill")
    st = json.loads(p.read_text())
    assert st["phase"] == "[phase 2/7] producer distill"


def test_default_status_path_derives_from_remote_basename() -> None:
    assert rpt.default_status_path("/workspace/train.log") == "train.status"
    assert rpt.default_status_path("/workspace/runs/exp0026") == "exp0026.status"


def test_content_sink_without_local_log_is_stdout_only(tmp_path, capsys) -> None:
    # local_log=None disables mirroring (stdout still streams) and must not raise
    sink = rpt.make_content_sink(None)
    sink("hello\n")
    assert capsys.readouterr().out == "hello\n"


def test_content_sink_degrades_to_stdout_on_io_error(tmp_path, capsys) -> None:
    # the mirror is OPTIONAL observability created AFTER the detached job has launched — an
    # unwritable path (read-only cwd, bad parent, full disk) must degrade to a warning, never
    # raise, or main()'s except-handler would stop the pod and kill a live remote run.
    bad = tmp_path / "a-file" / "run.live.log"  # parent is a regular file -> cannot create
    (tmp_path / "a-file").write_text("x")
    sink = rpt.make_content_sink(str(bad))  # setup failure must not raise
    sink("training continues\n")            # streaming must not raise
    assert "training continues\n" in capsys.readouterr().out


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


def test_resolve_clean_commit_covers_synced_tree_and_launcher(tmp_path) -> None:
    """Finding-9.1/10.1: the launcher (the only place with .git — the pod excludes it) resolves a full
    40-char HEAD only when EVERY producing path is clean: the synced subtree AND the launcher file
    itself (outside that subtree — it resolves the commit, rewrites the command, and launches). A dirty
    subtree, a dirty launcher, a missing HEAD, or a non-repo aborts (SystemExit)."""
    import subprocess

    import pytest

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True)

    pkg = tmp_path / "tokenomics"
    pkg.mkdir()
    launcher = tmp_path / "run_pod_training.py"

    with pytest.raises(SystemExit):  # not a git repo
        rpt.resolve_clean_commit(pkg, launcher)

    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (pkg / "a.py").write_text("x = 1\n")
    launcher.write_text("# launcher v1\n")
    git("add", "-A"); git("commit", "-qm", "init")
    sha = rpt.resolve_clean_commit(pkg, launcher)  # both clean -> full sha
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    # Finding-10.1: the SUBTREE is clean but the LAUNCHER was modified after HEAD -> must still abort
    launcher.write_text("# launcher v2 (tampered post-commit)\n")
    with pytest.raises(SystemExit):
        rpt.resolve_clean_commit(pkg, launcher)

    git("checkout", "--", "run_pod_training.py")  # restore launcher; now dirty the subtree instead
    (pkg / "dirty.py").write_text("y = 2\n")
    with pytest.raises(SystemExit):
        rpt.resolve_clean_commit(pkg, launcher)


def test_apply_source_commit_requires_and_fills_the_placeholder() -> None:
    """Finding-10.1: the verified commit is injected ONLY via an explicit {SOURCE_COMMIT} placeholder;
    a --verify-source-commit launch whose --train lacks it is a mistake and aborts."""
    import pytest

    out = rpt.apply_source_commit("-m m --source-commit {SOURCE_COMMIT} --x 1", "a" * 40)
    assert "--source-commit " + "a" * 40 in out and "{SOURCE_COMMIT}" not in out
    with pytest.raises(SystemExit):
        rpt.apply_source_commit("-m m --no-placeholder", "a" * 40)


def test_launch_cmd_exports_pod_image_into_the_run() -> None:
    """Finding-10.2: the resolved pod image is exported into the remote command as EXP0039_POD_IMAGE
    (a `VAR=val ` prefix on the python run) so the run's env manifest records the image it ran under;
    no exports -> unchanged command."""
    cmd = rpt.build_launch_cmd(
        remote_dir="/workspace/tokenomics", python_bin_path=".venv/bin/python",
        train_args="-m tokenomics.eval.bounded_rollout_orderer --x 1",
        log_file="/workspace/train.log", pid_file="/workspace/train.pid",
        exit_file="/workspace/train.exit",
        env_exports={"EXP0039_POD_IMAGE": "runpod/base:0.6.2"})
    # the prefix sets env for the immediately-following python run (unescaped form)
    assert rpt._env_prefix({"EXP0039_POD_IMAGE": "runpod/base:0.6.2"}) == \
        "EXP0039_POD_IMAGE='runpod/base:0.6.2' "
    assert rpt._env_prefix({}) == "" and rpt._env_prefix({"K": ""}) == ""  # empty -> not exported
    # and it rides in the launch command ahead of python (single quotes are shell-escaped by _shquote)
    assert "EXP0039_POD_IMAGE=" in cmd and "runpod/base:0.6.2" in cmd
    bare = rpt.build_launch_cmd(
        remote_dir="/workspace/tokenomics", python_bin_path=".venv/bin/python",
        train_args="-m x", log_file="/l", pid_file="/p", exit_file="/e")
    assert "EXP0039_POD_IMAGE" not in bare
