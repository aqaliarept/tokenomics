"""run_pod_training.py REST and SSH-discovery fixes (task 11.2): an explicit User-Agent on every
RunPod REST call (Cloudflare blocks Python-urllib's default with 403 / error 1010), and SSH
discovery that falls back to `runpodctl pod get` when `ssh info` misreports a running CPU pod.
No network and no runpodctl: urlopen and `run` are stubbed."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_pod_training", _ROOT / "run_pod_training.py")
assert _spec and _spec.loader
rpt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rpt
_spec.loader.exec_module(rpt)


class _Resp(io.BytesIO):
    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list:
    seen: list = []

    def fake_urlopen(req: object, timeout: float = 0) -> _Resp:
        seen.append(req)
        return _Resp(json.dumps({"id": "pod123", "imageName": "img:1"}).encode())

    monkeypatch.setattr(rpt.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rpt, "_runpod_api_key", lambda: "SECRET-KEY-123")
    monkeypatch.setattr(rpt, "log", lambda msg: seen.append(("log", msg)))
    return seen


def test_every_rest_call_sends_the_explicit_user_agent(captured: list) -> None:
    spec = rpt.PodSpec(image="img:1", compute_type="CPU", vcpu=16, mem_gb=32,
                       container_disk_gb=30, cpu_flavor="cpu5c")
    assert rpt.create_pod(spec, "n") == "pod123"
    assert rpt.resolve_pod_image("pod123") == "img:1"
    try:
        rpt.fetch_pod_spec("pod123")
    except Exception:  # the stub body is not a full pod spec; only the request matters here
        pass
    logs = [m for kind, m in (c for c in captured if isinstance(c, tuple))]
    reqs = [c for c in captured if not isinstance(c, tuple)]
    assert len(reqs) == 3
    for req in reqs:
        assert req.get_header("User-agent") == rpt.RUNPOD_USER_AGENT
        assert "python-urllib" not in rpt.RUNPOD_USER_AGENT.lower()
        assert req.get_header("Authorization") == "Bearer SECRET-KEY-123"
    assert reqs[0].get_header("Content-type") == "application/json"
    assert logs and not any("SECRET-KEY-123" in m for m in logs)  # the key is never logged


def _stub_run(monkeypatch: pytest.MonkeyPatch,
              outputs: dict[str, tuple[int, str, str]]) -> list[list[str]]:
    """`run` stub: outputs maps "ssh info" / "pod get" to (returncode, stdout, stderr)."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool = True,
                 capture: bool = False) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        code, out, err = outputs[" ".join(cmd[1:3])]
        if check and code:
            raise subprocess.CalledProcessError(code, cmd, out, err)
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr=err)

    monkeypatch.setattr(rpt, "run", fake_run)
    return calls


_SSH = {"ip": "1.2.3.4", "port": 2222, "ssh_key": {"path": "/k"}}
_NOT_READY = {"error": "pod not ready: no container reported yet (image pull, container create "
                       "or boot)", "id": "p", "runtimeStatus": "initializing"}
_GET = (0, json.dumps({"id": "p", "ssh": _SSH}), "")


def test_resolve_ssh_uses_ssh_info_when_it_reports_an_endpoint(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run(monkeypatch, {"ssh info": (0, json.dumps(_SSH), ""), "pod get": _GET})
    ssh = rpt.resolve_ssh("p", None)
    assert (ssh.ip, ssh.port, ssh.key) == ("1.2.3.4", 2222, "/k")
    assert [c[1:3] for c in calls] == [["ssh", "info"]]


@pytest.mark.parametrize("code", [0, 1])
def test_resolve_ssh_falls_back_to_pod_get_only_when_not_ready(
        monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    """runpodctl 3.0.1 reports "pod not ready: no container reported yet" for a running CPU
    pod; `pod get` already carries the same ssh block (whatever ssh info's exit code)."""
    calls = _stub_run(monkeypatch, {"ssh info": (code, json.dumps(_NOT_READY), ""),
                                    "pod get": _GET})
    ssh = rpt.resolve_ssh("p", "/override")
    assert (ssh.ip, ssh.port, ssh.key) == ("1.2.3.4", 2222, "/override")
    assert [c[1:3] for c in calls] == [["ssh", "info"], ["pod", "get"]]


def test_not_ready_without_any_endpoint_is_still_not_ready(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """No endpoint in `pod get` either raises KeyError: wait_for_ssh treats it as not ready."""
    _stub_run(monkeypatch, {"ssh info": (0, json.dumps(_NOT_READY), ""),
                            "pod get": (0, json.dumps({"id": "p"}), "")})
    with pytest.raises(KeyError):
        rpt.resolve_ssh("p", None)


@pytest.mark.parametrize("ssh_info,raises", [
    ((1, "", "Error: unauthorized: invalid api key"), subprocess.CalledProcessError),
    ((2, "unknown flag: --bogus", ""), subprocess.CalledProcessError),
    ((1, json.dumps({"error": "pod not found"}), ""), subprocess.CalledProcessError),
    ((0, "garbled <html>", ""), json.JSONDecodeError),
    ((0, json.dumps({"error": "rate limited"}), ""), KeyError),
])
def test_other_ssh_info_failures_are_not_masked(monkeypatch: pytest.MonkeyPatch,
                                                ssh_info: tuple, raises: type) -> None:
    """Review concern 122: authorization errors, bad arguments, unknown pods, malformed output
    and unrecognised JSON errors fail as they did before the fallback, never via `pod get`."""
    calls = _stub_run(monkeypatch, {"ssh info": ssh_info, "pod get": _GET})
    with pytest.raises(raises) as err:
        rpt.resolve_ssh("p", None)
    assert [c[1:3] for c in calls] == [["ssh", "info"]]
    if raises is subprocess.CalledProcessError:
        assert (err.value.returncode, err.value.stderr) == (ssh_info[0], ssh_info[2])
