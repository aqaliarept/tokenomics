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
    monkeypatch.setattr(rpt, "_runpod_api_key", lambda: "KEY")
    monkeypatch.setattr(rpt, "log", lambda msg: None)
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
    assert len(captured) == 3
    for req in captured:
        assert req.get_header("User-agent") == rpt.RUNPOD_USER_AGENT
        assert "python-urllib" not in rpt.RUNPOD_USER_AGENT.lower()
        assert req.get_header("Authorization") == "Bearer KEY"
    assert captured[0].get_header("Content-type") == "application/json"


def _stub_run(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, str]) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool = True,
                 capture: bool = False) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[" ".join(cmd[1:3])], stderr="")

    monkeypatch.setattr(rpt, "run", fake_run)
    return calls


_SSH = {"ip": "1.2.3.4", "port": 2222, "ssh_key": {"path": "/k"}}
_NOT_READY = {"error": "pod not ready: no container reported yet (image pull, container create "
                       "or boot)", "id": "p", "runtimeStatus": "initializing"}


def test_resolve_ssh_uses_ssh_info_when_it_reports_an_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_run(monkeypatch, {"ssh info": json.dumps(_SSH)})
    ssh = rpt.resolve_ssh("p", None)
    assert (ssh.ip, ssh.port, ssh.key) == ("1.2.3.4", 2222, "/k")
    assert [c[1:3] for c in calls] == [["ssh", "info"]]


def test_resolve_ssh_falls_back_to_pod_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """runpodctl 3.0.1 reports "no container reported yet" for a running CPU pod; `pod get`
    already carries the same ssh block."""
    calls = _stub_run(monkeypatch, {"ssh info": json.dumps(_NOT_READY),
                                    "pod get": json.dumps({"id": "p", "ssh": _SSH})})
    ssh = rpt.resolve_ssh("p", "/override")
    assert (ssh.ip, ssh.port, ssh.key) == ("1.2.3.4", 2222, "/override")
    assert [c[1:3] for c in calls] == [["ssh", "info"], ["pod", "get"]]


def test_resolve_ssh_without_any_endpoint_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """No endpoint anywhere raises KeyError, which wait_for_ssh treats as "not ready yet"."""
    _stub_run(monkeypatch, {"ssh info": json.dumps(_NOT_READY), "pod get": "not json"})
    with pytest.raises(KeyError):
        rpt.resolve_ssh("p", None)
