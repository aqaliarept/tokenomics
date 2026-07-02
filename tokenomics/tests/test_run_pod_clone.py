"""Tests for the clone-a-pod lifecycle in run_pod_training.py.

`--clone-from <template>` enrols a FRESH pod copying an existing pod's characteristics
(image, GPU/CPU shape, disks, volume), runs the experiment, then stops AND terminates the
created pod — for when an existing pod's host is stuck and `pod start` keeps failing. These
tests cover the pure helpers (spec extraction, create-command build, created-id parse) so
the billable lifecycle is verified without touching runpodctl.
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


_GPU_POD_JSON = {
    "id": "tmpl123",
    "name": "old-gpu-pod",
    "desiredStatus": "EXITED",
    "imageName": "runpod/pytorch:2.4.0-cu124",
    "gpuCount": 2,
    "vcpuCount": 16,
    "memoryInGb": 62,
    "containerDiskInGb": 40,
    "volumeInGb": 50,
    "volumeMountPath": "/workspace",
    "machine": {"gpuTypeId": "NVIDIA GeForce RTX 3090", "secureCloud": True,
                "dataCenterId": "US-OR-1"},
}

_CPU_POD_JSON = {
    "id": "cpu9",
    "name": "old-cpu-pod",
    "imageName": "runpod/base:0.6.1",
    "gpuCount": 0,
    "vcpuCount": 25,
    "memoryInGb": 50,
    "containerDiskInGb": 30,
    "volumeInGb": 20,
    "volumeMountPath": "/workspace",
    "networkVolumeId": "netvolABC",
    "cpuFlavorId": "cpu5c",
    "machine": {"secureCloud": False, "dataCenterId": "EU-RO-1"},
}


def test_pod_spec_from_json_extracts_gpu_characteristics() -> None:
    s = rpt.pod_spec_from_json(_GPU_POD_JSON)
    assert s.image == "runpod/pytorch:2.4.0-cu124"
    assert s.compute_type == "GPU"
    assert s.gpu_type == "NVIDIA GeForce RTX 3090"
    assert s.gpu_count == 2
    assert s.container_disk_gb == 40
    assert s.volume_gb == 50
    assert s.volume_path == "/workspace"
    assert s.secure_cloud is True
    assert s.data_center_id == "US-OR-1"
    assert s.network_volume_id is None


def test_pod_spec_from_json_extracts_cpu_and_network_volume() -> None:
    s = rpt.pod_spec_from_json(_CPU_POD_JSON)
    assert s.compute_type == "CPU"       # gpuCount 0 -> CPU pod
    assert s.gpu_type is None
    assert s.vcpu == 25 and s.mem_gb == 50
    assert s.cpu_flavor == "cpu5c"
    assert s.network_volume_id == "netvolABC"
    assert s.secure_cloud is False
    assert s.data_center_id == "EU-RO-1"


def test_build_create_body_for_gpu_pod_carries_gpu_shape() -> None:
    body = rpt.build_create_body(rpt.pod_spec_from_json(_GPU_POD_JSON), name="clone-x")
    assert body["name"] == "clone-x"
    assert body["imageName"] == "runpod/pytorch:2.4.0-cu124"
    assert body["computeType"] == "GPU"
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 3090"]
    assert body["gpuCount"] == 2
    assert body["containerDiskInGb"] == 40
    assert body["volumeInGb"] == 50
    assert body["volumeMountPath"] == "/workspace"
    assert "22/tcp" in body["ports"]      # driver needs SSH to reach the pod
    assert body["cloudType"] == "SECURE"


def test_build_create_body_for_cpu_pod_carries_vcpu_flavor_and_network_volume() -> None:
    body = rpt.build_create_body(rpt.pod_spec_from_json(_CPU_POD_JSON), name="clone-cpu")
    assert body["computeType"] == "CPU"
    # REST honours vCPU + flavor (the CLI could not — it gave 2 vCPU).
    assert body["vcpuCount"] == 25
    assert body["cpuFlavorIds"] == ["cpu5c"]
    # CPU pod: no gpu keys.
    assert "gpuTypeIds" not in body and "gpuCount" not in body
    # a network-volume pod attaches the volume + its datacenter, and does NOT also request a
    # per-pod volumeInGb (the network volume provides the storage).
    assert body["networkVolumeId"] == "netvolABC"
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert "volumeInGb" not in body
    assert body["cloudType"] == "COMMUNITY"  # secureCloud False -> community


def test_build_create_body_normalizes_a_cloned_port_list() -> None:
    # a real RunPod REST payload has ports as a JSON list; cloning a template that exposes
    # ports must not blow up (regression: re.split on a list raised TypeError).
    raw = dict(_CPU_POD_JSON, ports=["8888/http", "22/tcp"])
    body = rpt.build_create_body(rpt.pod_spec_from_json(raw), name="clone-ports")
    assert body["ports"][0] == "22/tcp"          # SSH port always first
    assert "8888/http" in body["ports"]
    assert body["ports"].count("22/tcp") == 1     # not duplicated


def test_build_create_body_accepts_a_port_string_too() -> None:
    raw = dict(_CPU_POD_JSON, ports="8888/http,22/tcp")
    body = rpt.build_create_body(rpt.pod_spec_from_json(raw), name="c")
    assert set(body["ports"]) == {"22/tcp", "8888/http"}


def test_do_pulls_returns_failed_specs() -> None:
    # bad spec (no REMOTE:LOCAL colon) is reported as failed, not silently skipped, so the
    # driver keeps a created pod instead of terminating it and losing the artifacts.
    failed = rpt._do_pulls(ssh=None, pull_specs=["no-colon-here"])
    assert failed == ["no-colon-here"]


def test_do_pulls_reports_rsync_failure(monkeypatch) -> None:
    def boom(ssh, remote, local):
        raise RuntimeError("rsync exit 23")
    monkeypatch.setattr(rpt, "rsync_down", boom)
    failed = rpt._do_pulls(ssh="fake", pull_specs=["/remote/out:./local/out"])
    assert failed == ["/remote/out:./local/out"]


def test_parse_created_pod_id_from_json_and_text() -> None:
    assert rpt.parse_created_pod_id('{"id": "newpod42", "desiredStatus": "RUNNING"}') == "newpod42"
    assert rpt.parse_created_pod_id('pod "newpod42" created for $0.44/hr') == "newpod42"
    assert rpt.parse_created_pod_id('pod newpod42 created') == "newpod42"


def test_parse_created_pod_id_raises_on_garbage() -> None:
    import pytest
    with pytest.raises(RuntimeError):
        rpt.parse_created_pod_id("something went wrong, no id here!!!")


import argparse


def _args(**kw):
    base = dict(image="runpod/pytorch:py3.11", compute_type="CPU", vcpu=16, mem=32,
                disk=30, cpu_flavor="cpu5c", volume_size=0, volume_path="/workspace",
                gpu_type=None, gpu_count=1, data_center=None, network_volume_id=None,
                community=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_pod_spec_from_args_defaults_to_the_cpu_shape() -> None:
    s = rpt.pod_spec_from_args(_args())
    assert s.compute_type == "CPU"
    assert s.vcpu == 16 and s.mem_gb == 32 and s.container_disk_gb == 30
    assert s.cpu_flavor == "cpu5c"
    assert s.image == "runpod/pytorch:py3.11"
    assert s.gpu_type is None
    assert s.secure_cloud is True   # --community not set -> secure cloud


def test_create_body_from_explicit_cpu_flags_sets_vcpu_and_omits_zero_volume() -> None:
    body = rpt.build_create_body(rpt.pod_spec_from_args(_args()), name="mine-run")
    assert body["computeType"] == "CPU"
    assert body["vcpuCount"] == 16 and body["cpuFlavorIds"] == ["cpu5c"]
    assert body["containerDiskInGb"] == 30
    assert "22/tcp" in body["ports"]
    # volume_size 0 -> no per-pod volume requested (/workspace on container disk).
    assert "volumeInGb" not in body


def test_create_body_requests_volume_when_sized() -> None:
    body = rpt.build_create_body(rpt.pod_spec_from_args(_args(volume_size=40)), name="r")
    assert body["volumeInGb"] == 40
