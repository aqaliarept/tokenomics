"""Run provenance helpers used by the EXP-0042 teacher (eval/provenance.py).

Ported from the closed EXP-0039 graph-dump tests when the helpers moved here: the input
pin must refuse a wrong producer/archive before any rollout compute, and the source and
commit bindings must identify exactly the code that produced a run.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tokenomics.eval import provenance as pv


def test_sha256_file_hashes_full_content(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"provenance")
    assert pv.sha256_file(str(f)) == hashlib.sha256(b"provenance").hexdigest()


def test_input_manifest_validation_fails_a_wrong_input_before_running() -> None:
    """A wrong staged input must fail at launch, not after hours of teacher rollout:
    `assert_inputs_match` raises on any sha mismatch or missing role."""
    expected = {"archive": "aaa", "producer_ckpt": "bbb"}
    good = {"archive": {"path": "a", "sha256": "aaa"},
            "producer_ckpt": {"path": "p", "sha256": "bbb"}}
    pv.assert_inputs_match(good, expected)  # exact match: no raise
    with pytest.raises(ValueError):  # a swapped producer is rejected
        pv.assert_inputs_match({**good, "producer_ckpt": {"path": "p", "sha256": "WRONG"}}, expected)
    with pytest.raises(ValueError):  # a None role is rejected, not silently skipped
        pv.assert_inputs_match({**good, "producer_ckpt": None}, expected)
    with pytest.raises(ValueError):  # an absent role is rejected too
        pv.assert_inputs_match({"archive": good["archive"]}, expected)


def test_source_fingerprint_is_deterministic_and_records_commit() -> None:
    """Hashes every `.py` in the package tree, git-independent (reproducible on the pod)."""
    fp = pv.source_fingerprint()
    assert len(fp["tree_sha256"]) == 64 and fp["commit"] is None and fp["n_files"] > 4
    assert pv.source_fingerprint()["tree_sha256"] == fp["tree_sha256"]
    assert pv.source_fingerprint("cafe")["commit"] == "cafe"


def test_resolve_verified_commit_binds_a_clean_tree_to_a_real_commit(tmp_path: Path) -> None:
    """Resolves a full 40-char HEAD only when the object exists and the producing tree is
    clean; a non-repo or an untracked file raises."""

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                       capture_output=True, text=True)

    with pytest.raises(ValueError):  # not a git repo
        pv.resolve_verified_commit(str(tmp_path))

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    sha = pv.resolve_verified_commit(str(tmp_path))  # clean tree -> full sha
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    (tmp_path / "b.py").write_text("y = 2\n")  # untracked file -> dirty producing tree
    with pytest.raises(ValueError):
        pv.resolve_verified_commit(str(tmp_path))


def test_decision_digest_is_stable_signed_int64_and_identity_sensitive() -> None:
    d = pv.decision_digest("taskX", 2, 7, "order-9")
    assert d == pv.decision_digest("taskX", "2", "7", "order-9")  # ints and CSV strings agree
    assert -(2 ** 63) <= d < 2 ** 63 and d != 0
    assert d != pv.decision_digest("taskX", 2, 8, "order-9")
    assert d != pv.decision_digest("taskY", 2, 7, "order-9")


def test_environment_manifest_records_runtime_identity() -> None:
    env = pv.environment_manifest()
    assert env["python"] and env["numpy"] and env["machine"]
    assert {"platform", "implementation", "torch", "pettingzoo", "gymnasium", "image",
            "numpy_blas", "torch_build"} <= set(env)
