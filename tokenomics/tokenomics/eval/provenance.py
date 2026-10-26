"""Run provenance: input pins, source-tree identity, and runtime environment.

Binds an experiment's outputs to exactly what produced them — SHA-256 of every staged
input, a git-independent hash of the package source tree (reproducible on a pod, where
`.git/` is not synced), the verified launcher-side commit, and the numeric environment.
Used by the EXP-0042 teacher; extracted from the closed EXP-0039 graph dump, whose
encoder and dump are preserved on the `archive-1` branch.
"""

from __future__ import annotations

import hashlib

def sha256_file(path: str) -> str:
    """Full-content SHA-256 of a file (streamed). Fingerprints the dump's INPUTS — archive,
    producer checkpoint, actor checkpoint — so a read can prove the dump was built from the
    pre-registered files, not arbitrary ones of compatible shape (Finding-5.1). A wrong producer
    silently changes teacher labels; a wrong actor silently weakens the licensing floor."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint(commit: "str | None" = None) -> dict:
    """Immutable identity of the code producing a run: SHA-256 over the bytes of EVERY `.py` file in
    the synced `tokenomics` package tree (sorted by relative path), plus an optional git commit
    recorded verbatim (Finding-6.1/7.1). Hashing the whole tree — not a hand-picked module list —
    covers all transitive producers of the trajectory and labels (`marginal_value_labels`,
    `joint_stack_env`, curriculum, actor scoring, domain deps); a four-module subset silently
    ignored changes to them. The hash is git-independent so it is reproducible ON the pod (where
    `.git/` is not synced); the commit is injected by the launcher (which does have git). Stamped
    into both specs AND cross-checked by `_load_dump` before training, so a dump built from source A
    can never earn a verdict under source B."""
    import hashlib as _hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]  # inner tokenomics package dir
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    h = _hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return {"tree_sha256": h.hexdigest(), "n_files": len(files), "commit": commit}


def resolve_verified_commit(tree_dir: "str | None" = None) -> str:
    """Launcher-side binding of the live source tree to a real, clean git revision (Finding-8.1).
    Needs git (NOT available on the pod — resolve LOCALLY and pass the result via `--source-commit`).
    Returns the full 40-char HEAD commit of the repo containing `tree_dir` after verifying: (1) HEAD
    resolves and the object EXISTS (`git cat-file -e`), and (2) the producing tree is CLEAN — no
    modified or untracked files under `tree_dir` (`git status --porcelain`). A clean tree guarantees
    the live `source_fingerprint` corresponds to that commit, so the artifact is recoverable from the
    claimed revision; a forged/short/nonexistent id or a dirty tree raises here rather than producing
    an unrecoverable run. `_validate_commit` (pod-side) only checks shape; this is the real binding."""
    import subprocess
    from pathlib import Path

    d = str(Path(tree_dir).resolve() if tree_dir else Path(__file__).resolve().parents[1])

    def git(*a: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)

    head = git("rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise ValueError(f"cannot resolve HEAD in {d}: {head.stderr.strip() or 'not a git repo'}")
    sha = head.stdout.strip()
    if len(sha) != 40 or git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        raise ValueError(f"HEAD {sha!r} is not a full existing commit object")
    dirty = git("status", "--porcelain", "--", d).stdout.strip()
    if dirty:
        raise ValueError(f"producing tree {d} is dirty — commit before launch so the artifact is "
                         f"recoverable from {sha}:\n{dirty}")
    return sha


def _blas_identity() -> dict:
    """Best-effort numeric-backend identity (Finding-9.2). Identical package versions on a different
    CPU arch or BLAS/build are NOT the same numerical environment, so also capture the numpy BLAS name
    and a hash of torch's build config (MKL/OpenBLAS, ISA, threading). Best-effort: None if unavailable."""
    import hashlib

    numpy_blas = torch_build = None
    try:
        import numpy as _np
        cfg = _np.show_config(mode="dicts")  # numpy>=1.26
        numpy_blas = ((cfg.get("Build Dependencies", {}) or {}).get("blas", {}) or {}).get("name")
    except Exception:
        pass
    try:
        import torch as _t
        torch_build = hashlib.sha1(_t.__config__.show().encode()).hexdigest()[:12]
    except Exception:
        pass
    return {"numpy_blas": numpy_blas, "torch_build": torch_build}


def environment_manifest() -> dict:
    """Runtime environment identity for a run (Finding-8.2/9.2; HANDOVER.md:413 requires it in RESULT).
    Deps are open lower bounds (`torch>=2.2`, …) and the pod installs current packages from the
    network, so labels/metrics/determinism depend on versions AND the machine arch + numeric backend,
    which must be captured and compared across the dump and probe stages. `image` records the pod image
    tag/digest when the launcher exports it (audit). Records python/platform/machine + the versions and
    BLAS/build config that actually shape the trajectory and the fit."""
    import os
    import platform
    import sys

    def _ver(mod: str) -> "str | None":
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "implementation": sys.implementation.name,
        "numpy": _ver("numpy"), "torch": _ver("torch"),
        "pettingzoo": _ver("pettingzoo"), "gymnasium": _ver("gymnasium"),
        "image": os.environ.get("EXP0039_POD_IMAGE"),
        **_blas_identity(),
    }


def assert_inputs_match(manifest: dict, expected: dict) -> None:
    """Abort a dump BEFORE it spends teacher-rollout compute if a staged input's SHA-256 does not
    match the pre-registered pin (Finding-6.3). `manifest` is {role: {path, sha256} | None} (as the
    dump records), `expected` is {role: sha256}. A missing role or a hash mismatch raises — so a
    wrong staged checkpoint fails at launch, not after the full ~4 core-hour dump."""
    for role, want in expected.items():
        got = manifest.get(role)
        got_sha = got.get("sha256") if isinstance(got, dict) else None
        if got_sha != want:
            raise ValueError(
                f"input '{role}' sha256 {got_sha} != pinned {want}; refusing to dump from an "
                "unpinned/mismatched input (a wrong producer changes labels, a wrong actor weakens "
                "the floor)")


def decision_digest(task_id, seed, cycle, order_id) -> int:
    """Stable signed-int64 digest of a decision's IDENTITY (task_id, seed, cycle, order_id).
    Stamped per-graph in the shard as an AUDIT array (never a model feature) so a graph payload
    is provably bound to its `decisions.csv` row: `_load_dump` recomputes this from the CSV row
    and rejects any graph whose stored digest disagrees. Components are stringified so the write
    side (ints) and the read side (CSV strings) agree. 0 is reserved as an 'unbound' sentinel."""
    blob = f"{task_id}|{seed}|{cycle}|{order_id}".encode()
    return int.from_bytes(hashlib.sha1(blob).digest()[:8], "big", signed=True)
