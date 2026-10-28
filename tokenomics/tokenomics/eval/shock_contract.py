"""M2 exit driver: `python -m tokenomics.eval.shock_contract` (add-shocks-i-drift design D11;
project-mission "Champion runs on a shocked ruleset unretrained").

The exit question is the contract, not the margin: do the current champion checkpoints load and
complete shocked episodes, unretrained, with zero contract failures? Two stacks run shocked,
each episode with its shock-free twin (D6), horizon 100:
- `portable`: the gate subject (M1 D1: EXP-0035 `producer_candidate.pt` + EXP-0012
  `orderer_scorer.pt`, `joint_stack_env`) on every registry v2 development shocked gate world,
  every registered episode seed (2 sizes x 4 worlds x 10 = 80 episodes + 80 twins).
- `flat`: the `champion.json` stack, loaded as the manifest declares (greedy v4 producer,
  orderer sampled from its learned log_std, product streams, orderer obs version 3), on its
  held-out topologies `build_scale_config(seed, depth)` with the profile's generators applied by
  `apply_shock_profile` (drift only when consumers are randomized); 10 hash-derived episode
  seeds per topology (4 x 10 = 40 episodes + 40 twins). The flat stack cannot run in the gate
  (fixed widths, M1 D1), hence a driver of its own.

The portable rows reuse the gate's row format and job runner (`gate_cli._execute`: digest
pairing as rows arrive, append-only `episodes.csv`, resume on matching pins) and the contract
checks of `eval/contract.py`. Output is the EXP layout: `spec.json` (pins and provenance),
`episodes.csv`, `resilience.csv`, `summary.csv` (per stack), `contract_failures.csv`, a
`RESULT.md` skeleton. The exit holds when `contract_failures.csv` has no rows and every planned
row is present; deaths, units and resilience are reported and do not gate. A recorded run pins
the "shocks-i" profile hash; the test-only `shocks-i-smoke` profile (D7) and every filter run
only with `--dry-run`, which marks the run as not an acceptance read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence

from .baselines import Policy
from .contract import (
    CHECKPOINT,
    FAILURES_FILE,
    ContractMonitor,
    ContractRunError,
    append_failure,
    ensure_failures_file,
    job_failure,
    read_failures,
    run_failure,
)
from .gate import (
    CANDIDATE,
    SHOCKED,
    TWIN,
    DigestMismatch,
    EpisodeRow,
    is_twin,
    resilience_table,
    resilience_to_csv,
    row_from_stats,
    row_to_dict,
)
from .gate_cli import (
    CONTRACT_INVALID,
    INVALID,
    Job,
    Key,
    _execute,
    _open_spec,
    _prepare_resume,
    _write_spec,
    check_rows,
    grid_resilience,
    job_key,
    job_grid,
    read_rows,
    test_only_problem,
)
from .provenance import sha256_file, source_fingerprint
from .registry import (
    DEVELOPMENT,
    REGISTRY_V2,
    Registry,
    WorldEntry,
    build_worlds_v2,
    episode_seed,
    load_registry,
)
from .shock_profiles import (
    SHOCKS_I,
    TEST_ONLY_PROFILES,
    apply_shock_profile,
    load_shock_profile,
    load_shock_profiles,
)

DRIVER = "tokenomics.eval.shock_contract"
PORTABLE = "portable"
FLAT = "flat"
STACKS = (PORTABLE, FLAT)
HORIZON = 100  # D11: a recorded run plays the full gate horizon
FLAT_EPISODES = 10  # D11: hash-derived episode seeds per flat topology
# D11's recorded flat grid: the champion's native held-out topologies at depth 3. A recorded
# run refuses a manifest that declares another grid (`real_run_problems`).
FLAT_DEPTH = 3
FLAT_TOPOLOGIES = (6, 7, 8, 9)
DEFAULT_CHAMPION = Path(__file__).resolve().parents[2] / "champion.json"


def flat_world_id(topology: int, depth: int) -> str:
    return f"flat-scale-d{depth}-s{topology}"


# --- worlds ---


def portable_worlds(registry: Registry, *, profile: str, sizes: Sequence[int] | None = None,
                    worlds_per_size: int | None = None) -> list[WorldEntry]:
    """The registry's development shocked gate worlds, in registry order. A world whose generator
    names another profile is rewritten to `profile` (a dry run's smoke profile); a real run
    refuses that before this matters (`real_run_problems`)."""
    out: list[WorldEntry] = []
    per_size: dict[int, int] = {}
    for e in registry.gate_worlds(DEVELOPMENT):
        if e.regime != SHOCKED or (sizes and e.size not in sizes):
            continue
        if worlds_per_size is not None and per_size.get(e.size, 0) >= worlds_per_size:
            continue
        per_size[e.size] = per_size.get(e.size, 0) + 1
        params = e.generator.get("params", {})
        if params.get("shock_profile") != profile:
            e = replace(e, generator={**e.generator,
                                      "params": {**params, "shock_profile": profile}})
        out.append(e)
    return out


def flat_config(topology: int, depth: int, profile: str):  # -> Configuration
    """A flat-stack world: the champion's native topology with the profile's shocks."""
    from .scale import build_scale_config

    cfg = build_scale_config(topology, depth=depth)
    return apply_shock_profile(cfg, load_shock_profile(profile), len(cfg.producer_configs))


def flat_grid(champion: Mapping[str, object], profile: str, *, sizes: Sequence[int] | None,
              worlds_per_size: int | None, episodes_per_world: int | None) -> dict[str, dict]:
    """world -> grid entry for the flat stack: topology, size, the hash-derived episode seeds and
    the profile's resilience window/tolerance (every world here is shocked, so every episode
    has a twin)."""
    depth = int(champion["depth"])  # type: ignore[arg-type]
    res = load_shock_profile(profile).resilience
    grid: dict[str, dict] = {}
    per_size: dict[int, int] = {}
    for t in champion["held_out_topology_seeds"]:  # type: ignore[union-attr]
        size = len(flat_config(int(t), depth, profile).producer_configs)
        if sizes and size not in sizes:
            continue
        if worlds_per_size is not None and per_size.get(size, 0) >= worlds_per_size:
            continue
        per_size[size] = per_size.get(size, 0) + 1
        w = flat_world_id(int(t), depth)
        seeds = [episode_seed(w, j) for j in range(FLAT_EPISODES)]
        grid[w] = {"stack": FLAT, "topology": int(t), "depth": depth, "size": size,
                   "regime": SHOCKED,
                   "episodes": seeds[:episodes_per_world] if episodes_per_world else seeds,
                   "twin": {"window": res.window, "tolerance": res.tolerance}}
    return grid


def expected_keys(grid: Mapping[str, Mapping]) -> set[Key]:
    """Every world here is shocked: one candidate and one twin per (world, episode)."""
    return {(w, ep, subject, 0) for w, g in grid.items() for ep in g["episodes"]
            for subject in (CANDIDATE, TWIN)}


def build_jobs(grid: Mapping[str, Mapping], worlds: Mapping[str, WorldEntry], horizon: int,
               profile: str, checkpoints: Mapping[str, Mapping[str, object]],
               profiles_version: str) -> list[Job]:
    """One candidate job and its twin per (world, episode). A job carries what a worker needs to
    rebuild its world and policies (plain data, so it pickles). `build` names the profile file's
    content hash too, so a worker's config cache never serves a world built from other inputs."""
    jobs: list[Job] = []
    for w, g in grid.items():
        if g["stack"] == PORTABLE:
            e = worlds[w]
            build = {"generator": e.generator, "world_seed": e.world_seed}
        else:
            build = {"topology": g["topology"], "depth": g["depth"], "shock_profile": profile}
        build["shock_profiles_version"] = profiles_version
        for ep in g["episodes"]:
            cand = {"world": w, "size": g["size"], "regime": g["regime"], "episode": ep,
                    "subject": CANDIDATE, "replicate": 0, "stack": g["stack"],
                    "horizon": horizon, "build": build, "checkpoints": checkpoints[g["stack"]]}
            jobs += [cand, {**cand, "subject": TWIN}]
    return jobs


# --- stacks ---


def load_champion(path: str | Path) -> dict:
    """`champion.json` with its checkpoint paths resolved against the manifest's directory."""
    path = Path(path)
    champ = json.loads(path.read_text())
    for k in ("checkpoint", "producer_checkpoint"):
        champ[k] = str((path.parent / champ[k]).resolve())
    return champ


def _portable_policies(ck: Mapping[str, object]) -> tuple[Policy, Policy]:
    from ..train.orderer_scorer import load_scorer, scorer_ordering_policy
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

    return (scorer_producing_policy(load_producer_score_model(str(ck["producer"]))),
            scorer_ordering_policy(load_scorer(str(ck["orderer"]))))


def _flat_actors(ck: Mapping[str, object]):  # -> (producer actor, orderer actor, bound)
    from ..train.clone_fidelity import load_producer_actor
    from ..train.mlp_actor import load_order_actor

    prod, bound = load_producer_actor(str(ck["producer"]))
    order, order_bound = load_order_actor(str(ck["orderer"]))
    if order_bound != bound:
        raise ValueError(f"champion producer bound {bound} != orderer bound {order_bound}")
    return prod, order, bound


def _flat_policies(ck: Mapping[str, object]) -> tuple[Policy, Policy]:
    """As deployed (EXP-0007A spec "deploy"): the v4 producer greedy, the orderer sampled from
    its learned log_std with the agent's action-space RNG."""
    from ..train.orderer_improve import _greedy_closure, _stochastic_closure

    prod, order, bound = _flat_actors(ck)
    return _greedy_closure(prod, bound), _stochastic_closure(order, bound)


def _flat_env(cfg, horizon: int, ck: Mapping[str, object]):  # -> TokenomicsAECEnv
    """The env `champion.json` declares; refuses an orderer obs version the env would not serve."""
    from ..env import TokenomicsAECEnv

    env = TokenomicsAECEnv(config=cfg, horizon=horizon,
                           product_stream_ordering=bool(ck["product_stream"]))
    if env.spec_.orderer_obs_version != ck["orderer_obs_version"]:
        raise ValueError(f"champion declares orderer obs version {ck['orderer_obs_version']}, "
                         f"the env serves {env.spec_.orderer_obs_version}")
    return env


def _check_flat(ck: Mapping[str, object], champion: Mapping[str, object], profile: str) -> None:
    """Trial load in the parent: both actors load and their widths match the declared env on
    the first held-out topology, so a wrong checkpoint fails as one `run` row, not 80 job rows."""
    prod, order, _ = _flat_actors(ck)
    topology = int(champion["held_out_topology_seeds"][0])  # type: ignore[index]
    env = _flat_env(flat_config(topology, int(champion["depth"]), profile), 1, ck)  # type: ignore[arg-type]
    agents = {role: next(a for a in env.possible_agents if env._role[a][0] == role)
              for role in ("prod", "order")}
    for role, actor in (("prod", prod), ("order", order)):
        a = agents[role]
        want = (env.observation_space(a).shape[0], env.action_space(a).shape[0])
        got = (actor.actor[0].in_features, actor.log_std.numel())
        if got != want:
            raise ValueError(f"champion {role} actor (obs, act) {got} != env {want}")


def checked_stacks(out: Path, stacks: Sequence[str], *, prod_ckpt: str | None,
                   orderer_ckpt: str | None, champion_path: str | Path, profile: str,
                   dry_run: bool, log: Callable[[str], None]
                   ) -> tuple[dict[str, dict[str, object]], dict[str, str], dict | None]:
    """Hash and trial-load each stack's inputs in the parent (D10 run-level failures). Returns
    (per-stack worker inputs, input sha256 by role, the resolved champion manifest). Each stack
    that fails is one `run` row; then the run ends invalid with `ContractRunError` before any
    job."""
    ckpts: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    paths: dict[str, str | None] = {}
    champion = None
    failed = 0
    for stack in stacks:
        try:
            if stack == PORTABLE:
                paths |= {"portable_producer": prod_ckpt, "portable_orderer": orderer_ckpt}
                if not (prod_ckpt and orderer_ckpt):
                    raise ValueError("the portable stack needs --prod-ckpt and --orderer-ckpt")
                hashes |= {"portable_producer": sha256_file(prod_ckpt),
                           "portable_orderer": sha256_file(orderer_ckpt)}
                ck: dict[str, object] = {
                    "producer": prod_ckpt, "orderer": orderer_ckpt,
                    "sha256": [hashes["portable_producer"], hashes["portable_orderer"]]}
                _portable_policies(ck)
            else:
                paths["champion"] = str(champion_path)
                hashes["champion"] = sha256_file(str(champion_path))
                champion = load_champion(champion_path)
                paths |= {"flat_producer": champion["producer_checkpoint"],
                          "flat_orderer": champion["checkpoint"]}
                hashes |= {"flat_producer": sha256_file(champion["producer_checkpoint"]),
                           "flat_orderer": sha256_file(champion["checkpoint"])}
                ck = {"producer": champion["producer_checkpoint"],
                      "orderer": champion["checkpoint"],
                      "product_stream": champion["product_stream"],
                      "orderer_obs_version": champion["orderer_obs_version"],
                      "sha256": [hashes["champion"], hashes["flat_producer"],
                                 hashes["flat_orderer"]]}
                _check_flat(ck, champion, profile)
            ckpts[stack] = ck
        except Exception as exc:  # noqa: BLE001 - recorded, then the run ends invalid
            failed += 1
            append_failure(out, run_failure(CHECKPOINT, RuntimeError(
                f"{stack} stack: {type(exc).__name__}: {exc}")))
    if failed:
        spec_path = out / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {
            "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "exit",
            "inputs": {k: {"path": v, "sha256": hashes.get(k)} for k, v in paths.items()},
            "date": datetime.now(timezone.utc).isoformat()[:10]}
        finish_invalid(out, spec, len(read_failures(out)), None, dry_run, log)
        raise ContractRunError(f"{failed} stack(s) failed to hash or load their checkpoints; "
                               f"recorded in {out / FAILURES_FILE}")
    return ckpts, hashes, champion


# --- workers ---

# Per-process caches, keyed by everything the value is built from (a worker, or the parent with
# --workers 1, may serve several runs): the world's full build spec, and the stack's checkpoint
# paths, contents (sha256 taken in the parent) and declared env flags.
_POLICIES: dict[tuple[str, str], tuple[Policy, Policy]] = {}
_CONFIGS: dict[tuple[str, str, bool], object] = {}


def _identity(d: Mapping[str, object]) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _config(job: Job):  # -> Configuration
    from .resilience import shock_free_twin
    from .task_archive import REPRODUCERS

    twin = is_twin(job["subject"])
    key = (job["stack"], _identity(job["build"]), twin)
    if key not in _CONFIGS:
        b = job["build"]
        if job["stack"] == PORTABLE:
            g = b["generator"]
            cfg = REPRODUCERS[g["generator"]](b["world_seed"], **g.get("params", {}))
            cfg.validate()
        else:
            cfg = flat_config(b["topology"], b["depth"], b["shock_profile"])
        _CONFIGS[key] = shock_free_twin(cfg) if twin else cfg
    return _CONFIGS[key]


def _policies(job: Job) -> tuple[Policy, Policy]:
    stack = job["stack"]
    key = (stack, _identity(job["checkpoints"]))
    if key not in _POLICIES:
        import torch

        torch.set_num_threads(1)
        load = _portable_policies if stack == PORTABLE else _flat_policies
        _POLICIES[key] = load(job["checkpoints"])
    return _POLICIES[key]


def run_job(job: Job) -> dict[str, str]:
    """One episode's row (`gate.row_to_dict`), or, when anything in the job raises or a contract
    check fails, its failure row: the job stops and the others go on (as `gate_cli.run_job`)."""
    try:
        return _run_job(job)
    except Exception as exc:  # noqa: BLE001 - every job failure is recorded, never fatal
        return job_failure(job, exc).to_dict()


def _run_job(job: Job) -> dict[str, str]:
    from ..env.shock_schedule import build_schedule, canonical_events, shock_digest
    from .joint_stack import joint_stack_env
    from .runner import run_episode

    cfg = _config(job)
    horizon = job["horizon"]
    prod, order = _policies(job)
    env = (joint_stack_env(cfg, horizon) if job["stack"] == PORTABLE
           else _flat_env(cfg, horizon, job["checkpoints"]))
    monitor = ContractMonitor(env)
    run_episode(env, prod, order, seed=job["episode"], monitor=monitor)
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=horizon)
    events = build_schedule(cfg, job["episode"], horizon)
    if shock_digest(events) != stats["shock_digest"]:
        raise AssertionError(f"{job['world']}/{job['episode']}: rebuilt shock schedule differs "
                             "from the episode's")  # internal invariant
    schedule = json.dumps(canonical_events(events), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
    return row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                      job["subject"], job["replicate"], stats,
                                      shock_schedule=schedule))


# --- output ---

SUMMARY_COLUMNS = ("stack", "worlds", "episodes", "twins", "deaths", "twin_deaths",
                   "shock_events", "units_mean", "twin_units_mean", "units_ratio",
                   "resilience_rows", "recovered_rows", "overlapped_rows")


def summary_table(rows: Sequence[EpisodeRow], resilience: Sequence[Mapping[str, str]],
                  grid: Mapping[str, Mapping]) -> list[dict[str, object]]:
    """Per stack: deaths, units against the twins, resilience counts. Reported, never gating."""
    out = []
    for stack in STACKS:
        worlds = {w for w, g in grid.items() if g["stack"] == stack}
        if not worlds:
            continue
        cand = [r for r in rows if r.world in worlds and r.subject == CANDIDATE]
        twins = [r for r in rows if r.world in worlds and is_twin(r.subject)]
        res = [x for x in resilience if x["world"] in worlds]
        units, twin_units = sum(r.units for r in cand), sum(r.units for r in twins)
        out.append({
            "stack": stack, "worlds": len(worlds), "episodes": len(cand), "twins": len(twins),
            "deaths": sum(r.dead for r in cand), "twin_deaths": sum(r.dead for r in twins),
            "shock_events": sum(r.shock_events for r in cand),
            "units_mean": fmean(r.units for r in cand) if cand else "",
            "twin_units_mean": fmean(r.units for r in twins) if twins else "",
            "units_ratio": units / twin_units if twin_units else "",
            "resilience_rows": len(res),
            "recovered_rows": sum(x["recovered"] == "true" for x in res),  # gate._res_cell
            "overlapped_rows": sum(x["overlapped"] == "true" for x in res),
        })
    return out


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


RESULT_SKELETON = """# {title}

{banner}**Contract: {exit}.** {failures} contract failure(s); {episodes} shocked episode(s) and
{twins} shock-free twin(s) on stacks {stacks}.

Shock profile `{profile}` (profile file {profiles_version}), horizon {horizon}. Produced by
`python -m {driver}` (see `spec.json`).

## Question

Do the current champion checkpoints load and complete shocked episodes, unretrained, with zero
contract failures (project-mission "Champion runs on a shocked ruleset unretrained")?

## Result

TODO: plain-English reading: contract failures (`contract_failures.csv`), deaths and units against
the twins (`summary.csv`), per-kind resilience (`resilience.csv`). Deaths, units and resilience are
reported, not gating.

## Conclusions

TODO
"""
INVALID_SKELETON = """# {title}

{banner}**INVALID: the contract does not hold.** {failures} contract failure(s) in
`contract_failures.csv`, {missing} episode row(s) missing.

Produced by `python -m {driver}` (see `spec.json`).

## Diagnosis

TODO
"""
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered stacks/worlds/episodes, a "
              "non-default horizon, the test-only smoke profile or no source commit; see "
              "`spec.json`.\n\n")


def finish_invalid(out: Path, spec: dict, failures: int, missing: int | None, dry_run: bool,
                   log: Callable[[str], None]) -> None:
    spec["status"] = INVALID
    spec["result"] = {"status": INVALID, "contract_failures": failures, "missing_rows": missing,
                      "exit_holds": False, "acceptance_read": False}
    out.mkdir(parents=True, exist_ok=True)
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(INVALID_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", failures=failures,
            missing="unknown" if missing is None else missing, driver=DRIVER))
    log(f"[shock-contract] INVALID run: {failures} contract failure(s), "
        f"{'unknown' if missing is None else missing} missing row(s); see {FAILURES_FILE}")


# --- run ---


def _is_commit(s: str) -> bool:
    return len(s) == 40 and all(c in "0123456789abcdef" for c in s)


def _world_identity(e: WorldEntry) -> tuple:
    return (e.id, e.size, e.world_seed, tuple(e.episode_seeds), _identity(e.generator))


def portable_grid_problem(worlds: Sequence[WorldEntry]) -> str | None:
    """D11's recorded portable grid is exactly the registry v2 draft's (`build_worlds_v2`)
    development shocked gate worlds, each with all its episode seeds: 4 worlds at each of 12
    and 42 producers x 10 = 80 episodes. A registry file that drops, adds or alters any of
    them (a subset, a screened replacement, other seeds) would make the exit read on another
    grid. None = the grid is the draft's."""
    want = {_world_identity(e) for e in portable_worlds(build_worlds_v2(), profile=SHOCKS_I)}
    got = {_world_identity(e) for e in worlds}
    if got == want:
        return None

    def ids(xs: set[tuple]) -> list[str]:
        return sorted({x[0] for x in xs})

    return (f"portable worlds differ from the registry v2 draft's development shocked gate "
            f"worlds (D11: 4 per size at 12 and 42, all 10 episode seeds each): missing or "
            f"altered {ids(want - got)}, unexpected or altered {ids(got - want)}")


def flat_grid_problem(champion: Mapping[str, object] | None) -> str | None:
    """D11's recorded flat grid: depth `FLAT_DEPTH`, topologies `FLAT_TOPOLOGIES`. None = the
    manifest declares it, or it could not be read (then its load fails as a `run` row)."""
    if champion is None:
        return None
    got = (champion.get("depth"), tuple(champion.get("held_out_topology_seeds") or ()))
    if got == (FLAT_DEPTH, FLAT_TOPOLOGIES):
        return None
    return (f"champion manifest declares depth {got[0]} topologies {list(got[1])}, not D11's "
            f"depth {FLAT_DEPTH} topologies {list(FLAT_TOPOLOGIES)}")


def _readable_champion(path: str | Path) -> dict | None:
    try:
        return load_champion(path)
    except Exception:  # noqa: BLE001 - the checkpoint stage records it as a `run` row
        return None


def real_run_problems(*, stacks: Sequence[str], profile: str, horizon: int, source_commit: str,
                      registry: Registry, worlds: Sequence[WorldEntry],
                      filtered: bool, champion: Mapping[str, object] | None = None) -> list[str]:
    """Why this may not be a recorded run (D7, D11); empty = admitted. Beyond the flags, the
    input files must give D11's exact grid (`portable_grid_problem`, `flat_grid_problem`);
    `champion` is the manifest as read, or None when it is unreadable."""
    named = sorted({e.generator.get("params", {}).get("shock_profile") or "none"
                    for e in registry.gate_worlds(DEVELOPMENT) if e.regime == SHOCKED})
    checks = (
        (profile in TEST_ONLY_PROFILES,
         f"shock profile {profile!r} is test-only and runs only with --dry-run"),
        (profile not in TEST_ONLY_PROFILES and profile != SHOCKS_I,
         f"a recorded run uses the pre-registered {SHOCKS_I!r} profile, not {profile!r}"),
        (PORTABLE in stacks and named != [SHOCKS_I],
         f"registry {registry.name!r} shocked development worlds name profiles {named}, "
         f"not [{SHOCKS_I!r}]"),
        (PORTABLE in stacks and registry.version_tag < 2,
         f"registry {registry.name!r} is v{registry.version_tag}, not a v2 registry"),
        (tuple(stacks) != STACKS, "--stacks is set"),
        (filtered, "world/episode filters are set"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
    )
    problems = [msg for bad, msg in checks if bad]
    smoke = test_only_problem(worlds)
    grids = ((PORTABLE, lambda: portable_grid_problem(worlds)),
             (FLAT, lambda: flat_grid_problem(champion)))
    found = [p() for stack, p in grids if stack in stacks]
    return problems + ([smoke] if smoke else []) + [p for p in found if p]


def run_contract(
    *,
    out_dir: str | Path,
    stacks: Sequence[str] = STACKS,
    prod_ckpt: str | None = None,
    orderer_ckpt: str | None = None,
    champion_path: str | Path = DEFAULT_CHAMPION,
    registry: Registry | None = None,
    profile: str = SHOCKS_I,
    horizon: int = HORIZON,
    workers: int = 1,
    sizes: Sequence[int] | None = None,
    worlds_per_size: int | None = None,
    episodes_per_world: int | None = None,
    source_commit: str = "",
    dry_run: bool = False,
    run_job_fn: Callable[[Job], dict[str, str]] = run_job,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run (or resume) the exit run into `out_dir`; return the `spec.json` result block.

    Everything that can refuse runs before any file in `out_dir` is touched. Contract failures
    are recorded, never refused: a stack whose checkpoints do not hash or load is a `run` row
    and raises `ContractRunError` before any job; a failed job is a `job` row and the other jobs
    finish; a digest mismatch is a `run` row and raises `DigestMismatch`. A run with any failure
    row (also from an earlier session in `out_dir`) or a missing row ends `invalid`."""
    stacks = tuple(s for s in STACKS if s in stacks)
    if not stacks:
        raise ValueError(f"no stack selected; choose from {STACKS}")
    registry = registry if registry is not None else load_registry(REGISTRY_V2)
    worlds = (portable_worlds(registry, profile=profile, sizes=sizes,
                              worlds_per_size=worlds_per_size) if PORTABLE in stacks else [])
    if not dry_run:
        problems = real_run_problems(
            stacks=stacks, profile=profile, horizon=horizon, source_commit=source_commit,
            registry=registry, worlds=worlds,
            filtered=any(f is not None for f in (sizes, worlds_per_size, episodes_per_world)),
            champion=_readable_champion(champion_path) if FLAT in stacks else None)
        if problems:
            raise ValueError("not a valid recorded run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    profiles = load_shock_profiles()
    profiles.get(profile)  # unknown name -> KeyError before any write
    out = Path(out_dir)
    ckpts, hashes, champion = checked_stacks(
        out, stacks, prod_ckpt=prod_ckpt, orderer_ckpt=orderer_ckpt, champion_path=champion_path,
        profile=profile, dry_run=dry_run, log=log)
    grid: dict[str, dict] = {}
    if PORTABLE in stacks:
        grid |= {w: {"stack": PORTABLE, **g}
                 for w, g in job_grid(worlds, episodes_per_world, twins=True).items()}
    if FLAT in stacks:
        assert champion is not None
        grid |= flat_grid(champion, profile, sizes=sizes, worlds_per_size=worlds_per_size,
                          episodes_per_world=episodes_per_world)
    if not grid:
        raise ValueError("no worlds selected")
    if any("twin" not in g for g in grid.values()):
        raise ValueError("a selected world has no shocks: "
                         f"{sorted(w for w, g in grid.items() if 'twin' not in g)}")
    source = source_fingerprint(source_commit or None)
    pins = {
        "stacks": list(stacks), "horizon": horizon, "dry_run": dry_run,
        "shock_profile": profile, "shock_profiles_version": profiles.version,
        "registry_version": registry.version if PORTABLE in stacks else None,
        "checkpoints": hashes,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    expected = expected_keys(grid)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "exit",
        "pins": pins,
        "inputs": _inputs(ckpts, champion_path, hashes),
        "champion": champion,
        "registry": None if PORTABLE not in stacks else {
            "name": registry.name, "content_hash": registry.version,
            "shock_profiles": registry.shock_profiles},
        "source": source,
    })
    jobs = build_jobs(grid, {e.id: e for e in worlds}, horizon, profile, ckpts, profiles.version)
    ensure_failures_file(out)

    def glog(s: str) -> None:
        log(s.replace("[gate]", "[shock-contract]", 1))

    try:
        _execute(out, spec, done, jobs, expected, workers, (horizon, None, None), run_job_fn,
                 glog)
    except DigestMismatch:
        finish_invalid(out, spec, len(read_failures(out)), None, dry_run, log)
        raise
    rows = read_rows(out / "episodes.csv")
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    failures = read_failures(out)
    if failures or missing:
        finish_invalid(out, spec, len(failures), len(missing), dry_run, log)
        return spec["result"]
    resilience = resilience_table(rows, grid_resilience(grid))
    (out / "resilience.csv").write_text(resilience_to_csv(resilience))
    summary = summary_table(rows, resilience, grid)
    _write_csv(out / "summary.csv", SUMMARY_COLUMNS, summary)
    spec["status"] = "complete"
    spec["result"] = {"status": "complete", "contract_failures": 0, "missing_rows": 0,
                      "exit_holds": True, "acceptance_read": not dry_run,
                      "episodes": sum(int(s["episodes"]) for s in summary),
                      "twins": sum(int(s["twins"]) for s in summary),
                      "deaths": sum(int(s["deaths"]) for s in summary),
                      "resilience_rows": len(resilience)}
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "",
            exit="HOLDS (dry run)" if dry_run else "HOLDS", failures=0,
            episodes=spec["result"]["episodes"], twins=spec["result"]["twins"],
            stacks=", ".join(stacks), profile=profile, profiles_version=profiles.version[:12],
            horizon=horizon, driver=DRIVER))
    log(f"[shock-contract] {'DRY RUN ' if dry_run else ''}contract holds: 0 failures, "
        f"{spec['result']['episodes']} episodes + {spec['result']['twins']} twins, "
        f"{spec['result']['deaths']} deaths, {len(resilience)} resilience rows")
    return spec["result"]


def _inputs(ckpts: Mapping[str, Mapping[str, object]], champion_path: str | Path,
            hashes: Mapping[str, str]) -> dict[str, dict[str, object]]:
    paths: dict[str, object] = {}
    if PORTABLE in ckpts:
        paths |= {"portable_producer": ckpts[PORTABLE]["producer"],
                  "portable_orderer": ckpts[PORTABLE]["orderer"]}
    if FLAT in ckpts:
        paths |= {"champion": str(champion_path), "flat_producer": ckpts[FLAT]["producer"],
                  "flat_orderer": ckpts[FLAT]["orderer"]}
    return {k: {"path": str(v), "sha256": hashes[k]} for k, v in paths.items()}


def _csv_ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",")] if s else None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m {DRIVER}", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stacks", help=f"dry-run filter, comma list of {', '.join(STACKS)}")
    ap.add_argument("--prod-ckpt", help="portable producer (EXP-0035 producer_candidate.pt)")
    ap.add_argument("--orderer-ckpt", help="portable orderer (EXP-0012 orderer_scorer.pt)")
    ap.add_argument("--champion", default=str(DEFAULT_CHAMPION),
                    help="flat stack manifest; checkpoint paths resolve against its directory")
    ap.add_argument("--registry", default=str(REGISTRY_V2))
    ap.add_argument("--shock-profile", default=SHOCKS_I,
                    help=f"default {SHOCKS_I}; the test-only shocks-i-smoke needs --dry-run")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="dry run: any other value")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sizes", help="dry-run filter, e.g. 12")
    ap.add_argument("--worlds-per-size", type=int, help="dry-run filter: first N per stack/size")
    ap.add_argument("--episodes-per-world", type=int, help="dry-run filter")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--verify-source-commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke run: allows filters, another horizon, the smoke profile and no "
                         "source commit; marked as not an acceptance read")
    args = ap.parse_args(argv)
    stacks = tuple(args.stacks.split(",")) if args.stacks else STACKS
    unknown = sorted(set(stacks) - set(STACKS))
    if unknown:
        ap.error(f"--stacks: unknown {unknown}; choose from {STACKS}")
    if args.stacks and not args.dry_run:
        ap.error("--stacks is a dry-run filter")
    source_commit = args.source_commit
    if args.verify_source_commit:
        from .provenance import resolve_verified_commit

        source_commit = resolve_verified_commit()
    try:
        result = run_contract(
            out_dir=args.out_dir, stacks=stacks, prod_ckpt=args.prod_ckpt,
            orderer_ckpt=args.orderer_ckpt, champion_path=args.champion,
            registry=load_registry(args.registry), profile=args.shock_profile,
            horizon=args.horizon, workers=args.workers, sizes=_csv_ints(args.sizes),
            worlds_per_size=args.worlds_per_size, episodes_per_world=args.episodes_per_world,
            source_commit=source_commit, dry_run=args.dry_run)
    except (ContractRunError, DigestMismatch) as exc:
        print(f"[shock-contract] INVALID run: {exc}", file=sys.stderr)
        return CONTRACT_INVALID
    return 0 if result.get("exit_holds") else CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
