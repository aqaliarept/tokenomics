"""Orderer oracle-relabel representability smoke (EXP-0032 v4, step 1).

Answers the standing EXP-0026 question BEFORE spending a pod run: *can the portable orderer
scorer head represent the mining bar's per-cycle search routing at all?* The Stage-4 orderer
distilled only from the `proportional` throughput teacher and could never absorb the
search-oracle headroom the mined gap used; v4 relabels the orderer from the bar's own search
(`collect_oracle_orderer_samples`). If a fresh scorer distilled on those oracle targets cannot
even reproduce the search on the SAME task it was trained on, the limit is architectural (no
run needed); if it recovers a large share, the pod run is green-lit.

Per task, with the producer held at the champion (deployment-matched), it reports four on-task
consumer-unit numbers:
  - `champ_cu`  : champion producer + champion orderer          (the stack being improved)
  - `search_cu` : champion producer + the bar's search orderer  (the teacher on-task = ceiling)
  - `oracle_cu` : champion producer + oracle-relabeled orderer  (the distilled candidate)
  - `bar_cu`    : entity_planner + mpc_search (the archived reference gap was measured against)
and two recovery ratios:
  - `recovered_vs_teacher = (oracle_cu - champ_cu) / (search_cu - champ_cu)`  — the
    representability answer (can the head reproduce the search it was trained on?),
  - `recovered_vs_bar     = (oracle_cu - champ_cu) / (bar_cu   - champ_cu)`   — recovery of the
    archived gap (bounded below 1 by the producer-side share entity_planner keeps).

CLOUD run (CLAUDE.md HARD RULE): this is a **decision-gating verification** — full-horizon
per-cycle search, distill to convergence, `eval_seeds > 1`, multiple tasks (~14 heavy search
episodes). It is NOT a local smoke despite touching only 1-2 tasks; classify by compute profile,
not task count. Launch it wrapped with `run_pod_training.py`. The full v4 promotion candidate is
the follow-on pod `run_stage4_replay` step.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from .curriculum import TaskRecord, _episode_seeds, replay
from ..train.orderer_scorer import (
    OrdererScoreModel,
    collect_oracle_orderer_samples,
    fit_scorer,
    save_scorer,
    scorer_sample_metrics,
    scorer_ordering_policy,
)

# The bar's search config (mirrors `curriculum._bar_mpc_search` / the collector defaults) so the
# oracle teacher, the on-task search ceiling, and the archived bar all use one search.
_SEARCH = dict(candidates=("even_split", "proportional"), n_samples=8, sample_scale=1.0,
               lookahead=6)


def _mean_units(run) -> float:
    return run["consumer_completed_units"]


def _safe_task_dir_name(task_id: str) -> str:
    """Filesystem-safe, still recognizable task id for debug artifact directories."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in task_id)[:220]


def _safe_metric_key(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_") or "unknown"


def _capacity_state(env) -> dict:
    """Compact per-capacity-type state for a deploy-collapse trace."""
    by_type = defaultdict(lambda: {
        "current": 0, "max": 0, "reserved": 0, "requested": 0,
        "producers_alive": 0, "producers_total": 0,
    })
    inventory = inbound = blocked = 0
    for p in env.system.producing_agents.values():
        inventory += p.inventory_units()
        inbound += p.outstanding_inbound_units()
        blocked += len(p.reservations)
        reserved = p.reserved_capacity()
        types = set(p.capacities) | set(p.max_capacities) | set(reserved) | set(p.requested_capacity_by_type)
        for t in types:
            slot = by_type[t]
            cur = p.capacities.get(t, 0)
            slot["current"] += cur
            slot["max"] += p.max_capacities.get(t, 0)
            slot["reserved"] += reserved.get(t, 0)
            slot["requested"] += p.requested_capacity_by_type.get(t, 0)
            slot["producers_total"] += int(t in p.max_capacities)
            slot["producers_alive"] += int(cur > 0)
    return {
        "inventory_units": inventory,
        "inbound_units": inbound,
        "blocked_reservations": blocked,
        "capacity_by_type": dict(by_type),
    }


def _trace_deployed_orderer_episode(cfg, prod_pol, order_pol, *, seed: int, horizon: int) -> tuple[dict, list[dict]]:
    """Run the deployed distilled orderer once and return stats plus cycle trace rows."""
    from .joint_stack import joint_stack_env

    env = joint_stack_env(cfg, horizon)
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)

    rows: list[dict] = []
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    prev_cycle = 0
    prev_cu = prev_done = prev_rej = 0
    for agent in env.agent_iter(max_iter=max_iter):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        action = prod_pol(env, agent) if role == "prod" else order_pol(env, agent)
        env.step(action)
        if env.current_cycle > prev_cycle:
            s = env.episode_stats
            cap_state = _capacity_state(env)
            by_type = cap_state.pop("capacity_by_type")
            row = {
                "seed": seed,
                "cycle": env.current_cycle,
                "dead": bool(s["dead"]),
                "consumer_completed_units_total": s["consumer_completed_units"],
                "consumer_completed_units_delta": s["consumer_completed_units"] - prev_cu,
                "replenishment_completed_delta": s["replenishment_completed"] - prev_done,
                "replenishment_rejected_delta": s["replenishment_rejected"] - prev_rej,
                **cap_state,
            }
            current_totals = [v["current"] for v in by_type.values()]
            row["min_capacity_type_current"] = min(current_totals) if current_totals else 0
            for cap_type, vals in sorted(by_type.items()):
                prefix = f"capacity_{_safe_metric_key(cap_type)}"
                for name, val in vals.items():
                    row[f"{prefix}_{name}"] = val
            rows.append(row)
            prev_cycle = env.current_cycle
            prev_cu = s["consumer_completed_units"]
            prev_done = s["replenishment_completed"]
            prev_rej = s["replenishment_rejected"]

    stats = dict(env.episode_stats)
    trace = env.capacity_trace
    stats["cap_final"] = trace[-1] if trace else 0.0
    stats["cap_min"] = min(trace) if trace else 0.0
    return stats, rows


def _trace_summary(trace_rows: list[dict]) -> dict:
    if not trace_rows:
        return {
            "trace_cycles": 0,
            "trace_dead": False,
            "trace_dead_seed_count": 0,
            "trace_first_dead_cycle": None,
            "trace_first_zero_capacity_cycle": None,
            "trace_zero_capacity_seed_count": 0,
            "trace_min_capacity_type_current": None,
        }
    first_dead = next((r["cycle"] for r in trace_rows if r["dead"]), None)
    first_zero = next((r["cycle"] for r in trace_rows if r["min_capacity_type_current"] <= 0), None)
    dead_seeds = {r["seed"] for r in trace_rows if r["dead"]}
    zero_capacity_seeds = {r["seed"] for r in trace_rows if r["min_capacity_type_current"] <= 0}
    return {
        "trace_cycles": len(trace_rows),
        "trace_dead": bool(dead_seeds),
        "trace_dead_seed_count": len(dead_seeds),
        "trace_first_dead_cycle": first_dead,
        "trace_first_zero_capacity_cycle": first_zero,
        "trace_zero_capacity_seed_count": len(zero_capacity_seeds),
        "trace_min_capacity_type_current": min(r["min_capacity_type_current"] for r in trace_rows),
    }


def _champ_search_units(cfg, prod_pol, champ_order_model, *, horizon: int, seeds: list[int]
                        ) -> tuple[float, float]:
    """Mean on-task units for the two producer-frozen crosses that bound the orderer headroom:
    `champ_prod + champ_ord` (the deployed stack) and `champ_prod + per-cycle search` (the
    teacher ceiling). Shared by the headroom scan and the representability row so the two never
    drift. `seeds` MUST be the archive-anchored episode seeds (`_episode_seeds`), never
    `range(eval_seeds)` — the mined gap lives at the record's demand trace, and evaluating any
    other seed measures a different task (EXP-0036 probe bug: champ alive at seeds 0-2 while
    the archived gap had it dead at seeds 77-79)."""
    from .joint_stack import joint_stack_env
    from .orderer_oracle import search_orderer_episode
    from .baselines import ORDERING_BASELINES
    from .runner import run_episode

    cand = {n: ORDERING_BASELINES[n] for n in _SEARCH["candidates"]}
    champ_order = scorer_ordering_policy(champ_order_model)
    champ, search = [], []
    for es in seeds:
        champ.append(_mean_units(run_episode(joint_stack_env(cfg, horizon), prod_pol, champ_order, seed=es)))
        stats, _ = search_orderer_episode(joint_stack_env(cfg, horizon), prod_pol, cand,
                                          _SEARCH["n_samples"], _SEARCH["sample_scale"],
                                          _SEARCH["lookahead"], es)
        search.append(stats["consumer_completed_units"])
    return statistics.mean(champ), statistics.mean(search)


def _named_search_units(cfg, prod_pol, *, horizon: int, seeds: list[int],
                        alphas: tuple[float, ...] = ()) -> float:
    """Mean on-task units for the per-cycle search restricted to the NAMED candidates only
    (`n_samples=0`) — same per-cycle choice/switching the full search has, minus the Gaussian
    directed-sample routings. Compared to the full `search_cu`, this isolates whether the orderer
    headroom lives INSIDE a named-policy vocabulary (per-cycle even_split/proportional switching, a
    learnable categorical teacher) or comes from the continuous sampled routings (which a named
    representation would throw away). With `alphas`, the vocabulary also carries the
    `ordering_blend` interpolations (A0b / D2: does the alpha-blend family recover the continuous
    residual hard selection leaves behind?). Frozen champion producer; archive-anchored seeds."""
    from .joint_stack import joint_stack_env
    from .orderer_oracle import blend_vocabulary, search_orderer_episode

    cand = blend_vocabulary(alphas)
    units = []
    for es in seeds:
        stats, _ = search_orderer_episode(joint_stack_env(cfg, horizon), prod_pol, cand,
                                          0, _SEARCH["sample_scale"], _SEARCH["lookahead"], es)
        units.append(stats["consumer_completed_units"])
    return statistics.mean(units)


def headroom_row(record: TaskRecord, prod_pol, champ_order_model, *, horizon: int,
                 eval_seeds: int, log=lambda *a, **k: None) -> dict:
    """Cheap per-task orderer-headroom probe (no distill/oracle/bar): `teacher_headroom =
    champ_prod+search − champ_prod+champ_ord`, under the FROZEN producer. EXP-0032 showed the
    archived mining `gap`/`score` does not survive the augmented EXP-0030 producer (it already
    closed most of it), so probe tasks must be re-selected by *measured* headroom, not archived
    score. `headroom_frac` normalizes by `champ_cu` for a topology-agnostic floor."""
    from .runner import run_episode
    from .joint_stack import joint_stack_env
    from .baselines import ORDERING_BASELINES

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    champ_cu, search_cu = _champ_search_units(cfg, prod_pol, champ_order_model,
                                              horizon=horizon, seeds=seeds)
    # Named-baseline orderers for the A0 STOP-gate table: `oracle ~= even_split` => no headroom.
    base_cu = {}
    for name in ("even_split", "proportional"):
        pol = ORDERING_BASELINES[name]
        base_cu[name] = statistics.mean(
            _mean_units(run_episode(joint_stack_env(cfg, horizon), prod_pol, pol, seed=es))
            for es in seeds)
    head = search_cu - champ_cu
    return {
        "task_id": record.task_id, "world_seed": record.world_seed,
        "champ_cu": round(champ_cu, 2), "search_cu": round(search_cu, 2),
        "even_split_cu": round(base_cu["even_split"], 2),
        "proportional_cu": round(base_cu["proportional"], 2),
        "teacher_headroom": round(head, 2),
        "headroom_frac": round(head / champ_cu, 4) if champ_cu > 1e-9 else None,
    }


def a0_probe_row(
    record: TaskRecord, prod_pol, champ_order_model, *, iters: int, horizon: int,
    eval_seeds: int, k_searches: int = 4, debug_dir=None, log=lambda *a, **k: None,
) -> dict:
    """Stage A0 decision row (EXP-0036 amendment): H3 STOP-gate baselines + H1's discriminating
    check. Emits the headroom columns (champ / search-oracle / even_split / proportional units) and
    the variance-explained of a scorer fit on the SINGLE-search target vs on the K-averaged
    (denoised) target — both on the frozen champion producer at the archived episode seeds.

    Read order: headroom first (`search_cu ~= even_split_cu` => STOP, no orderer lever), then
    `ve_averaged` vs `ve_single` (a jump => the single target was tie-arbitrary noise the search
    can denoise => proceed to Stage A1). Neither denoises but headroom exists => Stage A2."""
    from ..train.orderer_scorer import (
        collect_averaged_oracle_orderer_samples, collect_oracle_orderer_samples,
        OrdererScoreModel, fit_scorer, scorer_sample_metrics)

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    head = headroom_row(record, prod_pol, champ_order_model, horizon=horizon,
                        eval_seeds=eval_seeds, log=log)

    # Representation diagnostic: how much of the search's over-even_split headroom does a per-cycle
    # search restricted to NAMED candidates recover? ~1 => a named-policy teacher captures the lever.
    search_named_cu = _named_search_units(cfg, prod_pol, horizon=horizon, seeds=seeds)
    full_head = head["search_cu"] - head["even_split_cu"]
    named_capture_frac = (round((search_named_cu - head["even_split_cu"]) / full_head, 4)
                          if full_head > 1e-9 else None)

    collect_kw = dict(prod=prod_pol, candidates=_SEARCH["candidates"], n_samples=_SEARCH["n_samples"],
                      sample_scale=_SEARCH["sample_scale"], lookahead=_SEARCH["lookahead"],
                      horizon=horizon, log=log)
    single = collect_oracle_orderer_samples([cfg] * len(seeds), seeds, **collect_kw)
    averaged = collect_averaged_oracle_orderer_samples([cfg] * len(seeds), seeds,
                                                       k_searches=k_searches, **collect_kw)
    ve_single = scorer_sample_metrics(
        fit_scorer(OrdererScoreModel(), single, iters=iters, log=log), single)["sample_variance_explained"]
    ve_averaged = scorer_sample_metrics(
        fit_scorer(OrdererScoreModel(), averaged, iters=iters, log=log), averaged)["sample_variance_explained"]

    if debug_dir is not None:
        import gzip
        import pickle
        from pathlib import Path

        task_dir = Path(debug_dir) / _safe_task_dir_name(record.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        for name, sample_set in (("single", single), ("averaged", averaged)):
            with gzip.open(task_dir / f"samples_{name}.pkl.gz", "wb") as fh:
                pickle.dump(sample_set, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "task_id": record.task_id, "world_seed": record.world_seed,
        "champ_cu": head["champ_cu"], "search_cu": head["search_cu"],
        "even_split_cu": head["even_split_cu"], "proportional_cu": head["proportional_cu"],
        "search_named_cu": round(search_named_cu, 2), "named_capture_frac": named_capture_frac,
        "teacher_headroom": head["teacher_headroom"], "headroom_frac": head["headroom_frac"],
        "k_searches": k_searches,
        "ve_single": ve_single, "ve_averaged": ve_averaged,
        "n_samples_single": len(single), "n_samples_averaged": len(averaged),
    }


def a0b_diag_row(
    record: TaskRecord, prod_pol, *, iters: int, horizon: int, eval_seeds: int,
    alphas: tuple[float, ...] = (0.75, 0.5, 0.25), log=lambda *a, **k: None,
) -> dict:
    """A0b decision row — the two questions A0's `named_capture_frac` left open (it proved the
    headroom is REACHABLE inside a named vocabulary via a lookahead oracle, not that a student
    can realize it):

    D2 (blend value): re-run the named-only search with the `ordering_blend` alphas added.
    `blend_capture_frac` >> `named_capture_frac` => the continuous residual is blend-recoverable,
    keep the alpha head; ~equal => plain categorical suffices, drop alpha.

    D1 (learnability): fit `OrdererChoiceModel` on the committed vocabulary choices from the
    train episode seeds and read top-1 accuracy on the HELD-OUT last seed vs the majority-class
    floor (decisive-margin split separately). Near-majority => the categorical target inherits
    the same unpredictability wall as the continuous one => A1b would fail; well above => green.

    No champion involved — both questions compare search variants and observation-predictability
    under the frozen producer at the archive-anchored episode seeds."""
    from .runner import run_episode
    from .joint_stack import joint_stack_env
    from .baselines import ORDERING_BASELINES
    from .orderer_oracle import search_orderer_episode
    from ..train.orderer_scorer import (
        GlobalChoiceModel, OrdererChoiceModel, choice_metrics, collect_choice_samples,
        fit_choice_model)

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    even_split_cu = statistics.mean(
        _mean_units(run_episode(joint_stack_env(cfg, horizon), prod_pol,
                                ORDERING_BASELINES["even_split"], seed=es))
        for es in seeds)
    cand = {n: ORDERING_BASELINES[n] for n in _SEARCH["candidates"]}
    search_cu = statistics.mean(
        search_orderer_episode(joint_stack_env(cfg, horizon), prod_pol, cand,
                               _SEARCH["n_samples"], _SEARCH["sample_scale"],
                               _SEARCH["lookahead"], es)[0]["consumer_completed_units"]
        for es in seeds)
    search_named_cu = _named_search_units(cfg, prod_pol, horizon=horizon, seeds=seeds)
    search_blend_cu = _named_search_units(cfg, prod_pol, horizon=horizon, seeds=seeds,
                                          alphas=alphas)
    full_head = search_cu - even_split_cu
    capture = lambda cu: round((cu - even_split_cu) / full_head, 4) if full_head > 1e-9 else None

    heldout = len(seeds) >= 2
    train_seeds = list(seeds[:-1]) if heldout else list(seeds)
    collect_kw = dict(prod=prod_pol, alphas=alphas, lookahead=_SEARCH["lookahead"],
                      horizon=horizon, log=log)
    train_samples, train_gsamples, vocab = collect_choice_samples(
        [cfg] * len(train_seeds), train_seeds, **collect_kw)
    heldout_samples, heldout_gsamples, _ = collect_choice_samples([cfg], [seeds[-1]],
                                                                  **collect_kw)
    if train_samples:
        model = fit_choice_model(OrdererChoiceModel(n_choices=len(vocab)), train_samples,
                                 iters=iters, log=log)
        m = choice_metrics(model, heldout_samples)
    else:
        m = {"top1": None, "majority": None, "top1_decisive": None, "majority_decisive": None,
             "n_decisive": 0}
    # A0c discriminator: same protocol from the FULL global state (one sample per searched
    # cycle). Beats the majority floor where the stream view failed => the D1 wall is missing
    # CONTEXT; fails too => the committed choice is rollout NOISE — no observation can help.
    if train_gsamples:
        gmodel = GlobalChoiceModel(n_choices=len(vocab),
                                   state_dim=train_gsamples[0].view.shape[0])
        gmodel.set_norm([g.view for g in train_gsamples])
        fit_choice_model(gmodel, train_gsamples, iters=iters, log=log)
        gm = choice_metrics(gmodel, heldout_gsamples)
    else:
        gm = {"top1": None, "majority": None, "top1_decisive": None, "majority_decisive": None,
              "n_decisive": 0}

    return {
        "task_id": record.task_id, "world_seed": record.world_seed,
        "search_cu": round(search_cu, 2), "even_split_cu": round(even_split_cu, 2),
        "search_named_cu": round(search_named_cu, 2), "search_blend_cu": round(search_blend_cu, 2),
        "named_capture_frac": capture(search_named_cu),
        "blend_capture_frac": capture(search_blend_cu),
        "choice_top1": m["top1"], "choice_majority": m["majority"],
        "choice_top1_decisive": m["top1_decisive"],
        "choice_majority_decisive": m["majority_decisive"],
        "n_choice_decisive": m["n_decisive"],
        "n_choice_train": len(train_samples), "n_choice_heldout": len(heldout_samples),
        "gchoice_top1": gm["top1"], "gchoice_majority": gm["majority"],
        "gchoice_top1_decisive": gm["top1_decisive"],
        "gchoice_majority_decisive": gm["majority_decisive"],
        "n_gchoice_train": len(train_gsamples), "n_gchoice_heldout": len(heldout_gsamples),
        "choice_heldout": heldout, "vocab": vocab,
    }


def representability_row(
    record: TaskRecord, prod_pol, champ_order_model, *, iters: int, horizon: int,
    eval_seeds: int, debug_dir=None, log=lambda *a, **k: None,
) -> dict:
    """One task's representability row: distill an orderer on this task's oracle-relabel
    targets (champion producer drives collection), then eval champion / search-teacher /
    oracle / bar on-task. Producer fixed at champion throughout (isolates the orderer)."""
    from .curriculum import REFERENCE_BARS

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    task_dir = None
    if debug_dir is not None:
        from pathlib import Path

        task_dir = Path(debug_dir) / _safe_task_dir_name(record.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

    # 1. collect oracle-relabel samples on this task AT THE ARCHIVED EPISODE SEEDS (the same
    # demand traces every eval below averages — collect/deploy seed mismatch craters the oracle).
    samples = collect_oracle_orderer_samples(
        [cfg] * len(seeds), seeds, prod=prod_pol, horizon=horizon,
        candidates=_SEARCH["candidates"], n_samples=_SEARCH["n_samples"],
        sample_scale=_SEARCH["sample_scale"], lookahead=_SEARCH["lookahead"], log=log)
    log(f"{record.task_id}: {len(samples)} oracle samples")
    oracle_model = fit_scorer(OrdererScoreModel(), samples, iters=iters, log=log)
    sample_metrics = scorer_sample_metrics(oracle_model, samples)

    if debug_dir is not None:
        import gzip
        import json
        import pickle

        save_scorer(oracle_model, str(task_dir / "orderer_oracle.pt"))
        with gzip.open(task_dir / "samples.pkl.gz", "wb") as fh:
            pickle.dump(samples, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (task_dir / "sample_metrics.json").write_text(json.dumps(sample_metrics, indent=2) + "\n")

    oracle_order = scorer_ordering_policy(oracle_model)
    bar = REFERENCE_BARS["entity_planner+mpc_search"]

    # champ + search (the frozen-producer bounds) share one code path with the headroom scan.
    champ_cu, search_cu = _champ_search_units(cfg, prod_pol, champ_order_model,
                                              horizon=horizon, seeds=seeds)
    oracle, barcu, trace_rows = [], [], []
    for es in seeds:
        stats, rows = _trace_deployed_orderer_episode(cfg, prod_pol, oracle_order,
                                                      seed=es, horizon=horizon)
        oracle.append(_mean_units(stats))
        trace_rows.extend({"task_id": record.task_id, **r} for r in rows)
        barcu.append(_mean_units(bar(cfg, es, horizon)))
    trace_metrics = _trace_summary(trace_rows)
    if task_dir is not None:
        import json

        from .curriculum import write_matrix_csv

        write_matrix_csv(trace_rows, task_dir / "deploy_trace.csv")
        (task_dir / "deploy_trace_summary.json").write_text(
            json.dumps(trace_metrics, indent=2) + "\n")

    oracle_cu = statistics.mean(oracle)
    bar_cu = statistics.mean(barcu)
    teacher_head = search_cu - champ_cu
    bar_head = bar_cu - champ_cu
    recovered = oracle_cu - champ_cu
    return {
        "task_id": record.task_id, "world_seed": record.world_seed,
        "champ_cu": round(champ_cu, 2), "search_cu": round(search_cu, 2),
        "oracle_cu": round(oracle_cu, 2), "bar_cu": round(bar_cu, 2),
        "teacher_headroom": round(teacher_head, 2), "bar_headroom": round(bar_head, 2),
        "recovered": round(recovered, 2),
        "recovered_vs_teacher": round(recovered / teacher_head, 3) if teacher_head > 1e-9 else None,
        "recovered_vs_bar": round(recovered / bar_head, 3) if bar_head > 1e-9 else None,
        "n_samples": len(samples),
        **sample_metrics,
        **trace_metrics,
    }


def run_representability_verification(archive_path: str, producer_ckpt: str, orderer_ckpt: str,
                                      task_ids: list[str], *, iters: int, horizon: int,
                                      eval_seeds: int, out_dir=None, debug_dir=None,
                                      log=print) -> list[dict]:
    """Load champion producer + orderer, pick `task_ids` from the mined archive, and emit one
    representability row per task. The producer is held at champion (frozen at EXP-0030).

    This is a **decision-gating verification run** (full-horizon per-cycle search, distill to
    convergence, `eval_seeds > 1`, multi-task) — a cloud run per CLAUDE.md, not a local smoke.
    When `out_dir` is given, writes the EXP layout (`summary.csv`, `spec.json`, `RESULT.md`) so a
    pod run can be pulled back into a fresh `EXP-NNNN` dir."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champ_order_model = load_scorer(orderer_ckpt)
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {
        "producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt,
        "task_ids": task_ids, "iters": iters, "horizon": horizon,
        "eval_seeds": eval_seeds, "search": _SEARCH,
        "debug_dir": str(debug_dir) if debug_dir is not None else None,
    }
    rows = []
    for tid in task_ids:
        row = representability_row(by_id[tid], prod_pol, champ_order_model,
                                   iters=iters, horizon=horizon, eval_seeds=eval_seeds,
                                   debug_dir=debug_dir, log=log)
        log(f"[repr] {row}")
        rows.append(row)
        if out_dir is not None:
            _emit_representability(out_dir, rows, meta={**meta, "complete": False,
                                                        "completed_task_ids": [r["task_id"] for r in rows]})
    if out_dir is not None:
        _emit_representability(out_dir, rows, meta={**meta, "complete": True,
                                                    "completed_task_ids": [r["task_id"] for r in rows]})
    return rows


def run_a0_probe(archive_path: str, producer_ckpt: str, orderer_ckpt: str, task_ids: list[str], *,
                 iters: int, horizon: int, eval_seeds: int, k_searches: int, out_dir=None,
                 debug_dir=None, log=print) -> list[dict]:
    """Stage A0 orderer-harvest gate (EXP-0036 amendment). Per probe task emit one decision row:
    H3 headroom baselines (champ / search-oracle / even_split / proportional units) + H1's VE of a
    single-search vs K-averaged distill. Champion producer frozen throughout (isolates the orderer).

    Decision-gating verification (full-horizon per-cycle search x K, distill to convergence,
    multi-seed/multi-task) => a CLOUD run per CLAUDE.md, not a local smoke. `out_dir` writes the
    pullable EXP layout (`summary.csv` + `spec.json` + `RESULT.md`)."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champ_order_model = load_scorer(orderer_ckpt)
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {
        "producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt, "task_ids": task_ids,
        "iters": iters, "horizon": horizon, "eval_seeds": eval_seeds, "k_searches": k_searches,
        "search": _SEARCH, "debug_dir": str(debug_dir) if debug_dir is not None else None,
    }
    rows = []
    for tid in task_ids:
        row = a0_probe_row(by_id[tid], prod_pol, champ_order_model, iters=iters, horizon=horizon,
                           eval_seeds=eval_seeds, k_searches=k_searches, debug_dir=debug_dir, log=log)
        log(f"[a0] {row}")
        rows.append(row)
        if out_dir is not None:
            _emit_a0(out_dir, rows, meta={**meta, "complete": False,
                                          "completed_task_ids": [r["task_id"] for r in rows]})
    if out_dir is not None:
        _emit_a0(out_dir, rows, meta={**meta, "complete": True,
                                      "completed_task_ids": [r["task_id"] for r in rows]})
    return rows


def _emit_a0(out_dir, rows: list[dict], meta: dict) -> None:
    """Write the Stage A0 EXP layout: `summary.csv` (per-task decision rows), `spec.json`, and a
    `RESULT.md` whose read order is headroom-first (`search_cu ~= even_split_cu` => STOP) then the
    VE jump (`ve_averaged` over `ve_single` => single target was tie-arbitrary noise => Stage A1)."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    def _m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    mean_head = _m("teacher_headroom")
    mean_single, mean_avg = _m("ve_single"), _m("ve_averaged")
    mean_named = _m("named_capture_frac")
    lines = [
        "# EXP-0036 Stage A0 orderer-harvest gate", "",
        "Learnability-gated orderer harvest (EXP-0036 amendment). Per probe task: H3 headroom",
        "baselines + a representation diagnostic + H1 variance-explained of a single-search vs",
        "K-averaged distill, champion producer frozen.", "",
        "**Read order:**", "",
        "1. **Headroom (H3, first):** `search_cu` vs `even_split_cu` per task. If oracle ~= even_split",
        "   there is no orderer lever in this regime => STOP (accept champion or re-mine tighter).",
        "2. **Representation:** `named_capture_frac` = share of the search headroom a per-cycle",
        "   NAMED-only search (even_split/proportional switching, no Gaussian samples) recovers.",
        "   ~1 => the lever is per-cycle named-policy choice => a learnable categorical/blend teacher",
        "   head captures it (switch representation). ~0 => the win is continuous routings => keep the",
        "   continuous target and denoise it (H1).",
        "3. **Denoise (H1):** does `ve_averaged` jump over `ve_single`? A jump => the single-search",
        "   target was tie-arbitrary noise the K-average cancels => proceed to Stage A1. Neither moves",
        "   but headroom exists => Stage A2 (deploy-time search). Neither moves, no headroom => track done.", "",
        f"**Means:** teacher_headroom={mean_head}, named_capture_frac={mean_named}, "
        f"ve_single={mean_single}, ve_averaged={mean_avg}.", "",
        "## Per-task", "", "```json", json.dumps(rows, indent=2), "```", "",
        "## Conclusions", "", "_TODO (human): read headroom first, then the VE jump; pick the branch._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def run_a0b_diag(archive_path: str, producer_ckpt: str, task_ids: list[str], *, iters: int,
                 horizon: int, eval_seeds: int, alphas: tuple[float, ...] = (0.75, 0.5, 0.25),
                 out_dir=None, log=print) -> list[dict]:
    """Stage A0b — the two pre-A1b checks A0's named-capture left open: D2 (does the alpha-blend
    family recover the continuous residual hard selection leaves?) and D1 (is the committed
    vocabulary choice predictable from the stream view on a held-out seed?). No orderer champion
    involved — both compare search variants / observation-predictability under the frozen producer.

    Decision-gating verification (full-horizon per-cycle search x 4 variants, choice fit,
    multi-seed/multi-task) => a CLOUD run per CLAUDE.md, not a local smoke. `out_dir` writes the
    pullable EXP layout (`summary.csv` + `spec.json` + `RESULT.md`)."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {
        "producer_ckpt": producer_ckpt, "task_ids": task_ids, "iters": iters, "horizon": horizon,
        "eval_seeds": eval_seeds, "alphas": list(alphas), "search": _SEARCH,
    }
    rows = []
    for tid in task_ids:
        row = a0b_diag_row(by_id[tid], prod_pol, iters=iters, horizon=horizon,
                           eval_seeds=eval_seeds, alphas=alphas, log=log)
        log(f"[a0b] {row}")
        rows.append(row)
        if out_dir is not None:
            _emit_a0b(out_dir, rows, meta={**meta, "complete": False,
                                           "completed_task_ids": [r["task_id"] for r in rows]})
    if out_dir is not None:
        _emit_a0b(out_dir, rows, meta={**meta, "complete": True,
                                       "completed_task_ids": [r["task_id"] for r in rows]})
    return rows


def _emit_a0b(out_dir, rows: list[dict], meta: dict) -> None:
    """Write the Stage A0b EXP layout: `summary.csv` (per-task decision rows), `spec.json`, and a
    `RESULT.md` whose read order is blend-capture (D2: is the alpha head worth having?) then
    held-out choice accuracy vs the majority floor (D1: is the categorical target learnable?)."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    def _m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    lines = [
        "# EXP-0036 Stage A0b pre-A1b checks (blend value + choice learnability)", "",
        "A0's `named_capture_frac` proved the orderer headroom is REACHABLE inside a named-policy",
        "vocabulary via the lookahead oracle. It did NOT prove (a) that the alpha-blend family adds",
        "anything over hard selection, nor (b) that the committed choice is predictable from the",
        "stream observation — the two assumptions Stage A1b's head rests on. This run tests both.", "",
        "**Read order:**", "",
        "1. **Blend value (D2):** `blend_capture_frac` vs `named_capture_frac`. A clear gain => the",
        "   continuous residual is blend-recoverable => keep the alpha head. ~equal => plain",
        "   categorical selection suffices => drop alpha.",
        "2. **Choice learnability (D1):** held-out `choice_top1` vs `choice_majority` (and the",
        "   decisive-margin split). Near-majority => the categorical target inherits the same",
        "   unpredictability wall (A1b would fail; fall back to A1 denoise / A2 deploy-time search).",
        "   Well above majority => A1b is green-lit.",
        "3. **Global-state probe (A0c, read only if D1 failed):** held-out `gchoice_top1` vs",
        "   `gchoice_majority` — the same protocol from the FULL global state, one sample per",
        "   searched cycle. Beats its majority floor => the D1 wall is missing CONTEXT (enrich the",
        "   orderer obs with global features). Fails too => the committed choice is rollout NOISE;",
        "   no observation can predict it — go soft labels / deploy-time search, not enrichment.", "",
        f"**Means:** named_capture_frac={_m('named_capture_frac')}, "
        f"blend_capture_frac={_m('blend_capture_frac')}, choice_top1={_m('choice_top1')}, "
        f"choice_majority={_m('choice_majority')}, "
        f"choice_top1_decisive={_m('choice_top1_decisive')}, "
        f"choice_majority_decisive={_m('choice_majority_decisive')}, "
        f"gchoice_top1={_m('gchoice_top1')}, gchoice_majority={_m('gchoice_majority')}, "
        f"gchoice_top1_decisive={_m('gchoice_top1_decisive')}, "
        f"gchoice_majority_decisive={_m('gchoice_majority_decisive')}.", "",
        "## Per-task", "", "```json", json.dumps(rows, indent=2), "```", "",
        "## Conclusions", "", "_TODO (human): read blend value first, then learnability; pick the branch._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def _emit_representability(out_dir, rows: list[dict], meta: dict) -> None:
    """Write the EXP layout for a pod pull: `summary.csv` (per-task rows), `spec.json` (run
    knobs), and a `RESULT.md` skeleton whose gate line is the mean `recovered_vs_teacher`
    (>= ~0.5 => the scorer head can represent the search => green-light the full v4 pod run)."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    tvals = [r["recovered_vs_teacher"] for r in rows if r["recovered_vs_teacher"] is not None]
    mean_t = statistics.mean(tvals) if tvals else None
    verdict = ("green-light full v4 pod run" if mean_t is not None and mean_t >= 0.5 else
               "architecture limit likely real — escalate (richer stream obs / deploy-time search)")
    lines = [
        "# EXP-0032 v4 orderer oracle-relabel representability", "",
        "Decision gate for v4 (EXP-0032): can the portable orderer scorer head represent the",
        "mining bar's per-cycle search routing? Distill a fresh orderer on each task's",
        "oracle-relabel targets (champion producer drives collection), then compare on-task",
        "champion / search-teacher / oracle / bar consumer units. Producer frozen at champion.", "",
        f"**Gate:** mean `recovered_vs_teacher` = "
        f"{mean_t:.3f} (>= ~0.5 => head represents the search)." if mean_t is not None
        else "**Gate:** no teacher headroom on these tasks (degenerate).", "",
        f"**Verdict:** {verdict}.", "",
        "## Per-task", "", "```json", json.dumps(rows, indent=2), "```", "",
        "## Conclusions", "", "_TODO (human): confirm go/no-go on the full v4 `run_stage4_replay` run._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def _qualifies(row: dict, floor_frac: float) -> bool:
    """A task is an informative representability probe iff its measured orderer headroom clears
    the floor (a fraction of `champ_cu`). Below the floor the recovery ratio is noise (EXP-0032:
    a 0.33u denominator turned a −10u miss into −31)."""
    f = row.get("headroom_frac")
    return f is not None and f >= floor_frac


def scan_headroom(archive_path: str, producer_ckpt: str, orderer_ckpt: str, *, horizon: int,
                  eval_seeds: int, floor_frac: float = 0.02, out_dir=None, log=print) -> list[dict]:
    """Measure per-task orderer headroom over the WHOLE archive under the frozen producer, rank by
    it, and flag the tasks that clear `floor_frac` — the valid representability probes. If few/none
    clear the floor, the augmented producer has already closed the mined gap and the orderer lever
    is largely spent (STOP, don't escalate). Cloud run (per-cycle search × archive)."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champ_order_model = load_scorer(orderer_ckpt)
    records = _load_archive(archive_path)
    rows = []
    for rec in records:
        row = headroom_row(rec, prod_pol, champ_order_model, horizon=horizon,
                           eval_seeds=eval_seeds, log=log)
        row["qualifies"] = _qualifies(row, floor_frac)
        log(f"[headroom] {row}")
        rows.append(row)
    rows.sort(key=lambda r: r["teacher_headroom"], reverse=True)
    if out_dir is not None:
        _emit_headroom(out_dir, rows, floor_frac, meta={
            "producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt,
            "horizon": horizon, "eval_seeds": eval_seeds, "floor_frac": floor_frac,
            "search": _SEARCH})
    return rows


def _emit_headroom(out_dir, rows: list[dict], floor_frac: float, meta: dict) -> None:
    """Write the headroom-scan EXP layout: `summary.csv` (per-task, headroom-sorted), `spec.json`,
    and a `RESULT.md` naming the qualifying probe task_ids (or, if few, the STOP signal)."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    qual = [r for r in rows if r.get("qualifies")]
    verdict = (f"{len(qual)}/{len(rows)} tasks clear the {floor_frac:.0%} headroom floor — "
               "re-run representability on these" if qual else
               f"NO task clears the {floor_frac:.0%} floor — the augmented producer already "
               "closed the mined orderer gap; the orderer lever is spent (STOP, don't escalate)")
    lines = [
        "# EXP-0032b orderer-headroom scan (probe re-selection)", "",
        "EXP-0032 was an invalid probe (near-zero headroom). This scan measures `teacher_headroom",
        "= champ_prod+search − champ_prod+champ_ord` over the whole archive under the FROZEN",
        f"EXP-0030 producer, and keeps tasks with `headroom_frac >= {floor_frac:.0%}`.", "",
        f"**Result:** {verdict}.", "",
        "## Qualifying probe task_ids", "",
        ("```\n" + "\n".join(r["task_id"] for r in qual) + "\n```") if qual else "_none_", "",
        "## Per-task (headroom-sorted)", "", "```json", json.dumps(rows, indent=2), "```", "",
        "## Conclusions", "",
        "_TODO (human): if qualifying tasks exist, re-run representability on them; else STOP._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="headroom-scan mode: measure teacher_headroom over the whole archive and "
                         "flag tasks clearing --floor-frac (probe re-selection); ignores --task-ids")
    ap.add_argument("--floor-frac", type=float, default=0.02,
                    help="with --scan: min headroom_frac (of champ_cu) for a task to qualify")
    ap.add_argument("--a0", action="store_true",
                    help="Stage A0 mode (EXP-0036): per --task-ids emit H3 headroom baselines + H1 "
                         "variance-explained of a single-search vs --k-searches averaged distill")
    ap.add_argument("--k-searches", type=int, default=4,
                    help="with --a0: independent searches averaged per cycle for the denoised target")
    ap.add_argument("--a0b", action="store_true",
                    help="Stage A0b mode: per --task-ids emit blend-capture (named+alpha-blend "
                         "search vs full search) + held-out committed-choice top-1 vs majority "
                         "(the two pre-A1b checks); needs no --orderer-ckpt")
    ap.add_argument("--alphas", default="0.75,0.5,0.25",
                    help="with --a0b: comma-separated ordering_blend alphas added to the vocabulary")
    ap.add_argument("--archive", required=True, help="mined task_archive.jsonl (Stage 3 output)")
    ap.add_argument("--producer-ckpt", required=True, help="champion producer score model (frozen)")
    ap.add_argument("--orderer-ckpt", default=None,
                    help="champion orderer score model (required unless --a0b)")
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated archived task ids (required unless --scan)")
    ap.add_argument("--iters", type=int, default=300, help="orderer distill iters")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--out-dir", default=None, help="EXP dir to write summary.csv + RESULT.md (pod pull)")
    ap.add_argument("--debug-dir", default=None,
                    help="optional dir for per-task scorer checkpoints, samples.pkl.gz, and deploy traces")
    args = ap.parse_args()

    if not args.a0b and not args.orderer_ckpt:
        ap.error("--orderer-ckpt is required unless --a0b")

    if args.a0b:
        if not args.task_ids:
            ap.error("--task-ids is required with --a0b")
        rows = run_a0b_diag(
            args.archive, args.producer_ckpt, [t for t in args.task_ids.split(",")],
            iters=args.iters, horizon=args.horizon, eval_seeds=args.eval_seeds,
            alphas=tuple(float(a) for a in args.alphas.split(",")), out_dir=args.out_dir)
        print("\ntask_id,search_cu,even_split_cu,search_named_cu,search_blend_cu,"
              "named_capture_frac,blend_capture_frac,choice_top1,choice_majority,"
              "choice_top1_decisive,choice_majority_decisive")
        for r in rows:
            print(f"{r['task_id']},{r['search_cu']},{r['even_split_cu']},{r['search_named_cu']},"
                  f"{r['search_blend_cu']},{r['named_capture_frac']},{r['blend_capture_frac']},"
                  f"{r['choice_top1']},{r['choice_majority']},{r['choice_top1_decisive']},"
                  f"{r['choice_majority_decisive']}")
        return

    if args.scan:
        rows = scan_headroom(args.archive, args.producer_ckpt, args.orderer_ckpt,
                             horizon=args.horizon, eval_seeds=args.eval_seeds,
                             floor_frac=args.floor_frac, out_dir=args.out_dir)
        qual = [r for r in rows if r.get("qualifies")]
        print(f"\n{len(qual)}/{len(rows)} tasks clear the {args.floor_frac:.0%} headroom floor")
        for r in qual:
            print(f"  {r['teacher_headroom']:>6} ({r['headroom_frac']:.1%})  {r['task_id']}")
        return

    if args.a0:
        if not args.task_ids:
            ap.error("--task-ids is required with --a0")
        rows = run_a0_probe(
            args.archive, args.producer_ckpt, args.orderer_ckpt,
            [t for t in args.task_ids.split(",")], iters=args.iters, horizon=args.horizon,
            eval_seeds=args.eval_seeds, k_searches=args.k_searches,
            out_dir=args.out_dir, debug_dir=args.debug_dir)
        print("\ntask_id,search_cu,even_split_cu,search_named_cu,named_capture_frac,ve_single,ve_averaged")
        for r in rows:
            print(f"{r['task_id']},{r['search_cu']},{r['even_split_cu']},{r['search_named_cu']},"
                  f"{r['named_capture_frac']},{r['ve_single']},{r['ve_averaged']}")
        return

    if not args.task_ids:
        ap.error("--task-ids is required unless --scan")
    rows = run_representability_verification(
        args.archive, args.producer_ckpt, args.orderer_ckpt,
        [t for t in args.task_ids.split(",")], iters=args.iters,
        horizon=args.horizon, eval_seeds=args.eval_seeds,
        out_dir=args.out_dir, debug_dir=args.debug_dir)
    print("\ntask_id,champ_cu,search_cu,oracle_cu,bar_cu,recovered_vs_teacher,recovered_vs_bar")
    for r in rows:
        print(f"{r['task_id']},{r['champ_cu']},{r['search_cu']},{r['oracle_cu']},{r['bar_cu']},"
              f"{r['recovered_vs_teacher']},{r['recovered_vs_bar']}")
    tvals = [r["recovered_vs_teacher"] for r in rows if r["recovered_vs_teacher"] is not None]
    if tvals:
        print(f"\nmean recovered_vs_teacher: {statistics.mean(tvals):.3f} "
              f"(>= ~0.5 => head can represent the search; green-light pod run)")


if __name__ == "__main__":
    main()
