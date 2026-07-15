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

import numpy as np

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
    train_samples, train_gsamples, _train_traces, vocab = collect_choice_samples(
        [cfg] * len(train_seeds), train_seeds, **collect_kw)
    heldout_samples, heldout_gsamples, _heldout_traces, _ = collect_choice_samples(
        [cfg], [seeds[-1]], **collect_kw)
    if train_samples:
        model = fit_choice_model(OrdererChoiceModel(n_choices=len(vocab)), train_samples,
                                 iters=iters, log=log)
        m = choice_metrics(model, heldout_samples)
    else:
        m = {"top1": None, "majority": None, "top1_decisive": None, "majority_decisive": None,
             "n_decisive": 0}
    # A0c discriminator: same protocol from the ADR-0004 global snapshot (one sample per
    # searched cycle). Beats the majority floor where the stream view failed => the D1 wall is
    # missing CONTEXT. NOTE the input is a lossy ROLL-UP (encode_global sums capacity across
    # types, no queue/TTL/inventory-composition/history) at ~200 train cycles/task, so a fail
    # only kills enrichment with THESE features — not "no observation can help" (A0d probes
    # a lossless serialization ± history before that call is made).
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


def a0t_diag_row(
    record: TaskRecord, prod_pol, champion_pol, *, iters: int, horizon: int, eval_seeds: int,
    alphas: tuple[float, ...] = (0.75, 0.5, 0.25), log=lambda *a, **k: None,
) -> tuple[dict, dict]:
    """A0t trace-regret row — the read A0b/A0c's aggregate top-1 could not give: WHICH held-out
    decisions a student gets wrong vs the search teacher and what each miss COSTS.

    Per held-out searched cycle (units = lookahead-window rollout units from that state, a local
    proxy, not episode units):
    - probe_regret  = units[teacher argmax] - units[global-probe pick]   (>= 0 by construction)
    - champ_regret  = units[teacher argmax] - units[champion rollout]    (< 0 = champion beats
      the whole vocabulary at that state)
    - local headroom = units[teacher argmax] - units[even_split] — the normalizer: regret_frac
      is the share of the teacher's per-cycle edge a student loses, weighting misses by cost
      instead of counting them. Ties that don't matter stop polluting the read.

    Returns `(row, detail)`: `row` = per-task aggregates for summary.csv; `detail` = per-cycle
    records (candidate table, picks, regrets, state snapshot) for artifacts + the top-regret
    dump a human reads to infer missing features. Champion producer frozen throughout."""
    import torch

    from ..train.orderer_scorer import GlobalChoiceModel, collect_choice_samples, fit_choice_model

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    if len(seeds) < 2:
        raise ValueError("a0t needs eval_seeds >= 2 (train seeds + one held-out seed)")
    collect_kw = dict(prod=prod_pol, alphas=alphas, lookahead=_SEARCH["lookahead"],
                      horizon=horizon, champion=champion_pol, log=log)
    _, train_gsamples, train_traces, vocab = collect_choice_samples(
        [cfg] * (len(seeds) - 1), list(seeds[:-1]), **collect_kw)
    _, heldout_gsamples, heldout_traces, _ = collect_choice_samples(
        [cfg], [seeds[-1]], **collect_kw)

    gmodel = GlobalChoiceModel(n_choices=len(vocab), state_dim=train_gsamples[0].view.shape[0])
    gmodel.set_norm([g.view for g in train_gsamples])
    fit_choice_model(gmodel, train_gsamples, iters=iters, log=log)
    gmodel.eval()

    even = vocab.index("even_split")
    cycles: list[dict] = []
    for g, t in zip(heldout_gsamples, heldout_traces):
        pred = int(torch.argmax(gmodel.forward_view(g.view)).item())
        best = t.cand_units[t.choice]
        cycles.append({
            "seed": t.seed, "cycle": t.cycle, "cand_units": t.cand_units, "choice": t.choice,
            "margin": t.margin, "probe_pred": pred,
            "probe_regret": best - t.cand_units[pred],
            "champion_units": t.champion_units,
            "champ_regret": (best - t.champion_units if t.champion_units is not None else None),
            "local_headroom": best - t.cand_units[even],
            "state": t.state,
        })
    p_reg = [c["probe_regret"] for c in cycles]
    c_reg = [c["champ_regret"] for c in cycles if c["champ_regret"] is not None]
    head = sum(c["local_headroom"] for c in cycles)
    row = {
        "task_id": record.task_id, "world_seed": record.world_seed, "vocab": vocab,
        "n_traces_train": len(train_traces), "n_traces_heldout": len(heldout_traces),
        "probe_top1": round(statistics.mean(
            1.0 if c["probe_pred"] == c["choice"] else 0.0 for c in cycles), 4) if cycles else None,
        "probe_regret_total": round(sum(p_reg), 2), "probe_regret_mean": round(
            statistics.mean(p_reg), 4) if p_reg else None,
        "probe_regret_max": round(max(p_reg), 2) if p_reg else None,
        "n_probe_miss": sum(1 for r in p_reg if r > 0),
        "champ_regret_total": round(sum(c_reg), 2) if c_reg else None,
        "champ_regret_mean": round(statistics.mean(c_reg), 4) if c_reg else None,
        "champ_regret_max": round(max(c_reg), 2) if c_reg else None,
        "n_champ_worse": sum(1 for r in c_reg if r > 0),
        "n_champ_better": sum(1 for r in c_reg if r < 0),
        "local_headroom_total": round(head, 2),
        "probe_regret_frac": round(sum(p_reg) / head, 4) if head > 1e-9 else None,
        "champ_regret_frac": round(sum(c_reg) / head, 4) if (c_reg and head > 1e-9) else None,
    }
    detail = {"task_id": record.task_id, "world_seed": record.world_seed, "vocab": vocab,
              "heldout_seed": seeds[-1], "cycles": cycles}
    return row, detail


# A0f cumulative feature ladder (HANDOVER Task 1 block order). `gstate` is the lossy A0c
# reference input, fit standalone; the ladder then grows from the local stream block so each
# row isolates ONE added observation family. `history` stacks the previous k `compact` vectors.
_A0F_LADDER = ("per_type_capacity", "order_queue", "inventory", "inbound", "history",
               "relational")


def _a0f_block_sets() -> list[tuple[str, tuple[str, ...]]]:
    sets: list[tuple[str, tuple[str, ...]]] = [("gstate", ("gstate",)), ("stream", ("stream",))]
    cum = ["stream"]
    for b in _A0F_LADDER:
        cum.append(b)
        sets.append((f"+{b}", tuple(cum)))
    return sets


def a0f_feature_row(
    record: TaskRecord, prod_pol, *, iters: int, horizon: int, eval_seeds: int,
    alphas: tuple[float, ...] = (0.75, 0.5, 0.25), k_history: int = 4,
    log=lambda *a, **k: None,
) -> tuple[list[dict], dict]:
    """A0f missing-observation probe (EXP-0036 HANDOVER Task 1): does the D1 wall come from
    missing STATE, and if so which observation family carries it? Per cumulative block set,
    fit a `ValueTableModel` (full candidate table target — no hard-argmax tie noise, HANDOVER
    Task 4 target form) on the train-seed cycles and read held-out `regret_frac`. A block
    that sharply drops regret names the missing family; none dropping it (relational
    included) pushes the diagnosis to label/objective noise or trajectory coverage (Task 3).
    `train_regret_frac` separates "cannot even fit the train states" (representation/label
    wall) from "fits train, fails held-out" (coverage/generalization wall).

    Returns `(rows, detail)`: one row per block set for summary.csv; `detail` = both trace
    sets serialized (tables, snapshots, features) so probes can be re-fit locally without a
    pod re-run (the A0t analysis pain: ad-hoc scripts over prose)."""
    import torch

    from ..train.orderer_scorer import (
        ValueTableModel, collect_choice_samples, fit_value_table, history_features,
        value_table_metrics)

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    if len(seeds) < 2:
        raise ValueError("a0f needs eval_seeds >= 2 (train seeds + one held-out seed)")
    collect_kw = dict(prod=prod_pol, alphas=alphas, lookahead=_SEARCH["lookahead"],
                      horizon=horizon, with_features=True, log=log)
    _, _, train_traces, vocab = collect_choice_samples(
        [cfg] * (len(seeds) - 1), list(seeds[:-1]), **collect_kw)
    _, _, held_traces, _ = collect_choice_samples([cfg], [seeds[-1]], **collect_kw)
    if not train_traces or not held_traces:
        raise ValueError(f"{record.task_id}: no searched cycles collected")

    even = vocab.index("even_split")
    y_train = np.asarray([t.cand_units for t in train_traces], dtype=np.float32)
    y_held = np.asarray([t.cand_units for t in held_traces], dtype=np.float32)
    hist_train = history_features(train_traces, k=k_history)
    hist_held = history_features(held_traces, k=k_history)

    def matrix(traces, hist, blocks: tuple[str, ...]) -> np.ndarray:
        cols = [hist if b == "history" else np.stack([t.features[b] for t in traces])
                for b in blocks]
        return np.concatenate(cols, axis=1)

    rows: list[dict] = []
    for name, blocks in _a0f_block_sets():
        x_train = matrix(train_traces, hist_train, blocks)
        x_held = matrix(held_traces, hist_held, blocks)
        torch.manual_seed(0)  # same init across block sets — regret deltas read as feature value
        model = ValueTableModel(state_dim=x_train.shape[1], n_choices=len(vocab))
        model.set_norm(x_train)
        fit_value_table(model, x_train, y_train, iters=iters, log=log)
        m_train = value_table_metrics(model, x_train, y_train, even)
        m_held = value_table_metrics(model, x_held, y_held, even)
        row = {
            "task_id": record.task_id, "world_seed": record.world_seed,
            "block_set": name, "blocks": list(blocks), "dim": int(x_train.shape[1]),
            "n_train": len(train_traces), "n_heldout": len(held_traces),
            "train_regret_frac": m_train["regret_frac"], "train_top1": m_train["top1"],
            "train_winner_hit": m_train["winner_hit"],
            "regret_frac": m_held["regret_frac"], "top1": m_held["top1"],
            "winner_hit": m_held["winner_hit"], "regret_total": m_held["regret_total"],
            "headroom_total": m_held["headroom_total"],
        }
        log(f"[a0f] {row}")
        rows.append(row)

    def serialize(traces) -> list[dict]:
        return [{"seed": t.seed, "cycle": t.cycle, "cand_units": t.cand_units,
                 "choice": t.choice, "margin": t.margin, "state": t.state,
                 "features": {k: v.tolist() for k, v in t.features.items()}}
                for t in traces]

    detail = {"task_id": record.task_id, "world_seed": record.world_seed, "vocab": vocab,
              "heldout_seed": seeds[-1], "k_history": k_history,
              "train": serialize(train_traces), "heldout": serialize(held_traces)}
    return rows, detail


def run_a0f_probe(archive_path: str, producer_ckpt: str, task_ids: list[str], *, iters: int,
                  horizon: int, eval_seeds: int, alphas: tuple[float, ...] = (0.75, 0.5, 0.25),
                  k_history: int = 4, out_dir=None, log=print) -> list[dict]:
    """Stage A0f missing-observation probe over archived tasks (HANDOVER Task 1). Needs no
    orderer ckpt — the probe competes against the search teacher, not the champion.

    Decision-gating verification (full-horizon per-cycle search collection, multi-seed/
    multi-task, distill per block set) => a CLOUD run per CLAUDE.md, not a local smoke."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {"producer_ckpt": producer_ckpt, "task_ids": task_ids, "iters": iters,
            "horizon": horizon, "eval_seeds": eval_seeds, "alphas": list(alphas),
            "k_history": k_history, "search": _SEARCH}
    rows: list[dict] = []
    details: list[dict] = []
    for tid in task_ids:
        t_rows, detail = a0f_feature_row(by_id[tid], prod_pol, iters=iters, horizon=horizon,
                                         eval_seeds=eval_seeds, alphas=alphas,
                                         k_history=k_history, log=log)
        rows += t_rows
        details.append(detail)
        if out_dir is not None:
            _emit_a0f(out_dir, rows, details, meta={**meta, "complete": False,
                                                    "completed_task_ids": sorted({r['task_id'] for r in rows})})
    if out_dir is not None:
        _emit_a0f(out_dir, rows, details, meta={**meta, "complete": True,
                                                "completed_task_ids": sorted({r['task_id'] for r in rows})})
    return rows


def _emit_a0f(out_dir, rows: list[dict], details: list[dict], meta: dict) -> None:
    """Write the Stage A0f EXP layout. RESULT.md leads with the Task 1 definition of done —
    a per-task table of held-out `regret_frac` per cumulative feature block — plus the branch
    guidance (sharp drop => missing family; no drop incl. relational => labels/coverage).
    Full traces with features -> `artifacts/feature_traces.json.gz` for local re-fitting."""
    import gzip
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")
    with gzip.open(out / "artifacts" / "feature_traces.json.gz", "wt") as f:
        json.dump(details, f)

    lines = [
        "# EXP-0036 Stage A0f missing-observation probe (which feature block carries the choice)", "",
        "HANDOVER Task 1: the orderer fits its training seeds and fails held-out — is the wall",
        "missing STATE, missing trajectory coverage, or bad labels? Per cumulative block set a",
        "value-table probe (full candidate table target, no hard-argmax tie noise) is fit on the",
        "train-seed cycles; every row reports held-out regret_frac — the share of the teacher's",
        "per-cycle edge over even_split the probe loses (cost-weighted, ties free).", "",
        "**Read order:**", "",
        "1. Per task, scan `regret_frac` down the ladder. A block that sharply reduces it names",
        "   the missing observation family — that block is the enrichment candidate.",
        "2. No flat block reduces it but `+relational` does => the signal is relational, compress",
        "   that block into the smallest stable feature set (Task 2).",
        "3. Nothing reduces it: check `train_regret_frac`. High => the target is not even fittable",
        "   on train states with full features (label/objective noise — Task 4 soft targets).",
        "   Low => fits train, fails held-out => trajectory/coverage wall (Task 3 DAgger states).",
        "4. `gstate` row = the lossy A0c input under the VALUE target — compares target forms",
        "   at fixed features against the A0t hard-argmax probe (regret_frac 0.98/0.86/0.56).", "",
        "## Regret ladder per task", "",
    ]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    for tid, t_rows in by_task.items():
        lines += [f"### {tid}", "",
                  "```",
                  f"{'block_set':<22} {'dim':>5} {'train_rf':>9} {'heldout_rf':>10} "
                  f"{'top1':>6} {'winner_hit':>10}"]
        for r in t_rows:
            fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
            lines.append(f"{r['block_set']:<22} {r['dim']:>5} {fmt(r['train_regret_frac']):>9} "
                         f"{fmt(r['regret_frac']):>10} {fmt(r['top1']):>6} "
                         f"{fmt(r['winner_hit']):>10}")
        lines += ["```", ""]
    lines += ["## Per-row detail", "", "```json", json.dumps(rows, indent=2), "```", "",
              "## Conclusions", "",
              "_TODO (human): name the missing block family (or the labels/coverage branch)",
              "from the ladder; pick Task 2/3/4 accordingly._", ""]
    (out / "RESULT.md").write_text("\n".join(lines))


# --- Stage A0w: window/episode-level blend-policy distillation (HANDOVER Task 5) ---
# A0f/A0g fixed the hard-label problem but still predicted a PER-CYCLE candidate table whose
# winner ordering does not transfer across demand seeds. A0w changes the target time scale:
# the value of each CONSTANT blend policy over an episode / rolling window. One row per
# episode/window (not per searched cycle); the set head is deployed with a conservative
# switch rule (default blend, switch only when the predicted advantage is large). The report
# leads with the fixed-blend baselines: if a constant blend already closes most of the gap,
# the learned model must beat that constant, not just champion/even_split.

_A0W_WINDOWS = (5, 10, 20)


def _fixed_blend_episode(cfg, prod_pol, order_pol, *, seed: int, horizon: int,
                         collect_features: bool) -> dict:
    """One full episode under a FIXED ordering policy + frozen champion producer. Returns
    per-cycle consumer-unit `deltas`, the total, whether the system died, and (only when
    asked, for the reference trajectory) the per-cycle `cycle_set_features` snapshot taken at
    each cycle start pre-decision — matching what the deployed meta-orderer observes."""
    from .joint_stack import joint_stack_env
    from ..train.orderer_scorer import cycle_set_features

    env = joint_stack_env(cfg, horizon)
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    cur = -1
    marks: list[float] = []      # consumer units at the start of each new cycle
    feats: list[dict] = []
    for agent in env.agent_iter(max_iter=max_iter):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        if env.current_cycle != cur:
            cur = env.current_cycle
            marks.append(env.episode_stats["consumer_completed_units"])
            if collect_features:
                feats.append(cycle_set_features(env))
        role = env._role[agent][0]
        env.step(prod_pol(env, agent) if role == "prod" else order_pol(env, agent))
    total = float(env.episode_stats["consumer_completed_units"])
    bounds = marks + [total]
    deltas = [float(bounds[i + 1] - bounds[i]) for i in range(len(marks))]
    return {"deltas": deltas, "feats": feats, "total": total,
            "dead": bool(env.system.is_dead() or env.episode_stats.get("dead"))}


def episode_value_table(totals_by_policy: dict[str, float], order: list[str]) -> np.ndarray:
    """Full-episode value table: one true total-units entry per candidate, in vocab order."""
    return np.asarray([totals_by_policy[name] for name in order], dtype=np.float32)


def window_value_rows(deltas_by_policy: dict[str, list[float]], ref_len: int, window: int,
                      order: list[str]) -> list[tuple[int, np.ndarray]]:
    """Rolling-window value tables aligned to the reference trajectory's cycle index. Each
    policy's per-cycle deltas are padded to `ref_len` with 0.0 (post-death = no units) so the
    window sums are comparable across constant policies; one row per full window start
    `0..ref_len-window`. Returns `(start_cycle, target_vector)` in vocab order."""
    if window > ref_len:
        return []
    padded = {name: (list(d)[:ref_len] + [0.0] * max(0, ref_len - len(d)))
              for name, d in deltas_by_policy.items()}
    rows: list[tuple[int, np.ndarray]] = []
    for start in range(0, ref_len - window + 1):
        tgt = np.asarray([float(sum(padded[name][start:start + window])) for name in order],
                         dtype=np.float32)
        rows.append((start, tgt))
    return rows


def _const_regret_frac(tables: np.ndarray, pick: int, even: int) -> float | None:
    """Held-out regret_frac of ALWAYS picking candidate `pick` (the constant-blend baseline),
    normalized by the even_split headroom — same frame as the model's `regret_frac`."""
    y = np.asarray(tables, dtype=np.float64)
    head = float((y.max(axis=1) - y[:, even]).sum())
    reg = float((y.max(axis=1) - y[:, pick]).sum())
    return round(reg / head, 4) if head > 1e-9 else None


def _decision_regret_frac(pred: np.ndarray, tables: np.ndarray, even: int, default_idx: int,
                          threshold: float):
    """Held-out regret_frac and switch rate of the conservative deploy rule applied to the
    model's predicted tables (default blend unless a candidate clears `threshold`)."""
    from ..train.orderer_scorer import blend_decision

    y = np.asarray(tables, dtype=np.float64)
    picks = np.asarray([blend_decision(p, default_idx, threshold) for p in pred])
    head = float((y.max(axis=1) - y[:, even]).sum())
    reg = float((y.max(axis=1) - y[np.arange(len(y)), picks]).sum())
    switch = float((picks != default_idx).mean()) if len(picks) else 0.0
    return (round(reg / head, 4) if head > 1e-9 else None), round(switch, 4)


def a0w_task5_row(record: TaskRecord, prod_pol, champ_order_model, *, iters: int, horizon: int,
                  eval_seeds: int, alphas: tuple[float, ...], windows: tuple[int, ...],
                  threshold: float, default_blend: str, deploy_window: int,
                  log=lambda *a, **k: None) -> tuple[list[dict], dict]:
    """Per-task A0w row set (HANDOVER Task 5). Collects each fixed blend policy's full-episode
    per-cycle unit deltas (frozen champion producer, archive-anchored seeds), builds
    episode-level and rolling-window value tables, fits the producer-count-agnostic set head
    per granularity, and reads held-out regret vs the best constant-blend baseline and the
    conservative decision rule. The `deploy` row rolls out champion producer + the learned
    meta-orderer on the held-out seed and reports units/death/closure vs search and vs the
    default and best-constant blends."""
    import torch

    from .joint_stack import joint_stack_env
    from .orderer_oracle import blend_vocabulary
    from .runner import run_episode
    from ..train.orderer_scorer import (
        ProducerSetValueModel, aggregate_set_window, fit_set_value_table,
        set_table_ordering_policy, set_value_table_metrics)

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    if len(seeds) < 2:
        raise ValueError("a0w needs eval_seeds >= 2 (train seeds + one held-out seed)")
    vocab = blend_vocabulary(alphas)
    names = list(vocab)
    even = names.index("even_split")
    ref_name = default_blend if default_blend in names else names[0]
    default_idx = names.index(ref_name)

    per_seed: dict[int, dict] = {}
    for es in seeds:
        deltas_by_policy: dict[str, list[float]] = {}
        totals_by_policy: dict[str, float] = {}
        ref_feats: list[dict] = []
        dead_any = False
        for name in names:
            d = _fixed_blend_episode(cfg, prod_pol, vocab[name], seed=es, horizon=horizon,
                                     collect_features=(name == ref_name))
            deltas_by_policy[name] = d["deltas"]
            totals_by_policy[name] = d["total"]
            dead_any = dead_any or d["dead"]
            if name == ref_name:
                ref_feats = d["feats"]
        per_seed[es] = {"deltas": deltas_by_policy, "totals": totals_by_policy,
                        "feats": ref_feats, "dead": dead_any}
        log(f"[a0w] {record.task_id} seed {es}: ref_cycles={len(ref_feats)} "
            f"totals={{{', '.join(f'{n}:{totals_by_policy[n]:.0f}' for n in names)}}}")

    def rows_for(granularity: str, window: int | None, es: int) -> list[tuple[dict, np.ndarray]]:
        ps = per_seed[es]
        if window is None:  # episode: one row, feature = summary over the whole ref trajectory
            if not ps["feats"]:
                return []
            return [(aggregate_set_window(ps["feats"]),
                     episode_value_table(ps["totals"], names))]
        out = []
        for start, tgt in window_value_rows(ps["deltas"], len(ps["feats"]), window, names):
            out.append((aggregate_set_window(ps["feats"][start:start + window]), tgt))
        return out

    grains: list[tuple[str, int | None]] = [("episode", None)] + [
        (f"window{w}", w) for w in windows]
    rows: list[dict] = []
    detail: dict = {"task_id": record.task_id, "world_seed": record.world_seed,
                    "vocab": names, "heldout_seed": seeds[-1], "granularities": {}}
    for gname, window in grains:
        tr = [r for es in seeds[:-1] for r in rows_for(gname, window, es)]
        ho = rows_for(gname, window, seeds[-1])
        if not tr or not ho:
            log(f"[a0w] {record.task_id} {gname}: skipped (train={len(tr)} heldout={len(ho)})")
            continue
        ctx_tr = np.stack([f["context"] for f, _ in tr])
        prod_tr = np.stack([f["producers"] for f, _ in tr])
        y_tr = np.stack([t for _, t in tr])
        ctx_ho = np.stack([f["context"] for f, _ in ho])
        prod_ho = np.stack([f["producers"] for f, _ in ho])
        y_ho = np.stack([t for _, t in ho])

        torch.manual_seed(0)
        model = ProducerSetValueModel(context_dim=ctx_tr.shape[1],
                                      producer_dim=prod_tr.shape[2], n_choices=len(names))
        model.set_norm(ctx_tr, prod_tr.reshape(-1, prod_tr.shape[2]))
        fit_set_value_table(model, ctx_tr, prod_tr, y_tr, iters=iters, log=log)
        m_tr = set_value_table_metrics(model, ctx_tr, prod_tr, y_tr, even)
        m_ho = set_value_table_metrics(model, ctx_ho, prod_ho, y_ho, even)

        best_const = int(y_tr.mean(axis=0).argmax())  # best fixed blend chosen on TRAIN
        const_rf = _const_regret_frac(y_ho, best_const, even)
        with torch.no_grad():
            pred_ho = model.forward_tables(torch.from_numpy(ctx_ho.astype(np.float32)),
                                           torch.from_numpy(prod_ho.astype(np.float32))).numpy()
        dec_rf, switch = _decision_regret_frac(pred_ho, y_ho, even, default_idx, threshold)
        row = {
            "task_id": record.task_id, "world_seed": record.world_seed, "granularity": gname,
            "window": window if window is not None else "", "context_dim": ctx_tr.shape[1],
            "producer_dim": prod_tr.shape[2], "n_train": len(tr), "n_heldout": len(ho),
            "train_regret_frac": m_tr["regret_frac"], "regret_frac": m_ho["regret_frac"],
            "top1": m_ho["top1"], "winner_hit": m_ho["winner_hit"],
            "best_const_blend": names[best_const], "const_regret_frac": const_rf,
            "decision_regret_frac": dec_rf, "decision_switch_rate": switch,
            "default_blend": ref_name, "threshold": threshold,
        }
        log(f"[a0w] {row}")
        rows.append(row)
        detail["granularities"][gname] = {
            "best_const_blend": names[best_const], "regret_frac": m_ho["regret_frac"],
            "const_regret_frac": const_rf, "decision_regret_frac": dec_rf}
        if window == deploy_window:
            detail["deploy_model"] = model  # reused below for the rollout

    # Deployed rollout on the held-out seed: champion producer + learned meta-orderer.
    ho_seed = seeds[-1]
    deploy_model = detail.pop("deploy_model", None)
    if deploy_model is not None:
        learned = set_table_ordering_policy(deploy_model, vocab, default_idx, threshold,
                                            deploy_window)
        env = joint_stack_env(cfg, horizon)
        learned_stats = run_episode(env, prod_pol, learned, seed=ho_seed)
        default_stats = run_episode(joint_stack_env(cfg, horizon), prod_pol,
                                    vocab[ref_name], seed=ho_seed)
        best_const_name = detail["granularities"].get(f"window{deploy_window}", {}).get(
            "best_const_blend", ref_name)
        bc_stats = run_episode(joint_stack_env(cfg, horizon), prod_pol,
                               vocab[best_const_name], seed=ho_seed)
        champ_cu, search_cu = _champ_search_units(cfg, prod_pol, champ_order_model,
                                                  horizon=horizon, seeds=[ho_seed])
        head = search_cu - champ_cu
        learned_cu = learned_stats["consumer_completed_units"]
        rows.append({
            "task_id": record.task_id, "world_seed": record.world_seed, "granularity": "deploy",
            "window": deploy_window, "heldout_seed": ho_seed,
            "learned_cu": round(learned_cu, 2), "default_blend_cu": round(default_stats["consumer_completed_units"], 2),
            "best_const_cu": round(bc_stats["consumer_completed_units"], 2),
            "champ_cu": round(champ_cu, 2), "search_cu": round(search_cu, 2),
            "learned_dead": bool(learned_stats["dead"]),
            "default_blend": ref_name, "best_const_blend": best_const_name,
            "closure_vs_search": round((learned_cu - champ_cu) / head, 4) if head > 1e-9 else None,
        })
    detail.pop("granularities", None)
    return rows, detail


def run_a0w_task5(archive_path: str, producer_ckpt: str, orderer_ckpt: str, task_ids: list[str],
                  *, iters: int, horizon: int, eval_seeds: int,
                  alphas: tuple[float, ...] = (0.75, 0.5, 0.25), windows: tuple[int, ...] = _A0W_WINDOWS,
                  threshold: float = 0.25, default_blend: str = "blend_0.5", deploy_window: int = 10,
                  out_dir=None, log=print) -> list[dict]:
    """Stage A0w window/episode blend-policy distillation over archived tasks (HANDOVER Task 5).

    CLOUD run (CLAUDE.md HARD RULE): full-horizon fixed-policy episodes for every candidate x
    seed, per-granularity distill, a deployed rollout + search ceiling per task — a
    decision-gating, multi-seed/multi-task verification. Build/TDD local on a tiny subset;
    launch the full 81-row archive run wrapped with run_pod_training.py."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champ_order_model = load_scorer(orderer_ckpt)
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {"producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt, "task_ids": task_ids,
            "iters": iters, "horizon": horizon, "eval_seeds": eval_seeds, "alphas": list(alphas),
            "windows": list(windows), "threshold": threshold, "default_blend": default_blend,
            "deploy_window": deploy_window, "search": _SEARCH}
    rows: list[dict] = []
    details: list[dict] = []
    for tid in task_ids:
        t_rows, detail = a0w_task5_row(
            by_id[tid], prod_pol, champ_order_model, iters=iters, horizon=horizon,
            eval_seeds=eval_seeds, alphas=alphas, windows=windows, threshold=threshold,
            default_blend=default_blend, deploy_window=deploy_window, log=log)
        rows += t_rows
        details.append(detail)
        if out_dir is not None:
            _emit_a0w(out_dir, rows, details, meta={**meta, "complete": False})
    if out_dir is not None:
        _emit_a0w(out_dir, rows, details, meta={**meta, "complete": True})
    return rows


def _emit_a0w(out_dir, rows: list[dict], details: list[dict], meta: dict) -> None:
    """Write the Stage A0w EXP layout. RESULT.md leads with the fixed-blend baseline frame
    (learned model must beat the best CONSTANT blend, not just champion/even_split), separates
    episode- from window-level regret, and reports the deployed rollout + closure."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    fit_rows = [r for r in rows if r.get("granularity") != "deploy"]
    dep_rows = [r for r in rows if r.get("granularity") == "deploy"]
    lines = [
        "# EXP-0036 Stage A0w window/episode blend-policy distillation (HANDOVER Task 5)", "",
        "A0f/A0g predicted a PER-CYCLE candidate table whose winner ordering did not transfer",
        "across demand seeds. A0w changes the target time scale: the value of each CONSTANT",
        "blend policy over an episode / rolling window, one row per window. The set head is",
        "deployed with a conservative switch rule (default blend unless the predicted advantage",
        "clears a threshold).", "",
        "**Read order (definition of done):**", "",
        "1. Compare `regret_frac` (learned model) against `const_regret_frac` (best single blend",
        "   chosen on train). If a constant blend already closes most of the gap, the learned",
        "   model must beat that constant — not just champion/even_split.",
        "2. `decision_regret_frac` = the conservative deploy rule's held-out regret (what ships).",
        "3. Episode vs window rows are separated; `deploy` rows are the rollout, not table error.",
        "4. `closure_vs_search = (learned_cu - champ_cu)/(search_cu - champ_cu)` on the held-out",
        "   seed; compare `learned_cu` to `default_blend_cu` and `best_const_cu`.", "",
        "## Held-out regret per granularity", "",
        "```",
        f"{'task':<28} {'grain':<9} {'train_rf':>9} {'heldout_rf':>10} {'const_rf':>9} "
        f"{'decision_rf':>11} {'best_const':>12} {'switch':>7}",
    ]
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    for r in fit_rows:
        lines.append(f"{r['task_id'][-28:]:<28} {r['granularity']:<9} "
                     f"{fmt(r['train_regret_frac']):>9} {fmt(r['regret_frac']):>10} "
                     f"{fmt(r['const_regret_frac']):>9} {fmt(r['decision_regret_frac']):>11} "
                     f"{str(r['best_const_blend']):>12} {fmt(r.get('decision_switch_rate')):>7}")
    lines += ["```", "", "## Deployed rollout (held-out seed)", "", "```",
              f"{'task':<28} {'learned':>8} {'default':>8} {'best_const':>10} {'champ':>7} "
              f"{'search':>7} {'closure':>8} {'dead':>5}"]
    for r in dep_rows:
        lines.append(f"{r['task_id'][-28:]:<28} {r['learned_cu']:>8} {r['default_blend_cu']:>8} "
                     f"{r['best_const_cu']:>10} {r['champ_cu']:>7} {r['search_cu']:>7} "
                     f"{fmt(r['closure_vs_search']):>8} {str(r['learned_dead']):>5}")
    lines += ["```", "", "## Per-row detail", "", "```json", json.dumps(rows, indent=2), "```", "",
              "## Conclusions", "",
              "_TODO (human): does the learned window/episode model beat the best constant blend",
              "by more than seed noise, on table regret AND deployed units? If not, the per-cycle",
              "search winner is not a stable supervised target at any tested time scale._", ""]
    (out / "RESULT.md").write_text("\n".join(lines))


def a0u_rows(task_id: str, world_seed: int, episodes: list[dict], vocab: list[str],
             onset: float = 5.0) -> tuple[list[dict], list[dict]]:
    """A0u failure-chain rows from champion-trajectory deploy traces. Per cycle the
    counterfactual best INCLUDES the champion (`cf_best = max(vocab table + champion)`), so
    `champ_regret >= 0` and regret > 0 means some vocabulary entry beat the champion's own
    continuation at a state the champion actually visited. `onset_cycle` = first cycle whose
    regret clears `onset` (where the game starts going wrong); `peak_cycle` = argmax regret.
    Returns `(rows, details)` — aggregates for summary.csv, per-cycle chains for artifacts +
    the RESULT.md dump."""
    rows: list[dict] = []
    details: list[dict] = []
    for ep in episodes:
        cycles = []
        for t in ep["traces"]:
            cf_best = max(max(t.cand_units), t.champion_units)
            cycles.append({
                "cycle": t.cycle, "cand_units": t.cand_units, "choice": t.choice,
                "teacher_pick": vocab[t.choice], "margin": t.margin,
                "champion_units": t.champion_units,
                "champ_regret": cf_best - t.champion_units,
                "state": t.state,
            })
        regs = [c["champ_regret"] for c in cycles]
        onset_cycle = next((c["cycle"] for c in cycles if c["champ_regret"] >= onset), None)
        peak_i = max(range(len(cycles)), key=lambda i: regs[i]) if cycles else None
        rows.append({
            "task_id": task_id, "world_seed": world_seed, "seed": ep["seed"],
            "champ_cu": ep["consumer_completed_units"],
            "rejected_units": ep.get("consumer_rejected_units"),
            "dead": ep["dead"], "cycles": ep["cycles"], "n_traces": len(cycles),
            "regret_total": round(sum(regs), 2),
            "regret_max": round(max(regs), 2) if regs else None,
            "onset_cycle": onset_cycle,
            "peak_cycle": cycles[peak_i]["cycle"] if peak_i is not None else None,
            "n_regret_pos": sum(1 for r in regs if r > 0),
        })
        details.append({"task_id": task_id, "world_seed": world_seed, "seed": ep["seed"],
                        "vocab": vocab, "cycles": cycles})
    return rows, details


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
        "   `gchoice_majority` — the same protocol from the ADR-0004 global snapshot, one sample",
        "   per searched cycle. Beats its majority floor => the D1 wall is missing CONTEXT (enrich",
        "   the orderer obs with global features). Fails too => enrichment with THESE features is",
        "   dead — but the snapshot is a lossy roll-up (capacity summed across types, no queue/TTL/",
        "   inventory-composition/history) at ~200 train cycles/task, so it does NOT prove the",
        "   choice is unpredictable from any observation; discriminate further with a lossless",
        "   state serialization ± history (A0d) before choosing enrichment vs deploy-time search.", "",
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


def run_a0t_diag(archive_path: str, producer_ckpt: str, orderer_ckpt: str, task_ids: list[str],
                 *, iters: int, horizon: int, eval_seeds: int,
                 alphas: tuple[float, ...] = (0.75, 0.5, 0.25), out_dir=None, log=print,
                 ) -> list[dict]:
    """Stage A0t trace-regret analysis: per task, collect held-out CycleTraces (full candidate
    tables + state snapshots + champion scored at the same states), fit the global-state probe
    on train seeds, and emit per-cycle regret rows + a top-regret dump for feature inference.

    Decision-gating verification (full-horizon per-cycle search + champion rollout per cycle,
    multi-seed/multi-task, distill to convergence) => a CLOUD run per CLAUDE.md."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer, scorer_ordering_policy

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champion_pol = scorer_ordering_policy(load_scorer(orderer_ckpt))
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {
        "producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt, "task_ids": task_ids,
        "iters": iters, "horizon": horizon, "eval_seeds": eval_seeds, "alphas": list(alphas),
        "search": _SEARCH,
    }
    rows, details = [], []
    for tid in task_ids:
        row, detail = a0t_diag_row(by_id[tid], prod_pol, champion_pol, iters=iters,
                                   horizon=horizon, eval_seeds=eval_seeds, alphas=alphas, log=log)
        log(f"[a0t] {row}")
        rows.append(row)
        details.append(detail)
        if out_dir is not None:
            _emit_a0t(out_dir, rows, details, meta={**meta, "complete": False,
                                                    "completed_task_ids": [r["task_id"] for r in rows]})
    if out_dir is not None:
        _emit_a0t(out_dir, rows, details, meta={**meta, "complete": True,
                                                "completed_task_ids": [r["task_id"] for r in rows]})
    return rows


def _emit_a0t(out_dir, rows: list[dict], details: list[dict], meta: dict,
              top_n: int = 5) -> None:
    """Write the Stage A0t EXP layout. RESULT.md leads with the REGRET read (probe_regret_frac /
    champ_regret_frac — cost-weighted misses, not top-1) and dumps the `top_n` highest-regret
    held-out cycles per task WITH candidate tables and state snapshots — the material a human
    reads to infer which missing observation would have explained the teacher's pick. Full
    per-cycle traces (all cycles, all snapshots) go to `artifacts/traces.json.gz`."""
    import gzip
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")
    with gzip.open(out / "artifacts" / "traces.json.gz", "wt") as f:
        json.dump(details, f)

    def _m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    lines = [
        "# EXP-0036 Stage A0t trace-regret analysis (which decisions go wrong, and what they cost)", "",
        "A0b/A0c only measured aggregate top-1 vs the majority floor — that cannot separate",
        "'misses ties that don't matter' from 'misses the calls that carry the units'. Here every",
        "held-out searched cycle keeps its full candidate table, so a student's miss is weighted",
        "by REGRET (units[teacher pick] - units[student pick], lookahead-window units — a local",
        "proxy, not episode units). The champion orderer is scored at the SAME states by the same",
        "rollout (champ_regret; negative = champion beats the whole vocabulary there).", "",
        "**Read order:**", "",
        "1. **probe_regret_frac** (probe_regret_total / local_headroom_total): the share of the",
        "   teacher's per-cycle edge over even_split the global-state probe loses. ~0 despite low",
        "   top-1 => misses are ties, the D1 'wall' costs nothing, distillation is fine as-is.",
        "   Large => the misses are expensive; go to 2.",
        "2. **Top-regret cycles** (below, with state snapshots): read WHAT the teacher saw coming",
        "   that the pick missed — arriving inbound stock, per-type bottleneck, TTL cliff. Infer",
        "   the missing feature from evidence, not from a menu.",
        "3. **champ_regret_frac / n_champ_better:** where the deployed champion stands at the",
        "   same states — large positive => champion's misses are costly too (consistent with its",
        "   0-unit collapses); many negatives => the vocabulary itself is the ceiling, widen it.", "",
        f"**Means:** probe_top1={_m('probe_top1')}, probe_regret_frac={_m('probe_regret_frac')}, "
        f"champ_regret_frac={_m('champ_regret_frac')}, "
        f"local_headroom_total={_m('local_headroom_total')}.", "",
        "## Per-task", "", "```json", json.dumps(rows, indent=2), "```", "",
    ]
    for det in details:
        vocab = det["vocab"]
        worst = sorted(det["cycles"], key=lambda c: c["probe_regret"], reverse=True)[:top_n]
        lines += [f"## Top-regret held-out cycles — {det['task_id']} (seed {det['heldout_seed']})", ""]
        for c in worst:
            table = ", ".join(f"{n}={u:.1f}" for n, u in zip(vocab, c["cand_units"]))
            champ = ("n/a" if c["champion_units"] is None
                     else f"{c['champion_units']:.1f} (regret {c['champ_regret']:.1f})")
            lines += [
                f"### cycle {c['cycle']} — probe_regret {c['probe_regret']:.1f}, "
                f"margin {c['margin']:.1f}",
                f"- candidates: {table}",
                f"- teacher pick: `{vocab[c['choice']]}`; probe pick: `{vocab[c['probe_pred']]}`; "
                f"champion: {champ}",
                "", "```json", json.dumps(c["state"], indent=1), "```", "",
            ]
    lines += ["## Conclusions", "",
              "_TODO (human): read regret_frac first, then the top-regret cycles; name the",
              "missing feature (or conclude deploy-time search) from what the teacher saw._", ""]
    (out / "RESULT.md").write_text("\n".join(lines))


def run_a0u_chains(archive_path: str, producer_ckpt: str, orderer_ckpt: str,
                   task_ids: list[str], *, horizon: int, eval_seeds: int,
                   alphas: tuple[float, ...] = (0.75, 0.5, 0.25), onset: float = 5.0,
                   out_dir=None, log=print) -> list[dict]:
    """Stage A0u failure chains: replay every archived episode seed WITH THE CHAMPION and emit
    per-cycle counterfactual regret chains (see `collect_deploy_traces` / `a0u_rows`). No
    distillation — but full-horizon per-cycle search across tasks × seeds => CLOUD run."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import collect_deploy_traces, load_scorer, scorer_ordering_policy

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champion_pol = scorer_ordering_policy(load_scorer(orderer_ckpt))
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    meta = {"producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt, "task_ids": task_ids,
            "horizon": horizon, "eval_seeds": eval_seeds, "alphas": list(alphas),
            "onset": onset, "search": _SEARCH}
    rows: list[dict] = []
    details: list[dict] = []
    for tid in task_ids:
        record = by_id[tid]
        cfg = replay(record)
        seeds = _episode_seeds(record.episode_seed, eval_seeds)
        episodes, vocab = collect_deploy_traces(
            [cfg] * len(seeds), list(seeds), prod=prod_pol, champion=champion_pol,
            alphas=alphas, lookahead=_SEARCH["lookahead"], horizon=horizon, log=log)
        t_rows, t_details = a0u_rows(tid, record.world_seed, episodes, vocab, onset=onset)
        for r in t_rows:
            log(f"[a0u] {r}")
        rows += t_rows
        details += t_details
        if out_dir is not None:
            _emit_a0u(out_dir, rows, details, meta={**meta, "complete": False,
                                                    "completed_task_ids": sorted({r['task_id'] for r in rows})})
    if out_dir is not None:
        _emit_a0u(out_dir, rows, details, meta={**meta, "complete": True,
                                                "completed_task_ids": sorted({r['task_id'] for r in rows})})
    return rows


def _emit_a0u(out_dir, rows: list[dict], details: list[dict], meta: dict,
              top_n: int = 3) -> None:
    """Write the Stage A0u EXP layout. RESULT.md renders, per badly-played episode, the failure
    CHAIN: a progress line (cycle -> delivered units, counterfactual-best vs champion regret) so
    the collapse onset is visible at a glance, then the `top_n` highest-regret cycles with their
    candidate tables and state snapshots. Full chains -> `artifacts/deploy_traces.json.gz`."""
    import gzip
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")
    with gzip.open(out / "artifacts" / "deploy_traces.json.gz", "wt") as f:
        json.dump(details, f)

    lines = [
        "# EXP-0036 Stage A0u champion failure chains (the badly played games, cycle by cycle)", "",
        "A0t showed the champion is decent per-decision at TEACHER-visited states yet delivers 0",
        "units on some episodes — so the failure must live on its OWN trajectory. Each chain below",
        "replays a champion episode and, at every cycle it actually visited, scores the blend",
        "vocabulary + the champion's own continuation by the same lookahead rollout.",
        "`champ_regret` > 0 = a vocabulary entry beat the champion's continuation AT ITS OWN STATE;",
        "`onset_cycle` = where the game starts going wrong; the snapshots say why.", "",
        "**Read order:** per episode: (1) the progress line — does delivered flatline before or",
        "after the regret onset? (2) the top-regret snapshots at/after onset — what did the",
        "counterfactual route differently? (3) across episodes — is the onset mechanism the SAME",
        "(one repeated mistake => one targeted fix) or VERSATILE (many mechanisms => deploy-time",
        "search, which fixes all of them by construction)?", "",
        "## Per-episode rows", "", "```json", json.dumps(rows, indent=2), "```", "",
    ]
    for det in details:
        cyc = det["cycles"]
        lines += [f"## Failure chain — {det['task_id']} (episode seed {det['seed']})", "",
                  "```", "cycle  delivered  rejected  champ_units  cf_best  regret  teacher_pick"]
        for c in cyc:
            pr = c["state"].get("progress", {})
            cf_best = max(max(c["cand_units"]), c["champion_units"])
            mark = "  <-- onset" if c["champ_regret"] >= meta.get("onset", 5.0) else ""
            lines.append(f"{c['cycle']:>5}  {pr.get('units_delivered', 0):>9}  "
                         f"{pr.get('units_rejected', 0):>8}  {c['champion_units']:>11.1f}  "
                         f"{cf_best:>7.1f}  {c['champ_regret']:>6.1f}  {c['teacher_pick']}{mark}")
        lines += ["```", ""]
        worst = sorted(cyc, key=lambda c: c["champ_regret"], reverse=True)[:top_n]
        for c in worst:
            table = ", ".join(f"{n}={u:.1f}" for n, u in zip(det["vocab"], c["cand_units"]))
            lines += [
                f"### cycle {c['cycle']} — champ_regret {c['champ_regret']:.1f}",
                f"- candidates: {table}; champion continuation: {c['champion_units']:.1f}",
                f"- counterfactual pick here: `{c['teacher_pick']}`",
                "", "```json", json.dumps(c["state"], indent=1), "```", "",
            ]
    lines += ["## Conclusions", "",
              "_TODO (human): compare onset mechanisms across episodes — same or versatile?_", ""]
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
                    help="with --a0b/--a0t: comma-separated ordering_blend alphas added to the vocabulary")
    ap.add_argument("--a0f", action="store_true",
                    help="Stage A0f mode (HANDOVER Task 1): per --task-ids fit a value-table "
                         "probe on cumulative observation blocks and emit held-out regret_frac "
                         "per block set (missing-observation ranking); needs no --orderer-ckpt")
    ap.add_argument("--k-history", type=int, default=4,
                    help="with --a0f: previous cycles stacked into the history block")
    ap.add_argument("--a0w", action="store_true",
                    help="Stage A0w mode (HANDOVER Task 5): per --task-ids fit window/episode "
                         "blend-policy value tables (constant-blend returns, not per-cycle "
                         "search argmaxes), report held-out regret vs the best constant blend "
                         "and the deployed rollout closure; needs --orderer-ckpt (champion)")
    ap.add_argument("--windows", default="5,10,20",
                    help="with --a0w: comma-separated rolling-window lengths (cycles)")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="with --a0w: conservative switch threshold in the model's output units")
    ap.add_argument("--default-blend", default="blend_0.5",
                    help="with --a0w: the robust default blend held until a switch clears --threshold")
    ap.add_argument("--deploy-window", type=int, default=10,
                    help="with --a0w: the window model deployed in the held-out rollout")
    ap.add_argument("--a0t", action="store_true",
                    help="Stage A0t mode: per --task-ids emit held-out trace-REGRET rows (probe + "
                         "champion vs the search teacher, full candidate tables + state snapshots) "
                         "and a top-regret cycle dump; needs --orderer-ckpt (champion)")
    ap.add_argument("--a0u", action="store_true",
                    help="Stage A0u mode: replay archived episodes WITH the champion and emit "
                         "per-cycle counterfactual failure CHAINS (progress line, regret onset, "
                         "top-regret snapshots); needs --orderer-ckpt (champion)")
    ap.add_argument("--onset", type=float, default=5.0,
                    help="with --a0u: champ_regret threshold marking the failure onset cycle")
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

    if not (args.a0b or args.a0f) and not args.orderer_ckpt:
        ap.error("--orderer-ckpt is required unless --a0b/--a0f")

    if args.a0f:
        if not args.task_ids:
            ap.error("--task-ids is required with --a0f")
        rows = run_a0f_probe(
            args.archive, args.producer_ckpt, [t for t in args.task_ids.split(",")],
            iters=args.iters, horizon=args.horizon, eval_seeds=args.eval_seeds,
            alphas=tuple(float(a) for a in args.alphas.split(",")),
            k_history=args.k_history, out_dir=args.out_dir)
        print("\ntask_id,block_set,dim,train_regret_frac,regret_frac,top1,winner_hit")
        for r in rows:
            print(f"{r['task_id']},{r['block_set']},{r['dim']},{r['train_regret_frac']},"
                  f"{r['regret_frac']},{r['top1']},{r['winner_hit']}")
        return

    if args.a0w:
        if not args.task_ids:
            ap.error("--task-ids is required with --a0w")
        rows = run_a0w_task5(
            args.archive, args.producer_ckpt, args.orderer_ckpt,
            [t for t in args.task_ids.split(",")], iters=args.iters, horizon=args.horizon,
            eval_seeds=args.eval_seeds, alphas=tuple(float(a) for a in args.alphas.split(",")),
            windows=tuple(int(w) for w in args.windows.split(",")), threshold=args.threshold,
            default_blend=args.default_blend, deploy_window=args.deploy_window,
            out_dir=args.out_dir)
        print("\ntask_id,granularity,regret_frac,const_regret_frac,decision_regret_frac,"
              "best_const_blend,closure_vs_search")
        for r in rows:
            print(f"{r['task_id']},{r['granularity']},{r.get('regret_frac')},"
                  f"{r.get('const_regret_frac')},{r.get('decision_regret_frac')},"
                  f"{r.get('best_const_blend')},{r.get('closure_vs_search')}")
        return

    if args.a0u:
        if not args.task_ids:
            ap.error("--task-ids is required with --a0u")
        rows = run_a0u_chains(
            args.archive, args.producer_ckpt, args.orderer_ckpt,
            [t for t in args.task_ids.split(",")], horizon=args.horizon,
            eval_seeds=args.eval_seeds, alphas=tuple(float(a) for a in args.alphas.split(",")),
            onset=args.onset, out_dir=args.out_dir)
        print("\ntask_id,seed,champ_cu,dead,cycles,regret_total,regret_max,onset_cycle,peak_cycle")
        for r in rows:
            print(f"{r['task_id']},{r['seed']},{r['champ_cu']},{r['dead']},{r['cycles']},"
                  f"{r['regret_total']},{r['regret_max']},{r['onset_cycle']},{r['peak_cycle']}")
        return

    if args.a0t:
        if not args.task_ids:
            ap.error("--task-ids is required with --a0t")
        rows = run_a0t_diag(
            args.archive, args.producer_ckpt, args.orderer_ckpt,
            [t for t in args.task_ids.split(",")], iters=args.iters, horizon=args.horizon,
            eval_seeds=args.eval_seeds, alphas=tuple(float(a) for a in args.alphas.split(",")),
            out_dir=args.out_dir)
        print("\ntask_id,probe_top1,probe_regret_frac,champ_regret_frac,local_headroom_total,"
              "n_probe_miss,n_champ_worse,n_champ_better")
        for r in rows:
            print(f"{r['task_id']},{r['probe_top1']},{r['probe_regret_frac']},"
                  f"{r['champ_regret_frac']},{r['local_headroom_total']},{r['n_probe_miss']},"
                  f"{r['n_champ_worse']},{r['n_champ_better']}")
        return

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
