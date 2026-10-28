"""Task 3.3 one-shot diagnostic (single episode per world, local, `--workers 1`).

For the informed routers of EXP-0049, on the first episode seed of a world (horizon 100,
`entity_planner` producer, joint_stack), count per routing decision with two or more eligible
producers:
- `spare_rotating_whole`: how often the rotation's first producer cannot take the whole order
  (spare < quantity), i.e. how often the skip changes the receiver against `round_robin`;
- the spare split between the eligible producers (share of the larger spare), which is what
  `spare_weighted_whole` draws by and what `two_choice_spare` compares at fanout 2;
- how often the eligible producers' spares tie (where `two_choice_spare` keeps its first draw);
- the smaller spare's mean and 10th percentile, and how often the order is at least as large as
  the smaller spare (where the less-loaded producer could not take it whole).

Run from /srv/research/tokenomics/tokenomics:
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/spare_33.py \
      dev-sparse_wide-12-shocked-00 dev-sparse_wide-42-shocked-00
"""

from __future__ import annotations

import json
import sys
from statistics import fmean

import numpy as np

from tokenomics.domain.system import order_kind
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.routing_audit import select_worlds
from tokenomics.eval.routing_diagnostics import (DIAGNOSTIC_ROUTERS, _slots, spare_units,
                                                 stable_hash)
from tokenomics.eval.routing_trace import CONSUMER, order_class
from tokenomics.eval.runner import run_episode


def main() -> None:
    worlds = {w.id: w for w in select_worlds(load_registry(REGISTRY_V2))}
    for wid in sys.argv[1:]:
        entry = worlds[wid]
        stats = {CONSUMER: [], "producer": []}

        def probe(env, agent: str) -> np.ndarray:  # noqa: ANN001 - test-style probe
            cycle = env.system.cycle_counter
            pids = env.spec_.producer_ids
            for k, oid, q, elig in _slots(env, agent):
                if len(elig) < 2:
                    continue
                product = env.system.orders[oid].product
                spare = [spare_units(env, pids[i], product) for i in elig]
                first = (cycle + stable_hash(agent, k)) % len(elig)
                total = sum(spare)
                cls = order_class(order_kind(env.system.orders[oid]))
                stats[cls if cls == CONSUMER else "producer"].append({
                    "skip": spare[first] < q and any(s >= q for s in spare),
                    "none_fit": all(s < q for s in spare),
                    "tie": len(set(spare)) == 1,
                    "larger_share": max(spare) / total if total else 0.5,
                    "q": q, "min_spare": min(spare)})
            return DIAGNOSTIC_ROUTERS["spare_rotating_whole"](env, agent)

        env = joint_stack_env(entry.config(), 100)
        run_episode(env, PRODUCING_BASELINES["entity_planner"], probe, seed=entry.episode_seeds[0])
        for cls, rows in stats.items():
            if not rows:
                continue
            print(json.dumps({
                "world": wid, "class": cls, "decisions": len(rows),
                "skip_rate": round(fmean(r["skip"] for r in rows), 4),
                "none_fit_rate": round(fmean(r["none_fit"] for r in rows), 4),
                "tie_rate": round(fmean(r["tie"] for r in rows), 4),
                "mean_larger_spare_share": round(fmean(r["larger_share"] for r in rows), 4),
                "mean_quantity": round(fmean(r["q"] for r in rows), 2),
                "mean_min_spare": round(fmean(r["min_spare"] for r in rows), 1),
                "p10_min_spare": sorted(r["min_spare"] for r in rows)[len(rows) // 10],
                "q_ge_min_spare_rate": round(fmean(r["q"] >= r["min_spare"] for r in rows), 4)}),
                  flush=True)


if __name__ == "__main__":
    main()
