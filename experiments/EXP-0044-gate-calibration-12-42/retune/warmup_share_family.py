"""Warm-up diagnostic for any family/member (DIAGNOSTICS entry 18): for every development gate
world of `family` at `size`, the given member's shortfall in cycles 1-20 vs 21-100 (horizons 20
and 100, first episode seed, replicate 0).
usage: warmup_share_family.py FAMILY SIZE MEMBER"""
import json, sys
from tokenomics.eval.registry import DEVELOPMENT, load_registry
from tokenomics.eval.roster import load_roster, member_env, member_policies
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_seed

family, size, member = sys.argv[1], int(sys.argv[2]), sys.argv[3]
reg, ros = load_registry(), load_roster()
m = {x.id: x for x in ros.members}[member]
for e in reg.gate_worlds(DEVELOPMENT):
    if (e.family, e.size) != (family, size):
        continue
    ep, cfg, out = e.episode_seeds[0], e.config(), {}
    for h in (20, 100):
        env = member_env(m, cfg, h)
        prod, order = member_policies(m)
        run_episode(env, prod, order, seed=ep,
                    action_seed=replicate_seed(e.id, ep, 0) if m.random else None)
        st = env.episode_stats
        out[h] = (sum(st["consumer_ordered_units"].values()),
                  sum(st["consumer_fulfilled_units"].values()), st["dead"])
    (o20, f20, _), (o100, f100, dead) = out[20], out[100]
    print(json.dumps({"world": e.id, "member": member, "fill100": round(f100 / o100, 4),
                      "short_first20": o20 - f20, "short_total": o100 - f100,
                      "rate_after20": round((f100 - f20) / (o100 - o20), 4), "dead": dead}),
          flush=True)
