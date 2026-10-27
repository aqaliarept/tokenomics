"""10.3 follow-up diagnostic: share of the best baseline's shortfall that falls in the first 20
cycles, on every dense development gate world of one size (argv[1]). entity_planner/random,
first episode seed, replicate 0, horizons 20 and 100."""
import json, sys
from tokenomics.eval.registry import DEVELOPMENT, load_registry
from tokenomics.eval.roster import load_roster, member_env, member_policies
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_seed

size = int(sys.argv[1])
reg, ros = load_registry(), load_roster()
m = {x.id: x for x in ros.members}["entity_planner/random"]
for e in reg.gate_worlds(DEVELOPMENT):
    if (e.family, e.size) != ("dense", size):
        continue
    ep, cfg, out = e.episode_seeds[0], e.config(), {}
    for h in (20, 100):
        env = member_env(m, cfg, h)
        prod, order = member_policies(m)
        run_episode(env, prod, order, seed=ep, action_seed=replicate_seed(e.id, ep, 0))
        st = env.episode_stats
        out[h] = (sum(st["consumer_ordered_units"].values()),
                  sum(st["consumer_fulfilled_units"].values()), st["dead"])
    (o20, f20, _), (o100, f100, dead) = out[20], out[100]
    print(json.dumps({"world": e.id, "fill100": round(f100 / o100, 4),
                      "short_first20": o20 - f20, "short_total": o100 - f100,
                      "rate_after20": round((f100 - f20) / (o100 - o20), 4), "dead": dead}),
          flush=True)
