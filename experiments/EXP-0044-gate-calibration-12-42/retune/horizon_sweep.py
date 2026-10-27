"""10.3 follow-up diagnostic (local, one-shot): where does the best baseline's unfilled demand
come from? One episode (first episode seed, replicate 0) of entity_planner/random on three
development worlds, rerun at increasing horizons; the increment between horizons gives the fill
rate of each cycle window. Run from tokenomics/: ../.venv/bin/python <this file>."""
from tokenomics.eval.registry import load_registry
from tokenomics.eval.roster import load_roster, member_env, member_policies
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_seed

reg, ros = load_registry(), load_roster()
m = {x.id: x for x in ros.members}["entity_planner/random"]
for wid in ("dev-dense-12-calm-00", "dev-dense-12-shocked-00", "dev-dense-42-calm-00"):
    e = reg.entry(wid)
    ep, cfg, prev = e.episode_seeds[0], e.config(), None
    print(wid)
    for h in (3, 5, 10, 20, 50, 98, 100):
        env = member_env(m, cfg, h)
        prod, order = member_policies(m)
        run_episode(env, prod, order, seed=ep, action_seed=replicate_seed(wid, ep, 0))
        st = env.episode_stats
        o = sum(st["consumer_ordered_units"].values())
        f = sum(st["consumer_fulfilled_units"].values())
        inc = "" if prev is None else (f"  window {prev[2]}->{h}: ordered {o - prev[0]} "
                                       f"fulfilled {f - prev[1]} rate "
                                       f"{(f - prev[1]) / max(1, o - prev[0]):.3f}")
        print(f"  h={h:3d} ordered {o:5} fulfilled {f:5} fill {f / o:.4f}{inc}")
        prev = (o, f, h)
