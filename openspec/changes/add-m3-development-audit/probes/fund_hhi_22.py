"""Task 2.2 check: the fixed fund_share_herfindahl on the 18 traced cells of task 2.1."""
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import ROSTER_V2, load_roster
from tokenomics.eval.routing_audit import ordering_policy, router_table
from tokenomics.eval.routing_trace import RoutingTracer
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_seed

R = ["even_split", "proportional", "blend_0.5", "random", "round_robin", "even_split_rotating",
     "proportional_load_aware", "proportional_whole", "random_whole"]
reg, table = load_registry(REGISTRY_V2), {r["router"]: r for r in router_table(load_roster(ROSTER_V2))}
for w in ("dev-sparse_wide-42-shocked-00", "dev-sparse_wide-12-calm-03"):
    e = reg.entry(w); ep = e.episode_seeds[0]; cfg = e.config()
    for r in R:
        env = joint_stack_env(cfg, 100); tr = RoutingTracer(); tr.attach(env)
        seed = replicate_seed(w, ep, 0) if table[r]["random"] else None
        st = run_episode(env, PRODUCING_BASELINES["entity_planner"], ordering_policy(r), seed=ep, action_seed=seed)
        fs = tr.fund_shares(); t = sum(fs.values())
        sm = tr.summary()
        print(f"{w} {r:24s} units {st['consumer_completed_units']:5d} per_cycle {sm['fund_share_herfindahl']:.4f} "
              f"episode {sm['fund_share_herfindahl_episode']:.4f} (from series {sum((v/t)**2 for v in fs.values()):.4f}) "
              f"cycles_with_grants {tr._c.fund_hhi_n} granted_producers {len(tr._c.fund_granted)} 1/N {1/len(cfg.producer_configs):.4f}")
