"""Producer finalization step 2: why rf4 loses ~400 units vs rf3 on atlas 12-0391 (EXP-0084).
Per blocked upgrade: the producer, its type, max, degradation %, restoration, the exact ceiling
(floor(100 restores / d)), and whether the view estimate agrees with the exact rule; per producer
the capacity path under rf3 and rf4; horizon 100 and 200. Local, read-only wrappers (eval side)."""
import math
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tokenomics.domain import system as sys_mod  # noqa: E402
from tokenomics.eval import baselines as b  # noqa: E402
from tokenomics.eval.decoder_check import (MWWK, decoder_arm, decoder_policies, job_draw_key,  # noqa: E402
                                           select_views, world_entry)
from tokenomics.eval.joint_stack import joint_stack_env  # noqa: E402
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster  # noqa: E402
from tokenomics.eval.routing_audit import build_jobs, job_config  # noqa: E402
from tokenomics.eval.routing_diagnostics import set_draw_key  # noqa: E402
from tokenomics.eval.runner import run_episode  # noqa: E402
from tokenomics.eval.teacher_audit import load_atlas  # noqa: E402

ATLAS = "/srv/research/tokenomics/experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
WORLD, EPISODE = "ws-atlas-v1-12-0391", 1977552299
LOG: dict = {}
_cc, _act = sys_mod.System.complete_cycle, sys_mod.System.producing_agent_action


def cc(self):
    res = _cc(self)
    LOG["caps"][self.cycle_counter - 1] = {s.producer_id: (dict(s.capacities), dict(s.max_capacities))
                                           for s in res.end_capacities}
    LOG["units"].append(res.consumer_completed_units)
    return res


def act(self, pid, cmd):
    p = self.producing_agents[pid]
    LOG["sys"] = self
    if cmd.do_upgrade:
        LOG["upgrades"][pid] += 1
    return _act(self, pid, cmd)


sys_mod.System.complete_cycle, sys_mod.System.producing_agent_action = cc, act
real_over = b.over_sustainable_max


def over(view):
    hit = real_over(view)
    if hit:
        p = LOG["sys"].producing_agents[view.producer_id] if hasattr(view, "producer_id") else None
        LOG["blocked_views"].append(view)
    return hit


b.over_sustainable_max = over


def play(name: str, horizon: int) -> dict:
    LOG.clear()
    LOG.update(caps={}, units=[], upgrades=Counter(), blocked_views=[], sys=None)
    view = select_views(load_atlas(ATLAS), [WORLD])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    job = [j for j in build_jobs([world_entry(view)], [arm], 1, horizon, None, {})
           if j["replicate"] == 0 and j["episode"] == EPISODE][0]
    env = joint_stack_env(job_config(job), horizon)
    set_draw_key(env, job_draw_key(job))
    LOG["sys"] = env.system
    prod = b.EXTRA_PRODUCING[name]
    run_episode(env, prod, decoder_policies(job)[1], seed=job["episode"], action_seed=job["action_seed"])
    return {"units": sum(LOG["units"]), "units_trace": list(LOG["units"]), "caps": dict(LOG["caps"]),
            "upgrades": Counter(LOG["upgrades"]), "blocked": list(LOG["blocked_views"]),
            "system": LOG["sys"], "dead": env.episode_stats["dead"]}


def main() -> None:
    for horizon in (100, 200):
        r3 = play("entity_planner_buyahead_k4_rf3", horizon)
        r4 = play("entity_planner_buyahead_k4_rf4", horizon)
        s = r3["system"]
        print(f"== horizon {horizon}: rf3 {r3['units']} dead {r3['dead']} | rf4 {r4['units']} dead {r4['dead']}")
        if horizon == 100:
            blocked_by = Counter()
            est_vs_exact = Counter()
            for v in r4["blocked"]:
                for row in v.capacity_rows:
                    if row.upgrade_effect > 0 and row.restore_effect > 0 and row.max > 0:
                        est = math.ceil(row.degradation * (row.max + row.upgrade_effect) / row.max)
                        if est > row.restore_effect:
                            blocked_by[(v.producer_id if hasattr(v, 'producer_id') else '?', row.capacity_type, row.max)] += 1
            print(" blocked (producer?, type, max) top:", blocked_by.most_common(8))
            for pid, p in s.producing_agents.items():
                t = p.upgrade.type if p.upgrade else None
                if t is None or p.restoration is None:
                    continue
                ceil_exact = math.floor(100 * p.restoration.restores / p.degradation) if p.degradation else None
                m3 = r3["caps"][max(r3["caps"])][pid][1].get(t)
                m4 = r4["caps"][max(r4["caps"])][pid][1].get(t)
                c3 = [r3["caps"][c][pid][0].get(t) for c in (25, 50, 75, max(r3["caps"]))]
                c4 = [r4["caps"][c][pid][0].get(t) for c in (25, 50, 75, max(r4["caps"]))]
                same = p.restoration.type == t
                print(f" {pid} type {t} restores-type-same {same} d% {p.degradation} restores {p.restoration.restores} up +{p.upgrade.increases} "
                      f"ceiling {ceil_exact} | max rf3 {m3} rf4 {m4} | cap@25/50/75/end rf3 {c3} rf4 {c4} | upgrades asked rf3 {r3['upgrades'][pid]} rf4 {r4['upgrades'][pid]}")
            d = [x - y for x, y in zip(r3["units_trace"], r4["units_trace"])]
            print(" units rf3 - rf4 by 20-cycle window:", [sum(d[i:i + 20]) for i in range(0, len(d), 20)])


if __name__ == "__main__":
    main()


def exact_counterfactual() -> None:
    """Diagnostic only: the cap with the exact degradation % (the producer's own config value)
    instead of the view estimate, on 12-0391 at horizons 100 and 200."""
    def exact(view):
        p = LOG["sys"].producing_agents[view.producer_id]
        for r in view.capacity_rows:
            if r.upgrade_effect > 0 and r.restore_effect > 0 and r.max > 0:
                if math.ceil(p.degradation * (r.max + r.upgrade_effect) / 100) > r.restore_effect:
                    return True
        return False
    b.over_sustainable_max = exact
    for horizon in (100, 200):
        r = play("entity_planner_buyahead_k4_rf4", horizon)
        print(f"exact cap, horizon {horizon}: units {r['units']} dead {r['dead']} "
              f"p1_1 max end {r['caps'][max(r['caps'])]['p1_1'][1]}")
    b.over_sustainable_max = over


if __name__ == "__main__" and "--exact" in sys.argv:
    exact_counterfactual()
