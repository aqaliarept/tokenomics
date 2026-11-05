"""Task 4.2e: why EXP-0081's members die on train 12-0016 / 12-0004 (rf3 lives at every key).

Replays rf3 and dying members on the recorded keys and theta (the probe's noise stream), with
class-level read-only wrappers on System / ProducingAgent (eval-side, never in domain/), and
logs per cycle: capacities, restoration / replenishment / expansion requests and their fate,
funds, inventory, and the first unmakeable product. Also ablates each member's three
corrections (theta_k / theta_b / theta_e alone)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from tokenomics.domain import producing_agent as pa_mod
from tokenomics.domain import system as sys_mod
from tokenomics.train import producer_es as pe
from tokenomics.train.order_es import member_thetas, noise
from tokenomics.train.producer_residual import N_IN

E = "/srv/research/tokenomics/experiments/"
LOG: dict = {}


def _install() -> None:
    S, P = sys_mod.System, pa_mod.ProducingAgent
    cc, act = S.complete_cycle, S.producing_agent_action
    inv_c, inv_r, rep_c, rep_r = (P.investment_completed, P.investment_rejected,
                                  P.replenishment_completed, P.replenishment_rejected)

    def complete_cycle(self):
        c = self.cycle_counter
        LOG["funds"][c] = {pid: p.funds for pid, p in self.producing_agents.items()}
        LOG["fund"][c] = self.investment_fund
        res = cc(self)
        LOG["caps"][c] = {s.producer_id: (dict(s.capacities), dict(s.max_capacities))
                          for s in res.end_capacities}
        LOG["inv"][c] = {pid: (p.inventory_units(), p.inventory_limit, dict(p.inventory))
                         for pid, p in self.producing_agents.items()}
        LOG["units"][c] = res.consumer_completed_units
        if self.is_dead() and LOG["death"] is None:
            for product, ps in self.process_sheets.items():
                decl = {pid: {t: p.capacities.get(t, 0) for t in ps.require}
                        for pid, p in self.producing_agents.items() if product in p.produces}
                if not any(all(v > 0 for v in d.values()) for d in decl.values()):
                    LOG["death"] = {"cycle": c, "product": product, "declaring": decl}
                    break
        return res

    def producing_agent_action(self, pid, cmd):
        LOG["cmd"][self.cycle_counter][pid] = {
            "restore": cmd.do_restoration, "upgrade": cmd.do_upgrade,
            "expand": cmd.do_inventory_expansion,
            "buy": {g: q for g, q in cmd.replenishments.items() if q > 0}}
        return act(self, pid, cmd)

    def wrap(fn, tag):
        def w(self, request):
            LOG["events"].append((LOG["cyc"](), self.id, tag,
                                  getattr(getattr(request, "type", None), "name", "REPL"),
                                  getattr(request, "quantity", None)))
            return fn(self, request)
        return w

    S.complete_cycle, S.producing_agent_action = complete_cycle, producing_agent_action
    P.investment_completed = wrap(inv_c, "done")
    P.investment_rejected = wrap(inv_r, "rejected")
    P.replenishment_completed = wrap(rep_c, "done")
    P.replenishment_rejected = wrap(rep_r, "rejected")


def play(job: dict) -> dict:
    holder = {}
    LOG.clear()
    LOG.update(funds={}, fund={}, caps={}, inv={}, units={}, cmd=defaultdict(dict), events=[],
               death=None, cyc=lambda: holder["sys"].cycle_counter if "sys" in holder else -1)
    orig_init = sys_mod.System.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        holder["sys"] = self
    sys_mod.System.__init__ = init
    try:
        row = pe.run_producer_job(job)
    finally:
        sys_mod.System.__init__ = orig_init
    out = {k: v for k, v in LOG.items() if k != "cyc"}
    out["row"] = {k: row.get(k) for k in ("units", "dead", "min_capacity_ratio", "r_start", "r_end",
                                          "investment_rejection_rate")}
    out["system"] = holder["sys"]
    return out


def main() -> None:
    _install()
    ts = pe.load_training(E + "EXP-0058-es-gradient-probe/bar")
    nm, _ = pe.load_normaliser(E + "EXP-0080-producer-bar-noise/bar")
    a, b = pe.probe_batches(ts, pe.MINIBATCH)
    eps = noise(0, 0, pe.PAIRS, pe.THETA_DIM, pe.NOISE_STREAM_PURPOSE["probe"])
    sig = np.full(pe.THETA_DIM, pe.SIGMA)
    thetas = member_thetas(np.zeros(pe.THETA_DIM), eps, sig)
    target = sys.argv[1]  # world suffix, e.g. 12-0016
    members = sys.argv[2].split(",")
    (we,) = [k for k in b if k[0].endswith(target)]
    jobs = {"rf3": pe._member_jobs(ts, pe.PROBE, {"c": np.zeros(pe.THETA_DIM)}, [we], 0,
                                   pe.HORIZON, 1, normaliser=nm)[0]}
    for m in members:
        th = thetas[m]
        jobs[m] = pe._member_jobs(ts, pe.PROBE, {m: th}, [we], 0, pe.HORIZON, 1, normaliser=nm)[0]
        for part, sl in (("k", slice(0, N_IN)), ("b", slice(N_IN, 2 * N_IN)),
                         ("e", slice(2 * N_IN, None))):
            t = np.zeros(pe.THETA_DIM)
            t[sl] = th[sl]
            jobs[f"{m}/{part}"] = pe._member_jobs(ts, pe.PROBE, {m: t}, [we], 0,
                                                 pe.HORIZON, 1, normaliser=nm)[0]
    res = {}
    for lab, j in jobs.items():
        r = play(j)
        res[lab] = r
        d = r["death"]
        print(lab, r["row"], "DEATH" if d else "", json.dumps(d, default=str) if d else "")
    import pickle
    pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "system"} for k, v in res.items()},
                open(f"/tmp/death_trace_{target}.pkl", "wb"))
    s = res["rf3"]["system"]
    print("world", we, "producers", {pid: sorted(p.produces) for pid, p in s.producing_agents.items()})
    def mech(m):
        return None if m is None else {k: (v if isinstance(v, (int, float, str)) else str(v))
                                       for k, v in vars(m).items()}
    print("restoration", {pid: mech(p.restoration) for pid, p in list(s.producing_agents.items())[:5]})
    print("upgrade", mech(s.producing_agents["p0_0"].upgrade))
    print("expansion", {pid: mech(p.inventory_expansion) for pid, p in list(s.producing_agents.items())[::4]})
    print("degradation", {pid: p.degradation for pid, p in s.producing_agents.items()})
    print("sheets", {k: str(v) for k, v in list(s.process_sheets.items())[:6]})

if __name__ == "__main__":
    main()
