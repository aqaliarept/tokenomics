"""Task 4.2e counterfactual (diagnostic only, no rule changed in code): veto any upgrade whose
post-upgrade degradation ceil(d% x (max + increases)) exceeds what one restoration returns, via
an eval-side wrapper on System.producing_agent_action; replay rf3 and dying members."""
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import death_trace as dt  # noqa: E402
from tokenomics.domain import system as sys_mod  # noqa: E402
from tokenomics.train import producer_es as pe  # noqa: E402
from tokenomics.train.order_es import member_thetas, noise  # noqa: E402

VETO = {"on": False, "n": 0}
_act = sys_mod.System.producing_agent_action


def act(self, pid, cmd):
    p = self.producing_agents[pid]
    if VETO["on"] and cmd.do_upgrade and p.upgrade is not None and p.restoration is not None:
        t = p.upgrade.type
        if math.ceil(p.degradation * (p.max_capacities[t] + p.upgrade.increases) / 100) \
                > p.restoration.restores:
            cmd.do_upgrade = False
            VETO["n"] += 1
    return _act(self, pid, cmd)


sys_mod.System.producing_agent_action = act
dt._install()  # its logging wrapper calls `act`
ts = pe.load_training(dt.E + "EXP-0058-es-gradient-probe/bar")
nm, _ = pe.load_normaliser(dt.E + "EXP-0080-producer-bar-noise/bar")
a, b = pe.probe_batches(ts, 24)
eps = noise(0, 0, pe.PAIRS, pe.THETA_DIM, pe.NOISE_STREAM_PURPOSE["probe"])
th = member_thetas(np.zeros(pe.THETA_DIM), eps, np.full(pe.THETA_DIM, pe.SIGMA))
for world, members in (("12-0016", ["c", "-00", "+03", "-05", "+04", "-11"]), ("12-0004", ["c", "+06"])):
    (we,) = [k for k in b if k[0].endswith(world)]
    for H in (100, 200):
        for m in members:
            out = []
            for on in (False, True):
                VETO.update(on=on, n=0)
                theta = np.zeros(pe.THETA_DIM) if m == "c" else th[m]
                j = pe._member_jobs(ts, pe.PROBE, {m: theta}, [we], 0, H, 1, normaliser=nm)[0]
                r = dt.play(j)
                out.append(f"{'veto' if on else 'as-is'}: units {r['row']['units']} dead "
                           f"{r['row']['dead']} at {r['death'] and r['death']['cycle']} "
                           f"(vetoed {VETO['n']})")
            print(world, f"H{H}", "rf3" if m == "c" else m, " | ".join(out), flush=True)
