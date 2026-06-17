"""Scale training/eval lab (scratch, not shipped). Beat the best baseline on the big
(depth-3 scale) environment. Usable as: python -m tokenomics._scale_lab <cmd> ..."""
from __future__ import annotations

import sys
import tempfile

from tokenomics.eval.baselines import (ordering_even_split, ordering_proportional,
                                       ordering_obs_price, producing_noop,
                                       producing_obs_stock_aware, producing_stock_aware)
from tokenomics.eval.runner import run_episode
from tokenomics.eval.scale import build_scale_config
from tokenomics.env import TokenomicsAECEnv

QUIET = lambda *a, **k: None
TRAIN_SEED = 0
HELDOUT = range(10000, 10006)


def _eval(prod, order, seeds, depth=3, horizon=100):
    u, r, d = [], [], []
    for s in seeds:
        st = run_episode(TokenomicsAECEnv(config=build_scale_config(s, depth=depth), horizon=horizon), prod, order, s)
        u.append(st["consumer_completed_units"]); r.append(st["return"]); d.append(1.0 if st["dead"] else 0.0)
    return sum(u)/len(u), sum(r)/len(r), sum(d)/len(d)


def baselines():
    prods = {"obs_stock_aware": producing_obs_stock_aware(0.6), "stock_aware(WB)": producing_stock_aware(0.6)}
    ords = {"even_split": ordering_even_split, "proportional(WB)": ordering_proportional, "obs_price": ordering_obs_price}
    for tag, seeds in (("FIXED seed0", [TRAIN_SEED]), ("HELD-OUT", HELDOUT)):
        print(f"--- {tag} ---")
        for pn, pp in prods.items():
            for on, op in ords.items():
                u, r, d = _eval(pp, op, seeds)
                print(f"  {pn:16s} x {on:16s} units={u:6.1f} return={r:8.0f} dead={d:.2f}")
        un, rn, dn = _eval(producing_noop, ordering_proportional, seeds)
        print(f"  {'noop':16s} x {'proportional':16s} units={un:6.1f} return={rn:8.0f} dead={dn:.2f}")


def train_eval(**kw):
    """Train one config and report fixed-seed + held-out eval. kw passed to a fresh
    train() call; `dr` and `depth` handled here."""
    from tokenomics.train.smoke_ppo import (Curriculum, load_policies, train,
                                            staged_config)
    depth = kw.pop("depth", 3)
    dr_mode = kw.pop("dr", None)  # None | "seeds"
    dwell = int(kw.pop("dwell", 20))  # iters to hold each random topology
    net_base = build_scale_config(TRAIN_SEED, depth=depth)
    use_cur = kw.pop("curriculum", True)
    cur_kw = {k[4:]: kw.pop(k) for k in list(kw) if k.startswith("cur_")}  # cur_gate=, cur_d_step=, ...
    cur = Curriculum(**cur_kw) if use_cur else None
    dr = None
    if dr_mode == "seeds":
        # Topology DR with DWELL. A fresh random scale topology (the held-out family),
        # EASED by the curriculum's live difficulty. Two subtleties this addresses:
        #  (1) easing: train() applies staged_config to base_cfg then hands it here;
        #      replacing it with a full-difficulty random config discards the easing, so
        #      survival never bootstraps. Re-apply the same easing to the random topo.
        #  (2) dwell: a NEW topology every iter forces train() to rebuild rollout envs
        #      every iter, so in-progress episodes never reach horizon and NEVER
        #      complete (at eased difficulty nothing dies in one short rollout either).
        #      The gate then starves (no episodes => no evidence => pinned at 0.0) and
        #      the policy only ever sees cycles 0-9. Hold each topology for `dwell` iters
        #      so the (seed, difficulty) pair is stable -> staged_config returns an EQUAL
        #      config -> envs persist -> episodes run full-length and the gate gets
        #      evidence; rotate the seed every `dwell` iters for generalization breadth.
        state = {"seed": None, "left": 0}
        def dr(c, rng):
            if state["left"] <= 0:
                state["seed"] = int(rng.integers(1, 9999))
                state["left"] = dwell
            state["left"] -= 1
            cfg = build_scale_config(state["seed"], depth=depth)
            d = cur._gated_d if (cur is not None and cur.gate) else 1.0
            return staged_config(cfg, d)
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    train(config=net_base, horizon=100, num_envs=4, curriculum=cur, domain_random=dr,
          eval_seeds=2, save=f.name, log=print, **kw)
    pp, op = load_policies(f.name)
    fu, fr, fd = _eval(pp, op, [TRAIN_SEED], depth=depth)
    hu, hr, hd = _eval(pp, op, HELDOUT, depth=depth)
    print(f"\nRESULT fixed_seed0: units={fu:.1f} return={fr:.0f} dead={fd:.2f}")
    print(f"RESULT held_out:    units={hu:.1f} return={hr:.0f} dead={hd:.2f}")
    print(f"CHECKPOINT {f.name}")


def _parse(argv):
    kw = {}
    for a in argv:
        k, v = a.split("=", 1)
        if v in ("True", "False"):
            kw[k] = v == "True"
        elif v == "None":
            kw[k] = None
        elif "." in v or "e" in v:
            try: kw[k] = float(v)
            except ValueError: kw[k] = v
        else:
            try: kw[k] = int(v)
            except ValueError: kw[k] = v
    return kw


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "baselines":
        baselines()
    elif cmd == "train":
        train_eval(**_parse(sys.argv[2:]))
