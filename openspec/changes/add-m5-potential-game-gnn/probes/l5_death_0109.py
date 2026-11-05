"""EXP-0093 stop-branch diagnostic: why cand-s0 (EXP-0092 s0, generation 40) dies on atlas 12-0109
episode 1213081781 replicate 1 where k5_rf5 lives (k4 dies there too). Replays both arms on the L5
read's job path (producer_read.read_jobs) with death_trace's read-only wrappers. Local, one episode."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import death_trace as dt  # noqa: E402
from tokenomics.eval.teacher_audit import load_atlas  # noqa: E402
from tokenomics.train import producer_es as pe  # noqa: E402
from tokenomics.train import producer_read as rd  # noqa: E402

X = Path("/srv/research/tokenomics/experiments")
WORLD, EP, REP = "ws-atlas-v1-12-0109", 1213081781, 1
from tokenomics.train.order_es_validate import VALIDATION_STAGE as STAGE  # noqa: E402


def main() -> None:
    dt._install()
    cand = rd.load_candidate(X / "EXP-0092-l4-k5rf5" / "s0", X / "EXP-0090-k5rf5-bar-noise" / "bar",
                             real=True)
    atlas = load_atlas(X / "EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json")
    views = rd.read_views(atlas, rd.VALIDATE, [WORLD], None)
    entries = [rd.entry_for(v, rd.READ_EPISODES, STAGE) for v in views]
    arms = {"cand-s0": cand, "k5_rf5": {"producing": pe.BASE_PRODUCER}}
    for j in rd.read_jobs(entries, arms, pe.HORIZON, "validate"):
        if j["episode"] != EP or j["replicate"] != REP:
            continue
        j = {**j, "observe": False}
        r = dt.play(j)
        d = r["death"]
        print(j["arm"], r["row"], "DEATH" if d else "", json.dumps(d, default=str) if d else "")
        s = r["system"]
        if d:
            prod = d["product"]
            for pid in d["declaring"]:
                p = s.producing_agents[pid]
                t = next(iter(d["declaring"][pid]))
                caps = [r["caps"][c][pid][0].get(t) for c in sorted(r["caps"])][-15:]
                mx = [r["caps"][c][pid][1].get(t) for c in sorted(r["caps"])][-1]
                rest = [(c, tag) for c, q, tag, typ, _ in r["events"] if q == pid and typ == "RESTORATION"][-8:]
                print(f"  {pid} type {t} deg% {p.degradation} restores {p.restoration.restores} "
                      f"max {mx} cap last 15 {caps} restorations {rest}")
        else:
            for pid in ("p0_0", "p0_1", "p0_2", "p0_3"):
                if pid in r["caps"][max(r["caps"])]:
                    print(f"  {pid} end cap {r['caps'][max(r['caps'])][pid]}")


if __name__ == "__main__" and not ({"--why", "--no-e"} & set(sys.argv)):
    main()


def why() -> None:
    """Restoration fate at cycles 38-46 for p0_0: funds granted, the restoration's required product
    and who makes it, and the competing investment / replenishment demand on those makers."""
    dt._install()
    cand = rd.load_candidate(X / "EXP-0092-l4-k5rf5" / "s0", X / "EXP-0090-k5rf5-bar-noise" / "bar",
                             real=True)
    atlas = load_atlas(X / "EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json")
    views = rd.read_views(atlas, rd.VALIDATE, [WORLD], None)
    entries = [rd.entry_for(v, rd.READ_EPISODES, STAGE) for v in views]
    for arm, spec in (("cand-s0", cand), ("k5_rf5", {"producing": pe.BASE_PRODUCER})):
        j = [j for j in rd.read_jobs(entries, {arm: spec}, pe.HORIZON, "validate")
             if j["episode"] == EP and j["replicate"] == REP][0]
        r = dt.play({**j, "observe": False})
        s = r["system"]
        p = s.producing_agents["p0_0"]
        need = p.restoration.require
        makers = [q for q, a in s.producing_agents.items() if need in a.produces]
        print(f"== {arm}: p0_0 restoration needs product {need} (cost {p.restoration.cost}), made by {makers}")
        for c in range(36, 47):
            if c not in r["funds"]:
                continue
            cmds = r["cmd"].get(c, {})
            exp = sum(1 for x in cmds.values() if x["expand"])
            buy = sum(sum(x["buy"].values()) for x in cmds.values())
            ups = sum(1 for x in cmds.values() if x["upgrade"])
            rest = [q for q, x in cmds.items() if x["restore"]]
            mk = {q: r["caps"][c][q][0] for q in makers if c in r["caps"]}
            print(f"  c{c} fund {r['fund'][c]} p0_0 funds {r['funds'][c]['p0_0']} | restore asked "
                  f"{sorted(rest)} | expansions {exp} upgrades {ups} replenish units {buy} | makers cap {mk}")
        print("  p0_0 restoration events", [(c, t) for c, q, t, typ, _ in r["events"]
                                            if q == "p0_0" and typ == "RESTORATION"][-6:])


if __name__ == "__main__" and "--why" in sys.argv:
    why()


def no_expansion() -> None:
    """DIAGNOSTICS 30 check 2: cand-s0 with theta_e = 0 (theta_k, theta_b kept) on the same episode."""
    import numpy as np

    from tokenomics.train.producer_residual import N_IN

    dt._install()
    cand = rd.load_candidate(X / "EXP-0092-l4-k5rf5" / "s0", X / "EXP-0090-k5rf5-bar-noise" / "bar",
                             real=True)
    th = np.asarray(cand["theta"])
    th[2 * N_IN:] = 0.0
    cand = {**cand, "theta": [float(x) for x in th], "theta_sha256": pe.theta_sha256(th)}
    atlas = load_atlas(X / "EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json")
    views = rd.read_views(atlas, rd.VALIDATE, [WORLD], None)
    entries = [rd.entry_for(v, rd.READ_EPISODES, STAGE) for v in views]
    for j in rd.read_jobs(entries, {"cand-s0-no-e": cand}, pe.HORIZON, "validate"):
        r = dt.play({**j, "observe": False})
        print(j["episode"], j["replicate"], r["row"]["units"], "dead", r["row"]["dead"],
              "invrej", round(float(r["row"]["investment_rejection_rate"]), 3))


if __name__ == "__main__" and "--no-e" in sys.argv:
    no_expansion()
