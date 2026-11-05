"""EXP-0091 reported-only analyses (supervisor, before launch; never change the L3 reading):
1. per generation: |theta_k|, |theta_b|, |theta_e| (checkpoints) and the mean effective buffer
   k_g = 5 exp(clip(theta_k . phi_g, +-2)) over the input rows, evaluated on EXP-0090's recorded bar
   feature rows (phi under the base, normalised by the bar's normaliser) -- an approximation:
   the features the centre itself would see differ;
2. at generation 20: centre vs entity_planner_buyahead_k6_rf5 on the 16 monitor world-episodes at
   replicates 0 / 1 (k6 replayed on the pod, `k6_monitor/`: EXP-0088's k6 rows are EXP-0057 worlds,
   not the training monitor set), and vs the base; deaths per monitored generation.
Usage (from tokenomics/): python ../experiments/EXP-0091-pilot-k5rf5/reported_only.py"""
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from tokenomics.train import producer_es as pe
from tokenomics.train.producer_residual import K_CLIP, N_IN, Normaliser

HERE = Path(__file__).resolve().parent
RUN = HERE / "pilot-s0"
EXP0090 = HERE.parent / "EXP-0090-k5rf5-bar-noise"


def main() -> dict:
    nm = Normaliser.from_dict(json.loads((EXP0090 / "bar" / "normaliser.json").read_text()))
    phi = np.concatenate([np.load(p)["phi"] for p in sorted(
        (EXP0090 / "artifacts" / "bar_fit_rows").glob("*.npz"))])
    z = np.stack([nm.input(f) for f in phi])
    out = {"per_generation": []}
    for ck in sorted((RUN / "checkpoints").glob("gen-*.json")):
        d = pe.load_checkpoint(ck)
        th = np.asarray(d["theta"])
        tk, tb, te = th[:N_IN], th[N_IN:2 * N_IN], th[2 * N_IN:]
        kg = 5.0 * np.exp(np.clip(z @ tk, -K_CLIP, K_CLIP))
        out["per_generation"].append({
            "generation": d["generation"], "norm_theta_k": float(np.linalg.norm(tk)),
            "norm_theta_b": float(np.linalg.norm(tb)), "norm_theta_e": float(np.linalg.norm(te)),
            "mean_k_g": float(kg.mean()), "median_k_g": float(np.median(kg)),
            "share_k_g_above_5_5": float((kg > 5.5).mean()), "theta_k": [float(x) for x in tk]})
    mon = pe.read_csv(RUN / "monitor_episodes.csv")
    k6 = {(r["world"], r["episode"], r["replicate"]): r
          for r in pe.read_csv(HERE / "k6_monitor" / "k6_monitor_episodes.csv")}
    key = lambda r: (r["world"], r["episode"], r["replicate"])  # noqa: E731
    base = {key(r): r for r in mon if r["kind"] == "monitor_base"}
    deaths = {}
    for r in mon:
        if r["kind"] == "monitor":
            deaths.setdefault(r["generation"], [0, 0])
            deaths[r["generation"]][0] += r["dead"] == "true"
            deaths[r["generation"]][1] += 1
    out["monitor_deaths"] = {g: {"dead": a, "episodes": n} for g, (a, n) in deaths.items()}
    out["base_deaths"] = sum(r["dead"] == "true" for r in base.values())
    out["k6_deaths"] = sum(r["dead"] == "true" for r in k6.values())
    cen = {key(r): r for r in mon if r["kind"] == "monitor" and r["generation"] == "20"}
    assert set(cen) == set(k6) == set(base), "monitor keys differ"
    assert all(k6[k]["demand_digest"] == cen[k]["demand_digest"] for k in cen)
    wes = sorted({(k[0], k[1]) for k in cen})

    def comp(a: dict, b: dict) -> float:  # equal weight over world-episodes, mean over replicates
        return fmean(fmean(float(a[(w, e, r)]["units"]) - float(b[(w, e, r)]["units"])
                           for r in ("0", "1")) for w, e in wes)

    out["generation_20"] = {
        "center_minus_base": comp(cen, base), "k6_minus_base": comp(k6, base),
        "center_minus_k6": comp(cen, k6),
        "units_mean": {"center": fmean(float(r["units"]) for r in cen.values()),
                       "k6": fmean(float(r["units"]) for r in k6.values()),
                       "base": fmean(float(r["units"]) for r in base.values())},
        "holding_mean": {n: fmean(float(r["holding_units_per_cycle"] or 0) for r in d.values())
                         for n, d in (("center", cen), ("k6", k6), ("base", base))},
        "world_episodes": len(wes)}
    return out


if __name__ == "__main__":
    doc = main()
    (HERE / "reported_only.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "per_generation"}, indent=1))
    for g in doc["per_generation"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k != "theta_k"})
