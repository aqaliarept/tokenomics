"""EXP-0075 offline read (task 7.17): EXP-0073's pre-registered reading (timing / scarcity /
mixed per stack; 12 decides, 42 reported; rules in ../EXP-0073-ttl-sweep/read_ttl_sweep.py,
unchanged) on the fixed-TTL runs, plus the per-TTL loss-class table from the supply observer.
Usage: python read_exp0075.py <EXP-0075 dir> <EXP-0057 episodes.csv>"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(1 << 26)  # the st_producer_cycles JSON cell

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "read_ttl_sweep", HERE.parent / "EXP-0073-ttl-sweep" / "read_ttl_sweep.py")
r73 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r73)  # type: ignore[union-attr]

TABLE = {  # the supervisor's columns; `other` = every remaining class
    "input_late": ("b2_late",),
    "input_died_upstream": ("b2_lost_ttl", "b2_lost_ttl_outage"),
    "input_outbid_or_no_fit_upstream": ("b2_lost_displaced", "b2_lost_no_fit"),
    "contention": ("b4_contention",),
    "capacity": ("a_held", "a_degraded", "a_max", "a_dropped"),
    "outbid": ("c_displaced",),
}


def class_table(exp: Path) -> dict:
    out: dict = {}
    for t in r73.TTLS:
        rows = list(csv.DictReader((exp / f"ttl{t}" / "supply_trace.csv").open()))
        cls = [k for k in rows[0] if k not in ("size", "world", "episode", "arm", "replicate",
                                               "units", "lost")]
        agg: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            a = agg[(r["size"], r["arm"])]
            a["episodes"] += 1
            a["units"] += float(r["units"])
            a["lost"] += float(r["lost"])
            for k in cls:
                a[k] += float(r[k])
        for (size, arm), a in agg.items():
            named = {g: sum(a[k] for k in ks) for g, ks in TABLE.items()}
            named["other"] = a["lost"] - sum(named.values())
            out.setdefault(f"{size}/{arm}", {})[t] = {
                "episodes": a["episodes"], "units": a["units"], "lost": a["lost"], **named}
    for cell, per in out.items():  # recovery of the late-input loss against TTL 2
        base = per[2]["input_late"] + per[2]["input_died_upstream"]
        for t in (3, 4):
            now = per[t]["input_late"] + per[t]["input_died_upstream"]
            per[t]["late_or_died_recovered_share"] = (base - now) / base if base else None
    return out


def registered_reading(exp: Path, exp0057: Path) -> dict:
    """`read_ttl_sweep.main` with the per-world noise floors taken from EXP-0073's ttl2 spec
    (the same atlas v2 floors; supply_trace's pins.sample omits them). Rules unchanged."""
    floors = {w["world"]: float(w["noise_floor"]) for w in json.loads(
        (HERE.parent / "EXP-0073-ttl-sweep" / "ttl2" / "spec.json").read_text())["pins"]["sample"]}
    real_loads = json.loads

    def loads(text, *a, **k):  # inject the floors into each run's spec as it is read
        obj = real_loads(text, *a, **k)
        if isinstance(obj, dict) and "pins" in obj and "sample" in obj.get("pins", {}):
            for w in obj["pins"]["sample"]:
                w.setdefault("noise_floor", floors[w["world"]])
        return obj

    r73.json.loads = loads
    try:
        return r73.main(exp, exp0057)
    finally:
        r73.json.loads = real_loads


def main(exp: Path, exp0057: Path) -> dict:
    res = registered_reading(exp, exp0057)  # TTL-2 neutrality vs EXP-0057 + the reading
    res["loss_classes"] = class_table(exp)
    res["driver_neutrality"] = {
        t: {k: json.loads((exp / f"ttl{t}" / "reading.json").read_text()).get(k)
            for k in ("status", "neutral_rows", "compared_rows", "neutral_rows_exp0073",
                      "compared_rows_exp0073")} for t in r73.TTLS}
    return res


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2])), indent=1, default=str))
