"""Task 4.2e: summarise death_trace's pickled logs (rf3 vs members, by tier and cycle window)."""
import pickle, sys
from collections import Counter, defaultdict
d = pickle.load(open(f"/tmp/death_trace_{sys.argv[1]}.pkl", "rb"))
tier = lambda pid: pid[:2]
for lab, r in d.items():
    if "/" in lab and lab.split("/")[1] not in ("e",): continue
    death = r["death"]["cycle"] if r["death"] else None
    exp_req = Counter(); rest_req = Counter(); ev = Counter()
    for c, cmds in r["cmd"].items():
        for pid, cmd in cmds.items():
            if cmd["expand"]: exp_req[tier(pid)] += 1
            if cmd["restore"]: rest_req[tier(pid)] += 1
    for c, pid, tag, typ, q in r["events"]:
        ev[(tier(pid), typ, tag)] += 1
    fund_p0 = sum(sum(v for k, v in f.items() if k.startswith("p0")) for f in r["funds"].values())
    fund_all = sum(sum(f.values()) for f in r["funds"].values())
    cap0 = {c: sum(v[0].get("capacity-0", 0) for k, v in caps.items() if k.startswith("p0"))
            for c, caps in r["caps"].items()}
    print(f"== {lab}: death {death} units {r['row']['units']}")
    print("  expansions requested by tier", dict(exp_req), "| restorations requested", dict(rest_req))
    print("  outcomes", {f"{a}/{b}/{c}": n for (a, b, c), n in sorted(ev.items())})
    print("  p0 share of funds received", round(fund_p0 / max(1, fund_all), 3), "total", fund_all)
    print("  p0 capacity-0 sum by cycle", {c: cap0[c] for c in sorted(cap0) if c % 10 == 0 or c == max(cap0)})
