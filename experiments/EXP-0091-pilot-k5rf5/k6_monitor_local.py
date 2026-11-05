"""EXP-0091 reported-only k6 arm, rerun locally with 1 worker (the pod's run of the same jobs
failed: its spawned workers re-ran the runpy-launched script as __main__). Same job path and keys as
the monitor's base rows; 32 episodes. Usage (from tokenomics/): python <this> """
from pathlib import Path

from tokenomics.train import producer_es as pe

E = Path(__file__).resolve().parent
X = E.parent
ts = pe.load_training(X / "EXP-0058-es-gradient-probe" / "bar")
bar, _, _ = pe.load_bar(X / "EXP-0090-k5rf5-bar-noise" / "bar")
ts = pe.matched_to_bar(ts, bar, False)
jobs = [pe.make_job(ts, kind="monitor_k6", producing="entity_planner_buyahead_k6_rf5", theta=None,
                    generation="k6", member="k6", world=w, episode=e, replicate=r,
                    horizon=pe.HORIZON)
        for w, e in pe.monitor_set(ts) for r in pe.MONITOR_REPLICATES]
out = E / "k6_monitor"
out.mkdir(parents=True, exist_ok=True)
rows, failures = pe.play(out, "k6_monitor_episodes.csv", jobs, 1, pe.run_producer_job, bar, print)
print(f"[k6_monitor] {len(rows)} / {len(jobs)} rows, {failures} failures", flush=True)
