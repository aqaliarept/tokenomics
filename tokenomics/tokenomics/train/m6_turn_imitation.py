"""Task 4.3 (add-m6-joint-mpg-gnn, ADR-0062): the producer GNN's turn heads imitate today's acceptance, fund
spend order and claim scheduling (pre-registration m6_prereg_v12; notes/4.3-draft.md).

- **Start:** EXP-0100's stack, frozen (trunk, investment heads, orderer). Only `TurnHeads` is fitted, from a
  seeded fresh init.
- **Labels:** the default turn policies (`default_turn_decision`, `default_fund_scores`):
  - the accept bit per routed bid;
  - one priority order per accept turn over carried reservations, pending claims and accepted bids (declined
    bids' priorities never act, so they carry no order label);
  - one order per fund turn over the funding requests.
- **Losses:** BCE on accept bits; a pairwise logistic loss on every ordered pair within a turn.
- **Data:** the start stack with the default turns plays the 112 M4 training world-episodes × r0 / r1 (fit) and
  the 16 monitor world-episodes × r0 / r1 (held out). The trunk is frozen, so the workers compute its
  embeddings once per recorded producer-cycle (`TurnCache`'s sharing) and the fit runs only the heads, batched.
- **Offline gate** (held-out turns): accept agreement ≥ 0.99; pairwise rank agreement ≥ 0.99 for the accept-turn
  order and for the fund order.
- **Closed loop** (only after the offline gate passes):
  - the candidate (greedy GNN turn heads) against the reference (default turns), the same start stack,
    16 monitor world-episodes × r0 / r1, M5's floors;
  - `pass` iff g ≥ −F, no more deaths, run-down and starvation shares no worse;
  - 42 producers reported on the same rule (atlas v2 train-split 42 worlds, fresh seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import m6_gnn as g
from . import m6_imitation as im

PREREG_V12_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v12.json"
PREREG_V12_SHA256: str | None = "3c0714f7d7c5e3bd80db8870025e732a2de5dc5de54d5f46858b571ed303d67d"  # frozen before EXP-0117 data
CFG = {"horizon": 100, "replicates": [0, 1], "epochs": 20, "lr": 1e-3, "batch_turns": 512,
       "max_pairs_per_turn": 400, "seed": 0, "gate": {"accept": 0.99, "rank_accept": 0.99,
                                                      "rank_fund": 0.99},
       "r42": {"worlds": 8, "episodes": 1, "stage": "m6-turn-imit-42"}}
START = "EXP-0100 artifacts/fit/models.pt (im.EXP0100_MODELS_SHA)"


def registered_prereg_v12() -> dict:
    return {
        "format": "m6-prereg-v12", "change": "add-m6-joint-mpg-gnn", "task": "4.3",
        "stage": "EXP-0117: turn-head imitation of the default acceptance, fund order and claim scheduling",
        "start": START, "fitted": "TurnHeads only (fresh seeded init); trunk, investment heads, orderer frozen",
        "labels": "default_turn_decision (accept bits; one priority order over reservations, pending claims "
                  "and accepted bids) and default_fund_scores (fund order)",
        "loss": "BCE on accept bits + pairwise logistic on every ordered pair within a turn "
                f"(≤ {CFG['max_pairs_per_turn']} sampled pairs per turn in training)",
        "data": "start stack + default turns: 112 M4 training world-episodes × r0/r1 (fit), 16 monitor "
                "world-episodes × r0/r1 (held out), horizon 100, 12 producers",
        "offline_gate": "held-out turns: accept agreement ≥ 0.99, pairwise rank agreement ≥ 0.99 for the "
                        "accept-turn order and for the fund order; reported: share of turns whose whole "
                        "schedule (accept set and order) equals the default's; failing → no closed loop",
        "closed_loop": "candidate (greedy GNN turn heads, TurnCache) vs reference (default turns), same start "
                       "stack, 16 monitor world-episodes × r0/r1, M5's floors (g, F); pass iff g ≥ −F, deaths, "
                       "run-down and starvation shares no worse; 42 reported (atlas v2 train-split 42 worlds, "
                       f"{CFG['r42']['worlds']} worlds × {CFG['r42']['episodes']} fresh episode × r0/r1)",
        "config": CFG,
    }


def freeze() -> str:
    from ..eval.provenance import sha256_file

    PREREG_V12_PATH.write_text(json.dumps(registered_prereg_v12(), indent=1, sort_keys=True) + "\n")
    return sha256_file(str(PREREG_V12_PATH))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V12_SHA256 is None:
        return ["no pinned m6_prereg_v12.json (`--mode freeze`)"]
    if not PREREG_V12_PATH.exists() or sha256_file(str(PREREG_V12_PATH)) != PREREG_V12_SHA256:
        return ["m6_prereg_v12.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V12_PATH.read_text()) != registered_prereg_v12():
        return ["the v12 pre-registration differs from the code"]
    return []


# --- episodes ---


def _turn_env(job: dict) -> Any:
    from ..eval.joint_stack import joint_stack_env
    from ..eval.routing_audit import job_config
    from ..eval.routing_diagnostics import DrawKey, set_draw_key

    env = joint_stack_env(job_config(job), job["horizon"], producer_turns=True)
    set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
    return env


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().numpy().astype(np.float32)


def record_job(job: dict) -> dict:
    """The start stack with the default turns on one keyed world-episode; every fund turn and accept turn
    with something to decide is recorded: its turn features, the frozen trunk's capacity / input embeddings
    and pooled embedding, and the default labels."""
    from ..env.producer_turns import default_fund_scores, encode_accept_action
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _turn_env(job)
    pm, om = im.load_models(job["models"])
    pm.eval()
    cache = g.TurnCache()
    recs: list[dict] = []

    def emb(pid: str) -> tuple[Any, dict]:
        view, _f, (h, pooled) = cache.get(pm, env, pid)
        return view, {"h_cap": _np(h["capacity"]), "h_in": _np(h["input"]), "pooled": _np(pooled)}

    def fund(e: Any, agent: str) -> np.ndarray:
        from ..env.producer_turns import default_fund_policy

        pid = e._role[agent][1]
        reqs = e.system.funding_requests(pid)
        if reqs:
            view, x = emb(pid)
            t = g.fund_features(e.system.producing_agents[pid], reqs, e.system.funding_budget(pid), view)
            recs.append({"kind": "fund", **x, "fund": t["fund"], "fund_cap": t["fund_cap"],
                         "fund_in": t["fund_in"], "y": np.asarray(default_fund_scores(len(reqs)),
                                                                   np.float32)})
        return default_fund_policy(e, agent)

    def accept(e: Any, agent: str) -> np.ndarray:
        pid = e._role[agent][1]
        p = e.system.producing_agents[pid]
        d = p.default_turn_decision()
        lists = p.turn_lists()
        if lists.bids or lists.reservations or lists.pending:
            view, x = emb(pid)
            t = g.accept_features(p, lists, view)
            recs.append({"kind": "accept", **x, **{k: t[k] for k in ("bid", "bid_cap", "bid_in", "claim",
                                                                     "claim_cap", "claim_in", "n_res")},
                         "acc": np.asarray(d.accept, np.float32),
                         "y_bid": np.asarray(d.bid_priority, np.float32),
                         "y_claim": np.asarray(d.reservation_priority + d.pending_priority, np.float32)})
        return encode_accept_action(d)

    s = run_episode(env, g.gnn_producing_policy(pm), g.gnn_ordering_policy(om), seed=job["episode"],
                    action_seed=None, fund_policy=fund, accept_policy=accept)
    return {"job": job["job_id"], "records": recs, "units": float(s["consumer_completed_units"]),
            "dead": bool(s["dead"])}


def gate_job(job: dict) -> dict[str, str]:
    """One closed-loop episode: `job["arm"]` = `candidate` (GNN turn heads) or `reference` (default turns)."""
    from ..eval.gate import row_from_stats, row_to_dict
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _turn_env(job)
    pm, om = im.load_models(job["models"])
    kw = {}
    if job["arm"] == "candidate":
        fp, ap = g.gnn_turn_policies(pm)
        kw = {"fund_policy": fp, "accept_policy": ap}
    st = run_episode(env, g.gnn_producing_policy(pm), g.gnn_ordering_policy(om), seed=job["episode"],
                     action_seed=None, **kw)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     f"{job['arm']}@h{job['horizon']}", job["replicate"], st))
    row.update({"arm": job["arm"], "set": job["set"], "units": repr(float(st["consumer_completed_units"])),
                "beyond_k": str(env.turn_beyond_k)})
    return row


# --- the fit (batched over turns) ---


def _cat(recs: Sequence[dict], kind: str) -> dict[str, Any]:
    """Concatenate `kind` nodes over turns, offsetting edges into the concatenated trunk rows."""
    xs, ec, ei, node_turn, hc, hi, pooled = [], [], [], [], [], [], []
    n = rc = ri = 0
    for k, r in enumerate(recs):
        x = r[kind]
        xs.append(x)
        ec += [(n + a, rc + b) for a, b in r[f"{kind}_cap"]]
        ei += [(n + a, ri + b) for a, b in r[f"{kind}_in"]]
        node_turn += [k] * x.shape[0]
        hc.append(r["h_cap"])
        hi.append(r["h_in"])
        pooled.append(r["pooled"])
        n, rc, ri = n + x.shape[0], rc + r["h_cap"].shape[0], ri + r["h_in"].shape[0]
    hid = recs[0]["pooled"].shape[0] // 4 if recs else g.HIDDEN
    return {"x": np.concatenate(xs, 0) if xs else np.zeros((0, 1), np.float32), "ec": ec, "ei": ei,
            "turn": torch.tensor(node_turn, dtype=torch.long),
            "h_cap": torch.from_numpy(np.concatenate(hc, 0)) if hc else torch.zeros((0, hid)),
            "h_in": torch.from_numpy(np.concatenate(hi, 0)) if hi else torch.zeros((0, hid)),
            "pooled": torch.from_numpy(np.stack(pooled, 0)) if pooled else torch.zeros((0, 4 * hid))}


def batch_out(heads: g.TurnHeads, kind: str, b: Mapping[str, Any]) -> torch.Tensor:
    """`TurnHeads.node_out` over a concatenated batch (identical per node; tested)."""
    x = b["x"]
    if not x.shape[0]:
        return torch.zeros((0, heads.out[kind].out_features))
    e = heads.enc[kind](heads.norm(kind, x))
    n = e.shape[0]
    cc = g._mean_ctx(b["ec"], n, b["h_cap"], True)
    ci = g._mean_ctx(b["ei"], n, b["h_in"], True)
    z = torch.cat([e, cc, ci, b["pooled"][b["turn"]]], 1)
    return heads.out[kind](torch.relu(heads.mix[kind](z)))


def _pairs(y: np.ndarray, mask: np.ndarray | None, cap: int | None, rng: np.random.Generator | None
           ) -> list[tuple[int, int]]:
    """Ordered pairs (i above j) among the items in `mask` (all if None); `cap` sampled with `rng`."""
    idx = np.arange(len(y)) if mask is None else np.nonzero(mask)[0]
    out = [(int(i), int(j)) for i in idx for j in idx if y[i] > y[j]]
    if cap is not None and len(out) > cap and rng is not None:
        out = [out[k] for k in rng.choice(len(out), cap, replace=False)]
    return out


def turn_targets(recs: Sequence[dict], rng: np.random.Generator | None, cap: int | None
                 ) -> dict[str, Any]:
    """Global indices of the BCE targets and the ordered pairs of a batch of turns."""
    fund = [r for r in recs if r["kind"] == "fund"]
    acc = [r for r in recs if r["kind"] == "accept"]
    fp, off = [], 0
    for r in fund:
        fp += [(off + i, off + j) for i, j in _pairs(r["y"], None, cap, rng)]
        off += len(r["y"])
    ap, bits, ob, oc = [], [], 0, 0
    nb_total = sum(len(r["acc"]) for r in acc)
    for r in acc:
        nb, nc = len(r["acc"]), len(r["y_claim"])
        y = np.concatenate([r["y_bid"], r["y_claim"]])
        mask = np.concatenate([r["acc"] > 0, np.ones(nc, bool)])
        # combined vector = [all bids of the batch, then all claims]: map local -> global
        loc = np.concatenate([ob + np.arange(nb), nb_total + oc + np.arange(nc)])
        ap += [(int(loc[i]), int(loc[j])) for i, j in _pairs(y, mask, cap, rng)]
        bits.append(r["acc"])
        ob, oc = ob + nb, oc + nc
    return {"fund": fund, "accept": acc, "fund_pairs": fp, "acc_pairs": ap,
            "bits": torch.from_numpy(np.concatenate(bits)) if bits else torch.zeros(0)}


def _losses(heads: g.TurnHeads, tg: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    loss, out = torch.zeros(()), {}
    if tg["fund"]:
        s = batch_out(heads, "fund", _cat(tg["fund"], "fund")).squeeze(-1)
        out["fund"] = s
        if tg["fund_pairs"]:
            i, j = torch.tensor(tg["fund_pairs"]).T
            loss = loss + torch.nn.functional.softplus(-(s[i] - s[j])).mean()
    if tg["accept"]:
        b = batch_out(heads, "bid", _cat(tg["accept"], "bid"))
        c = batch_out(heads, "claim", _cat(tg["accept"], "claim")).squeeze(-1)
        out["acc"], out["pri"] = b[:, 0], torch.cat([b[:, 1], c])
        if b.shape[0]:
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(b[:, 0], tg["bits"])
        if tg["acc_pairs"]:
            i, j = torch.tensor(tg["acc_pairs"]).T
            loss = loss + torch.nn.functional.softplus(-(out["pri"][i] - out["pri"][j])).mean()
    return loss, out


def fit(pm: g.ProducerGNN, train: Sequence[dict], epochs: int, lr: float, batch: int, cap: int,
        seed: int, log: Any = print) -> dict[str, Any]:
    torch.manual_seed(seed)
    pm.add_turn_heads()
    heads = pm.turn
    assert heads is not None
    blank = {"fund": np.zeros((0, g.FUND_DIM), np.float32), "bid": np.zeros((0, g.BID_DIM), np.float32),
             "claim": np.zeros((0, g.CLAIM_DIM), np.float32)}
    norm = heads.norm.fit([{**blank, **{k: r[k] for k in ("fund", "bid", "claim") if k in r}}
                           for r in train])
    opt = torch.optim.Adam(heads.parameters(), lr=lr)
    rng = np.random.default_rng([seed, 117])
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(train))
        tot, nb, t0 = 0.0, 0, time.perf_counter()
        for k in range(0, len(order), batch):
            tg = turn_targets([train[i] for i in order[k:k + batch]], rng, cap)
            loss, _ = _losses(heads, tg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        hist.append({"epoch": ep + 1, "loss": tot / max(1, nb), "seconds": round(time.perf_counter() - t0, 1)})
        log(f"[turn_imit] epoch {ep + 1}: loss {hist[-1]['loss']:.5f} ({hist[-1]['seconds']} s)")
    return {"norm": norm, "history": hist}


def agreement(pm: g.ProducerGNN, recs: Sequence[dict], batch: int = 2048, whole: bool = True
              ) -> dict[str, Any]:
    """Held-out agreement: accept bits, pairwise order (accept turns: claims + accepted bids; fund turns),
    and the share of turns whose whole schedule equals the default's."""
    heads = pm.turn
    assert heads is not None
    acc_ok = acc_n = ap_ok = ap_n = fp_ok = fp_n = 0
    whole_ok = whole_n = 0
    with torch.no_grad():
        for k in range(0, len(recs), batch):
            part = list(recs[k:k + batch])
            tg = turn_targets(part, None, None)
            _, out = _losses(heads, tg)
            if tg["fund_pairs"]:
                i, j = torch.tensor(tg["fund_pairs"]).T
                fp_ok += int((out["fund"][i] > out["fund"][j]).sum())
                fp_n += len(tg["fund_pairs"])
            if tg["accept"]:
                pred = out["acc"] > 0
                acc_ok += int((pred == (tg["bits"] > 0)).sum())
                acc_n += len(tg["bits"])
                if tg["acc_pairs"]:
                    i, j = torch.tensor(tg["acc_pairs"]).T
                    ap_ok += int((out["pri"][i] > out["pri"][j]).sum())
                    ap_n += len(tg["acc_pairs"])
            # whole schedule per turn (both kinds), from per-turn evaluations
            for r in part if whole else ():
                t1 = turn_targets([r], None, None)
                _, o = _losses(heads, t1)
                ok = True
                if r["kind"] == "fund" and t1["fund_pairs"]:
                    i, j = torch.tensor(t1["fund_pairs"]).T
                    ok = bool((o["fund"][i] > o["fund"][j]).all())
                if r["kind"] == "accept":
                    ok = bool(((o["acc"] > 0) == (t1["bits"] > 0)).all())
                    if ok and t1["acc_pairs"]:
                        i, j = torch.tensor(t1["acc_pairs"]).T
                        ok = bool((o["pri"][i] > o["pri"][j]).all())
                whole_ok, whole_n = whole_ok + int(ok), whole_n + 1
    rate = lambda a, b: a / b if b else None  # noqa: E731
    return {"accept": rate(acc_ok, acc_n), "accept_n": acc_n, "rank_accept": rate(ap_ok, ap_n),
            "rank_accept_pairs": ap_n, "rank_fund": rate(fp_ok, fp_n), "rank_fund_pairs": fp_n,
            "whole_schedule": rate(whole_ok, whole_n), "turns": whole_n}


def offline_pass(a: Mapping[str, Any]) -> bool:
    gt = CFG["gate"]
    return all(a[k] is None or a[k] >= gt[k] for k in ("accept", "rank_accept", "rank_fund"))


def closed_loop_reading(rows: Sequence[Mapping[str, str]], cells: Mapping, sset: str) -> dict[str, Any]:
    from .order_es_validate import row_flags
    from .producer_read import pilot_reading

    sel = [r for r in rows if r["set"] == sset]
    pr = pilot_reading([{**r, "kind": "monitor", "generation": "0"} for r in sel if r["arm"] == "candidate"]
                       + [{**r, "kind": "monitor_base"} for r in sel if r["arm"] == "reference"], cells, 0)

    def flags(arm: str) -> dict[str, Any]:
        f = [row_flags(r) for r in sel if r["arm"] == arm]
        u = [float(r["units"]) for r in sel if r["arm"] == arm]
        return {"rows": len(f), "deaths": sum(x[0] for x in f),
                "rundown_share": sum(x[1] for x in f) / len(f) if f else None,
                "starvation_share": sum(x[2] for x in f) / len(f) if f else None,
                "units": statistics.fmean(u) if u else None,
                "beyond_k": sum(int(r.get("beyond_k") or 0) for r in sel if r["arm"] == arm)}

    c, r = flags("candidate"), flags("reference")
    g_, f_ = pr.get("g_tr"), pr.get("F_tr")
    ok = (g_ is not None and f_ is not None and g_ >= -f_ and c["deaths"] <= r["deaths"]
          and (c["rundown_share"] or 0) <= (r["rundown_share"] or 0)
          and (c["starvation_share"] or 0) <= (r["starvation_share"] or 0))
    return {"g": g_, "F": f_, "candidate": c, "reference": r, "reading": "pass" if ok else "fail"}


# --- the run ---


def _jobs(ts: Any, wes: Sequence, kind: str, models: str, horizon: int) -> list[dict]:
    return [{**j, "job_id": f"{kind}|{j['world']}|{j['episode']}|{j['replicate']}"}
            for j in im._jobs(ts, wes, kind, CFG["replicates"], horizon, models)]


def _jobs_42(atlas: Mapping, models: str, horizon: int, max_worlds: int) -> tuple[list[dict], dict]:
    from ..eval.atlas import space_info
    from ..eval.teacher_audit import world_view
    from .order_es_validate import Cell, spread_from_dict
    from .producer_read import entry_for

    views = {world_view(e)["world"]: world_view(e) for e in atlas["worlds"]}
    worlds = sorted(w for w, v in views.items() if v["size"] == 42 and v["split"] == "train"
                    and v["class"] != "collapse")[:max_worlds]
    jobs, cells = [], {}
    for w in worlds:
        e = entry_for(views[w], CFG["r42"]["episodes"], CFG["r42"]["stage"])
        rec = next(x["record"] for x in atlas["worlds"] if world_view(x)["world"] == w)
        info = space_info(rec)
        key = f"{info.size}/{info.choice}"
        for ep in e.episode_seeds:
            cells[f"{e.id}#{int(ep)}"] = Cell(f"{e.id}#{int(ep)}", info.size, info.choice, info.differing,
                                               spread_from_dict(atlas["spreads"][key])
                                               if key in atlas["spreads"] else None)
            for r in CFG["replicates"]:
                jobs.append({"world": e.id, "size": e.size, "regime": e.regime, "world_seed": e.world_seed,
                             "generator": e.generator, "episode": int(ep), "replicate": r,
                             "horizon": horizon, "kind": "r42", "models": models,
                             "job_id": f"r42|{e.id}|{ep}|{r}"})
    return jobs, cells


def run(training_dir: str, atlas_path: str, models: str, out_dir: str, workers: int, dry_run: bool
        ) -> int:
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from .order_es_validate import monitor_cells

    if not dry_run:
        im._refuse(prereg_problems()
                   + ([] if sha256_file(models) == im.EXP0100_MODELS_SHA else ["--models is not EXP-0100's"])
                   + ([] if len(im.SOURCE_COMMIT) == 40 else ["a real run needs a 40-hex --source-commit"]))
    atlas = load_atlas(atlas_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts, train, mon = im._training(training_dir)
    horizon, epochs, n42 = CFG["horizon"], CFG["epochs"], CFG["r42"]["worlds"]
    if dry_run:  # a true smoke: 1 + 1 world-episodes, horizon 6, 2 epochs, 1 world at 42
        train, mon, horizon, epochs, n42 = train[:1], mon[:1], 6, 2, 1
    t0 = time.perf_counter()
    rec_train = im._pool(record_job, _jobs(ts, train, "turn_train", models, horizon), workers)
    rec_mon = im._pool(record_job, _jobs(ts, mon, "turn_monitor", models, horizon), workers)
    tr = [r for x in rec_train for r in x["records"]]
    ho = [r for x in rec_mon for r in x["records"]]
    collect = {"train_turns": len(tr), "heldout_turns": len(ho),
               "train_fund": sum(r["kind"] == "fund" for r in tr),
               "train_accept": sum(r["kind"] == "accept" for r in tr),
               "seconds": round(time.perf_counter() - t0, 1)}
    print(f"[turn_imit] collected {collect}", flush=True)
    pm, om = im.load_models(models)
    t1 = time.perf_counter()
    rep = fit(pm, tr, epochs, CFG["lr"], CFG["batch_turns"], CFG["max_pairs_per_turn"], CFG["seed"])
    rep["seconds"] = round(time.perf_counter() - t1, 1)
    cand = out / "models.pt"
    im.save_models(pm, om, cand)
    offline = {"train": agreement(pm, tr, whole=False), "heldout": agreement(pm, ho)}
    offline["pass"] = offline_pass(offline["heldout"])
    print(f"[turn_imit] offline {offline['heldout']} pass {offline['pass']}", flush=True)
    doc: dict[str, Any] = {"prereg": PREREG_V12_SHA256, "source_commit": im.SOURCE_COMMIT, "dry_run": dry_run,
                           "start_sha256": sha256_file(models), "candidate_sha256": sha256_file(str(cand)),
                           "collect": collect, "fit": rep, "offline": offline, "reading": None}
    if offline["pass"] or dry_run:  # a smoke exercises the closed loop whatever its offline result
        jobs = [{**j, "arm": a, "set": "monitor", "models": m} for a, m in (("candidate", str(cand)),
                                                                            ("reference", models))
                for j in _jobs(ts, mon, "turn_gate", models, horizon)]
        j42, cells42 = _jobs_42(atlas, models, horizon, n42)
        jobs += [{**j, "arm": a, "set": "r42", "models": m} for a, m in (("candidate", str(cand)),
                                                                         ("reference", models)) for j in j42]
        jobs.sort(key=lambda j: j["size"] != 42)  # the 42-producer episodes first (longest)
        rows = im._pool(gate_job, jobs, workers)
        with (out / "gate_rows.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        cells = monitor_cells(ts.training, atlas["spreads"])
        doc["closed_loop"] = {"monitor_12": closed_loop_reading(rows, cells, "monitor"),
                              "reported_42": closed_loop_reading(rows, cells42, "r42")}
        doc["reading"] = doc["closed_loop"]["monitor_12"]["reading"]
    else:
        doc["reading"] = "offline_fail"
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[turn_imit] reading {doc['reading']}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("freeze", "run"), required=True)
    ap.add_argument("--training-dir")
    ap.add_argument("--atlas")
    ap.add_argument("--models", help="EXP-0100's models.pt (the start)")
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "freeze":
        print(freeze())
        return 0
    im.SOURCE_COMMIT = a.source_commit
    return run(a.training_dir, a.atlas, a.models, a.out_dir, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
