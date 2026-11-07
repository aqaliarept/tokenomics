"""Task 4.3 (add-m6-joint-mpg-gnn, ADR-0062): the producer GNN's turn heads imitate today's acceptance, fund
spend order and claim scheduling (pre-registration m6_prereg_v12; notes/4.3-draft.md).

- **Start:** EXP-0100's stack, frozen (trunk, investment heads, orderer). Only `TurnHeads` is fitted, from a
  seeded fresh init.
- **Labels:** the default turn policies (`default_turn_decision`, `default_fund_scores`):
  - the accept bit per routed bid;
  - one priority order per accept turn over carried reservations, pending claims and bids. The gate's rank metric
    reads claims + accepted bids (a declined bid's priority never acts in the domain). The order-aware heads also
    train the declined bids' default ranks, because their priority places them in the ordered accept pass, and the
    displaced bids near the capacity threshold are the error class it must fix;
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

from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import m6_imitation as im

# v12 (3c0714f7…, EXP-0117) was stopped before its reading: its offline gate could not pass (review 2026-11-07:
# 3.5% of accept-turn pairs are feature-identical with position-based ranks; 48% of fund pairs are replenishment vs
# replenishment in creation order, which no feature carries). v13 drops those pairs.
PREREG_V12_SHA256 = "3c0714f7d7c5e3bd80db8870025e732a2de5dc5de54d5f46858b571ed303d67d"
PREREG_V13_SHA256 = "97474c2290f687b7320257b2f62b64c1c84cf5c12300a60541b0efd8ab37c186"  # EXP-0118, offline_fail
# v14 (supervisor 2026-11-07, DIAGNOSTICS 38): learned set context (self-attention over a turn's items) and
# bounded scores; the start chosen by EXP-0119's reading (F40 on `gain`, else EXP-0100).
PREREG_V14_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v14.json"
PREREG_V14_SHA256: str | None = "21b60013ae5eb20084bb9789c73ae68a7dce5ffb408a702c30b0e6de343f30bf"  # frozen before EXP-0120 data (re-frozen if the review changes the code)
CFG = {"horizon": 100, "replicates": [0, 1], "epochs": 20, "lr": 1e-3, "batch_turns": 512,
       "max_pairs_per_turn": 400, "score_l2": 0.0, "z_l2": 1e-4, "set_layers": 2, "bounded": True, "seed": 0, "gate": {"accept": 0.99, "rank_accept": 0.99,
                                                      "rank_fund": 0.99},
       "r42": {"worlds": 8, "episodes": 1, "stage": "m6-turn-imit-42"}}
START = "EXP-0100 artifacts/fit/models.pt (im.EXP0100_MODELS_SHA)"


def registered_prereg_v14() -> dict:
    return {
        "format": "m6-prereg-v14", "change": "add-m6-joint-mpg-gnn", "task": "4.3",
        "stage": "EXP-0120: turn-head imitation with learned set context (v13 + DIAGNOSTICS 38's fix)",
        "supersedes": f"m6_prereg_v13 ({PREREG_V13_SHA256[:8]}…, EXP-0118 offline_fail: a per-bid head cannot "
                      "imitate the cumulative auction)",
        "set_context": f"{CFG['set_layers']} self-attention layers (4 heads, pre-norm, no positional encoding, "
                       "padding masked) over each turn's items after their per-item encoding: the fund requests; "
                       "the bids and claims together. No hand-built set feature.",
        "bounded": "priorities and fund scores are 10·tanh(z/10); float32 rounds to exactly ±10 once |z| > ~90, "
                   f"so a penalty {CFG['z_l2']}·mean(z²) on the pre-tanh z keeps them apart; tied learnable "
                   "pairs reported; the metric stays on the bounded scores",
        "start": "chosen by EXP-0119's reading, recorded before data: deciding `gain` → F40 (EXP-0114 "
                 "candidate_it40's trunk and investment gates, the EXP-0110 start's replenish; the turn heads "
                 "read F40's trunk); otherwise EXP-0100's stack",
        "accept_errors": "reported split as DIAGNOSTICS 38: declined no_fit, declined displaced (fits alone), "
                         "accepted",
        "excluded_pairs": "pairs whose order the head's inputs cannot determine are left out of the loss and the "
                          "metric: feature-identical pairs (same row features and the same linked capacity / input "
                          "rows; the domain's tie-break keeps the default order) and, in the fund turn, "
                          "replenishment vs replenishment (creation order, arbitrary); excluded counts reported "
                          "by kind; the closed loop measures whether the unordered replenishment order matters",
        "metric_scores": "the offline metric scores the clipped (±10) outputs the closed loop acts on; saturation "
                         "(share of outputs at the bound) reported; an L2 penalty (score_l2) on fund scores and "
                         "priorities keeps the pairwise loss's growing margins inside the clip",
        "reported": "identical-bid pairs with different default accept bits (in the closed loop both bids get "
                    "the same bit, DIAGNOSTICS 36); held-out records and per-kind pair counts persisted (npz)",
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

    PREREG_V14_PATH.write_text(json.dumps(registered_prereg_v14(), indent=1, sort_keys=True) + "\n")
    return sha256_file(str(PREREG_V14_PATH))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V14_SHA256 is None:
        return ["no pinned m6_prereg_v14.json (`--mode freeze`)"]
    if not PREREG_V14_PATH.exists() or sha256_file(str(PREREG_V14_PATH)) != PREREG_V14_SHA256:
        return ["m6_prereg_v14.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V14_PATH.read_text()) != registered_prereg_v14():
        return ["the v14 pre-registration differs from the code"]
    return []


# --- episodes ---


def _turn_env(job: dict) -> Any:
    from ..eval.joint_stack import joint_stack_env
    from ..eval.routing_audit import job_config
    from ..eval.routing_diagnostics import DrawKey, set_draw_key

    env = joint_stack_env(job_config(job), job["horizon"], producer_turns=True)
    set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
    return env


def _producer_policy(job: dict, pm: g.ProducerGNN) -> Any:
    """The start stack's producing policy: the model's greedy gates and own replenish, or (F40) its
    greedy gates with the replenish of `job["replenish"]` (EXP-0116's `start_rep`)."""
    if not job.get("replenish"):
        return g.gnn_producing_policy(pm)
    from .m6_gap_probe import _policy

    pm.eval()
    pm0, _ = im.load_models(job["replenish"])
    return _policy("start_rep", pm, pm0, np.random.default_rng(0))


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
    linear = bool(job.get("linear", CFG.get("linear", False)))  # v15's linear fractions

    def emb(pid: str) -> tuple[Any, dict]:
        view, _f, (h, pooled) = cache.get(pm, env, pid)
        return view, {"h_cap": _np(h["capacity"]), "h_in": _np(h["input"]), "pooled": _np(pooled)}

    def fund(e: Any, agent: str) -> np.ndarray:
        from ..env.producer_turns import default_fund_policy

        pid = e._role[agent][1]
        reqs = e.system.funding_requests(pid)
        if reqs:
            view, x = emb(pid)
            t = g.fund_features(e.system.producing_agents[pid], reqs, e.system.funding_budget(pid), view,
                                linear)
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
            t = g.accept_features(p, lists, view, linear)
            recs.append({"kind": "accept", **x, **{k: t[k] for k in ("bid", "bid_cap", "bid_in", "claim",
                                                                     "claim_cap", "claim_in", "n_res")},
                         "acc": np.asarray(d.accept, np.float32),
                         "y_bid": np.asarray(d.bid_priority, np.float32),
                         "y_claim": np.asarray(d.reservation_priority + d.pending_priority, np.float32)})
        return encode_accept_action(d)

    s = run_episode(env, _producer_policy(job, pm), g.gnn_ordering_policy(om), seed=job["episode"],
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
    st = run_episode(env, _producer_policy(job, pm), g.gnn_ordering_policy(om), seed=job["episode"],
                     action_seed=None, **kw)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     f"{job['arm']}@h{job['horizon']}", job["replicate"], st))
    row.update({"arm": job["arm"], "set": job["set"], "units": repr(float(st["consumer_completed_units"])),
                "replenish": job.get("replenish") or "",
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


def _embed(heads: g.TurnHeads, kind: str, b: Mapping[str, Any]) -> torch.Tensor:
    if not b["x"].shape[0]:
        return torch.zeros((0, heads.width))
    return heads.node_embed(kind, b["x"], b["ec"], b["ei"], b["h_cap"], b["h_in"], b["pooled"][b["turn"]])


def batch_fund(heads: g.TurnHeads, b: Mapping[str, Any]) -> torch.Tensor:
    """`TurnHeads.turn_fund` over a concatenated batch of fund turns (equal per turn; tested)."""
    return heads.fund_out(_embed(heads, "fund", b), b["turn"])


def batch_accept(heads: g.TurnHeads, bb: Mapping[str, Any], bc: Mapping[str, Any]
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`TurnHeads.turn_accept` over a concatenated batch of accept turns (`bb` bids, `bc` claims, the
    same turn order); bids and claims of a turn attend to each other."""
    return heads.accept_out(_embed(heads, "bid", bb), bb["turn"], _embed(heads, "claim", bc), bc["turn"])


REPLENISH_COL = 1  # the fund row's "replenishment" kind one-hot column (g.FUND_KINDS index 1)
FITS_COL = 6  # the bid row's "fits the producer's free capacity alone" column (g.accept_features)


def node_keys(r: Mapping[str, Any], kind: str) -> list[tuple]:
    """What the head reads per node besides the turn's shared trunk / pooled embedding: the row features
    and the sets of linked capacity and input rows. Equal keys = equal head outputs."""
    x, ec, ei = r[kind], r[f"{kind}_cap"], r[f"{kind}_in"]
    return [(kind, *x[i].tolist(), "|", *sorted(b for a, b in ec if a == i), "|",
             *sorted(b for a, b in ei if a == i)) for i in range(x.shape[0])]


def _pairs(y: np.ndarray, mask: np.ndarray | None, cap: int | None, rng: np.random.Generator | None,
           keys: Sequence[tuple] | None = None, rep_rep: np.ndarray | None = None
           ) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Ordered pairs (i above j) among the items in `mask` (all if None) whose order the head can learn
    (v13): pairs with equal `keys` and, if `rep_rep` marks replenishment requests, replenishment vs
    replenishment are left out and counted. `cap` sampled with `rng` (after the exclusion)."""
    idx = np.arange(len(y)) if mask is None else np.nonzero(mask)[0]
    out, ex = [], {"rep_rep": 0, "identical": 0}
    for i in idx:
        for j in idx:
            if y[i] > y[j]:
                if rep_rep is not None and rep_rep[i] and rep_rep[j]:
                    ex["rep_rep"] += 1
                elif keys is not None and keys[i] == keys[j]:
                    ex["identical"] += 1
                else:
                    out.append((int(i), int(j)))
    kept = len(out)
    if cap is not None and len(out) > cap and rng is not None:
        out = [out[k] for k in rng.choice(len(out), cap, replace=False)]
    return out, {**ex, "kept": kept}


def turn_targets(recs: Sequence[dict], rng: np.random.Generator | None, cap: int | None,
                 rank_all_bids: bool = False) -> dict[str, Any]:
    """Global indices of the BCE targets and the learnable ordered pairs of a batch of turns, with the
    per-kind counts of kept and excluded pairs."""
    fund = [r for r in recs if r["kind"] == "fund"]
    acc = [r for r in recs if r["kind"] == "accept"]
    counts = {"fund_kept": 0, "fund_rep_rep": 0, "fund_identical": 0, "accept_kept": 0,
              "accept_identical": 0}
    fp, off = [], 0
    for r in fund:
        pr, ex = _pairs(r["y"], None, cap, rng, node_keys(r, "fund"), r["fund"][:, REPLENISH_COL] > 0)
        fp += [(off + i, off + j) for i, j in pr]
        off += len(r["y"])
        counts["fund_kept"] += ex["kept"]
        counts["fund_rep_rep"] += ex["rep_rep"]
        counts["fund_identical"] += ex["identical"]
    ap, bits, ob, oc = [], [], 0, 0
    nb_total = sum(len(r["acc"]) for r in acc)
    for r in acc:
        nb, nc = len(r["acc"]), len(r["y_claim"])
        y = np.concatenate([r["y_bid"], r["y_claim"]])
        mask = np.concatenate([np.ones(nb, bool) if rank_all_bids else r["acc"] > 0, np.ones(nc, bool)])
        # combined vector = [all bids of the batch, then all claims]: map local -> global
        loc = np.concatenate([ob + np.arange(nb), nb_total + oc + np.arange(nc)])
        pr, ex = _pairs(y, mask, cap, rng, node_keys(r, "bid") + node_keys(r, "claim"))
        ap += [(int(loc[i]), int(loc[j])) for i, j in pr]
        counts["accept_kept"] += ex["kept"]
        counts["accept_identical"] += ex["identical"]
        bits.append(r["acc"])
        ob, oc = ob + nb, oc + nc
    return {"fund": fund, "accept": acc, "fund_pairs": fp, "acc_pairs": ap, "counts": counts,
            "bits": torch.from_numpy(np.concatenate(bits)) if bits else torch.zeros(0)}


def _losses(heads: g.TurnHeads, tg: Mapping[str, Any], rank_only: bool = False
            ) -> tuple[torch.Tensor, dict[str, Any]]:
    """`rank_only`: the ranking terms (and the pre-tanh penalty) only, no accept BCE (the ordered heads'
    warm-up, so the order the accept pass reads is learned first)."""
    loss, out = torch.zeros(()), {}
    heads.pre_bound = []
    try:
        loss, out = _losses_inner(heads, tg, loss, out, rank_only)
        z = [x for x in heads.pre_bound if x.numel()]
        if CFG["z_l2"] and heads.bounded and z:
            loss = loss + CFG["z_l2"] * (torch.cat(z) ** 2).mean()
    finally:
        heads.pre_bound = None
    return loss, out


def _losses_inner(heads: g.TurnHeads, tg: Mapping[str, Any], loss: torch.Tensor, out: dict[str, Any],
                  rank_only: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    if tg["fund"]:
        s = batch_fund(heads, _cat(tg["fund"], "fund"))
        out["fund"] = s
        if tg["fund_pairs"]:
            i, j = torch.tensor(tg["fund_pairs"]).T
            loss = loss + torch.nn.functional.softplus(-(s[i] - s[j])).mean()
        if CFG["score_l2"]:
            loss = loss + CFG["score_l2"] * (s ** 2).mean()
    if tg["accept"]:
        a, bp, c = batch_accept(heads, _cat(tg["accept"], "bid"), _cat(tg["accept"], "claim"))
        out["acc"], out["pri"] = a, torch.cat([bp, c])
        if a.shape[0] and not rank_only:
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(a, tg["bits"])
        if tg["acc_pairs"]:
            i, j = torch.tensor(tg["acc_pairs"]).T
            loss = loss + torch.nn.functional.softplus(-(out["pri"][i] - out["pri"][j])).mean()
        if CFG["score_l2"] and out["pri"].numel():
            loss = loss + CFG["score_l2"] * (out["pri"] ** 2).mean()
    return loss, out


def fit(pm: g.ProducerGNN, train: Sequence[dict], epochs: int, lr: float, batch: int, cap: int,
        seed: int, log: Any = print, heads_cfg: Mapping[str, Any] | None = None,
        schedule: str | None = None, warmup: int | None = None) -> dict[str, Any]:
    """`heads_cfg` = TurnHeads config (default CFG's set_layers / bounded / width); `schedule`
    "cosine" decays the learning rate to 0 over the epochs (default CFG's, none in v14)."""
    torch.manual_seed(seed)
    hc = heads_cfg if heads_cfg is not None else {"set_layers": CFG["set_layers"], "bounded": CFG["bounded"],
                                                  "width": CFG.get("width"), "linear": CFG.get("linear", False),
                                                  "ordered": CFG.get("ordered", False)}
    pm.add_turn_heads(**hc)
    schedule = schedule if schedule is not None else CFG.get("schedule")
    heads = pm.turn
    assert heads is not None
    blank = {k: np.zeros((0, v), np.float32) for k, v in heads.dims.items()}
    norm = heads.norm.fit([{**blank, **{k: r[k] for k in ("fund", "bid", "claim") if k in r}}
                           for r in train])
    # ordered heads: the ranking loss alone for the first `warm` epochs (supervisor 2026-11-07), then everything;
    # the accept pass always reads the student's own order (no teacher-forcing)
    warm = (warmup if warmup is not None else CFG.get("rank_warmup", 3)) if heads.ordered else 0
    opt = torch.optim.Adam(heads.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, math.ceil(len(train) / batch)))
             if schedule == "cosine" else None)
    rng = np.random.default_rng([seed, 117])
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(train))
        tot, nb, t0 = 0.0, 0, time.perf_counter()
        for k in range(0, len(order), batch):
            tg = turn_targets([train[i] for i in order[k:k + batch]], rng, cap, rank_all_bids=heads.ordered)
            loss, _ = _losses(heads, tg, rank_only=ep < warm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        hist.append({"epoch": ep + 1, "loss": tot / max(1, nb), "seconds": round(time.perf_counter() - t0, 1)})
        log(f"[turn_imit] epoch {ep + 1}: loss {hist[-1]['loss']:.5f} ({hist[-1]['seconds']} s)")
    return {"norm": norm, "history": hist}


def _clip(x: torch.Tensor) -> torch.Tensor:
    """The scores the closed loop acts on (`fund_action` / `accept_action` clip to the action bound)."""
    return x.clamp(-ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND)


def identical_bid_pairs(r: Mapping[str, Any]) -> tuple[int, int]:
    """(pairs of feature-identical bids in one accept turn, of which the default accept bits differ)."""
    keys = node_keys(r, "bid")
    n = d = 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if keys[a] == keys[b]:
                n += 1
                d += int(r["acc"][a] != r["acc"][b])
    return n, d


def agreement(pm: g.ProducerGNN, recs: Sequence[dict], batch: int = 2048, whole: bool = True
              ) -> dict[str, Any]:
    """Held-out agreement on the clipped scores (v13): accept bits, pairwise order over the learnable pairs
    (accept turns: claims + accepted bids; fund turns), the share of turns whose whole schedule equals the
    default's; the excluded pair counts, the saturation of each output, the identical-bid bit conflicts."""
    heads = pm.turn
    assert heads is not None
    acc_ok = acc_n = ap_ok = ap_n = fp_ok = fp_n = 0
    whole_ok = whole_n = 0
    counts: dict[str, int] = {}
    sat = {"fund": [0, 0], "priority": [0, 0], "accept_logit": [0, 0]}
    ident = [0, 0]
    split = {c: [0, 0] for c in ("declined_no_fit", "declined_displaced", "accepted")}  # [wrong, n]
    ties = {"fund": 0, "accept": 0}  # learnable pairs whose bounded scores are equal
    with torch.no_grad():
        for k in range(0, len(recs), batch):
            part = list(recs[k:k + batch])
            tg = turn_targets(part, None, None)
            for c, v in tg["counts"].items():
                counts[c] = counts.get(c, 0) + v
            _, out = _losses(heads, tg)
            for name, key in (("fund", "fund"), ("priority", "pri"), ("accept_logit", "acc")):
                if key in out:
                    sat[name][0] += int((out[key].abs() >= ACTION_LOGIT_BOUND).sum())
                    sat[name][1] += int(out[key].numel())
            if tg["fund_pairs"]:
                f = _clip(out["fund"])
                i, j = torch.tensor(tg["fund_pairs"]).T
                fp_ok += int((f[i] > f[j]).sum())
                fp_n += len(tg["fund_pairs"])
                ties["fund"] += int((f[i] == f[j]).sum())
            if tg["accept"]:
                pred = out["acc"] > 0
                lab = tg["bits"] > 0
                acc_ok += int((pred == lab).sum())
                acc_n += len(tg["bits"])
                fits = torch.from_numpy(np.concatenate([r["bid"][:, FITS_COL] for r in tg["accept"]]) > 0)
                for c, m in (("declined_no_fit", ~lab & ~fits), ("declined_displaced", ~lab & fits),
                             ("accepted", lab)):
                    split[c][0] += int((pred[m] != lab[m]).sum())
                    split[c][1] += int(m.sum())
                if tg["acc_pairs"]:
                    pri = _clip(out["pri"])
                    i, j = torch.tensor(tg["acc_pairs"]).T
                    ap_ok += int((pri[i] > pri[j]).sum())
                    ap_n += len(tg["acc_pairs"])
                    ties["accept"] += int((pri[i] == pri[j]).sum())
                for r in tg["accept"]:
                    n_, d_ = identical_bid_pairs(r)
                    ident[0], ident[1] = ident[0] + n_, ident[1] + d_
            # whole schedule per turn (both kinds), from per-turn evaluations
            for r in part if whole else ():
                t1 = turn_targets([r], None, None)
                _, o = _losses(heads, t1)
                ok = True
                if r["kind"] == "fund" and t1["fund_pairs"]:
                    f = _clip(o["fund"])
                    i, j = torch.tensor(t1["fund_pairs"]).T
                    ok = bool((f[i] > f[j]).all())
                if r["kind"] == "accept":
                    ok = bool(((o["acc"] > 0) == (t1["bits"] > 0)).all())
                    if ok and t1["acc_pairs"]:
                        pri = _clip(o["pri"])
                        i, j = torch.tensor(t1["acc_pairs"]).T
                        ok = bool((pri[i] > pri[j]).all())
                whole_ok, whole_n = whole_ok + int(ok), whole_n + 1
    rate = lambda a, b: a / b if b else None  # noqa: E731
    return {"accept": rate(acc_ok, acc_n), "accept_n": acc_n, "rank_accept": rate(ap_ok, ap_n),
            "rank_accept_pairs": ap_n, "rank_fund": rate(fp_ok, fp_n), "rank_fund_pairs": fp_n,
            "whole_schedule": rate(whole_ok, whole_n) if whole else None, "turns": len(recs),
            "pairs": counts, "saturation": {k: rate(*v) for k, v in sat.items()},
            "identical_bid_pairs": ident[0], "identical_bid_pairs_bits_differ": ident[1],
            "accept_errors": {c: {"wrong": w, "n": n, "rate": rate(w, n)} for c, (w, n) in split.items()},
            "tied_pairs": ties}


# --- persisted records (npz; v13) ---

_ARRAYS = ("h_cap", "h_in", "pooled", "fund", "fund_cap", "fund_in", "y", "bid", "bid_cap", "bid_in", "claim",
           "claim_cap", "claim_in", "acc", "y_bid", "y_claim")


def save_records(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    """Every field concatenated over the turns with an offsets array (turn k's rows = off[k]:off[k+1]); edge
    lists as (n, 2) int arrays; `kind` 0 = fund, 1 = accept; `n_res` per turn."""
    out: dict[str, np.ndarray] = {"kind": np.asarray([r["kind"] == "accept" for r in recs], np.int8),
                                  "n_res": np.asarray([int(r.get("n_res", 0)) for r in recs], np.int64)}
    for f in _ARRAYS:
        parts = []
        for r in recs:
            v = r.get(f)
            if v is None:
                parts.append(np.zeros((0, 2) if f.endswith(("_cap", "_in")) and f not in ("h_cap", "h_in")
                                      else (0,), np.float32))
            elif isinstance(v, list):
                parts.append(np.asarray(v, np.int64).reshape(-1, 2))
            else:
                parts.append(np.asarray(v)[None] if f == "pooled" else np.asarray(v))
        lens = np.asarray([p.shape[0] for p in parts], np.int64)
        nonempty = [p for p in parts if p.shape[0]]
        out[f"off_{f}"] = np.concatenate([[0], np.cumsum(lens)])
        out[f] = np.concatenate(nonempty, 0) if nonempty else np.zeros((0,), np.float32)
    np.savez_compressed(path, **out)


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """The records of `save_records`, in order (the first `limit` only, if given)."""
    d = np.load(path)
    arrays = {f: (d[f], d[f"off_{f}"]) for f in _ARRAYS}  # decompress each array once
    kinds, n_res = d["kind"], d["n_res"]
    recs = []
    for k in range(len(kinds) if limit is None else min(limit, len(kinds))):
        r: dict[str, Any] = {"kind": "accept" if kinds[k] else "fund"}
        for f in _ARRAYS:
            arr, off = arrays[f]
            a, b = off[k], off[k + 1]
            fund_field = f in ("fund", "fund_cap", "fund_in", "y")
            if (r["kind"] == "fund") != fund_field and f not in ("h_cap", "h_in", "pooled"):
                continue
            v = arr[a:b]
            if f.endswith(("_cap", "_in")) and f not in ("h_cap", "h_in"):
                r[f] = [(int(x), int(y)) for x, y in v.reshape(-1, 2)]
            elif f == "pooled":
                r[f] = v[0]
            else:
                r[f] = v
        if r["kind"] == "accept":
            r["n_res"] = int(n_res[k])
        recs.append(r)
    return recs


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


def heldout_set(ts: Any, mon: Sequence, atlas: Mapping) -> tuple[list[tuple[str, int]], dict]:
    """The offline gate's and closed loop's world-episodes and their floor cells. v15 (CFG `heldout`): the
    monitor worlds with fresh episodes (never played: EXP-0121 tuned on EXP-0120's monitor episodes);
    before v15: the monitor world-episodes."""
    from ..eval.world_space import episode_seeds
    from .m6_f40_confirm import _cell
    from .order_es_validate import monitor_cells

    ho = CFG.get("heldout")
    if not ho:
        return list(mon), monitor_cells(ts.training, atlas["spreads"])
    records = {w["world"]: w["record"] for w in ts.training["worlds"]}
    wes = [(w, int(ep)) for w in sorted({w for w, _ in mon})
           for ep in episode_seeds(w, ho["episodes_per_world"], ho["stage"])]
    assert not set(wes) & set(mon), "a fresh held-out episode collides with a monitor episode"
    return wes, {f"{w}#{ep}": _cell(records[w], f"{w}#{ep}", atlas["spreads"]) for w, ep in wes}


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


# Real-run pins of v14's F40 start (review 2026-11-07): EXP-0119's reading itself, its candidate and start.
EXP0119_READING_SHA = "246a94a7ff4b853402b6101ad7818108127a9e5d5bfe52f0194042798d5a9bd9"
F40_MODELS_SHA = "ace0fabf585e5b3fec5ca2d003a8305cb2e3f6aaf76eb3ad5927543971b8b57e"  # EXP-0114 candidate_it40.pt
F40_START_SHA = "697d94c08321fffe08cb6a44bfeaba14988da21821cbed502ec78d54fb528d2c"  # EXP-0110 start_models.pt


def choose_start(exp0119_reading: str | None, models: str, f40_models: str | None,
                 f40_replenish: str | None) -> dict[str, Any]:
    """v14's start: F40 when EXP-0119's deciding reading is `gain`, else EXP-0100's stack."""
    from ..eval.provenance import sha256_file

    rd = json.loads(Path(exp0119_reading).read_text()) if exp0119_reading else None
    deciding = rd["reading"]["deciding"]["reading"] if rd else None
    f40 = deciding == "gain"
    if f40 and not (f40_models and f40_replenish):
        raise ValueError("EXP-0119 reads gain: --f40-models and --f40-replenish are needed")
    return {"exp0119_reading": deciding, "exp0119_sha256": sha256_file(exp0119_reading) if rd else None,
            "exp0119_dry_run": rd.get("dry_run") if rd else None,
            "exp0119_candidate_sha256": rd.get("candidate_sha256") if rd else None,
            "exp0119_start_sha256": rd.get("start_sha256") if rd else None,
            "start": "F40" if f40 else "EXP-0100", "models": f40_models if f40 else models,
            "replenish": f40_replenish if f40 else None,
            "models_sha256": sha256_file(f40_models if f40 else models),
            "replenish_sha256": sha256_file(f40_replenish) if f40 else None}


def start_problems(start: Mapping[str, Any]) -> list[str]:
    """A real run's start must be the registered one: EXP-0119's real reading (pinned SHA, not a dry run)
    and, on F40, exactly its candidate and start."""
    out = []
    if start["exp0119_sha256"] != EXP0119_READING_SHA:
        out.append("--exp0119-reading is not EXP-0119's registered reading")
    if start["exp0119_dry_run"] is not False:
        out.append("--exp0119-reading is a dry run's")
    if start["start"] == "F40":
        if not (start["models_sha256"] == start["exp0119_candidate_sha256"] == F40_MODELS_SHA):
            out.append("--f40-models is not EXP-0119's candidate (EXP-0114 candidate_it40.pt)")
        if not (start["replenish_sha256"] == start["exp0119_start_sha256"] == F40_START_SHA):
            out.append("--f40-replenish is not EXP-0119's start (EXP-0110 start_models.pt)")
    return out


def run(training_dir: str, atlas_path: str, models: str, out_dir: str, workers: int, dry_run: bool,
        exp0119_reading: str | None = None, f40_models: str | None = None,
        f40_replenish: str | None = None) -> int:
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas

    start = choose_start(exp0119_reading, models, f40_models, f40_replenish)
    if not dry_run:
        im._refuse(prereg_problems()
                   + ([] if sha256_file(models) == im.EXP0100_MODELS_SHA else ["--models is not EXP-0100's"])
                   + ([] if exp0119_reading else ["a real run needs --exp0119-reading (the start's choice)"])
                   + start_problems(start)
                   + ([] if len(im.SOURCE_COMMIT) == 40 else ["a real run needs a 40-hex --source-commit"]))
    print(f"[turn_imit] start {start['start']} (EXP-0119 {start['exp0119_reading']})", flush=True)
    models, rep_from = start["models"], start["replenish"]
    atlas = load_atlas(atlas_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts, train, mon = im._training(training_dir)
    mon, mon_cells = heldout_set(ts, mon, atlas)
    horizon, epochs, n42 = CFG["horizon"], CFG["epochs"], CFG["r42"]["worlds"]
    if dry_run:  # a true smoke: 1 + 1 world-episodes, horizon 6, 2 epochs, 1 world at 42
        train, mon, horizon, epochs, n42 = train[:1], mon[:1], 6, 2, 1
    t0 = time.perf_counter()
    rec_train = im._pool(record_job, [{**j, "replenish": rep_from}
                                      for j in _jobs(ts, train, "turn_train", models, horizon)], workers)
    rec_mon = im._pool(record_job, [{**j, "replenish": rep_from}
                                    for j in _jobs(ts, mon, "turn_monitor", models, horizon)], workers)
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
    save_records(out / "heldout_records.npz", ho)
    offline = {"train": agreement(pm, tr, whole=False), "heldout": agreement(pm, ho)}
    offline["pass"] = offline_pass(offline["heldout"])
    print(f"[turn_imit] offline {offline['heldout']} pass {offline['pass']}", flush=True)
    doc: dict[str, Any] = {"prereg": PREREG_V14_SHA256, "start": start,
                           "heldout": {"set": CFG.get("heldout") or "monitor", "world_episodes": len(mon)}, "source_commit": im.SOURCE_COMMIT, "dry_run": dry_run,
                           "start_sha256": sha256_file(models), "candidate_sha256": sha256_file(str(cand)),
                           "collect": collect, "fit": rep, "offline": offline, "reading": None}
    if offline["pass"] or dry_run:  # a smoke exercises the closed loop whatever its offline result
        jobs = [{**j, "arm": a, "set": "monitor", "models": m, "replenish": rep_from}
                for a, m in (("candidate", str(cand)), ("reference", models))
                for j in _jobs(ts, mon, "turn_gate", models, horizon)]
        j42, cells42 = _jobs_42(atlas, models, horizon, n42)
        jobs += [{**j, "arm": a, "set": "r42", "models": m, "replenish": rep_from}
                 for a, m in (("candidate", str(cand)), ("reference", models)) for j in j42]
        jobs.sort(key=lambda j: j["size"] != 42)  # the 42-producer episodes first (longest)
        rows = im._pool(gate_job, jobs, workers)
        with (out / "gate_rows.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        doc["closed_loop"] = {"monitor_12": closed_loop_reading(rows, mon_cells, "monitor"),
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
    ap.add_argument("--models", help="EXP-0100's models.pt")
    ap.add_argument("--exp0119-reading", help="EXP-0119's reading.json (chooses the start)")
    ap.add_argument("--f40-models", help="EXP-0114 candidate_it40.pt")
    ap.add_argument("--f40-replenish", help="EXP-0110 start_models.pt (F40's replenish)")
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "freeze":
        print(freeze())
        return 0
    im.SOURCE_COMMIT = a.source_commit
    return run(a.training_dir, a.atlas, a.models, a.out_dir, a.workers, a.dry_run, a.exp0119_reading,
               a.f40_models, a.f40_replenish)


if __name__ == "__main__":
    sys.exit(main())
