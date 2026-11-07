"""On-policy RL of every producer head (add-m6-joint-mpg-gnn 4.4 / 4.5; draft notes/4.4-4.5-draft.md, approved
2026-11-07).

One producer model (F40 + v15 turn heads) acts in all three producer turns. Deployment plays each head greedily;
training samples from the same heads (user decision 2: every head on-policy, played as deployed; DIAGNOSTICS 37):
- **investment gates:** Bernoulli(σ(logit / T_g)), T_g = v9 TEMPS (5% flips);
- **replenish:** frac = σ(z + σ_r ε), ε ~ N(0, 1) (deployed σ(z));
- **fund order:** Plackett–Luce on the bounded fund scores / τ_f (Gumbel-top-k sample);
- **accept-turn order:** Plackett–Luce on the bounded priorities / τ_p;
- **accept bits:** Bernoulli(σ(a / T_a)), a = the ordered pass's logit run **in the sampled order** (deployed: the pass
  in the greedy order, a > 0).
The orderer is frozen (the keyed softmax draw). Temperatures are calibrated offline on the start's recorded decisions
(`calibrate`), recorded with every run.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import nn

from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_stage0b as sb

B = ACTION_LOGIT_BOUND


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))  # no overflow at any temperature


FAMILIES = ("gates", "replenish", "fund_order", "accept_order", "accept_bits")


@dataclass(frozen=True)
class Temps:
    """The exploration temperatures (section A of the draft)."""

    gates: tuple[float, float, float] = (sb.TEMPS["restore"], sb.TEMPS["upgrade"], sb.TEMPS["expand"])
    replenish_sigma: float = 0.0
    fund_tau: float = 1.0
    order_tau: float = 1.0
    accept_t: float = 1.0


# --- the joint start as one model ---


class F40Model(nn.Module):
    """F40 + v15 as one producer model (user decision 2; draft section A). `pm` = EXP-0126's producer: F40's trunk,
    investment gates and the v15 turn heads. `rep` = the EXP-0110 start: its trunk frozen, only its replenish head
    trains. Input buying therefore starts exactly as F40 plays it (the start's replenish on the start's trunk) and
    can never drift with a moving trunk (DIAGNOSTICS 37). `pm`'s own replenish head is unused and frozen."""

    def __init__(self, pm: g.ProducerGNN, rep: g.ProducerGNN) -> None:
        super().__init__()
        self.pm, self.rep = pm, rep
        for n, p in rep.named_parameters():
            p.requires_grad_(n.startswith("replenish."))
        for p in pm.replenish.parameters():
            p.requires_grad_(False)
        self.features = pm.features
        self.turn = pm.turn

    def trainable(self) -> list[torch.Tensor]:
        return [p for p in self.parameters() if p.requires_grad]

    def heads(self, f: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(the gate logits of `pm`, the replenish pre-sigmoid logits of `rep`)."""
        return self.pm.heads(f)[0], self.rep.heads(f)[1]

    def fund_scores(self, *a: Any) -> torch.Tensor:
        return self.pm.fund_scores(*a)

    def accept_outputs(self, *a: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pm.accept_outputs(*a)

    def _trunk(self, f: dict[str, Any]) -> Any:
        return self.pm._trunk(f)

    def embed(self, f: dict[str, Any]) -> torch.Tensor:
        return self.pm.embed(f)


def load_start(turn_models: str, replenish_models: str) -> tuple[F40Model, g.OrdererGNN]:
    """(the F40 + v15 composite, the frozen orderer) from EXP-0126's models.pt and EXP-0110's start_models.pt."""
    pm, om = im.load_models(turn_models)
    rep, _ = im.load_models(replenish_models)
    for m in (pm, rep, om):
        m.eval()
    return F40Model(pm, rep), om


FORMAT = "m6-onpolicy-f40-v1"


def save_model(model: F40Model, om: g.OrdererGNN, path: Any) -> None:
    """The composite checkpoint. `pm`'s unused replenish head is not saved (a sentinel at load), and the format
    tag makes `m6_imitation.load_models` refuse it: deploy only through `load_model` + `deployed_policies`."""
    prod = {k: v for k, v in model.pm.state_dict().items() if not k.startswith("replenish.")}
    torch.save({"format": FORMAT, "norm_format": g.NORM_FORMAT, "producer": prod,
                "replenish_model": model.rep.state_dict(), "orderer": om.state_dict(),
                "turn_heads": model.pm.turn.config(), "producer_features": model.pm.features,
                "capacity_extra": model.pm.capacity_extra, "rep_capacity_extra": model.rep.capacity_extra,
                "hidden": g.HIDDEN, "rounds": g.ROUNDS}, path)


def load_model(path: Any) -> tuple[F40Model, g.OrdererGNN]:
    d = torch.load(path, weights_only=True)
    if d.get("format") != FORMAT or d.get("norm_format") != g.NORM_FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} checkpoint with normaliser {g.NORM_FORMAT}")
    pm = g.ProducerGNN(d["hidden"], d["rounds"], d["producer_features"], int(d["capacity_extra"]))
    pm.add_turn_heads(**d["turn_heads"])
    rep = g.ProducerGNN(d["hidden"], d["rounds"], d["producer_features"], int(d["rep_capacity_extra"]))
    om = g.OrdererGNN(d["hidden"], d["rounds"])
    missing, unexpected = pm.load_state_dict(d["producer"], strict=False)
    if unexpected or any(not k.startswith("replenish.") for k in missing):
        raise ValueError(f"{path}: producer weights do not match ({missing}, {unexpected})")
    with torch.no_grad():  # sentinel: the composite never uses pm's own replenish head
        for p in pm.replenish.parameters():
            p.fill_(float("nan"))
    rep.load_state_dict(d["replenish_model"])
    om.load_state_dict(d["orderer"])
    for m in (pm, rep, om):
        m.eval()
    return F40Model(pm, rep), om


def deployed_policies(model: F40Model, om: g.OrdererGNN) -> dict[str, Any]:
    """The deployed composite (the only deployment path for gates and validation): greedy investment action, the
    frozen orderer's keyed draw, the turn heads' greedy fund and accept policies (one trunk pass per cycle)."""
    fp, ap = g.gnn_turn_policies(model.pm)
    return {"producing": greedy_producing_policy(model), "ordering": g.gnn_ordering_policy(om),
            "fund": fp, "accept": ap}


def greedy_producing_policy(model: F40Model) -> Any:
    """The deployed investment action of the composite: greedy gates of `pm`, replenish σ(z) of `rep`."""
    def act(env: Any, agent: str) -> np.ndarray:
        pid = env._role[agent][1]
        f = g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid), model.features)
        with torch.no_grad():
            gates, z = model.heads(f)
        sp = env.spec_
        out = np.zeros(sp.producing_action_dim, dtype=np.float32)
        for gp, v in zip(f["inputs"], torch.sigmoid(z).numpy()):
            if gp in sp.input_index:
                out[sp.input_index[gp]] = float(v) * B
        out[sp.num_inputs:sp.num_inputs + 3] = np.where(gates.numpy() > 0, B, -B)
        return out

    return act


# --- Plackett–Luce ---


def pl_sample(scores: np.ndarray, tau: float, rng: np.random.Generator) -> list[int]:
    """A Plackett–Luce order of the items (first = highest), by the Gumbel-top-k trick on scores / τ."""
    if len(scores) == 0:
        return []
    gumbel = -np.log(-np.log(rng.uniform(1e-12, 1.0, len(scores))))
    return [int(i) for i in np.argsort(-(np.asarray(scores, np.float64) / tau + gumbel), kind="stable")]


def pl_logp(scores: torch.Tensor, order: Sequence[int], tau: float) -> torch.Tensor:
    """log P(order | scores / τ) under Plackett–Luce."""
    if len(order) <= 1:
        return torch.zeros(())
    s = scores[torch.as_tensor(list(order))] / tau
    # log Π_k exp(s_k) / Σ_{j ≥ k} exp(s_j): reverse cumulative logsumexp
    tail = torch.logcumsumexp(s.flip(0), 0).flip(0)
    return (s - tail).sum()


def greedy_order(scores: np.ndarray) -> list[int]:
    """The deployed order: descending score, exact ties in input order (the domain's stable sorts: fund stage 2 keeps
    today's order on equal scores, the accept turn's claims and bids are listed in the domain's key order)."""
    return [int(i) for i in np.argsort(-np.asarray(scores, np.float64), kind="stable")]


def learnable_adjacent(scores: np.ndarray, rep_: np.ndarray | None) -> list[tuple[int, int]]:
    """For each item of the greedy order, its nearest successor it has a learnable pair with (a replenishment
    request is never compared with another replenishment request: that order is arbitrary, v13)."""
    greedy = greedy_order(scores)
    adj = {}
    for k, a in enumerate(greedy):
        for b in greedy[k + 1:]:
            if rep_ is None or not (rep_[a] and rep_[b]):
                adj[a] = b
                break
    return list(adj.items())


def swaps(order: Sequence[int], adj: Sequence[tuple[int, int]]) -> int:
    pos = {i: k for k, i in enumerate(order)}
    return sum(pos[a] > pos[b] for a, b in adj)


def pl_kl(s: torch.Tensor, s0: torch.Tensor, order: Sequence[int], tau: float) -> torch.Tensor:
    """KL(PL(s/τ) ‖ PL(s0/τ)) along the sampled order: Σ_k KL(softmax(s[o_k:]/τ) ‖ softmax(s0[o_k:]/τ)), the exact
    categorical KL of each stage's choice given the earlier picks (≥ 0, 0 iff the stage distributions agree; its
    gradient pulls s toward s0). Review of ebc30f5: the sampled log-ratio did not pull toward the start."""
    if len(order) <= 1:
        return torch.zeros(())
    idx = torch.as_tensor(list(order))
    a, b = s[idx] / tau, s0[idx] / tau
    kl = torch.zeros(())
    for k in range(len(order) - 1):
        la, lb = torch.log_softmax(a[k:], 0), torch.log_softmax(b[k:], 0)
        kl = kl + (la.exp() * (la - lb)).sum()
    return kl


def rank_scores(order: Sequence[int], n: int) -> np.ndarray:
    """Scores in (−B, B] that sort descending into `order` (the env applies the turn's choice by score)."""
    out = np.zeros(n, np.float32)
    for rank, i in enumerate(order):
        out[i] = B - 2 * B * rank / max(n, 1)
    return out


# --- the per-decision log-probability (shared by the rollout, the PPO loss and the KL) ---


def prod_parts(pm: Any, d: Mapping[str, Any], temps: Temps) -> dict[str, torch.Tensor]:
    gates, z = pm.heads(d["f"])
    tl = gates / torch.as_tensor(temps.gates)
    out = {"gates": torch.distributions.Bernoulli(logits=tl).log_prob(torch.as_tensor(d["bits"])).sum(),
           "_tl": tl, "_z": z}
    if temps.replenish_sigma > 0 and z.numel():
        out["replenish"] = torch.distributions.Normal(z, temps.replenish_sigma).log_prob(
            torch.as_tensor(d["y"])).sum()
    return out


def fund_parts(pm: Any, d: Mapping[str, Any], temps: Temps) -> dict[str, torch.Tensor]:
    s = pm.fund_scores(d["f"], d["t"])
    return {"fund_order": pl_logp(s, d["order"], temps.fund_tau), "_s": s}


def accept_parts(pm: Any, d: Mapping[str, Any], temps: Temps) -> dict[str, torch.Tensor]:
    """Items in the pass's input order [claims (reservations, pending), bids]; the sampled order drives the pass."""
    heads = pm.turn
    assert heads is not None and heads.ordered
    h, pooled = pm._trunk(d["f"])
    t = d["t"]
    nb, nc = t["bid"].shape[0], t["claim"].shape[0]
    _a0, bp, cp = heads.turn_accept(t, h, pooled)
    pri = torch.cat([cp, bp])
    pass_pri = torch.as_tensor(rank_scores(d["order"], nb + nc))
    a, _bp, _cp = heads.turn_accept(t, h, pooled, pass_pri)
    out = {"accept_order": pl_logp(pri, d["order"], temps.order_tau), "_a": a / temps.accept_t, "_pri": pri}
    if nb:
        out["accept_bits"] = torch.distributions.Bernoulli(logits=a / temps.accept_t).log_prob(
            torch.as_tensor(d["bits"])).sum()
    return out


PARTS = {"prod": prod_parts, "fund": fund_parts, "acc": accept_parts}


def decision_logp(pm: Any, d: Mapping[str, Any], temps: Temps) -> tuple[torch.Tensor, dict]:
    parts = PARTS[d["kind"]](pm, d, temps)
    fam = {k: v for k, v in parts.items() if not k.startswith("_")}
    return sum(fam.values(), torch.zeros(())), parts


def decision_kl(parts: Mapping[str, torch.Tensor], parts0: Mapping[str, torch.Tensor], d: Mapping[str, Any],
                temps: Temps) -> torch.Tensor:
    """KL(π_θ ‖ π_start) of one decision: exact for the Bernoulli and Gaussian parts; for the Plackett–Luce orders the
    exact stage-wise categorical KL along the decision's sampled order (`pl_kl`)."""
    kind = d["kind"]
    ls = torch.nn.functional.logsigmoid

    def bern(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(a)
        return (p * (ls(a) - ls(b)) + (1 - p) * (ls(-a) - ls(-b))).sum()

    if kind == "prod":
        kl = bern(parts["_tl"], parts0["_tl"])
        if temps.replenish_sigma > 0 and parts["_z"].numel():
            kl = kl + ((parts["_z"] - parts0["_z"]) ** 2).sum() / (2 * temps.replenish_sigma ** 2)
        return kl
    if kind == "fund":
        return pl_kl(parts["_s"], parts0["_s"], d["order"], temps.fund_tau)
    kl = pl_kl(parts["_pri"], parts0["_pri"], d["order"], temps.order_tau)
    if "accept_bits" in parts:
        kl = kl + bern(parts["_a"], parts0["_a"])
    return kl


# --- the exploring policy ---


class OnPolicy:
    """Section A's training policy on the turn env (`producer_turns=True`). Records every producer decision (its
    kind, inputs, sampled actions, old log-probability, cycle) for the PPO update; with `score`, also the projected
    score per head family (the power check)."""

    def __init__(self, pm: Any, pm0: Any, om: g.OrdererGNN, rng: np.random.Generator,
                 temps: Temps, score: bool = False) -> None:
        self.pm, self.pm0, self.om, self.rng, self.temps, self.score = pm, pm0, om, rng, temps, score
        self.decisions: list[dict] = []
        self.counts = {"gate_flips": [0, 0], **{f"flip_{k}": [0, 0] for k in sb.GATES},
                       "fund_swaps": [0, 0], "order_swaps": [0, 0], "bit_flips": [0, 0]}
        self.rec = {r: {"cycle": [], "agent": [], "emb": [], "score": [], "family_sq": [], "score_norep": []}
                    for r in ("producer", "orderer")}
        # the orderer's keyed softmax draw (frozen); `ExplorePolicies` only reads the orderer here
        self._explore = sb.ExplorePolicies(pm.pm if isinstance(pm, F40Model) else pm, pm0, om, om, rng,
                                           score=score)

    def _view(self, env: Any, pid: str) -> tuple[Any, dict]:
        view = g._turn_view(env, pid)
        return view, g.producer_features(view, g.producer_context(env, pid), self.pm.features)

    def _record(self, env: Any, d: dict) -> None:
        d["cycle"] = int(env.current_cycle)
        if self.score:
            self._score(env, d)
        else:
            with torch.no_grad():
                lp, _ = decision_logp(self.pm, d, self.temps)
            d["old_lp"] = float(lp)
        self.decisions.append(d)

    def _score(self, env: Any, d: dict) -> None:
        from . import m6_stage0 as s0

        lp, parts = decision_logp(self.pm, d, self.temps)
        d["old_lp"] = float(lp.detach())
        ps = self.pm.trainable() if isinstance(self.pm, F40Model) else list(self.pm.parameters())
        proj = s0.projection("producer", sum(p.numel() for p in ps))
        total = np.zeros(proj.shape[0], np.float32)
        norep = np.zeros(proj.shape[0], np.float32)  # the score without the replenish part (4.4's counterfactual)
        fam_sq = dict.fromkeys(FAMILIES, 0.0)
        # a family without a gradient path (a 1-item order is fixed) scores 0
        fams = [k for k in parts if not k.startswith("_") and parts[k].requires_grad]
        for i, k in enumerate(fams):
            gr = torch.autograd.grad(parts[k], ps, allow_unused=True, retain_graph=i < len(fams) - 1)
            flat = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(gr, ps)])
            v = (proj @ flat).numpy().astype(np.float32)
            total += v
            if k != "replenish":
                norep += v
            fam_sq[k] = float((v.astype(np.float64) ** 2).sum())
        with torch.no_grad():
            emb = self.pm0.embed(d["f"]).numpy()
        r = self.rec["producer"]
        r["cycle"].append(d["cycle"])
        r["agent"].append(d.get("pid", ""))
        r["emb"].append(emb.astype(np.float32))
        r["score"].append(total)
        r["score_norep"].append(norep)
        r["family_sq"].append([fam_sq[k] for k in FAMILIES])

    def producing(self, env: Any, agent: str) -> np.ndarray:
        pid = env._role[agent][1]
        f = g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid), self.pm.features)
        with torch.no_grad():
            gates, z = self.pm.heads(f)
        tl = (gates / torch.as_tensor(self.temps.gates)).numpy()
        bits = (self.rng.random(3) < _sigmoid(tl)).astype(np.float32)
        greedy = gates.numpy() > 0
        self.counts["gate_flips"][0] += int((bits.astype(bool) != greedy).sum())
        self.counts["gate_flips"][1] += 3
        for i, k in enumerate(sb.GATES):
            self.counts[f"flip_{k}"][0] += int(bool(bits[i]) != bool(greedy[i]))
            self.counts[f"flip_{k}"][1] += 1
        zn = z.numpy()
        y = zn + self.temps.replenish_sigma * self.rng.standard_normal(zn.shape) if zn.size else zn
        self._record(env, {"kind": "prod", "f": f, "bits": bits, "y": y.astype(np.float32), "pid": pid})
        sp = env.spec_
        act = np.zeros(sp.producing_action_dim, dtype=np.float32)
        for gp, v in zip(f["inputs"], y):
            if gp in sp.input_index:
                act[sp.input_index[gp]] = float(_sigmoid(v)) * B
        act[sp.num_inputs:sp.num_inputs + 3] = np.where(bits > 0, B, -B)
        return act

    def fund(self, env: Any, agent: str) -> np.ndarray:
        from ..env.producer_turns import fund_slots

        pid = env._role[agent][1]
        s = env.system
        reqs = s.funding_requests(pid)
        out = np.full(fund_slots(env.spec_), -B, np.float32)
        if not reqs:
            return out
        view, f = self._view(env, pid)
        t = g.fund_features(s.producing_agents[pid], reqs, s.funding_budget(pid), view, self.pm.turn.linear)
        with torch.no_grad():
            sc = self.pm.fund_scores(f, t).numpy()
        order = pl_sample(sc, self.temps.fund_tau, self.rng)
        adj = learnable_adjacent(sc, t["fund"][:, 1] > 0)
        self.counts["fund_swaps"][0] += swaps(order, adj)
        self.counts["fund_swaps"][1] += len(adj)
        self._record(env, {"kind": "fund", "f": f, "t": t, "order": order, "pid": pid})
        n = min(len(reqs), out.shape[0])
        out[:n] = rank_scores(order, len(reqs))[:n]
        return out

    def accept(self, env: Any, agent: str) -> np.ndarray:
        from ..domain.producing_agent import TurnDecision
        from ..env.producer_turns import encode_accept_action

        pid = env._role[agent][1]
        p = env.system.producing_agents[pid]
        lists = p.turn_lists()
        nb, nr, npd = len(lists.bids), len(lists.reservations), len(lists.pending)
        if not (nb or nr or npd):
            return encode_accept_action(TurnDecision((), (), (), ()))
        view, f = self._view(env, pid)
        heads = self.pm.turn
        t = g.accept_features(p, lists, view, heads.linear, heads.arrival)
        with torch.no_grad():
            h, pooled = self.pm._trunk(f)
            _a, bp, cp = heads.turn_accept(t, h, pooled)
            pri = torch.cat([cp, bp]).numpy()
        order = pl_sample(pri, self.temps.order_tau, self.rng)
        adj = learnable_adjacent(pri, None)
        self.counts["order_swaps"][0] += swaps(order, adj)
        self.counts["order_swaps"][1] += len(adj)
        with torch.no_grad():
            a, _b, _c = heads.turn_accept(t, h, pooled, torch.as_tensor(rank_scores(order, len(pri))))
        logit = (a / self.temps.accept_t).numpy()
        bits = (self.rng.random(nb) < _sigmoid(logit)).astype(np.float32)
        self.counts["bit_flips"][0] += int((bits.astype(bool) != (a.numpy() > 0)).sum())
        self.counts["bit_flips"][1] += nb
        self._record(env, {"kind": "acc", "f": f, "t": t, "order": order, "bits": bits, "pid": pid})
        rs = rank_scores(order, nr + npd + nb)  # aligned to [reservations, pending, bids]
        return encode_accept_action(TurnDecision(tuple(bool(b) for b in bits), tuple(float(x) for x in rs[nr + npd:]),
                                                 tuple(float(x) for x in rs[:nr]),
                                                 tuple(float(x) for x in rs[nr:nr + npd])))

    def ordering(self, env: Any, agent: str) -> np.ndarray:
        out = self._explore.ordering(env, agent)
        if self.score:
            self.rec["orderer"] = self._explore.rec["orderer"]
        return out

    def rates(self) -> dict[str, float | None]:
        return {k: (v[0] / v[1] if v[1] else None) for k, v in self.counts.items()}


# --- calibration (offline, on the start's recorded decisions) ---


def _bisect(fn: Any, target: float, lo: float, hi: float, iters: int = 40, log: bool = True) -> float:
    """x in [lo, hi] with fn(x) ≈ target for an increasing fn."""
    for _ in range(iters):
        mid = math.sqrt(lo * hi) if log else (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi) if log else (lo + hi) / 2


def calibrate(pm: Any, prod_feats: Sequence[dict], turn_recs: Sequence[dict], seed: int = 0,
              target_frac: float = 0.02, target_swap: float = 0.05, target_flip: float = 0.05) -> dict[str, Any]:
    """σ_r: mean |σ(z + σ ε) − σ(z)| = target_frac per replenish row; τ_f / τ_p: the share of adjacent greedy-order
    pairs a Plackett–Luce sample swaps = target_swap; T_a: the mean flip probability of the accept bits (the pass in
    orders sampled at τ_p) = target_flip. Fixed draws, so the result is deterministic."""
    rng = np.random.default_rng([seed, 4401])
    with torch.no_grad():
        zs = np.concatenate([pm.heads(f)[1].numpy() for f in prod_feats if f["input"].shape[0]] or [np.zeros(0)])
    eps = rng.standard_normal((8, len(zs)))
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    frac = lambda s: float(np.abs(sig(zs + s * eps) - sig(zs)).mean()) if len(zs) else 0.0  # noqa: E731
    sigma_r = _bisect(frac, target_frac, 1e-4, 50.0)

    heads = pm.turn
    assert heads is not None
    fund_s, acc = [], []
    with torch.no_grad():
        for r in turn_recs:
            hh = {"capacity": torch.from_numpy(r["h_cap"]), "input": torch.from_numpy(r["h_in"])}
            pooled = torch.from_numpy(r["pooled"])
            if r["kind"] == "fund" and len(r["fund"]) > 1:
                # replenishment-vs-replenishment pairs carry no learned order (v13); they do not count
                rep_ = r["fund"][:, 1] > 0
                fund_s.append((heads.turn_fund(r, hh, pooled).numpy(), rep_))
            elif r["kind"] == "accept" and len(r["bid"]) + len(r["claim"]) > 0:
                _a, bp, cp = heads.turn_accept(r, hh, pooled)
                acc.append((r, hh, pooled, torch.cat([cp, bp]).numpy()))

    def swap_rate(scores: Sequence[tuple[np.ndarray, np.ndarray | None]], tau: float, draws: int = 4) -> float:
        """The share of learnable adjacent pairs of the greedy order that a sample swaps (the adjacency is taken
        among the items whose pairs are learnable: a replenishment request is compared with the next
        non-replenishment item, never with another replenishment)."""
        sw = n = 0
        r2 = np.random.default_rng([seed, 4402])
        for s, rep_ in scores:
            if len(s) < 2:
                continue
            adj = learnable_adjacent(s, rep_)
            for _ in range(draws):
                sw += swaps(pl_sample(s, tau, r2), adj)
                n += len(adj)
        return sw / n if n else 0.0

    tau_f = _bisect(lambda x: swap_rate(fund_s, x), target_swap, 1e-4, 1e3)
    tau_p = _bisect(lambda x: swap_rate([(x[3], None) for x in acc], x), target_swap, 1e-4, 1e3)
    r3 = np.random.default_rng([seed, 4403])
    logits = []
    with torch.no_grad():
        for r, hh, pooled, pri in acc:
            if not len(r["bid"]):
                continue
            o = pl_sample(pri, tau_p, r3)
            a, _b, _c = heads.turn_accept(r, hh, pooled, torch.as_tensor(rank_scores(o, len(pri))))
            logits.append(a.numpy())
    al = np.abs(np.concatenate(logits)) if logits else np.zeros(0)
    flip = lambda t: float((1 / (1 + np.exp(al / t))).mean()) if len(al) else 0.0  # noqa: E731
    accept_t = _bisect(flip, target_flip, 1e-4, 1e4)
    temps = Temps(replenish_sigma=sigma_r, fund_tau=tau_f, order_tau=tau_p, accept_t=accept_t)
    return {"temps": asdict(temps), "achieved": {"replenish_mean_abs_dfrac": frac(sigma_r),
                                                 "fund_swap": swap_rate(fund_s, tau_f),
                                                 "order_swap": swap_rate([(x[3], None) for x in acc], tau_p),
                                                 "accept_flip": flip(accept_t)},
            "n": {"replenish_rows": int(len(zs)), "fund_turns": len(fund_s), "accept_turns": len(acc),
                  "accept_bits": int(len(al))}}


def temps_from(doc: Mapping[str, Any]) -> Temps:
    t = doc["temps"]
    return Temps(tuple(t["gates"]), t["replenish_sigma"], t["fund_tau"], t["order_tau"], t["accept_t"])


# --- one PPO iteration (4.5) ---

CFG = {"seeds_per_iter": 96, "siblings": 4, "gamma": 0.99, "clip": 0.2, "kl_coef": 0.04, "lr": 3e-4,
       "grad_clip": 1.0, "steps": 100, "batch": 256, "horizon": 100, "replicate_base": 1000, "seed": 0,
       "lambda_scale": 0.25}


def turn_env(job: Mapping[str, Any]) -> Any:
    from . import m6_turn_imitation as ti

    return ti._turn_env(dict(job))


def play_seed(job: dict, pm: Any, pm0: Any, om: g.OrdererGNN, temps: Temps) -> dict:
    """The G siblings of one seed under the exploring policy; E3 advantages per cycle on every decision."""
    from ..env.potential import PotentialTracker, PotentialWeights
    from ..eval.runner import run_episode
    from . import m6_stage0 as s0

    eps, counts = [], []
    for i in range(CFG["siblings"]):
        jj = {**job, "replicate": CFG["replicate_base"] + 10 * job["iteration"] + i}
        env = turn_env(jj)
        tr = PotentialTracker(PotentialWeights(**job["weights"]))
        tr.attach(env)
        rng = np.random.default_rng([CFG["seed"], 45, job["iteration"], int(job["episode"]) % (1 << 31), i])
        pol = OnPolicy(pm, pm0, om, rng, temps)
        run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                    fund_policy=pol.fund, accept_policy=pol.accept)
        eps.append({"rewards": tr.rewards(float(job["unit_scale"])), "decisions": pol.decisions,
                    "units": tr.totals()["units"], "dead": bool(env.episode_stats["dead"])})
        counts.append(pol.counts)
    rtg = [s0._returns_to_go(e["rewards"], CFG["gamma"]) for e in eps]
    n = min(len(x) for x in rtg)
    m = np.stack([x[:n] for x in rtg])
    adv = m - (m.sum(0, keepdims=True) - m) / (CFG["siblings"] - 1)
    out = [{**d, "adv": float(adv[i, d["cycle"]])} for i, e in enumerate(eps) for d in e["decisions"]
           if d["cycle"] < n]
    tot = {k: [sum(c[k][0] for c in counts), sum(c[k][1] for c in counts)] for k in counts[0]}
    return {"decisions": out, "returns": [float(sum(e["rewards"])) for e in eps],
            "units": [e["units"] for e in eps], "dead": [e["dead"] for e in eps], "counts": tot}


def ppo_loss(pm: Any, pm0: Any, batch: Sequence[dict], temps: Temps
             ) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate on each decision's full log-probability (all heads that acted) + β · KL to the start."""
    pg, kl = torch.zeros(()), torch.zeros(())
    clipped = 0
    for d in batch:
        lp, parts = decision_logp(pm, d, temps)
        with torch.no_grad():
            _lp0, parts0 = decision_logp(pm0, d, temps)
        ratio = torch.exp(lp - d["old_lp"])
        a = d["adv"]
        pg = pg - torch.min(ratio * a, ratio.clamp(1 - CFG["clip"], 1 + CFG["clip"]) * a)
        kl = kl + decision_kl(parts, parts0, d, temps)
        clipped += int(abs(float(ratio.detach()) - 1) > CFG["clip"])
    n = max(1, len(batch))
    return pg / n + CFG["kl_coef"] * kl / n, {"pg": float(pg.detach()) / n, "kl": float(kl.detach()) / n,
                                               "clip_frac": clipped / n}


def time_iteration(pm: F40Model, pm0: F40Model, om: g.OrdererGNN, jobs: Sequence[dict], temps: Temps,
                   workers: int, steps: int) -> dict[str, Any]:
    """One iteration's cost on real data: the rollouts (in a pool, as the training iteration) and `steps` PPO
    steps on the collected decisions (one process, timed per step), for the 4.5 sizing."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    t0 = time.perf_counter()
    if workers <= 1:
        res = [play_seed(j, pm, pm0, om, temps) for j in jobs]
    else:
        im.set_sharing()  # tensors through /dev/shm files, not descriptors (EXP-0105: "Too many open files")
        state = {k: v.detach().clone() for k, v in pm.state_dict().items()}
        with ProcessPoolExecutor(workers, mp_context=get_context("fork"), initializer=im.set_sharing) as ex:
            res = list(ex.map(_play_job, [(j, temps, state) for j in jobs]))
    t_roll = time.perf_counter() - t0
    decs = [d for r in res for d in r["decisions"]]
    opt = torch.optim.Adam(pm.trainable(), lr=CFG["lr"])
    rng = np.random.default_rng(0)
    check_first_step(pm, decs, temps)
    t1 = time.perf_counter()
    for _ in range(steps):
        idx = rng.choice(len(decs), min(CFG["batch"], len(decs)), replace=False)
        loss, _st = ppo_loss(pm, pm0, [decs[i] for i in idx], temps)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pm.trainable(), CFG["grad_clip"])
        opt.step()
    t_step = (time.perf_counter() - t1) / max(1, steps)
    kinds = {k: sum(d["kind"] == k for d in decs) for k in ("prod", "fund", "acc")}
    return {"rollout_seconds": t_roll, "seeds": len(jobs), "episodes": len(jobs) * CFG["siblings"],
            "decisions": len(decs), "kinds": kinds, "seconds_per_step": t_step, "steps_timed": steps,
            "workers": workers, "deaths": sum(sum(r["dead"]) for r in res),
            "units": fmean(u for r in res for u in r["units"]) if res else None}


def check_first_step(pm: Any, decs: Sequence[dict], temps: Temps, n: int = 64) -> float:
    """Before an iteration's first update the acting model is the updating model: every ratio must be 1 (review of
    ebc30f5: a pool worker that kept playing the start would break this silently)."""
    with torch.no_grad():
        dev = max((abs(math.exp(float(decision_logp(pm, d, temps)[0]) - d["old_lp"]) - 1.0) for d in decs[:n]),
                  default=0.0)
    if dev > 1e-3:
        raise RuntimeError(f"the rollouts were not played by the current model (max |ratio - 1| = {dev:.3g})")
    return dev


_POOL_MODELS: dict[tuple, Any] = {}


def _play_job(arg: tuple[dict, Temps, Mapping[str, torch.Tensor]]) -> dict:
    """One seed in a pool worker, played by the current model (`state`, sent every iteration); the KL anchor
    stays the start."""
    job, temps, state = arg
    torch.set_num_threads(1)
    key = (job["turn_models"], job["replenish_models"])
    if key not in _POOL_MODELS:
        pm, om = load_start(*key)
        pm0, _ = load_start(*key)
        _POOL_MODELS[key] = (pm, pm0, om)
    pm, pm0, om = _POOL_MODELS[key]
    pm.load_state_dict(state)
    return play_seed(job, pm, pm0, om, temps)


def main(argv: Sequence[str] | None = None) -> int:
    """`--mode time`: one real 4.5 iteration (seeds × G siblings in a fork pool, then `--steps` PPO steps) for
    the sizing; the temperatures from a calibration file (`m6_stage0c` writes one)."""
    import argparse
    import json
    import sys as _sys
    from pathlib import Path

    from . import m6_turn_imitation as ti

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("time",), required=True)
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--exp0110", required=True)
    ap.add_argument("--turn-models", required=True)
    ap.add_argument("--replenish-models", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--seeds", type=int, default=CFG["seeds_per_iter"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=CFG["horizon"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cal = json.loads((Path(a.exp0110) / "calibration.json").read_text())
    temps = temps_from(json.loads(Path(a.calibration).read_text()))
    ts, train, _mon = im._training(a.training_dir)
    c = cal["per_size"]["12"]
    w = {"lam": c["lambda"] * CFG["lambda_scale"], "mu": c["mu"], "nu": float(a.horizon), "x": cal["x"], "y": cal["y"]}
    jobs = [{**j, "iteration": 1, "weights": w, "unit_scale": a.horizon / cal["bar"][f"{j['world']}|{j['episode']}"],
             "turn_models": a.turn_models, "replenish_models": a.replenish_models}
            for j in ti._jobs(ts, train[:a.seeds], "onpolicy_time", "x", a.horizon)
            if int(j["replicate"]) == 0]  # one job per seed; play_seed plays its G siblings
    model, om = load_start(a.turn_models, a.replenish_models)
    model0, _ = load_start(a.turn_models, a.replenish_models)
    t = time_iteration(model, model0, om, jobs, temps, a.workers, a.steps)
    t["iteration_seconds_estimate"] = t["rollout_seconds"] + CFG["steps"] * t["seconds_per_step"]
    Path(a.out).write_text(json.dumps(t, indent=1, default=float) + "\n")
    print(f"[onpolicy] time {json.dumps(t, default=float)}", flush=True)
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
