"""M6 Stage I: imitation of k5_rf5 + MWWk by the fully learned GNN policies (add-m6-joint-mpg-gnn D4).

`python -m tokenomics.train.m6_imitation --mode {timing,collect,fit,check}`
- `timing` (L0, local, tiny): seconds per decision of the GNN policies against k5_rf5 / MWWk, and a horizon-H
  episode with the GNN stack, at 12 and 42 producers. Re-sizes sizing.md before any pod.
- `collect`: k5_rf5 + MWWk on M4's pinned training world-episodes (EXP-0058 `bar/`), excluding the 16 monitor
  world-episodes, at r0. It records every producer decision (features, k5_rf5's raw action as the label) and
  every order-slot decision (features, MWW's normalised weights over the eligible producers as a soft label).
  Each label is a deterministic function of the agent's view: tested.
- `fit`: behaviour cloning, with BCE on the 3 producer toggles, Huber on the replenish fractions, and KL on the
  orderer's slot distributions. v2 (EXP-0100, after DIAGNOSTICS 31): an in-model median / IQR normaliser over
  signed-log1p'd counts, fitted on the training samples and stored in the checkpoint; gradient clipping; a
  synchronous data-parallel fit (deterministic) instead of v1's Hogwild.
- `check` (the pre-registered imitation check, D4):
  - the GNN stack and k5_rf5 + MWWk on the 16 monitor world-episodes at replicates 0 / 1;
  - g_imit = the equal-weight mean of (imitated − bar) units, F_imit = M5's floors rule;
  - **`faithful`** if g_imit ≥ −F_imit and no extra deaths, else **`short`** (a stop: the user hears it before
    any RL);
  - reported: exact-command agreement, gate accuracy and replenish MAE (producer), with per toggle head and
    replenish row class (`producer_heads`), and KL and top-1 agreement (orderer), on the monitor episodes' labels;
  - reported only (v2): the role-swap arms, GNN producer + MWWk and k5_rf5 + GNN orderer, on the same keys.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import producer_es as pe

BASE_PRODUCER = "entity_planner_buyahead_k5_rf5"
BASE_ORDERER = "mww_keyed"
IMIT_RULE = ("faithful if g_imit >= -F_imit (M5's floors rule on the 16 monitor world-episodes at "
             "replicates 0 / 1) and the imitated stack has no more deaths than k5_rf5 + MWWk; else "
             "short (stop: the user hears it before any RL)")
MONITOR_REPLICATES = (0, 1)
# the fit's pre-registered stopping rule and settings (v2, EXP-0100; v1 = EXP-0099's Hogwild fit).
# `batch` is per worker: a sync step averages `workers` shard gradients (global 64 x 16).
FIT = {"epochs": 20, "batch": 64, "lr": 3e-3, "seed": 0, "workers": 16, "mode": "auto",
       "clip_grad_norm": 1.0, "norm_decisions": 50_000, "sync_max_cost": 1.5}
MODE_RULE = ("auto: before the fit, one epoch of each mode on the full training samples (hogwild "
             "first, then sync; fresh models, the same seed); sync if its epoch costs <= "
             "sync_max_cost x hogwild's, else hogwild (with clipping); the mode and both times "
             "recorded in spec.json")
FIDELITY_METRICS = ("exact_command", "gate_accuracy", "replenish_mae_units", "orderer_kl",
                    "orderer_top1", "producer_heads")
# Reported-only role-swap arms (v2): each GNN actor beside the other base actor, same keys.
ARMS = {"monitor": ("gnn", "gnn"), "monitor_base": ("base", "base"),
        "swap_gnn_producer": ("gnn", "base"), "swap_gnn_orderer": ("base", "gnn")}
SWAP_ARMS = ("swap_gnn_producer", "swap_gnn_orderer")
TOGGLES = ("restore", "upgrade", "expand")
REGISTERED_ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
PREREG_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v2.json"
SOURCE_COMMIT = ""  # set by `main` (--source-commit); recorded in every output
# v1 (m6_prereg_v1.json, 38db5c0c..., EXP-0099) stays frozen and unedited. v2's fit may reuse the
# samples v1's collect wrote (EXP-0099 artifacts/collect_sha256.txt): features and labels are
# unchanged (the normaliser lives in the model).
PREREG_V1_SHA256 = "38db5c0c96cfd2a7fa14eae1e3bf2832092738d1d13ce2fdcade25ac245c26f6"
EXP0099_SAMPLES = {
    "samples_train.pkl": "5c404d4e3fba082fde1762b221bb965360440bc421d98b38c7c817bb867dc10b",
    "samples_monitor.pkl": "0cbfa88646d25a99a4af895f314311cd4349feb4a62f734429f0ef2a68132abb"}
PREREG_SHA256: str | None = "2460cd07755930225837f996450f359e10480d495e3fc3e5be540d597a76d4a3"  # frozen before EXP-0100


# --- pre-registration (frozen before Stage I's data) ---


def registered_prereg() -> dict:
    return {
        "format": "m6-prereg-v2", "change": "add-m6-joint-mpg-gnn", "stage": "I (imitation), D4",
        "status": "frozen before EXP-0100's data; real collect / fit / check refuse any other file",
        "supersedes": {"m6_prereg_v1.json": PREREG_V1_SHA256,
                       "why": "EXP-0099 `short`; the orderer fit diverged on unscaled features "
                              "(DIAGNOSTICS 31)"},
        "decisions": "notes/user-decisions.md (2026-11-06): fully learned, from imitation; the "
                     "user hears a `short` reading before any RL",
        "base": {"producer": BASE_PRODUCER, "orderer": BASE_ORDERER},
        "data": {"training": {**pe.TRAINING_PINS, "world_episodes": "the 128 training world-"
                              "episodes minus the 16 monitor ones (112), replicate 0"},
                 "horizon": pe.HORIZON, "order_ttl": 2,
                 "labels": {"producer": "k5_rf5's raw action: 3 toggles (raw > 0) and per input "
                                        "row the replenish fraction raw / BOUND",
                            "orderer": "MWW's normalised makeable weights over the eligible "
                                       "producers per order slot; slots with one eligible "
                                       "producer dropped"},
                 "samples": {"reuse": "EXP-0099's collected samples (v1 collect), only at these "
                                      "SHA-256s; else a v2 collect",
                             "sha256": EXP0099_SAMPLES}},
        "model": {"hidden": g.HIDDEN, "rounds": g.ROUNDS,
                  "losses": "BCE (toggles) + Huber delta 0.1 (replenish fractions, masked) + KL "
                            "(orderer slots)",
                  "normaliser": {"format": g.NORM_FORMAT, "clip": g.NORM_CLIP,
                                 "transform": "signed log1p on the registered count columns",
                                 "orderer_log1p": {k: list(v) for k, v in g.ORDERER_LOG1P.items()},
                                 "producer_log1p": {k: list(v) for k, v in
                                                    g.PRODUCER_LOG1P.items()},
                                 "centre_scale": "median; IQR, IQR = 0 -> std, std = 0 -> 1",
                                 "fitted_on": "every k-th training decision per actor, k = "
                                              "ceil(n / norm_decisions); stored in the "
                                              "checkpoint"}},
        "fit": {**FIT, "stopping_rule": f"fixed {FIT['epochs']} epochs (no early stopping); the "
                                        "checkpoint SHA-256 is the pinned artifact",
                "sync": "synchronous data parallel: per step each worker's shard-mean gradient, "
                        "averaged in worker order, clipped to the global norm, one Adam step; "
                        "deterministic",
                "hogwild": "v1's fit (shared weights, a shard and an Adam per process) plus "
                           "per-step clipping; not bit-reproducible",
                "mode_rule": MODE_RULE},
        "check": {"rule": IMIT_RULE,
                  "monitor_set": "schedule.json `monitor` (16 world-episodes)",
                  "replicates": list(MONITOR_REPLICATES),
                  "comparison": "the GNN stack vs k5_rf5 + MWWk at the same DrawKeys",
                  "reported_only_arms": {a: {"producer": ARMS[a][0], "orderer": ARMS[a][1]}
                                         for a in SWAP_ARMS},
                  "swap_reading": "each swap arm - k5_rf5 + MWWk with the same floor rule; "
                                  "deaths; reported, never gating",
                  "floor": "M5's floors rule (producer_read.pilot_reading: max(the atlas-spread "
                           "floor at the read's counts, own 3 SE), the n = 1 rule); atlas spreads "
                           "from the registered atlas v2",
                  "floor_source": {"atlas_content_hash": REGISTERED_ATLAS_HASH}},
        "reported": {"fidelity_metrics": list(FIDELITY_METRICS),
                     "on": "the training and the monitor episodes' labels",
                     "also": "units of both stacks and deaths"}}


def freeze_prereg(path: Path = PREREG_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    text = json.dumps(registered_prereg(), indent=1, sort_keys=True) + "\n"
    if not path.exists():
        path.write_text(text)
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == registered_prereg() else "MISMATCH"), \
        sha256_file(str(path))


# --- Stage I v3 (supervisor decision 2026-11-06, DIAGNOSTICS 32): the ceiling-margin feature, collect at
# 12 and 42 producers (training split only), a three-clause check; pre-registration v5 ---

PREREG_V5_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v5.json"
PREREG_V5_SHA256: str | None = "4fdf45ef0ed44c6f899af606eb8b5073f6dddf750f2b2cb6c5b8ab1d2f0526c9"  # re-frozen before Stage I v3's data (pre-data 63d85a94... superseded)
PREREG_V5_PREDATA_SHA = "63d85a948834ed9a569c71cad870aa7eb1d41f816f12ea946181bdcefbbdb569"
COLLECT_42 = {"episodes": 2, "stage": "m6-collect-42", "horizon": 200, "replicate": 0,
              "subsample": {"producer": 2, "orderer": 5}}
# the atlas v2 train-split 42-producer worlds, collapse excluded (never held-out or validation)
COLLECT_42_WORLDS = (
    "ws-atlas-v1-42-0000", "ws-atlas-v1-42-0013", "ws-atlas-v1-42-0031", "ws-atlas-v1-42-0034",
    "ws-atlas-v1-42-0037", "ws-atlas-v1-42-0039", "ws-atlas-v1-42-0040", "ws-atlas-v1-42-0041",
    "ws-atlas-v1-42-0043", "ws-atlas-v1-42-0052", "ws-atlas-v1-42-0054", "ws-atlas-v1-42-0073",
    "ws-atlas-v1-42-0076", "ws-atlas-v1-42-0081", "ws-atlas-v1-42-0093", "ws-atlas-v1-42-0113",
    "ws-atlas-v1-42-0119", "ws-atlas-v1-42-0122", "ws-atlas-v1-42-0126")
# EXP-0104's non-held-out 42 set (m6_drain.worlds "other": rho > 0.5, not collapse)
CHECK_42_WORLDS = (
    "ws-atlas-v1-42-0000", "ws-atlas-v1-42-0013", "ws-atlas-v1-42-0039", "ws-atlas-v1-42-0043",
    "ws-atlas-v1-42-0066", "ws-atlas-v1-42-0076", "ws-atlas-v1-42-0081", "ws-atlas-v1-42-0126")
CHECK_42 = {"episodes": 4, "stage": "m6-drain", "replicates": (0, 1), "horizon": 200}
EXP0100_MODELS_SHA = "5acf8b82df4686dd14a1da3a441a390f87fe32bf3c177e16ddd63fe417f43126"
V3_RULE = ("faithful iff (a) at 12: the v3 GNN stack vs k5_rf5 + MWWk on the 16 monitor world-episodes "
           "x replicates 0 / 1, g >= -F (M5's floors) and no more deaths (Stage I's rule); (b) at 42: on "
           "CHECK_42_WORLDS at horizon 200 (4 episodes, stage m6-drain, replicates 0 / 1), the producers "
           "past the sustainable ceiling at episode end, summed, are no more than the base's, and the "
           "units are not below the base's beyond the floor (L5's mean_comp floor); (c) the v3 stack vs "
           "the EXP-0100 stack on the monitor set, g >= -F. Else short, naming the clauses")


def registered_prereg_v5() -> dict:
    return {
        "format": "m6-prereg-v5", "change": "add-m6-joint-mpg-gnn",
        "stage": "Stage I v3: imitation with the ceiling-margin feature, collected at 12 and 42",
        "status": "frozen before Stage I v3's data; real collect / fit / check refuse any other file",
        "supersedes_predata": {"sha256": PREREG_V5_PREDATA_SHA,
                               "why": "review 2026-11-06, before any data: the margin's denominator "
                                      "max(restore_effect, 1) grew with scale on rows without a "
                                      "restoration (restore_effect 0: -deg_after); bounded to "
                                      "max(restore_effect, deg_after, 1)"},
        "keeps": {"m6_prereg_v2.json": PREREG_SHA256, "m6_prereg_v4.json": PREREG_V4_SHA256,
                  "note": "v1..v4 stay frozen; the critic-free Stage 0 is pre-registered as v6"},
        "decision": "supervisor 2026-11-06: fix the diagnosed cause (DIAGNOSTICS 32, EXP-0104) in the "
                    "imitation stage before RL; 42-0125 and every held-out world excluded",
        "cause": "EXP-0104: the imitated producer upgrades past the sustainable ceiling at 42; the "
                 "offline check found no clipping (|z| < 5): the gate misses an exact integer "
                 "threshold by one unit of degradation",
        "feature": {"version": g.FEATURES_V3,
                    "ceiling_margin": "(restore_effect - deg_after) / max(restore_effect, deg_after, "
                                      "1), deg_after = ceil(rate (max + upgrade_effect) / 100), per "
                                      "capacity row (m6_gnn.ceiling_margin); in [-1, 1]",
                    "normaliser": "as v2 (median / IQR, clip 5), fitted on the v3 training samples"},
        "base": {"producer": BASE_PRODUCER, "orderer": BASE_ORDERER},
        "data": {"12": {**pe.TRAINING_PINS, "world_episodes": "the 112 training world-episodes "
                                                              "(monitor excluded), replicate 0",
                        "horizon": pe.HORIZON},
                 "42": {**COLLECT_42, "worlds": list(COLLECT_42_WORLDS),
                        "subsample": "every 2nd cycle's producer decisions, every 5th cycle's "
                                     "orderer decisions (memory and fit time)"},
                 "monitor": "the 16 monitor world-episodes at r0 (fidelity metrics only)",
                 "labels": "as v2", "atlas_content_hash": REGISTERED_ATLAS_HASH},
        "model": {"hidden": g.HIDDEN, "rounds": g.ROUNDS, "producer_features": g.FEATURES_V3,
                  "losses": "as v2"},
        "fit": {**FIT, "mode_rule": MODE_RULE, "stopping_rule": f"fixed {FIT['epochs']} epochs"},
        "check": {"rule": V3_RULE, "check_42_worlds": list(CHECK_42_WORLDS), "check_42": {
            **CHECK_42, "replicates": list(CHECK_42["replicates"])},
            "reference_models_sha256": EXP0100_MODELS_SHA,
            "ceiling": "degradation_for(t) > restoration.restores on the restoration type",
            "overlap": "the 42 check worlds include collect worlds (other seeds) and validation "
                       "42-0066 (already read in EXP-0102)"},
        "reported": ["fidelity per head at 12 and on the 42 samples", "per-world 42 counts and units",
                     "the fit curve and the mode probe"]}


def freeze_prereg_v5(path: Path = PREREG_V5_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v5()
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


# --- Stage I v4 (supervisor decision 2026-11-06, DIAGNOSTICS 33): the margin's sign as a step input, a
# boundary weight on the upgrade gate, an offline gate before the closed-loop check; pre-registration v6 ---

PREREG_V6_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v6.json"
PREREG_V6_SHA256: str | None = "1bd23f29e657215365e84048e7e3e148c8c3c58dc18766a586aeb17841355db2"  # frozen before Stage I v4's data
EXP0105_COLLECT = {
    "samples_train.pkl": "6339c8c884d717f8a07893ea0b7385eee9873972b4c70ff1d0d323b57883cb9c",
    "samples_monitor.pkl": "42322622542b76de533988d31bec038600f2ea9482820ada86cc31aa29ab581e",
    "collect.json": "fb3303f4fd691bf3b727e8bb6168f93d2542713a88d602c322f5a31a2d699e9d"}
GATE_42_WORLDS = COLLECT_42_WORLDS[::6]  # held out of the v4 fit: the 42 gate set
DECISIONS_PER_42_EPISODE = 4200  # 42 producers x 100 recorded cycles (every 2nd of 200), no deaths
V4_GATE = {"below_zero_fire_max": 0.01, "above_zero_agreement_slack": 0.02,
           "heads_no_worse": ["restore", "expand"]}
BOUNDARY_WEIGHT_RULE = (
    "per size s, w_s = #(decisions with upgrade-row margin >= 0 and upgrade label 1) / #(decisions "
    "with margin < 0 and upgrade label 0) on the v4 training samples of size s; the upgrade gate's "
    "BCE term of each margin < 0, label-0 decision is weighted w_s, every other term 1. It equalises "
    "per size the total weight of the two error types the gate must separate (firing at margin < 0, "
    "missing a legitimate upgrade), with no free constant")
V4_RULE = (
    "step 1 (offline gate, on recorded k5_rf5 decisions: the 12-producer monitor set and the "
    "42-producer GATE_42_WORLDS held out of the fit): pass iff the v4 upgrade fire rate at margin < 0 "
    "is <= 1% at both sizes, the upgrade accuracy at margin >= 0 on the 12 monitor set is >= v2's "
    "(EXP-0100) - 0.02, and the restore and expand accuracies are >= v2's at both sizes; else "
    "gate_fail (stop, no closed-loop run). step 2 (only after a pass): EXP-0105's three-clause check "
    "(V3_RULE) unchanged with the v4 models: faithful or short naming the clauses")


def registered_prereg_v6() -> dict:
    return {
        "format": "m6-prereg-v6", "change": "add-m6-joint-mpg-gnn",
        "stage": "Stage I v4: the margin's sign as a step input, a boundary-weighted upgrade gate",
        "status": "frozen before Stage I v4's data; real fit / gate / check refuse any other file",
        "keeps": {"m6_prereg_v5.json": PREREG_V5_SHA256,
                  "note": "v1..v5 stay frozen; the critic-free Stage 0 becomes v7"},
        "decision": "supervisor 2026-11-06 after EXP-0106 (DIAGNOSTICS 33: the gate learns the ceiling "
                    "smoothly; a non-zero boundary rate compounds into irreversible upgrades)",
        "feature": {"version": g.FEATURES_V4, "added": "1[margin < 0] next to the v3 margin "
                                                       "(m6_gnn.v3_to_v4 on recorded v3 features)"},
        "data": {"reuse": "EXP-0105's collect, only at these SHA-256s", "sha256": EXP0105_COLLECT,
                 "held_out_of_fit": {"producer_42_worlds": list(GATE_42_WORLDS),
                                     "rule": "COLLECT_42_WORLDS[::6], both episodes (blocks of 2 x "
                                             f"{DECISIONS_PER_42_EPISODE} decisions in world order)",
                                     "orderer": "orderer samples kept whole (the gate is the "
                                                "producer's)"}},
        "loss": {"boundary_weight_rule": BOUNDARY_WEIGHT_RULE, "otherwise": "as v2 / v3"},
        "model": {"hidden": g.HIDDEN, "rounds": g.ROUNDS, "producer_features": g.FEATURES_V4},
        "fit": {**FIT, "mode_rule": MODE_RULE, "stopping_rule": f"fixed {FIT['epochs']} epochs"},
        "gate": {**V4_GATE, "reference_models_sha256": EXP0100_MODELS_SHA,
                 "measured_by": "m6_gate_probe.model_stats (fire / accuracy at margin < 0 and >= 0, "
                                "per-head accuracy)"},
        "rule": V4_RULE,
        "check": {"rule": V3_RULE, "check_42_worlds": list(CHECK_42_WORLDS),
                  "check_42": {**CHECK_42, "replicates": list(CHECK_42["replicates"])},
                  "overlap": "as v5: the check worlds include collect worlds and validation 42-0066"}}


def freeze_prereg_v6(path: Path = PREREG_V6_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v6()
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


# --- EXP-0108 (supervisor decision 2026-11-06): attribution after EXP-0107; pre-registration v7 ---

PREREG_V7_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v7.json"
PREREG_V7_SHA256: str | None = "970fa4aa24f3398b50d1d82a5e40a29be157d5ff8074de69d4338a0f25758275"  # frozen before EXP-0108's data
EXP0107_MODELS_SHA = "83a0e8571d5e2c6a0cab2d1792ee8082f0caeda9b3371a3b8e42e951af8da64d"
# label: (monitor kind, 42 arm, model source: "cand" = v4 producer + EXP-0100 orderer, "v4", "prev")
ATTRIBUTION_ARMS = {
    "C": ("monitor", "gnn", "cand"),
    "base": ("monitor_base", "k5_rf5", None),
    "v4p_mwwk": ("swap_gnn_producer", "gnn_p", "v4"),
    "k5_v4o": ("swap_gnn_orderer", "gnn_o", "v4"),
    "k5_o100": ("swap_gnn_orderer", "gnn_o", "prev")}
REPORTED_ARMS = ("v4p_mwwk", "k5_v4o", "k5_o100")
V7_RULE = ("the candidate C = the v4 producer (EXP-0107) + the EXP-0100 orderer, on EXP-0107's check sets, "
           "EXP-0105's three clauses unchanged (V3_RULE; (c) against the EXP-0100 stack): faithful or "
           "short naming the clauses. Reported only, never deciding: v4 producer + MWWk, k5_rf5 + v4 "
           "orderer, k5_rf5 + EXP-0100 orderer, each vs k5_rf5 + MWWk with the same floors (12: M5's "
           "floors on the monitor set; 42: L5's mean_comp floor), deaths and past-ceiling counts")


def registered_prereg_v7() -> dict:
    return {
        "format": "m6-prereg-v7", "change": "add-m6-joint-mpg-gnn",
        "stage": "EXP-0108: attribution of EXP-0107's shortfall; the composite candidate",
        "status": "frozen before EXP-0108's data; the real run refuses any other file",
        "keeps": {"m6_prereg_v6.json": PREREG_V6_SHA256,
                  "note": "v1..v6 stay frozen; the critic-free Stage 0 becomes v8"},
        "decision": "supervisor 2026-11-06 after EXP-0107 (offline gate passed; the v4 orderer "
                    "diverged at epochs 14-15, a DIAGNOSTICS 31 recurrence)",
        "candidate": {"producer": {"from": "EXP-0107 fit/models.pt", "sha256": EXP0107_MODELS_SHA},
                      "orderer": {"from": "EXP-0100 fit/models.pt", "sha256": EXP0100_MODELS_SHA},
                      "rationale": "the v4 orderer is a failed fit; EXP-0100's orderer is a faithful "
                                   "imitation of the same teacher (MWW), never diverged"},
        "arms": {k: {"monitor_kind": v[0], "arm_42": v[1], "models": v[2]}
                 for k, v in ATTRIBUTION_ARMS.items()},
        "reported_arms": list(REPORTED_ARMS),
        "rule": V7_RULE,
        "sets": {"monitor": "the 16 monitor world-episodes x replicates 0 / 1, horizon 100",
                 "check_42_worlds": list(CHECK_42_WORLDS),
                 "check_42": {**CHECK_42, "replicates": list(CHECK_42["replicates"])}},
        "phases": "deciding arms (C, base, the EXP-0100 stack) first and written; then the reported "
                  "arms"}


def freeze_prereg_v7(path: Path = PREREG_V7_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v7()
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


def prereg_problems(path: Path = PREREG_V7_PATH) -> list[str]:
    """Real runs are EXP-0108's: the pinned v7 (v6 governed EXP-0107, v5 EXP-0105)."""
    from ..eval.provenance import sha256_file

    if PREREG_V7_SHA256 is None:
        return ["no pinned EXP-0108 pre-registration (m6_prereg_v7.json; `--mode freeze-v7`)"]
    if not path.exists() or sha256_file(str(path)) != PREREG_V7_SHA256:
        return [f"{path.name} does not match the pinned SHA-256 {PREREG_V7_SHA256[:12]}"]
    if json.loads(path.read_text()) != registered_prereg_v7():
        return ["the pre-registration differs from the code"]
    return []


def upgrade_margin(f: dict) -> float | None:
    """The upgrade type's row margin (the row with upgrade_effect > 0), else None."""
    cap = f["capacity"]
    rows = np.nonzero(cap[:, 12] > 0)[0] if cap.shape[0] else []
    return float(cap[rows[0], g.MARGIN_COL]) if len(rows) else None


def split_v4(producer: list, n42: int, dry_run: bool = False) -> tuple[list, list, list]:
    """EXP-0105's recorded producer samples -> (42 train, 12 train, 42 gate), v4 features. The 42
    block is in COLLECT_42_WORLDS order, 2 episodes x DECISIONS_PER_42_EPISODE per world."""
    per_world = 2 * DECISIONS_PER_42_EPISODE
    if n42 != per_world * len(COLLECT_42_WORLDS):
        if not dry_run:
            raise ValueError(f"the 42 block ({n42}) is not {len(COLLECT_42_WORLDS)} worlds x "
                             f"{per_world} decisions")
        per_world = max(1, n42 // len(COLLECT_42_WORLDS))
    conv = [(g.v3_to_v4(f), y) for f, y in producer]
    tr42, gate42 = [], []
    for i, w in enumerate(COLLECT_42_WORLDS):
        block = conv[i * per_world:(i + 1) * per_world]
        (gate42 if w in GATE_42_WORLDS else tr42).extend(block)
    return tr42, conv[n42:], gate42


def boundary_weights(samples: list) -> tuple[list, dict]:
    """BOUNDARY_WEIGHT_RULE on one size's samples: (weighted samples, the counts and w)."""
    pos = neg = 0
    for f, y in samples:
        m = upgrade_margin(f)
        if m is None:
            continue
        if m >= 0 and y["toggles"][1] == 1:
            pos += 1
        elif m < 0 and y["toggles"][1] == 0:
            neg += 1
    w = pos / neg if neg else 1.0
    out = []
    for f, y in samples:
        m = upgrade_margin(f)
        if m is not None and m < 0 and y["toggles"][1] == 0:
            y = {**y, "gate_weight": np.asarray([1.0, w, 1.0], dtype=np.float32)}
        out.append((f, y))
    return out, {"above_zero_upgrades": pos, "below_zero_no_upgrades": neg, "w": w}


def real_problems(*, horizon: int, max_world_episodes: int | None, ts: Any = None,
                  atlas: Any = None, epochs: int | None = None, workers: int | None = None,
                  seed: int | None = None) -> list[str]:
    p = prereg_problems()
    if horizon != pe.HORIZON:
        p.append(f"horizon {horizon} != {pe.HORIZON}")
    if max_world_episodes:
        p.append("--max-world-episodes is set")
    if ts is not None and ts.pins != pe.TRAINING_PINS:
        p.append("the training set is not M4's pinned set")
    if atlas is not None and atlas.get("content_hash") != REGISTERED_ATLAS_HASH:
        p.append("the atlas is not the registered atlas v2")
    for k, v in (("epochs", epochs), ("workers", workers), ("seed", seed)):
        if v is not None and v != FIT[k]:
            p.append(f"--{k} {v} != the registered {FIT[k]}")
    return p


def _refuse(problems: list[str]) -> None:
    if problems:
        raise ValueError("not a valid real run (" + "; ".join(problems) + "); pass --dry-run")


# --- labels (deterministic functions of the agent's view; tested) ---


def producer_label(env: Any, pid: str, action: np.ndarray, inputs: Sequence[Any]
                   ) -> dict[str, np.ndarray]:
    """k5_rf5's raw action as the label: 3 toggles (raw > 0 fires) and, per input row, the
    replenish fraction raw / BOUND in [0, 1] (rows the env does not index are masked)."""
    sp = env.spec_
    tog = (np.asarray(action[sp.num_inputs:sp.num_inputs + 3]) > 0).astype(np.float32)
    frac, mask = [], []
    for gp in inputs:
        if gp in sp.input_index:
            frac.append(float(np.clip(action[sp.input_index[gp]] / ACTION_LOGIT_BOUND, 0, 1)))
            mask.append(1.0)
        else:
            frac.append(0.0)
            mask.append(0.0)
    return {"toggles": tog, "replenish": np.asarray(frac, np.float32),
            "mask": np.asarray(mask, np.float32)}


def orderer_labels(env: Any, slots: Sequence, feats: dict[str, Any]) -> list[np.ndarray]:
    """Per slot, MWW's normalised weights over the eligible producers, in the features' column
    order (MWWk draws from exactly these)."""
    from ..eval.routing_diagnostics import makeable_weights

    pids = env.spec_.producer_ids
    out = []
    for (k, oid, q, el), cols in zip(slots, feats["elig"]):
        if not el or not cols:
            out.append(np.zeros(0, np.float32))
            continue
        w = dict(zip((pids[i] for i in el), makeable_weights(env, oid, el)))
        v = np.asarray([w[feats["pids"][c]] for c in cols], np.float64)
        out.append((v / v.sum()).astype(np.float32))
    return out


# --- recording episodes ---


def _episode_env(job: dict) -> Any:
    from ..eval.joint_stack import joint_stack_env
    from ..eval.routing_audit import job_config
    from ..eval.routing_diagnostics import DrawKey, set_draw_key

    env = joint_stack_env(job_config(job), job["horizon"])
    set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
    return env


def record_job(job: dict) -> dict:
    """Play k5_rf5 + MWWk on one keyed world-episode, recording every decision's features and
    label; returns {"producer": [...], "orderer": [...], "units", "dead"}."""
    from ..eval.baselines import producing_policy
    from ..eval.rotation_routers import _slots
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _episode_env(job)
    base_p, base_o = producing_policy(BASE_PRODUCER), ordering_policy(BASE_ORDERER)
    prod_s: list = []
    ord_s: list = []

    sub = job.get("subsample") or {}
    kp, ko = int(sub.get("producer", 1)), int(sub.get("orderer", 1))

    def rec_p(e: Any, agent: str) -> np.ndarray:
        a = base_p(e, agent)
        if e.system.cycle_counter % kp:
            return a
        pid = e._role[agent][1]
        f = g.producer_features(e.system.producer_entity_view(pid), g.producer_context(e, pid),
                                g.LATEST_FEATURES)
        prod_s.append((f, producer_label(e, pid, a, f["inputs"])))
        return a

    def rec_o(e: Any, agent: str) -> np.ndarray:
        aid = e._role[agent][1]
        slots = _slots(e, agent) if e.system.cycle_counter % ko == 0 else []
        if slots:
            f = g.orderer_features(e.system.product_stream_view(aid), slots, e.spec_.producer_ids)
            ys = orderer_labels(e, slots, f)
            if any(len(y) > 1 for y in ys):  # a single eligible producer carries no label signal
                ord_s.append((f, ys))
        return base_o(e, agent)

    run_episode(env, rec_p, rec_o, seed=job["episode"], action_seed=None)
    st = env.episode_stats
    return {"producer": prod_s, "orderer": ord_s, "units": float(st["consumer_completed_units"]),
            "dead": bool(st["dead"]), "job_id": job.get("job_id")}


def play_job(job: dict) -> dict:
    """One keyed episode of arm `job["kind"]` (ARMS: who plays producer / orderer)."""
    from ..eval.baselines import producing_policy
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _episode_env(job)
    who_p, who_o = ARMS[job["kind"]]
    pm, om = load_models(job["models"]) if "gnn" in (who_p, who_o) else (None, None)
    p = g.gnn_producing_policy(pm) if who_p == "gnn" else producing_policy(BASE_PRODUCER)
    o = g.gnn_ordering_policy(om) if who_o == "gnn" else ordering_policy(BASE_ORDERER)
    run_episode(env, p, o, seed=job["episode"], action_seed=None)
    st = env.episode_stats
    return {"world": job["world"], "episode": str(job["episode"]),
            "replicate": str(job["replicate"]), "units": float(st["consumer_completed_units"]),
            "dead": "true" if st["dead"] else "false", "kind": job["kind"],
            "generation": str(job["generation"])}


def _jobs(ts: Any, wes: Sequence, kind: str, replicates: Sequence[int], horizon: int,
          models: str | None = None) -> list[dict]:
    return [{**pe.make_job(ts, kind=kind, producing=BASE_PRODUCER, theta=None, generation=0,
                           member=kind, world=w, episode=e, replicate=r, horizon=horizon),
             "models": models} for w, e in wes for r in replicates]


def _pool(fn: Any, jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                             initializer=pe._init_worker) as ex:
        return list(ex.map(fn, jobs))


# --- fit (behaviour cloning) ---


def bc_losses(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list
              ) -> tuple[torch.Tensor, torch.Tensor]:
    lp = torch.zeros(())
    for f, y in prod:
        gates, rep = pm(f)
        w = y.get("gate_weight")  # Stage I v4's boundary weight on the upgrade gate (prereg v6)
        lp = lp + torch.nn.functional.binary_cross_entropy_with_logits(
            gates, torch.from_numpy(y["toggles"]), reduction="sum",
            weight=None if w is None else torch.from_numpy(w))
        if rep.shape[0]:
            m = torch.from_numpy(y["mask"])
            lp = lp + (torch.nn.functional.huber_loss(rep, torch.from_numpy(y["replenish"]),
                                                      reduction="none", delta=0.1) * m).sum()
    lo = torch.zeros(())
    for f, ys in orde:
        for lg, y in zip(om(f), ys):
            if len(y):
                lo = lo - (torch.from_numpy(y) * torch.log_softmax(lg, 0)).sum()
    return lp / max(1, len(prod)), lo / max(1, len(orde))


class Stepper:
    """The optimiser step of a fit (DIAGNOSTICS 31 recurrence, EXP-0107).
    - `per_agent` False: one Adam and one global clip over every group's parameters, exactly the v1..v6
      fits (a diverging orderer's gradient rescales the producer's through the shared clip).
    - `per_agent` True: one Adam and one clip per group (agent), so no agent's gradient touches
      another's step.
    - `log_path`: one JSON line per step with, per group, the gradient norm before clipping and the
      weight norm after the step, plus the step's losses."""

    def __init__(self, groups: dict[str, list], lr: float | dict[str, float], clip: float | None,
                 per_agent: bool, log_path: str | Path | None = None) -> None:
        self.groups = {k: list(v) for k, v in groups.items()}
        lrs = lr if isinstance(lr, dict) else {k: lr for k in self.groups}
        if per_agent:
            self.steps = self.groups
            self.opts = {k: torch.optim.Adam(ps, lr=lrs[k]) for k, ps in self.groups.items()}
        else:
            self.steps = {"all": [p for ps in self.groups.values() for p in ps]}
            self.opts = {"all": torch.optim.Adam(
                self.steps["all"] if not isinstance(lr, dict) else
                [{"params": ps, "lr": lrs[k]} for k, ps in self.groups.items()],
                lr=lr if not isinstance(lr, dict) else next(iter(lrs.values())))}
        self.clip, self.per_agent, self.n = clip, per_agent, 0
        self.fh = None if log_path is None else open(log_path, "a")

    @staticmethod
    def _norm(ts: list) -> float:
        ts = [t for t in ts if t is not None]
        return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t.detach())
                                                           for t in ts]))) if ts else 0.0

    def zero_grad(self) -> None:
        for o in self.opts.values():
            o.zero_grad()

    def step(self, losses: dict[str, float] | None = None, epoch: int | None = None) -> None:
        rec: dict[str, Any] = {"step": self.n, "epoch": epoch, **(losses or {})}
        if self.fh is not None:
            for k, ps in self.groups.items():
                rec[f"{k}_grad_norm"] = self._norm([p.grad for p in ps])
        for k, ps in self.steps.items():
            if self.clip:
                torch.nn.utils.clip_grad_norm_(ps, self.clip)
            self.opts[k].step()
        if self.fh is not None:
            for k, ps in self.groups.items():
                rec[f"{k}_weight_norm"] = self._norm(list(ps))
            self.fh.write(json.dumps(rec) + "\n")
            self.fh.flush()
        self.n += 1

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()

    def state_dict(self) -> dict:
        return {"per_agent": self.per_agent, "n": self.n,
                "opts": {k: o.state_dict() for k, o in self.opts.items()}}

    def load_state_dict(self, d: dict) -> None:
        if d.get("per_agent") != self.per_agent:
            raise ValueError("the checkpoint's optimiser layout differs (per_agent)")
        self.n = int(d["n"])
        for k, o in self.opts.items():
            o.load_state_dict(d["opts"][k])


def _stepper(pm: g.ProducerGNN, om: g.OrdererGNN, lr: float, clip: float | None,
             per_agent: bool, step_log: str | Path | None) -> Stepper:
    return Stepper({"producer": list(pm.parameters()), "orderer": list(om.parameters())}, lr,
                   clip, per_agent, step_log)


def _epochs(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list, epochs: int, batch: int,
            lr: float, seed: int, log: Any, clip: float | None = None, per_agent: bool = False,
            step_log: str | Path | None = None) -> list[dict]:
    torch.set_num_threads(1)
    st = _stepper(pm, om, lr, clip, per_agent, step_log)
    rng = np.random.default_rng(seed)
    curve = []
    for ep in range(epochs):
        ip, io = rng.permutation(len(prod)), rng.permutation(len(orde))
        n = max(1, max(len(prod), len(orde)) // batch)
        tot = [0.0, 0.0]
        for b in range(n):
            bp = [prod[i] for i in ip[b * batch:(b + 1) * batch]]
            bo = [orde[i] for i in io[b * batch:(b + 1) * batch]]
            lp, lo = bc_losses(pm, om, bp, bo)
            st.zero_grad()
            if (lp + lo).requires_grad:
                (lp + lo).backward()
                st.step({"producer_loss": float(lp.detach()), "orderer_loss": float(lo.detach())},
                        ep + 1)
            tot[0] += float(lp.detach())
            tot[1] += float(lo.detach())
        curve.append({"epoch": ep + 1, "producer_loss": tot[0] / n, "orderer_loss": tot[1] / n})
        log(f"[m6_imitation] epoch {ep + 1}: producer {tot[0] / n:.4f} orderer {tot[1] / n:.4f}")
    st.close()
    return curve


@dataclass
class Shards:
    """The training samples split for the parallel fit (EXP-0105: 16 workers each holding the whole
    sample list ran a 32 GB pod out of memory). File w holds `samples[w::workers]`, so a shard fit
    is the same computation as the in-memory one; each worker loads only its own file. `norm`
    holds the normaliser's strided subsample, taken before the split."""

    paths: list[str]
    counts: tuple[int, int]  # total producer / orderer decisions
    norm: dict[str, list]


def write_shards(samples: dict, workers: int, out: Path, norm_decisions: int) -> Shards:
    out.mkdir(parents=True, exist_ok=True)
    prod, orde = samples["producer"], samples["orderer"]
    paths = []
    for w in range(workers):
        p = out / f"shard_{w:02d}.pkl"
        with p.open("wb") as fh:
            pickle.dump({"producer": prod[w::workers], "orderer": orde[w::workers]}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(str(p))
    return Shards(paths, (len(prod), len(orde)),
                  {"producer": _norm_feats(prod, norm_decisions),
                   "orderer": _norm_feats(orde, norm_decisions)})


def _shard(src: Any) -> tuple[list, list]:
    """A worker's shard: an in-memory pair, or the path of its shard file (loaded here, in the
    worker, so the parent never hands it the whole sample list)."""
    if isinstance(src, str):
        with open(src, "rb") as fh:
            d = pickle.load(fh)
        return d["producer"], d["orderer"]
    return src


SHARING_STRATEGY = "file_system"


def set_sharing() -> None:
    """Shared tensors through `/dev/shm` files, not file descriptors: with the default
    `file_descriptor` strategy, 16 workers x every shared storage exhausted the pod's open-file limit
    (EXP-0105 relaunch: `Too many open files` in `resource_sharer`). Also lifts the soft open-file
    limit to the hard one. Called in the parent before any `share_memory()` and in every worker."""
    import resource

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 65536
    if soft != resource.RLIM_INFINITY and soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except (ValueError, OSError):
            pass


def release_memory() -> None:
    """gc, then return freed heap to the OS (glibc), before forking the fit workers."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _hogwild(args: tuple) -> list[dict]:
    set_sharing()
    pm, om, src, epochs, batch, lr, seed, clip = args
    prod, orde = _shard(src)
    return _epochs(pm, om, prod, orde, epochs, batch, lr, seed, lambda s: None, clip)


def _sync_worker(w: int, pm: g.ProducerGNN, om: g.OrdererGNN, src: Any,
                 epochs: int, steps: int, batch: int, seed: int, gbuf: torch.Tensor,
                 lbuf: torch.Tensor, ready: Any, done: Any) -> None:
    """One shard of the synchronous fit: per step its shard-mean gradient into `gbuf[w]`."""
    torch.set_num_threads(1)
    set_sharing()
    assert torch.multiprocessing.get_sharing_strategy() == SHARING_STRATEGY
    prod, orde = _shard(src)
    params = [*pm.parameters(), *om.parameters()]
    for ep in range(epochs):
        rng = np.random.default_rng([seed, ep, w])
        ip, io = rng.permutation(len(prod)), rng.permutation(len(orde))
        for b in range(steps):
            ready.wait()  # the parent's step is applied to the shared weights
            for p in params:
                p.grad = None
            lp, lo = bc_losses(pm, om, [prod[i] for i in ip[b * batch:(b + 1) * batch]],
                               [orde[i] for i in io[b * batch:(b + 1) * batch]])
            if (lp + lo).requires_grad:
                (lp + lo).backward()
            gbuf[w] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)
                                  ).reshape(-1) for p in params])
            lbuf[w, 0], lbuf[w, 1] = float(lp.detach()), float(lo.detach())
            done.wait()


def _sync_fit(pm: g.ProducerGNN, om: g.OrdererGNN, srcs: list, counts: tuple[int, int],
              epochs: int, batch: int, lr: float, seed: int, workers: int, clip: float | None,
              log: Any, per_agent: bool = False, step_log: str | Path | None = None
              ) -> list[dict]:
    """Synchronous data-parallel fit (v2): shard w = samples[w::workers]; per step the workers'
    gradients are averaged in worker order, clipped, and one Adam step is applied to the shared
    weights. Deterministic given the seed (BLAS 1 thread, fixed reduction order)."""
    torch.set_num_threads(1)
    params = [*pm.parameters(), *om.parameters()]
    sizes = [p.numel() for p in params]
    pm.share_memory()
    om.share_memory()
    gbuf = torch.zeros(workers, sum(sizes)).share_memory_()
    lbuf = torch.zeros(workers, 2).share_memory_()
    ctx = torch.multiprocessing.get_context("fork")
    ready, done = ctx.Barrier(workers + 1, timeout=1800), ctx.Barrier(workers + 1, timeout=1800)
    steps = max(1, max(counts) // (batch * workers))
    procs = [ctx.Process(target=_sync_worker, args=(
        w, pm, om, srcs[w], epochs, steps, batch, seed, gbuf, lbuf, ready, done), daemon=True)
        for w in range(workers)]
    for pr in procs:
        pr.start()
    st = _stepper(pm, om, lr, clip, per_agent, step_log)
    curve = []
    try:
        for ep in range(epochs):
            tot = [0.0, 0.0]
            t0 = time.perf_counter()
            for _b in range(steps):
                ready.wait()
                done.wait()
                grad = gbuf[0].clone()
                for w in range(1, workers):
                    grad += gbuf[w]
                grad /= workers
                for p, chunk in zip(params, torch.split(grad, sizes)):
                    p.grad = chunk.view_as(p).clone()
                st.step({"producer_loss": float(lbuf[:, 0].mean()),
                         "orderer_loss": float(lbuf[:, 1].mean())}, ep + 1)
                tot[0] += float(lbuf[:, 0].mean())
                tot[1] += float(lbuf[:, 1].mean())
            curve.append({"epoch": ep + 1, "producer_loss": tot[0] / steps,
                          "orderer_loss": tot[1] / steps,
                          "seconds": round(time.perf_counter() - t0, 3)})
            log(f"[m6_imitation] sync epoch {ep + 1}: producer {tot[0] / steps:.4f} orderer "
                f"{tot[1] / steps:.4f} ({curve[-1]['seconds']} s)")
    finally:
        st.close()
        for pr in procs:
            pr.join(timeout=60)
            if pr.is_alive():
                pr.terminate()
    bad = [pr.exitcode for pr in procs if pr.exitcode]
    if bad:
        raise RuntimeError(f"sync fit worker failed (exit codes {bad})")
    return curve


def _norm_feats(samples: list, cap: int) -> list[dict]:
    k = max(1, -(-len(samples) // cap))  # ceil(n / cap)
    return [f for f, _y in samples[::k]]


def fit(samples: dict | Shards, epochs: int, batch: int = 64, lr: float = 3e-3, seed: int = 0,
        log: Any = print, workers: int = 1, mode: str = "sync", clip: float | None = 1.0,
        norm_decisions: int = 50_000, per_agent: bool = False,
        step_log: str | Path | None = None
        ) -> tuple[g.ProducerGNN, g.OrdererGNN, list[dict], dict]:
    """Behaviour cloning (v2). The normalisers are fitted first on every k-th training decision
    (DIAGNOSTICS 31). `workers` > 1: `mode` "sync" (deterministic data parallel, `_sync_fit`) or
    "hogwild" (shared weights, one shard and Adam per process; not bit-reproducible). Returns
    the models, the loss curve and {"normaliser": its record, "train_seconds": the training
    wall time without the normaliser fit}."""
    torch.manual_seed(seed)
    if workers > 1:
        set_sharing()
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    if isinstance(samples, Shards):
        if len(samples.paths) != workers:
            raise ValueError("the shard count differs from --workers")
        norm, counts, srcs = samples.norm, samples.counts, list(samples.paths)
    else:
        prod, orde = samples["producer"], samples["orderer"]
        norm = {"producer": _norm_feats(prod, norm_decisions),
                "orderer": _norm_feats(orde, norm_decisions)}
        counts = (len(prod), len(orde))
        srcs = [(prod[w::workers], orde[w::workers]) for w in range(max(1, workers))]
    info: dict[str, Any] = {"normaliser": {"producer": pm.norm.fit(norm["producer"]),
                                           "orderer": om.norm.fit(norm["orderer"])}}
    t0 = time.perf_counter()
    if workers <= 1:
        curve = _epochs(pm, om, *_shard(srcs[0]), epochs, batch, lr, seed, log, clip, per_agent,
                        step_log)
    elif mode == "sync":
        curve = _sync_fit(pm, om, srcs, counts, epochs, batch, lr, seed, workers, clip, log,
                          per_agent, step_log)
    else:
        curve = _hogwild_fit(pm, om, srcs, epochs, batch, lr, seed, workers, clip, log)
    info["train_seconds"] = round(time.perf_counter() - t0, 3)
    return pm, om, curve, info


def _hogwild_fit(pm: g.ProducerGNN, om: g.OrdererGNN, srcs: list, epochs: int, batch: int,
                 lr: float, seed: int, workers: int, clip: float | None, log: Any
                 ) -> list[dict]:
    pm.share_memory()
    om.share_memory()
    ctx = torch.multiprocessing.get_context("fork")
    shards = [(pm, om, srcs[i], epochs, batch, lr, seed + i, clip) for i in range(workers)]
    with ctx.Pool(workers) as pool:
        curves = pool.map(_hogwild, shards)
    curve = [{"epoch": e + 1,
              "producer_loss": fmean(c[e]["producer_loss"] for c in curves),
              "orderer_loss": fmean(c[e]["orderer_loss"] for c in curves)} for e in range(epochs)]
    log(f"[m6_imitation] hogwild x{workers}: final producer {curve[-1]['producer_loss']:.4f} "
        f"orderer {curve[-1]['orderer_loss']:.4f}")
    return curve


def choose_mode(samples: dict | Shards, workers: int, seed: int, log: Any = print) -> dict:
    """MODE_RULE: one epoch of each mode (hogwild, then sync), timed without the normaliser fit."""
    t = {}
    for m in ("hogwild", "sync"):
        t[m] = fit(samples, 1, batch=FIT["batch"], lr=FIT["lr"], seed=seed, log=lambda s: None,
                   workers=workers, mode=m, clip=FIT["clip_grad_norm"],
                   norm_decisions=FIT["norm_decisions"])[3]["train_seconds"]
    ratio = t["sync"] / max(t["hogwild"], 1e-9)
    mode = "sync" if ratio <= FIT["sync_max_cost"] else "hogwild"
    log(f"[m6_imitation] fit mode probe: hogwild {t['hogwild']:.1f} s, sync {t['sync']:.1f} s "
        f"(x{ratio:.2f}) -> {mode}")
    return {"rule": MODE_RULE, "epoch_seconds": t, "sync_over_hogwild": ratio, "mode": mode}


def _head_fidelity(conf: np.ndarray, rows: dict[str, list], split: dict[str, int], n: int) -> dict:
    """Per toggle head (restore / upgrade / expand) and replenish row class; which part breaks
    an exact command (v2 report)."""
    heads = {}
    for i, name in enumerate(TOGGLES):
        tp, fp, fn, tn = (int(x) for x in conf[i])
        heads[name] = {"label_rate": (tp + fn) / max(1, n), "pred_rate": (tp + fp) / max(1, n),
                       "accuracy": (tp + tn) / max(1, n), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    rep = {}
    for cls, du in rows.items():
        a = np.asarray(du, dtype=np.float64)
        rep[cls] = {"rows": int(a.size), "exact_share": float((a == 0).mean()) if a.size else None,
                    "within_1_unit_share": float((a <= 1).mean()) if a.size else None,
                    "mae_units": float(a.mean()) if a.size else None}
    return {"toggles": heads, "replenish_rows": rep,
            "commands": {k: v / max(1, n) for k, v in split.items()}}


def agreement(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list) -> dict:
    """Fidelity metrics on labelled decisions (reported, D4). Exact command = every toggle and
    every replenish row's rounded units equal the label's."""
    exact = gates_ok = n_gate = 0
    mae: list[float] = []
    conf = np.zeros((3, 4), dtype=np.int64)  # tp, fp, fn, tn per head
    rows: dict[str, list] = {"label_zero": [], "label_interior": [], "label_at_bound": []}
    split = {"exact": 0, "toggles_only_wrong": 0, "replenish_only_wrong": 0, "both_wrong": 0,
             "replenish_only_wrong_within_1_unit": 0}
    with torch.no_grad():
        for f, y in prod:
            gates, rep = pm(f)
            tg = (gates.numpy() > 0).astype(np.float32)
            yt = y["toggles"]
            gates_ok += int((tg == yt).sum())
            n_gate += 3
            conf[:, 0] += (tg == 1) & (yt == 1)
            conf[:, 1] += (tg == 1) & (yt == 0)
            conf[:, 2] += (tg == 0) & (yt == 1)
            conf[:, 3] += (tg == 0) & (yt == 0)
            mr = f["global"][0][9]  # log1p(max_replenish)
            units = np.expm1(mr)
            t_ok = bool((tg == yt).all())
            r_ok, r_near = True, True
            if rep.shape[0]:
                m = y["mask"] > 0
                du = np.abs(np.round(rep.numpy() * units) - np.round(y["replenish"] * units))[m]
                lab = y["replenish"][m]
                mae += du.tolist()
                for d, lv in zip(du.tolist(), lab.tolist()):
                    rows["label_zero" if lv == 0 else "label_at_bound" if lv >= 1
                         else "label_interior"].append(d)
                r_ok, r_near = bool((du == 0).all()), bool((du <= 1).all())
            ok = t_ok and r_ok
            exact += ok
            key = ("exact" if ok else "toggles_only_wrong" if r_ok else
                   "replenish_only_wrong" if t_ok else "both_wrong")
            split[key] += 1
            split["replenish_only_wrong_within_1_unit"] += int(t_ok and not r_ok and r_near)
        kl, top1, n_s = [], 0, 0
        for f, ys in orde:
            for lg, y in zip(om(f), ys):
                if len(y):
                    p = torch.softmax(lg, 0).numpy()
                    kl.append(float((y * (np.log(np.clip(y, 1e-9, 1)) - np.log(np.clip(p, 1e-9, 1))
                                          )).sum()))
                    top1 += int(np.argmax(p) == np.argmax(y))
                    n_s += 1
    return {"producer_decisions": len(prod), "exact_command": exact / max(1, len(prod)),
            "gate_accuracy": gates_ok / max(1, n_gate),
            "replenish_mae_units": fmean(mae) if mae else 0.0, "order_slots": n_s,
            "orderer_kl": fmean(kl) if kl else 0.0, "orderer_top1": top1 / max(1, n_s),
            "producer_heads": _head_fidelity(conf, rows, split, len(prod))}


def save_models(pm: g.ProducerGNN, om: g.OrdererGNN, path: Path) -> None:
    torch.save({"producer": pm.state_dict(), "orderer": om.state_dict(),
                "hidden": g.HIDDEN, "rounds": g.ROUNDS, "norm_format": g.NORM_FORMAT,
                "producer_features": pm.features, "capacity_extra": pm.capacity_extra,
                **({"turn_heads": True} if pm.turn is not None else {})}, path)


def load_models(path: str | Path) -> tuple[g.ProducerGNN, g.OrdererGNN]:
    d = torch.load(path, weights_only=True)
    # a checkpoint without `producer_features` predates v3 (EXP-0099 / EXP-0100): v2 features
    pm = g.ProducerGNN(d["hidden"], d["rounds"], d.get("producer_features", g.FEATURES_V2),
                       int(d.get("capacity_extra", 0)))
    if d.get("turn_heads"):  # ADR-0062 heads; a checkpoint without them loads as before
        pm.add_turn_heads()
    om = g.OrdererGNN(d["hidden"], d["rounds"])
    # a pre-normaliser checkpoint (EXP-0099) loads with the identity normaliser
    strict = d.get("norm_format") == g.NORM_FORMAT
    if not strict and "norm_format" in d:
        raise ValueError(f"unknown normaliser format {d['norm_format']}")
    pm.load_state_dict(d["producer"], strict=strict)
    om.load_state_dict(d["orderer"], strict=strict)
    if not strict:
        pm.norm.clip = om.norm.clip = None  # raw features, exactly as EXP-0099 played them
    return pm, om


# --- modes ---


def _training(training_dir: str) -> tuple[Any, list, list]:
    ts = pe.load_training(training_dir)
    mon = pe.monitor_set(ts)
    train = [we for we in ts.world_episodes if we not in set(mon)]
    return ts, train, mon


def collect_42_jobs(atlas: Mapping, max_worlds: int | None = None,
                    horizon: int | None = None) -> list[dict]:
    """COLLECT_42: k5_rf5 + MWWk on the train-split 42-producer worlds (never held-out, never
    validation), fresh seeds (stage m6-collect-42), subsampled recording."""
    from ..eval.teacher_audit import world_view
    from .producer_read import entry_for

    views = {world_view(e)["world"]: world_view(e) for e in atlas["worlds"]}
    found = sorted(w for w, v in views.items() if v["size"] == 42 and v["split"] == "train"
                   and v["class"] != "collapse")
    if tuple(found) != COLLECT_42_WORLDS:
        raise ValueError("the atlas's train-split 42 worlds differ from the registered list")
    jobs = []
    for w in COLLECT_42_WORLDS[:max_worlds] if max_worlds else COLLECT_42_WORLDS:
        e = entry_for(views[w], COLLECT_42["episodes"], COLLECT_42["stage"])
        for ep in e.episode_seeds:
            jobs.append({"world": e.id, "size": e.size, "regime": e.regime,
                         "world_seed": e.world_seed, "generator": e.generator, "episode": int(ep),
                         "replicate": COLLECT_42["replicate"],
                         "horizon": horizon or COLLECT_42["horizon"],
                         "subsample": COLLECT_42["subsample"], "kind": "imit_train_42",
                         "job_id": f"imit42|{e.id}|{ep}"})
    return jobs


def run_collect(training_dir: str, out_dir: str, workers: int, horizon: int,
                max_world_episodes: int | None, dry_run: bool = False,
                atlas_path: str | None = None, horizon_42: int | None = None) -> int:
    """Stage I v3: the 12-producer training and monitor world-episodes (as v2) plus, with an atlas,
    the 42-producer training worlds (COLLECT_42) into the training samples."""
    from ..eval.teacher_audit import load_atlas

    ts, train, mon = _training(training_dir)
    atlas = None if atlas_path is None else load_atlas(atlas_path)
    if not dry_run:
        _refuse(real_problems(horizon=horizon, max_world_episodes=max_world_episodes, ts=ts,
                              atlas=atlas)
                + ([] if atlas is not None else ["a v3 collect needs --atlas (the 42 worlds)"])
                + ([] if horizon_42 is None else ["--horizon-42 is set"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {"train": train[:max_world_episodes] if max_world_episodes else train,
            "monitor": mon[:max(1, max_world_episodes // 4)] if max_world_episodes else mon}
    counts = {}
    for name, wes in sets.items():
        jobs = _jobs(ts, wes, f"imit_{name}", (0,), horizon)
        if name == "train" and atlas is not None:
            jobs = collect_42_jobs(atlas, 1 if max_world_episodes else None, horizon_42) + jobs
        res = _pool(record_job, jobs, workers)
        counts[name] = {"world_episodes": len(jobs),
                        "producer_42": sum(len(r["producer"]) for r, j in zip(res, jobs)
                                           if j.get("size") == 42)}
        samples = {"producer": [s for r in res for s in r["producer"]],
                   "orderer": [s for r in res for s in r["orderer"]]}
        with (out / f"samples_{name}.pkl").open("wb") as fh:
            pickle.dump(samples, fh)
        print(f"[m6_imitation] {name}: {len(wes)} world-episodes, {len(samples['producer'])} "
              f"producer and {len(samples['orderer'])} orderer decisions", flush=True)
    (out / "collect.json").write_text(json.dumps(
        {"training_pins": ts.pins, "horizon": horizon, "train": len(sets["train"]),
         "dry_run": dry_run, "prereg": PREREG_V6_SHA256, "source_commit": SOURCE_COMMIT,
         "monitor": len(sets["monitor"]), "base": [BASE_PRODUCER, BASE_ORDERER],
         "features": g.LATEST_FEATURES, "collect_42": None if atlas is None else {
             **COLLECT_42, "worlds": list(COLLECT_42_WORLDS)}, "counts": counts},
        indent=1) + "\n")
    return 0


def sample_problems(collect_dir: Path, meta: dict) -> list[str]:
    """A real v2 fit reads a real v2 collect, or v1's (EXP-0099) samples at their pinned SHAs."""
    from ..eval.provenance import sha256_file

    if meta.get("dry_run"):
        return ["the collect run was a dry run"]
    if meta.get("prereg") == PREREG_V5_SHA256 and meta.get("features") == g.FEATURES_V3:
        return [f"{n} does not match EXP-0105's pinned SHA-256" for n, h in EXP0105_COLLECT.items()
                if sha256_file(str(collect_dir / n)) != h]
    if PREREG_V5_SHA256 is not None:
        return ["a Stage I v4 fit reads EXP-0105's (v5) collect only"]
    if meta.get("prereg") == PREREG_SHA256:
        return []
    if meta.get("prereg") != PREREG_V1_SHA256:
        return ["the samples come from neither the v2 nor the v1 pre-registration"]
    return [f"{n} does not match EXP-0099's pinned SHA-256" for n, h in EXP0099_SAMPLES.items()
            if sha256_file(str(collect_dir / n)) != h]


def run_fit(collect_dir: str, out_dir: str, epochs: int, seed: int, workers: int = 1,
            dry_run: bool = False, mode: str = FIT["mode"], per_agent: bool = False) -> int:
    """A fit always writes `steps.jsonl` (per step, per agent: gradient norm before clipping,
    weight norm, losses; it changes no computation). `per_agent` (one Adam + clip per agent) is
    off unless a pre-registration sets it (the v1..v6 fits shared one)."""
    c = Path(collect_dir)
    meta = json.loads((c / "collect.json").read_text())
    if not dry_run:
        _refuse(real_problems(horizon=meta["horizon"], max_world_episodes=None, epochs=epochs,
                              workers=workers, seed=seed)
                + ([] if mode == FIT["mode"] else [f"--fit-mode {mode} != {FIT['mode']}"])
                + sample_problems(c, meta))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (c / "samples_train.pkl").open("rb") as fh:
        tr = pickle.load(fh)
    v4 = None
    if meta.get("features") == g.FEATURES_V3:  # Stage I v4 on EXP-0105's recorded v3 samples
        n42 = int(meta["counts"]["train"]["producer_42"])
        tr42, tr12, gate42 = split_v4(tr["producer"], n42, dry_run)
        tr42, w42 = boundary_weights(tr42)
        tr12, w12 = boundary_weights(tr12)
        tr = {"producer": tr42 + tr12, "orderer": tr["orderer"]}
        v4 = {"weights": {"42": w42, "12": w12}, "producer_42_train": len(tr42),
              "producer_12_train": len(tr12), "producer_42_gate_held_out": len(gate42),
              "rule": BOUNDARY_WEIGHT_RULE}
        del tr42, tr12, gate42
    data: dict | Shards = tr
    shard_dir = out / "_shards"
    if workers > 1:  # each worker loads only its own shard file (EXP-0105's out-of-memory)
        data = write_shards(tr, workers, shard_dir, FIT["norm_decisions"])
        del tr
        release_memory()
    try:
        probe = choose_mode(data, workers, seed) if mode == "auto" else None
        ran = probe["mode"] if probe else mode
        t0 = time.perf_counter()
        pm, om, curve, info = fit(data, epochs, batch=FIT["batch"], lr=FIT["lr"], seed=seed,
                                  workers=workers, mode=ran, clip=FIT["clip_grad_norm"],
                                  norm_decisions=FIT["norm_decisions"], per_agent=per_agent,
                                  step_log=out / "steps.jsonl")
        fit_s = time.perf_counter() - t0
    finally:
        if shard_dir.exists():  # never pulled back: GBs of copies of the samples
            import shutil

            shutil.rmtree(shard_dir)
    save_models(pm, om, out / "models.pt")
    if workers > 1:
        with (c / "samples_train.pkl").open("rb") as fh:
            tr = pickle.load(fh)
        if v4 is not None:
            tr = {"producer": [(g.v3_to_v4(f), y) for f, y in tr["producer"]],
                  "orderer": tr["orderer"]}
    with (c / "samples_monitor.pkl").open("rb") as fh:
        mo = pickle.load(fh)
    if v4 is not None:
        mo = {"producer": [(g.v3_to_v4(f), y) for f, y in mo["producer"]], "orderer": mo["orderer"]}
    from ..eval.provenance import sha256_file

    spec = {"fit_mode": ran, "mode_probe": probe, "requested_mode": mode, "fit": FIT, "v4": v4,
            "per_agent": per_agent,
            "epochs": epochs, "seed": seed, "workers": workers, "dry_run": dry_run,
            "prereg": PREREG_V6_SHA256, "source_commit": SOURCE_COMMIT,
            "samples_from": {"prereg": meta.get("prereg"), "source_commit":
                             meta.get("source_commit")}}
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    doc = {"epochs": epochs, "seed": seed, "workers": workers, "mode": ran, "mode_probe": probe,
           "curve": curve, "fit_seconds": round(fit_s, 1),
           "train_seconds": info["train_seconds"], "dry_run": dry_run,
           "normaliser": info["normaliser"], "v4": v4,
           "samples_from": {"prereg": meta.get("prereg"), "source_commit":
                            meta.get("source_commit")},
           "prereg": PREREG_V6_SHA256, "models_sha256": sha256_file(str(out / "models.pt")),
           "source_commit": SOURCE_COMMIT,
           "train_agreement": agreement(pm, om, tr["producer"], tr["orderer"]),
           "monitor_agreement": agreement(pm, om, mo["producer"], mo["orderer"])}
    (out / "fit.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("train_agreement", "monitor_agreement")}), flush=True)
    return 0


def gate_v4_reading(stats: Mapping) -> dict:
    """V4_RULE step 1 over m6_gate_probe.model_stats results {set: {model: stats}} (pure)."""
    def below(st: Mapping) -> float:
        r = st["upgrade_margin_response"]["margin_lt_0"]
        return r["fire_rate"] if r["fire_rate"] is not None else 0.0

    failed = []
    for k in ("12_monitor", "42_gate"):
        if below(stats[k]["v4"]) > V4_GATE["below_zero_fire_max"]:
            failed.append(f"below_zero_fire_{k}")
        for h in V4_GATE["heads_no_worse"]:
            if stats[k]["v4"]["heads"][h]["accuracy"] < stats[k]["v2"]["heads"][h]["accuracy"]:
                failed.append(f"{h}_accuracy_{k}")
    a4 = stats["12_monitor"]["v4"]["upgrade_margin_response"]["margin_ge_0"]["accuracy"]
    a2 = stats["12_monitor"]["v2"]["upgrade_margin_response"]["margin_ge_0"]["accuracy"]
    if a4 < a2 - V4_GATE["above_zero_agreement_slack"]:
        failed.append("above_zero_upgrade_accuracy_12")
    return {"reading": "gate_pass" if not failed else "gate_fail", "failed": failed,
            "below_zero_fire": {k: {m: below(stats[k][m]) for m in stats[k]} for k in stats},
            "above_zero_upgrade_accuracy_12": {"v4": a4, "v2": a2}}


def run_gate(collect_dir: str, models: str, prev_models: str, out_dir: str,
             dry_run: bool = False) -> int:
    """V4_RULE step 1: the offline gate. Exit 0 on a pass, 3 on gate_fail (the chained check
    then does not run)."""
    from ..eval.provenance import sha256_file
    from .m6_gate_probe import model_stats

    c = Path(collect_dir)
    meta = json.loads((c / "collect.json").read_text())
    if not dry_run:
        _refuse(prereg_problems() + sample_problems(c, meta)
                + ([] if sha256_file(prev_models) == EXP0100_MODELS_SHA else
                   ["--prev-models is not EXP-0100's models"]))
    with (c / "samples_train.pkl").open("rb") as fh:
        tr = pickle.load(fh)
    _a, _b, gate42 = split_v4(tr["producer"], int(meta["counts"]["train"]["producer_42"]), dry_run)
    del tr, _a, _b
    release_memory()
    with (c / "samples_monitor.pkl").open("rb") as fh:
        mon = [(g.v3_to_v4(f), y) for f, y in pickle.load(fh)["producer"]]
    p4, _ = load_models(models)
    p2, _ = load_models(prev_models)
    if p4.features != g.FEATURES_V4 or p2.features != g.FEATURES_V2:
        raise ValueError("--models / --prev-models are not v4 / v2 producer checkpoints")

    def strip(f: dict) -> dict:
        return {**f, "capacity": f["capacity"][:, :g.CAPACITY_DIM[g.FEATURES_V2]]}

    stats = {k: {"v4": model_stats(p4, v, lambda f: f), "v2": model_stats(p2, v, strip)}
             for k, v in (("12_monitor", mon), ("42_gate", gate42))}
    doc = {**gate_v4_reading(stats), "rule": V4_RULE, "gate": V4_GATE, "stats": stats,
           "decisions": {"12_monitor": len(mon), "42_gate": len(gate42)}, "dry_run": dry_run,
           "prereg": PREREG_V6_SHA256, "models_sha256": sha256_file(models),
           "source_commit": SOURCE_COMMIT}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[m6_imitation] Stage I v4 gate: {doc['reading']} {doc['failed']} below-zero fire "
          f"{doc['below_zero_fire']}", flush=True)
    return 0 if doc["reading"] == "gate_pass" else 3


def check_42_jobs(atlas: Mapping, models: str, max_worlds: int | None = None,
                  episodes: int | None = None, horizon: int | None = None,
                  arms: Sequence[tuple[str, str | None, str]] | None = None) -> list[dict]:
    """Clause (b): the v3 GNN stack and k5_rf5 + MWWk on CHECK_42_WORLDS (EXP-0104's keys)."""
    from ..eval.teacher_audit import world_view
    from .order_es_worlds import X_RUNDOWN
    from .producer_read import entry_for

    views = {world_view(e)["world"]: world_view(e) for e in atlas["worlds"]}
    jobs = []
    for w in CHECK_42_WORLDS[:max_worlds] if max_worlds else CHECK_42_WORLDS:
        e = entry_for(views[w], episodes or CHECK_42["episodes"], CHECK_42["stage"])
        for ep in e.episode_seeds:
            for r in CHECK_42["replicates"]:
                for arm, mdl, label in arms or (("gnn", models, "gnn"), ("k5_rf5", None, "k5_rf5")):
                    jobs.append({"world": e.id, "size": e.size, "regime": e.regime,
                                 "world_seed": e.world_seed, "generator": e.generator,
                                 "episode": int(ep), "replicate": r,
                                 "horizon": horizon or CHECK_42["horizon"], "arm": arm,
                                 "label": label, "set": "check42", "models": mdl,
                                 "x": float(X_RUNDOWN)})
    return sorted(jobs, key=lambda j: j["arm"] == "k5_rf5")


def check_v3_reading(mon_rows: Sequence[dict], prev_rows: Sequence[dict], rows42: Sequence[dict],
                     mon_cells: Mapping, cells42: Mapping) -> dict:
    """V3_RULE over the played rows (pure)."""
    from .order_es_validate import mean_comp, world_mean
    from .producer_read import _units, pilot_reading

    base = [x for x in mon_rows if x["kind"] == "monitor_base"]
    cand = [x for x in mon_rows if x["kind"] == "monitor"]
    a = pilot_reading(cand + base, mon_cells, 0)
    c = pilot_reading(cand + [{**x, "kind": "monitor_base"} for x in prev_rows], mon_cells, 0)
    deaths = {"v3": sum(x["dead"] == "true" for x in cand),
              "base": sum(x["dead"] == "true" for x in base),
              "exp0100": sum(x["dead"] == "true" for x in prev_rows)}
    units = _units([{**r, "units": str(r["units"])} for r in rows42], lambda d: d["world"])
    worlds = sorted(w for w in units if "gnn" in units[w] and "k5_rf5" in units[w])
    bars = {w: world_mean(units[w]["k5_rf5"]) for w in worlds}
    comp = mean_comp(units, cells42, "gnn", "k5_rf5", {"gnn": 2, "k5_rf5": 2}, bars, worlds) \
        if worlds else None
    over = {arm: sum(int(r["over_ceiling"]) for r in rows42 if r["arm"] == arm)
            for arm in ("gnn", "k5_rf5")}
    ok_a = (a.get("g_tr") is not None and a["g_tr"] >= -a["F_tr"]
            and deaths["v3"] <= deaths["base"])
    ok_b = (comp is not None and over["gnn"] <= over["k5_rf5"] and comp.diff >= -comp.floor)
    ok_c = c.get("g_tr") is not None and c["g_tr"] >= -c["F_tr"]
    failed = [n for n, ok in (("a_12_faithful", ok_a), ("b_42_ceiling_units", ok_b),
                              ("c_no_regression", ok_c)) if not ok]
    per_world = {w: {arm: {"over_ceiling": sum(int(r["over_ceiling"]) for r in rows42
                                               if r["world"] == w and r["arm"] == arm),
                           "units": world_mean(units[w][arm]),
                           "deaths": sum(bool(r["dead"]) for r in rows42
                                         if r["world"] == w and r["arm"] == arm)}
                     for arm in ("gnn", "k5_rf5")} for w in worlds}
    return {"rule": V3_RULE, "reading": "faithful" if not failed else "short", "failed": failed,
            "a_12": {"g": a.get("g_tr"), "F": a.get("F_tr"), "deaths": deaths},
            "b_42": {"over_ceiling": over, "diff": None if comp is None else comp.diff,
                     "floor": None if comp is None else comp.floor, "worlds": worlds,
                     "deaths": {arm: sum(bool(r["dead"]) for r in rows42 if r["arm"] == arm)
                                for arm in ("gnn", "k5_rf5")}, "per_world": per_world},
            "c_vs_exp0100": {"g": c.get("g_tr"), "F": c.get("F_tr")}}


def compose_candidate(v4: str, prev: str, out: Path) -> str:
    """C = the v4 producer + the EXP-0100 orderer, as one deployable checkpoint."""
    pm, _om = load_models(v4)
    _pm, om = load_models(prev)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_models(pm, om, out)
    return str(out)


def run_attribution(training_dir: str, atlas: str, v4: str, prev: str, out_dir: str, workers: int,
                    horizon: int, max_world_episodes: int | None, dry_run: bool = False,
                    episodes_42: int | None = None, horizon_42: int | None = None) -> int:
    """EXP-0108 (V7_RULE): the deciding candidate C and the EXP-0100 stack against k5_rf5 + MWWk,
    written first; then the reported attribution arms."""
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas, world_view
    from .m6_drain import drain_job
    from .order_es_validate import mean_comp, monitor_cells, world_cells, world_mean
    from .producer_read import _units, pilot_reading

    ts, _train, mon = _training(training_dir)
    at = load_atlas(atlas)
    if not dry_run:
        _refuse(real_problems(horizon=horizon, max_world_episodes=max_world_episodes, ts=ts,
                              atlas=at)
                + ([] if sha256_file(v4) == EXP0107_MODELS_SHA else ["--models is not EXP-0107's"])
                + ([] if sha256_file(prev) == EXP0100_MODELS_SHA else
                   ["--prev-models is not EXP-0100's"])
                + ([] if episodes_42 is None and horizon_42 is None else
                   ["--episodes-42 / --horizon-42 is set"]))
    if max_world_episodes:
        mon = mon[:max_world_episodes]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cand = compose_candidate(v4, prev, out / "candidate_models.pt")
    src = {"cand": cand, "v4": v4, "prev": prev, None: None}
    mcells = monitor_cells(ts.training, at["spreads"])
    nw = 1 if max_world_episodes else None

    def play(labels: Sequence[str]) -> tuple[dict, dict]:
        mj = {lb: _jobs(ts, mon, ATTRIBUTION_ARMS[lb][0], MONITOR_REPLICATES, horizon,
                        src[ATTRIBUTION_ARMS[lb][2]]) for lb in labels}
        flat = [j for lb in labels for j in mj[lb]]
        rows = _pool(play_job, flat, workers)
        mr, i = {}, 0
        for lb in labels:
            mr[lb] = rows[i:i + len(mj[lb])]
            i += len(mj[lb])
        arms42 = [(ATTRIBUTION_ARMS[lb][1], src[ATTRIBUTION_ARMS[lb][2]], lb) for lb in labels]
        r42 = _pool(drain_job, check_42_jobs(at, cand, nw, episodes_42, horizon_42, arms42),
                    workers)
        by = {lb: [] for lb in labels}
        for r, j in zip(r42, check_42_jobs(at, cand, nw, episodes_42, horizon_42, arms42)):
            by[j["label"]].append({**r, "arm": "gnn" if j["label"] != "base" else "k5_rf5"})
        return mr, by

    views = [dict(world_view(e), rho=float(e["record"]["coordinates"]["rho"]))
             for e in at["worlds"] if world_view(e)["world"] in CHECK_42_WORLDS]
    cells42 = world_cells(views, at["spreads"])
    # phase 1: deciding (C, base) on both sets, the EXP-0100 stack on the monitor (clause c)
    mr, by = play(("C", "base"))
    prev_rows = _pool(play_job, _jobs(ts, mon, "monitor", MONITOR_REPLICATES, horizon, prev),
                      workers)
    dec = check_v3_reading(mr["C"] + mr["base"], prev_rows, by["C"] + by["base"], mcells, cells42)
    doc = {**dec, "rule": V7_RULE, "dry_run": dry_run, "prereg": PREREG_V7_SHA256,
           "candidate_models_sha256": sha256_file(cand), "v4_sha256": sha256_file(v4),
           "prev_sha256": sha256_file(prev), "source_commit": SOURCE_COMMIT,
           "units_mean": {"C": fmean(float(x["units"]) for x in mr["C"]),
                          "base": fmean(float(x["units"]) for x in mr["base"]),
                          "exp0100": fmean(float(x["units"]) for x in prev_rows)}}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"[m6_imitation] EXP-0108 candidate: {doc['reading']} {doc['failed']} a {doc['a_12']} b "
          f"{dict((k, doc['b_42'][k]) for k in ('over_ceiling', 'diff', 'floor'))} c "
          f"{doc['c_vs_exp0100']}", flush=True)
    # phase 2: the reported attribution arms, each vs the base rows of phase 1
    mr2, by2 = play(REPORTED_ARMS)
    att = {}
    for lb in REPORTED_ARMS:
        a = pilot_reading([{**x, "kind": "monitor"} for x in mr2[lb]] + mr["base"], mcells, 0)
        rows42 = [{**r, "arm": "gnn"} for r in by2[lb]] + by["base"]
        u = _units([{**r, "units": str(r["units"])} for r in rows42], lambda d: d["world"])
        ws = sorted(w for w in u if "gnn" in u[w] and "k5_rf5" in u[w])
        comp = mean_comp(u, cells42, "gnn", "k5_rf5", {"gnn": 2, "k5_rf5": 2},
                         {w: world_mean(u[w]["k5_rf5"]) for w in ws}, ws) if ws else None
        att[lb] = {"12": {"g": a.get("g_tr"), "F": a.get("F_tr"),
                          "units": fmean(float(x["units"]) for x in mr2[lb]),
                          "deaths": sum(x["dead"] == "true" for x in mr2[lb])},
                   "42": {"diff": None if comp is None else comp.diff,
                          "floor": None if comp is None else comp.floor,
                          "over_ceiling": sum(int(r["over_ceiling"]) for r in by2[lb]),
                          "deaths": sum(bool(r["dead"]) for r in by2[lb])}}
    (out / "attribution.json").write_text(json.dumps(
        {"reported_only": True, "arms": att, "base_units_12": doc["units_mean"]["base"]},
        indent=1, default=str) + "\n")
    with (out / "check42.jsonl").open("w") as fh:
        for lb, rs in {**by, **by2}.items():
            for r in rs:
                fh.write(json.dumps({"label": lb, **{k: v for k, v in r.items()
                                                     if k not in ("cap_trace", "units_trace")}})
                         + "\n")
    print(f"[m6_imitation] EXP-0108 attribution (reported): {json.dumps(att, default=str)}",
          flush=True)
    return 0


def run_check(training_dir: str, atlas: str, models: str, out_dir: str, workers: int,
              horizon: int, max_world_episodes: int | None, dry_run: bool = False,
              prev_models: str | None = None, episodes_42: int | None = None,
              horizon_42: int | None = None) -> int:
    """Stage I v3's pre-registered check (V3_RULE): (a) 12 monitor vs k5_rf5 + MWWk, (b) the 42
    ceiling / units clause, (c) no regression vs the EXP-0100 stack."""
    import csv

    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from .m6_drain import drain_job
    from .order_es_validate import monitor_cells, world_cells

    ts, _train, mon = _training(training_dir)
    at = load_atlas(atlas)
    if not dry_run:
        _refuse(real_problems(horizon=horizon, max_world_episodes=max_world_episodes, ts=ts,
                              atlas=at)
                + ([] if prev_models and sha256_file(prev_models) == EXP0100_MODELS_SHA else
                   ["--prev-models is not EXP-0100's models"])
                + ([] if episodes_42 is None and horizon_42 is None else
                   ["--episodes-42 / --horizon-42 is set"]))
    if max_world_episodes:
        mon = mon[:max_world_episodes]
    jobs42 = check_42_jobs(at, models, 1 if max_world_episodes else None, episodes_42, horizon_42)
    mon_jobs = [j for arm in ("monitor", "monitor_base")
                for j in _jobs(ts, mon, arm, MONITOR_REPLICATES, horizon, models)]
    prev_jobs = _jobs(ts, mon, "monitor", MONITOR_REPLICATES, horizon, prev_models) \
        if prev_models else []
    rows42 = _pool(drain_job, jobs42, workers)  # the 42 jobs are the longest: first
    rows = _pool(play_job, mon_jobs + prev_jobs, workers)
    mon_rows, prev_rows = rows[:len(mon_jobs)], rows[len(mon_jobs):]
    from ..eval.teacher_audit import world_view

    views = []
    for e in at["worlds"]:
        v = world_view(e)
        if v["world"] in {r["world"] for r in rows42}:
            v["rho"] = float(e["record"]["coordinates"]["rho"])
            views.append(v)
    doc = {**check_v3_reading(mon_rows, prev_rows, rows42,
                              monitor_cells(ts.training, at["spreads"]),
                              world_cells(views, at["spreads"])),
           "dry_run": dry_run, "prereg": PREREG_V6_SHA256, "models_sha256": sha256_file(models),
           "prev_models_sha256": None if not prev_models else sha256_file(prev_models),
           "source_commit": SOURCE_COMMIT, "world_episodes": len(mon), "horizon": horizon,
           "units_mean": {"v3": fmean(float(x["units"]) for x in mon_rows if x["kind"] == "monitor"),
                          "base": fmean(float(x["units"]) for x in mon_rows
                                        if x["kind"] == "monitor_base"),
                          "exp0100": fmean(float(x["units"]) for x in prev_rows) if prev_rows
                          else None}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "imitation_check.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    with (out / "episodes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("set", "kind", "world", "episode", "replicate", "units", "dead"))
        for x in mon_rows:
            w.writerow(("monitor", x["kind"], x["world"], x["episode"], x["replicate"], x["units"],
                        x["dead"]))
        for x in prev_rows:
            w.writerow(("monitor", "exp0100", x["world"], x["episode"], x["replicate"], x["units"],
                        x["dead"]))
    with (out / "check42.jsonl").open("w") as fh:
        for r in rows42:
            fh.write(json.dumps({k: v for k, v in r.items() if k not in ("cap_trace", "units_trace")})
                     + "\n")
    print(f"[m6_imitation] Stage I v3 check: {doc['reading']} {doc['failed']} a {doc['a_12']} "
          f"b {dict((k, doc['b_42'][k]) for k in ('over_ceiling', 'diff', 'floor'))} "
          f"c {doc['c_vs_exp0100']}", flush=True)
    return 0


# --- side effects (task 2.7, EXP-0101; pre-registered in tasks.md before its data) ---

EXP0100_MODELS_SHA256 = "5acf8b82df4686dd14a1da3a441a390f87fe32bf3c177e16ddd63fe417f43126"
SIDE_ARMS: dict[int, tuple[str, ...]] = {100: ("monitor", "swap_gnn_producer", "monitor_base"),
                                         200: ("monitor", "monitor_base")}



def side_arms(h: int, horizons: Sequence[int]) -> tuple[str, ...]:
    """The first horizon carries the deciding set (3 arms), later ones the reported pair (a dry
    run's small horizons map the same way)."""
    return SIDE_ARMS[100] if h == horizons[0] else SIDE_ARMS[200]
SIDE_RULE = ("clean if the GNN + GNN stack at horizon 100 is no worse than k5_rf5 + MWWk on any of L5's "
             "side-effect clauses (a) deaths, (b) run-down share r_end < X r_start, (c) starvation "
             "share any consumer fill < Y; else side_effect naming the clause(s). The swap arm and "
             "the horizon-200 pair: the same clauses, reported only")


def side_effect_job(job: dict) -> dict[str, str]:
    """One keyed episode of arm `job["kind"]` at `job["horizon"]` with the routing tracer and the
    contract monitor: the gate row, trace cells, fund shares and per-head toggle firing."""
    from ..eval.baselines import producing_policy
    from ..eval.contract import ContractMonitor
    from ..eval.gate import row_from_stats, row_to_dict
    from ..eval.routing_audit import ordering_policy, trace_cells
    from ..eval.routing_trace import RoutingTracer
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _episode_env(job)
    tracer = RoutingTracer()
    tracer.attach(env)
    monitor = ContractMonitor(env)
    who_p, who_o = ARMS[job["kind"]]
    pm, om = load_models(job["models"]) if "gnn" in (who_p, who_o) else (None, None)
    p = g.gnn_producing_policy(pm) if who_p == "gnn" else producing_policy(BASE_PRODUCER)
    o = g.gnn_ordering_policy(om) if who_o == "gnn" else ordering_policy(BASE_ORDERER)
    fired = np.zeros(3)
    n_dec = [0]

    def counted(e: Any, agent: str) -> np.ndarray:
        a = p(e, agent)
        k = e.spec_.num_inputs
        fired[:] += np.asarray(a[k:k + 3]) > 0
        n_dec[0] += 1
        return a

    run_episode(env, counted, o, seed=job["episode"], action_seed=None, monitor=monitor)
    st = env.episode_stats
    monitor.verify(dead=bool(st["dead"]), cycles=st["cycles"], horizon=job["horizon"])
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     f"{job['kind']}@h{job['horizon']}", job["replicate"], st))
    row.update(trace_cells(tracer))
    row.update({"arm": job["kind"], "horizon": str(job["horizon"]),
                "units": repr(float(st["consumer_completed_units"])),
                "fund_shares": json.dumps(tracer.fund_shares(), sort_keys=True),
                **{f"fire_{t}": repr(float(fired[i] / max(1, n_dec[0])))
                   for i, t in enumerate(TOGGLES)}})
    return row


def side_effect_reading(rows: Sequence[dict], horizons: Sequence[int] = tuple(SIDE_ARMS)) -> dict:
    """SIDE_RULE over the rows: per (arm, horizon) the clause flags and the reported measures;
    deciding = GNN + GNN at the first horizon."""
    from .order_es_validate import row_flags
    from .producer_read import STOCK_HEAVY

    def summary(rs: list) -> dict:
        f = [row_flags(d) for d in rs]
        n = len(rs)
        ratio = [float(d["r_end"]) / float(d["r_start"]) for d in rs
                 if d["r_start"] not in ("", "None") and d["r_end"] not in ("", "None")
                 and float(d["r_start"]) > 0]
        num = lambda k: fmean(float(d[k]) for d in rs if d.get(k) not in ("", "None", None))  # noqa: E731
        return {"rows": n, "deaths": sum(x[0] for x in f),
                "rundown_share": sum(x[1] for x in f) / n, "starvation_share": sum(x[2] for x in f) / n,
                "r_end_over_r_start": fmean(ratio) if ratio else None,
                "min_capacity_ratio": num("min_capacity_ratio"),
                "holding_units_per_cycle": num("holding_units_per_cycle"),
                "investment_rejection_rate": num("investment_rejection_rate"),
                "units": num("units"),
                "fire": {t: num(f"fire_{t}") for t in TOGGLES}}

    out: dict[str, Any] = {"rule": SIDE_RULE, "per_horizon": {}}
    for h in horizons:
        arms = side_arms(h, horizons)
        per = {a: summary([d for d in rows if d["arm"] == a and d["horizon"] == str(h)])
               for a in arms}
        base = per["monitor_base"]
        for a in arms:
            if a == "monitor_base" or not per[a]["rows"]:
                continue
            worse = [c for c, k in (("a_deaths", "deaths"), ("b_rundown", "rundown_share"),
                                    ("c_starvation", "starvation_share"))
                     if per[a][k] > base[k]]
            per[a]["clauses_worse"] = worse
            hb = base["holding_units_per_cycle"]
            per[a]["stock_heavy"] = bool(hb and per[a]["holding_units_per_cycle"] > STOCK_HEAVY * hb)
        out["per_horizon"][str(h)] = per
    dec = out["per_horizon"][str(horizons[0])]["monitor"]
    out["reading"] = "clean" if not dec["clauses_worse"] else "side_effect"
    out["clauses"] = dec["clauses_worse"]
    return out


def run_side_effects(training_dir: str, models: str, out_dir: str, workers: int,
                     max_world_episodes: int | None, dry_run: bool = False,
                     horizons: Sequence[int] = tuple(SIDE_ARMS)) -> int:
    """EXP-0101 (task 2.7): SIDE_ARMS on the monitor world-episodes x MONITOR_REPLICATES."""
    from ..eval.provenance import sha256_file

    ts, _train, mon = _training(training_dir)
    if not dry_run:
        _refuse(real_problems(horizon=pe.HORIZON, max_world_episodes=max_world_episodes, ts=ts)
                + ([] if sha256_file(models) == EXP0100_MODELS_SHA256 else
                   ["the models are not EXP-0100's (5acf8b82...)"])
                + ([] if tuple(horizons) == tuple(SIDE_ARMS) else ["--horizons is set"]))
    if max_world_episodes:
        mon = mon[:max_world_episodes]
    jobs = [j for h in horizons for arm in side_arms(h, horizons)
            for j in _jobs(ts, mon, arm, MONITOR_REPLICATES, h, models)]
    jobs.sort(key=lambda j: -j["horizon"])  # longest first
    rows = _pool(side_effect_job, jobs, workers)
    doc = {**side_effect_reading(rows, horizons), "dry_run": dry_run, "models_sha256": sha256_file(models),
           "prereg": "openspec/changes/add-m6-joint-mpg-gnn/tasks.md 2.7 (2026-11-06)",
           "source_commit": SOURCE_COMMIT, "world_episodes": len(mon),
           "replicates": list(MONITOR_REPLICATES)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "side_effects.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "episodes.csv").open("w", newline="") as fh:
        import csv

        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["arm"], r["horizon"], r["world"], r["episode"],
                                             r["replicate"])):
            w.writerow(r)
    print(f"[m6_imitation] side effects: {doc['reading']} {doc['clauses']}", flush=True)
    return 0


# --- validation read (EXP-0102; pre-registration v3, frozen before its data) ---

PREREG_V3_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v3.json"
PREREG_V3_SHA256: str | None = "409643f205aadce1fc504b07282f7fec295b3d1960d9f5268dcba3df095e2267"  # frozen before EXP-0102
VALIDATE_ARMS = {"gnn": "monitor", "gnn_p": "swap_gnn_producer", "k5_rf5": "monitor_base"}
VALIDATE_RULE = ("at 12 producers (deciding), GNN + GNN vs k5_rf5 + MWWk on the loaded atlas v2 "
                 "validation worlds (L5's set and floors): parity if g >= -floor, no extra deaths, "
                 "and run-down and starvation shares <= the base's; better if parity and g > floor; "
                 "else short. 42 producers and the GNN producer + MWWk arm: the same rule, reported "
                 "only")


def validation_worlds(atlas: Mapping) -> list[dict]:
    from .producer_read import VALIDATE, read_views

    return read_views(atlas, VALIDATE, None, None)


def registered_prereg_v3(atlas: Mapping) -> dict:
    from .order_es_validate import VALIDATION_STAGE
    from .producer_read import READ_EPISODES, READ_REPLICATES, RHO_LOADED

    views = validation_worlds(atlas)
    return {
        "format": "m6-prereg-v3", "change": "add-m6-joint-mpg-gnn",
        "stage": "validation of the imitated stack (EXP-0102), before MAPPO",
        "status": "frozen before EXP-0102's data; the real read refuses any other file",
        "keeps": {"m6_prereg_v2.json": PREREG_SHA256, "note": "v1 and v2 stay frozen, unedited"},
        "decision": "supervisor 2026-11-06: validate the imitated stack on unseen worlds before RL "
                    "(the user's parity goal); the held-out set is not touched",
        "models": {"sha256": EXP0100_MODELS_SHA256, "from": "EXP-0100 fit/models.pt"},
        "arms": {"gnn": "GNN producer + GNN orderer", "gnn_p": "GNN producer + MWWk (reported)",
                 "k5_rf5": "k5_rf5 + MWWk (the base)"},
        "worlds": {"split": "atlas v2 validation", "loaded": f"rho > {RHO_LOADED}, not collapse",
                   "atlas_content_hash": REGISTERED_ATLAS_HASH,
                   "ids": {str(sz): [v["world"] for v in views if v["size"] == sz]
                           for sz in (12, 42)},
                   "episodes": READ_EPISODES, "stage": VALIDATION_STAGE,
                   "replicates": list(READ_REPLICATES), "horizon": pe.HORIZON},
        "rule": VALIDATE_RULE,
        "floor": "L5's (producer_read._comparisons: order_es_validate.mean_comp, equal-weight world "
                 "mean, max(atlas floor, own 3 SE), both arms with 2 replicates)",
        "clauses": "deaths, run-down share (r_end < X r_start), starvation share (any consumer "
                   "fill < Y), X / Y = side-effect params v2",
        "reported": ["the 42-producer reading", "the GNN producer + MWWk arm", "holding, "
                     "investment rejection, min capacity ratio per arm"]}


def freeze_prereg_v3(atlas: Mapping, path: Path = PREREG_V3_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v3(atlas)
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


# held-out (EXP-0103; pre-registration v4, frozen before its data; one shot, ledgered)
PREREG_V4_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v4.json"
PREREG_V4_SHA256: str | None = "f06fdf0c1a76c13e1a8137f02057510692bd5dd4b8cf60e6951b1fe04f349b6d"  # frozen before EXP-0103
HELDOUT_LEDGER = Path(__file__).resolve().parents[3] / "experiments" / "m6_heldout_ledger.json"
HELDOUT_KEY = "held-out/m6-imitated"


def heldout_worlds(atlas: Mapping) -> list[dict]:
    from .producer_read import HELDOUT, read_views

    return read_views(atlas, HELDOUT, None, None)


def registered_prereg_v4(atlas: Mapping) -> dict:
    from .producer_read import HELDOUT_STAGE, READ_EPISODES, READ_REPLICATES, RHO_LOADED

    views = heldout_worlds(atlas)
    loaded = [v["world"] for v in views if v["rho"] > RHO_LOADED and v["class"] != "collapse"]
    return {
        "format": "m6-prereg-v4", "change": "add-m6-joint-mpg-gnn",
        "stage": "held-out read of the imitated stack (EXP-0103), before MAPPO",
        "status": "frozen before EXP-0103's data; the real read refuses any other file; one shot, "
                  "with a pending ledger entry committed before the launch",
        "keeps": {"m6_prereg_v3.json": PREREG_V3_SHA256, "m6_prereg_v2.json": PREREG_SHA256,
                  "note": "v1, v2 and v3 stay frozen, unedited"},
        "decision": "user 2026-11-06 (via the supervisor): held-out now, then MAPPO",
        "models": {"sha256": EXP0100_MODELS_SHA256, "from": "EXP-0100 fit/models.pt"},
        "arms": {"gnn": "GNN producer + GNN orderer", "gnn_p": "GNN producer + MWWk (reported)",
                 "k5_rf5": "k5_rf5 + MWWk (the base)"},
        "worlds": {"split": "atlas v2 held-out (M5's L6 set; never read before: no M5 held-out "
                            "ledger entry exists)",
                   "played": "every held-out world at 12 and 42",
                   "gating": f"the loaded ones (rho > {RHO_LOADED}, not collapse), as L5 / L6",
                   "atlas_content_hash": REGISTERED_ATLAS_HASH,
                   "ids": {str(sz): [v["world"] for v in views if v["size"] == sz]
                           for sz in (12, 42)},
                   "loaded_ids": {str(sz): [w for w in loaded if f"-{sz}-" in w]
                                  for sz in (12, 42)},
                   "episodes": READ_EPISODES, "stage": HELDOUT_STAGE,
                   "replicates": list(READ_REPLICATES), "horizon": pe.HORIZON},
        "rule": VALIDATE_RULE.replace("validation worlds", "held-out worlds"),
        "floor": "L5 / L6's (producer_read._comparisons: order_es_validate.mean_comp, equal-weight "
                 "world mean over the loaded worlds, max(atlas floor, own 3 SE), both arms with 2 "
                 "replicates)",
        "clauses": "deaths, run-down share (r_end < X r_start), starvation share (any consumer "
                   "fill < Y), X / Y = side-effect params v2",
        "multiplicity": "MAPPO's later held-out read (if any) is a second comparison on this same "
                        "set; it will be reported as such, never as a fresh one-shot read",
        "reported": ["the 42-producer reading", "the GNN producer + MWWk arm", "holding, "
                     "investment rejection, min capacity ratio per arm"]}


def freeze_prereg_v4(atlas: Mapping, path: Path = PREREG_V4_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v4(atlas)
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


def heldout_ledger_problems(path: Path = HELDOUT_LEDGER) -> list[str]:
    """One shot: exactly one pending entry for these models and v4, and no closed one."""
    if not path.exists():
        return [f"no held-out ledger {path.name} (commit a pending entry before the launch)"]
    es = [e for e in json.loads(path.read_text()).get("entries", []) if e.get("key") == HELDOUT_KEY]
    pend = [e for e in es if e.get("status") == "pending"
            and e.get("models_sha256") == EXP0100_MODELS_SHA256
            and e.get("prereg") == PREREG_V4_SHA256]
    if len(pend) != 1 or any(e.get("status") != "pending" for e in es):
        return ["the held-out ledger does not hold exactly one pending entry (and none closed) "
                f"for {HELDOUT_KEY}"]
    return []


def validate_job(job: dict) -> dict[str, str]:
    row = side_effect_job({**job, "kind": VALIDATE_ARMS[job["arm"]]})
    row["arm"] = job["arm"]
    return row


def validation_reading(dicts: Sequence[dict], cells: Mapping) -> dict:
    """VALIDATE_RULE per size and arm (12 / gnn deciding)."""
    from .producer_read import BASE, _comparisons, _guard_ok

    per = _comparisons(dicts, cells, ("gnn", "gnn_p"))
    out: dict[str, Any] = {"rule": VALIDATE_RULE, "per_size": per, "readings": {}}
    for size, d in per.items():
        for arm in ("gnn", "gnn_p"):
            c, f = d["comparisons"].get(arm), d["flags"]
            if c is None:
                r = None
            else:
                ok = c["diff"] >= -c["floor"] and _guard_ok(f[arm], f[BASE])
                r = ("better" if ok and c["diff"] > c["floor"] else "parity" if ok else "short")
            out["readings"][f"{size}/{arm}"] = r
    out["reading"] = out["readings"].get("12/gnn")
    return out


def run_validate(atlas_path: str, models: str, out_dir: str, workers: int,
                 dry_run: bool = False, max_worlds: int | None = None,
                 episodes: int | None = None, horizon: int = pe.HORIZON,
                 split: str = "validate", ledger: str | None = None) -> int:
    """EXP-0102 (`validate`): VALIDATE_ARMS on the loaded atlas v2 validation worlds (L5's set);
    EXP-0103 (`heldout`): every held-out world played, the loaded ones gating (M5's L6 set)."""
    import csv

    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from .order_es_validate import VALIDATION_STAGE
    from .producer_read import HELDOUT_STAGE, READ_EPISODES, READ_REPLICATES, _cells, entry_for

    atlas = load_atlas(atlas_path)
    eps = episodes or READ_EPISODES
    held = split == "heldout"
    path, pin, reg = ((PREREG_V4_PATH, PREREG_V4_SHA256, registered_prereg_v4) if held
                      else (PREREG_V3_PATH, PREREG_V3_SHA256, registered_prereg_v3))
    if not dry_run:
        p = []
        if pin is None or not path.exists() or sha256_file(str(path)) != pin:
            p.append(f"{path.name} is not the pinned file")
        elif json.loads(path.read_text()) != reg(atlas):
            p.append(f"the {path.name} pre-registration differs from the code / atlas")
        if held:  # the pod syncs only the package dir: the ledger is staged and passed in
            p += heldout_ledger_problems(Path(ledger) if ledger else HELDOUT_LEDGER)
        if atlas.get("content_hash") != REGISTERED_ATLAS_HASH:
            p.append("not the registered atlas v2")
        if sha256_file(models) != EXP0100_MODELS_SHA256:
            p.append("the models are not EXP-0100's")
        if max_worlds or episodes or horizon != pe.HORIZON:
            p.append("a filter is set (--max-world-episodes / --episodes / --horizon)")
        _refuse(p)
    views = heldout_worlds(atlas) if held else validation_worlds(atlas)
    if max_worlds:
        views = [v for sz in (12, 42) for v in [x for x in views if x["size"] == sz][:max_worlds]]
    cells = _cells(views, atlas)
    jobs = []
    for v in views:
        e = entry_for(v, eps, HELDOUT_STAGE if held else VALIDATION_STAGE)
        for ep in e.episode_seeds:
            for r in READ_REPLICATES:
                for arm in VALIDATE_ARMS:
                    jobs.append({"kind": arm, "arm": arm, "world": e.id, "size": e.size,
                                 "regime": e.regime, "world_seed": e.world_seed,
                                 "generator": e.generator, "episode": int(ep), "replicate": r,
                                 "horizon": horizon, "models": models, "generation": "0"})
    jobs.sort(key=lambda j: (-j["size"], j["arm"] == "k5_rf5"))  # longest first
    rows = _pool(validate_job, jobs, workers)
    doc = {**validation_reading(rows, cells), "dry_run": dry_run, "split": split,
           "prereg": pin, "models_sha256": sha256_file(models),
           "source_commit": SOURCE_COMMIT, "worlds": [v["world"] for v in views],
           "episodes": eps, "horizon": horizon, "rows": len(rows), "jobs": len(jobs)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reading.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["arm"], r["world"], r["episode"], r["replicate"])):
            w.writerow(r)
    print(f"[m6_imitation] {split}: {doc['reading']} {doc['readings']}", flush=True)
    return 0 if len(rows) == len(jobs) else 2


def run_timing(atlas: str, horizon: int, worlds: Sequence[str]) -> dict:
    """L0 timing probe: seconds per decision (GNN vs k5_rf5 / MWWk) and per horizon-H episode."""
    from ..eval.baselines import producing_policy
    from ..eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from ..eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from ..eval.routing_audit import build_jobs, ordering_policy
    from ..eval.runner import run_episode
    from ..eval.teacher_audit import load_atlas

    torch.set_num_threads(1)
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    out = {}
    views = select_views(load_atlas(atlas), list(worlds))
    for view in views:
        arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
        arm = {**arm, "producing": BASE_PRODUCER,
               "subject": member_id(BASE_PRODUCER, arm["ordering"])}
        job = {**build_jobs([world_entry(view)], [arm], 1, horizon, None, {})[0], "order_ttl": 2}
        res = {}
        for name, (p, o) in {"base": (producing_policy(BASE_PRODUCER), ordering_policy(BASE_ORDERER)),
                             "gnn": (g.gnn_producing_policy(pm), g.gnn_ordering_policy(om))}.items():
            env = _episode_env(job)
            tp, to, n_p, n_o = [0.0], [0.0], [0], [0]

            def timed(pol: Any, acc: list, cnt: list) -> Any:
                def f(e: Any, a: str) -> np.ndarray:
                    t0 = time.perf_counter()
                    r_ = pol(e, a)
                    acc[0] += time.perf_counter() - t0
                    cnt[0] += 1
                    return r_
                return f

            t0 = time.perf_counter()
            run_episode(env, timed(p, tp, n_p), timed(o, to, n_o), seed=job["episode"])
            res[name] = {"episode_s": time.perf_counter() - t0,
                         "producer_ms_per_decision": 1000 * tp[0] / max(1, n_p[0]),
                         "orderer_ms_per_decision": 1000 * to[0] / max(1, n_o[0]),
                         "decisions": [n_p[0], n_o[0]]}
        out[f"{view.world if hasattr(view, 'world') else job['world']} ({job['size']})"] = res
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=("timing", "collect", "fit", "check", "side-effects", "validate",
                            "heldout", "gate", "attribution", "freeze", "freeze-v3", "freeze-v4",
                            "freeze-v5", "freeze-v6", "freeze-v7"))
    p.add_argument("--prev-models", help="check: the EXP-0100 models (clause c)")
    p.add_argument("--episodes-42", type=int, help="dry run: 42 check episodes per world")
    p.add_argument("--horizon-42", type=int, help="dry run: the 42 collect / check horizon")
    p.add_argument("--episodes", type=int)
    p.add_argument("--ledger", help="heldout: the held-out ledger (a staged copy on the pod)")
    p.add_argument("--horizons", default=",".join(str(h) for h in SIDE_ARMS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="", help="recorded in every output (provenance)")
    p.add_argument("--training-dir")
    p.add_argument("--collect-dir")
    p.add_argument("--models")
    p.add_argument("--atlas")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=pe.HORIZON)
    p.add_argument("--epochs", type=int, default=FIT["epochs"])
    p.add_argument("--fit-mode", default=FIT["mode"], choices=("auto", "sync", "hogwild"))
    p.add_argument("--per-agent", action="store_true",
                   help="fit: one Adam + clip per agent (a pre-registration must set it)")
    p.add_argument("--seed", type=int, default=FIT["seed"])
    p.add_argument("--max-world-episodes", type=int)
    p.add_argument("--worlds", default="ws-atlas-v1-12-0111,ws-atlas-v1-42-0000")
    a = p.parse_args(argv)
    global SOURCE_COMMIT
    SOURCE_COMMIT = a.source_commit
    if not a.dry_run and a.mode in ("collect", "fit", "check", "side-effects", "validate",
                                    "heldout", "gate", "attribution") and \
            len(SOURCE_COMMIT) != 40:
        p.error("a real run needs a 40-hex --source-commit")
    if a.mode == "freeze":
        status, sha = freeze_prereg()
        print(f"[m6_imitation] pre-registration {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "timing":
        print(json.dumps(run_timing(a.atlas, a.horizon, a.worlds.split(",")), indent=1))
        return 0
    if a.mode == "collect":
        return run_collect(a.training_dir, a.out_dir, a.workers, a.horizon, a.max_world_episodes,
                           a.dry_run, a.atlas, a.horizon_42)
    if a.mode == "freeze-v7":
        status, sha = freeze_prereg_v7()
        print(f"[m6_imitation] pre-registration v7 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "attribution":
        return run_attribution(a.training_dir, a.atlas, a.models, a.prev_models, a.out_dir,
                               a.workers, a.horizon, a.max_world_episodes, a.dry_run,
                               a.episodes_42, a.horizon_42)
    if a.mode == "freeze-v6":
        status, sha = freeze_prereg_v6()
        print(f"[m6_imitation] pre-registration v6 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "gate":
        return run_gate(a.collect_dir, a.models, a.prev_models, a.out_dir, a.dry_run)
    if a.mode == "freeze-v5":
        status, sha = freeze_prereg_v5()
        print(f"[m6_imitation] pre-registration v5 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "freeze-v3":
        from ..eval.teacher_audit import load_atlas

        status, sha = freeze_prereg_v3(load_atlas(a.atlas))
        print(f"[m6_imitation] pre-registration v3 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "freeze-v4":
        from ..eval.teacher_audit import load_atlas

        status, sha = freeze_prereg_v4(load_atlas(a.atlas))
        print(f"[m6_imitation] pre-registration v4 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode in ("validate", "heldout"):
        return run_validate(a.atlas, a.models, a.out_dir, a.workers, a.dry_run,
                            a.max_world_episodes, a.episodes, a.horizon, a.mode, a.ledger)
    if a.mode == "side-effects":
        return run_side_effects(a.training_dir, a.models, a.out_dir, a.workers,
                                a.max_world_episodes, a.dry_run,
                                tuple(int(h) for h in a.horizons.split(",")))
    if a.mode == "fit":
        return run_fit(a.collect_dir, a.out_dir, a.epochs, a.seed, a.workers, a.dry_run,
                       a.fit_mode, a.per_agent)
    return run_check(a.training_dir, a.atlas, a.models, a.out_dir, a.workers, a.horizon,
                     a.max_world_episodes, a.dry_run, a.prev_models, a.episodes_42, a.horizon_42)


if __name__ == "__main__":
    sys.exit(main())
