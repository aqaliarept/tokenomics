"""Step-2 validation readings: `python -m tokenomics.train.order_es_validate`
(add-m4-orderer-redesign design D6 / D7 / D9, task 5.3; learned-orderer "Ordered ES readings",
"Paired references and noise floors", "Herding and concentration are measured", "History effect
is recorded", "Detectable effect pre-registered").

`--mode validate --runs <order_es train dirs> --bar-dir <bar> --atlas <atlas_v2.json>
--orderer-ckpt <EXP-0012> --episodes 4|8` plays, with the `entity_planner` producer on every atlas
v2 validation world (12 and 42 producers; never another split):
- every staged candidate (the run's `candidate.json`, choice-v1, agent-local history, R = 2) and
  its shared-only ablation (the same theta, history replaced by the product's agent mean; D9);
- A0 (choice-v1, R = 2), RR (`round_robin`), MWW (`makeable_weighted_whole`, R = 2) and TR (the
  trained orderer as deployed), all three under logits-v1.
Per-order arms share DrawKey(world, episode, replicate), so candidate, ablation and A0 pair by
replicate (common random numbers). Episode seeds come from the stage `VALIDATION_STAGE`.

Readings are pure functions of the rows and the runs' logs (`analyse`), with D7's precedence
`unresolved` -> `kill` -> `success` -> `plateau` -> `budget_limited` -> `other`; the seed-1 read
(`--seed-combine --reuse-references <val-seed0>`) reuses the seed-0 read's rows (digests must be
equal), reads the seed-1 candidate with the same table and combines the two seeds by D7's table.
Floors are per comparison: max(the atlas-spread floor at the read's counts, 3 SE of the read's
own paired differences), the own term only from >= 2 paired observations on a world (the n = 1
rule). Labels (`not_via_deconcentration`, `history_inert`, `herding`) follow D7's label table.

`--mode recompute --out-dir <read> --runs ... --bar-dir ...` rebuilds `reading.json` from
`episodes.csv` and the runs' logs (MATCH / MISMATCH, exit 0 / 2). `--mode power` applies the
frozen power rule to EXP-0052's confirm rows (task 5.4 records the outcome). `--mode freeze`
writes / checks the frozen pre-registration `train/registry/es_prereg_m4_v1.json`.

A real validate run needs a frozen pre-registration (v1 `PREREG_SHA256` or v2
`PREREG_V2_SHA256`, task 6.1f; `--prereg`, default v1) with every run trained under that same
file, the registered atlas and orderer, complete real 40-generation runs on the same
bar, horizon 100, 4 or 8 episodes, a 40-hex source commit and no filters. Contract failures,
missing rows or digest differences end it invalid (exit 2). It never writes the held-out ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..env.spaces import CHOICE_V1, LOGITS_V1, decode_pins
from ..eval.atlas import (COLLAPSE, FLOOR_BAR_SHARE, VALIDATION, Spread, quartile_cuts,
                          space_entry, space_info, standard_error)
from ..eval.contract import (ContractMonitor, ContractRunError, ensure_failures_file,
                             job_failure, read_failures)
from ..eval.decoder_check import arm_units, paired_diffs, world_mean
from ..eval.gate import DigestMismatch, row_from_stats, row_to_dict
from ..eval.gate_cli import (CONTRACT_INVALID, INVALID, Job, Key, _open_spec, _prepare_resume,
                             _write_spec, check_rows, job_grid, job_key, read_rows)
from ..eval.provenance import resolve_verified_commit, sha256_file, source_fingerprint
from ..eval.registry import WorldEntry
from ..eval.routing_audit import (EPISODE_COLUMNS, PRODUCER, TRAINED_ORDERER, _is_commit,
                                  _trained_entry, build_jobs, execute, job_config,
                                  read_episode_dicts, trace_cells, write_csv)
from ..eval.routing_diagnostics import ROUTER_KIND, STOCHASTIC, DrawKey, set_draw_key
from ..eval.routing_trace import RoutingTracer
from ..eval.shock_profiles import SHOCKS_I_SMOKE, SMOKE_START_WINDOW
from ..eval.teacher_audit import (REGISTERED_ATLAS_HASH, ROLE_CHECKPOINTS, checked_orderer,
                                  load_atlas, world_view)
from ..eval.world_space import config_from_record, episode_seeds, point_from_record
from . import order_es as es
from . import order_es_worlds as ow
from .per_order_scorer import (AGENT, BASE_MWW, BASE_TS, SHARED_ONLY, PerOrderPolicy,
                               base_kind_of, make_policy, theta_sha256)

DRIVER = "tokenomics.train.order_es_validate"
EXPERIMENT = "EXP-0060"
VALIDATE, RECOMPUTE, POWER, FREEZE = "validate", "recompute", "power", "freeze"
HORIZON = 100
READING_SIZE, REPORTED_SIZE = 12, 42
SIZES = (READING_SIZE, REPORTED_SIZE)
RHO_LOADED = 0.28
EXPECTED_LOADED = {READING_SIZE: 9, REPORTED_SIZE: 3}  # D9: validation, loaded, non-collapse
EPISODE_CHOICES = (4, 8)  # D7 power rule (task 5.4 picks one)
VALIDATION_STAGE = "m4-es-validate"  # episode-seed salt of every validation world-episode
STOCHASTIC_R = 2
MAIN_GENERATIONS = es.GENERATIONS["main"]
IMPROVEMENT_WINDOW = es.MONITOR_EVERY  # "the last 10 generations"

FLOOR_SE = 3.0
MIN_OWN_PAIRS = 2  # the n = 1 rule: the own term needs >= 2 paired observations on a world
KILL_RATIO = 1 / 3  # D5's kill rule: g_val < g_tr / 3
LOST_TO_RR_SHARE = 1 / 6
LABEL_SE = 3.0
HERDING_WORLD_SHARE = 1 / 3

# arms (labels in `episodes.csv` column `arm`)
A0, RR, MWW, TR = "A0", "RR", "MWW", "TR"
REFERENCES = (RR, MWW, TR)
SUCCESS_REFS = (A0, RR, MWW, TR)
ABLATION = "ablation"  # the reference key of a candidate's shared-only ablation
SHARED_SUFFIX = "/shared"
LEGACY_ROUTER = {RR: "round_robin", MWW: "makeable_weighted_whole", TR: TRAINED_ORDERER}

# groups (D9 tables)
LOADED, LIGHT = "loaded", "light"

# readings
UNRESOLVED, KILL, SUCCESS, PLATEAU = "unresolved", "kill", "success", "plateau"
BUDGET_LIMITED, OTHER, SEED_DISAGREEMENT = "budget_limited", "other", "seed_disagreement"
PRECEDENCE = (UNRESOLVED, KILL, SUCCESS, PLATEAU, BUDGET_LIMITED, OTHER)
NEXT = {UNRESOLVED: "diagnose, stop", KILL: "diagnose the cross-world cause, stop",
        SUCCESS: "seed 1, then held-out", PLATEAU: "seed 1, then the step-3 gate",
        BUDGET_LIMITED: "to the user (more generations need approval)",
        OTHER: "diagnose, stop", SEED_DISAGREEMENT: "diagnose, stop"}
FINAL_NEXT = {SUCCESS: "held-out read (group 8) once every herding / not_via_deconcentration "
                       "label is diagnosed",
              PLATEAU: "record history_inert and herding; ask the user to release tranche B1"}

# labels
NOT_VIA_DECONCENTRATION, HISTORY_INERT, HERDING = (
    "not_via_deconcentration", "history_inert", "herding")
LABELS_WITH = {SUCCESS: (NOT_VIA_DECONCENTRATION, HISTORY_INERT, HERDING),
               PLATEAU: (HISTORY_INERT, HERDING)}

# trace columns
TOP_SHARE = "trace_lead_concurrent_top_share_consumer"
TOP_SHARE_N = "trace_lead_concurrent_groups_consumer"
DISPERSION = "trace_routed_share_dispersion"
DISPERSION_N = "trace_routed_share_pairs"
SWING = "trace_share_swing_consumer"
SWING_N = "trace_share_swing_pairs_consumer"

# power rule (D7, frozen here; task 5.4 computes the outcome)
POWER_PAIRS = ((MWW, RR), (TR, RR))
POWER_FOUR, POWER_EIGHT = 0.02, 0.03
POWER_SUBJECTS = {RR: "entity_planner/round_robin", MWW: "entity_planner/makeable_weighted_whole",
                  TR: "entity_planner/trained_orderer"}

PREREG_PATH = es.PREREG_PATH
# SHA-256 of the frozen file (task 5.3): a real validate run refuses any other file, and any run
# trained under another file; task 5.4 records it in sizing.md
PREREG_SHA256 = "b9d449c15db6860950aab8dbc49bc0e3d67635d4d50e6f1c817d1e04fedcbabf"
# pre-registration v2 (task 6.1d): frozen by `--mode freeze --prereg-version 2`; a real probe /
# train run accepts either frozen file and plays the rule its version registers
PREREG_V2_PATH = es.PREREG_V2_PATH
PREREG_V2_SHA256 = "511e939a41213dd401d390637bc7e5be1c3725f32fb1de3a9e70d7b1385ceeae"
PREREG_V3_PATH = es.PREREG_V3_PATH
# pre-registration v3 (task 7.6, the re-based step 2): frozen by `--mode freeze --prereg-version 3`
PREREG_V3_SHA256 = "edd6efc2c325edac003e352c1a2e0a66e8ce72c67afacf86c8e1076ac3ffa850"
# the frozen pre-registrations a read accepts, by file SHA-256 -> version (task 6.1f). Their
# `readings` sections are identical (v2 changed training only), so what decides a read is the
# same under both; a read and its runs must share one file.
FROZEN_PREREGS = {PREREG_SHA256: 1, PREREG_V2_SHA256: 2, PREREG_V3_SHA256: 3}


def run_prereg_version(training_pins: Mapping[str, Any]) -> int:
    """The pre-registration version a train run's rule ran under: v2 runs pin
    `disqualification.prereg_version` (order_es `_v2_pins`); v1 runs pin none."""
    return int((training_pins.get("disqualification") or {}).get("prereg_version", 1))


def read_prereg_version(prereg_sha: str, prereg: Mapping[str, Any]) -> int:
    """A read's version: by the frozen file's SHA, else (a dry run's draft) by its format."""
    return FROZEN_PREREGS.get(prereg_sha, es.prereg_version(prereg))


def prereg_mix_problems(cands: Sequence["Candidate"], version: int) -> list[str]:
    """A v1 / v2 mix: a run trained under another version than the read's (any mode)."""
    return [f"pre-registration mix: run {c.label} was trained under v{c.prereg_version}, this "
            f"read uses v{version} (--prereg)" for c in cands if c.prereg_version != version]

ROW_EXTRA = ("arm", "ordering_decode", "history_mode", "theta_sha256", "episode_seconds",
             "job_seconds")
COLUMNS = EPISODE_COLUMNS + ROW_EXTRA


def _log(msg: str) -> None:
    print(msg, flush=True)


def ablation_label(label: str) -> str:
    return label + SHARED_SUFFIX


def run_label(arm: str, opt_seed: int) -> str:
    return f"{arm}-s{opt_seed}"


# --- pre-registration (the `readings` section) ---


def readings_registration() -> dict[str, Any]:
    """D6 selection, D7 floors / power rule / readings / seed table / labels and D9's sets and
    checks, as frozen in `es_prereg_m4_v1.json`'s `readings` section."""
    return es._json_norm({
        "format": "m4-es-readings-v1",
        "design": "add-m4-orderer-redesign D6 / D7 / D9 (task 5.3)",
        "units": "consumer units per world-episode (consumer_completed_units); a world's value "
                 "is the equal-weight mean over its episodes of the replicate means; a read is "
                 "the equal-weight mean over its worlds",
        "gains": {
            "g_tr": "center - A0 on the training-monitor set at the run's last generation (16 "
                    "fixed world-episodes, replicates 0 and 1, the run dir's monitor.csv, which "
                    "holds A0's rows in a resumed dir too), equal-weight over world-episodes",
            "g_val": "the candidate - A0 on the 12-producer loaded validation worlds (world "
                     "mean)",
        },
        "floors": {
            "rule": "per comparison (candidate vs A0, RR, MWW, TR and its shared-only ablation; "
                    "A2 vs A1; center vs A0 on the monitor set): max(atlas-spread floor at the "
                    "read's counts, own 3 SE)",
            "floor_se": FLOOR_SE, "min_own_pairs": MIN_OWN_PAIRS,
            "floor_bar_share": FLOOR_BAR_SHARE,
            "atlas_per_world": "SE_w = atlas.standard_error(n_w episodes, differing_w, the "
                               "compared stochastic arms at their R, the atlas v2 `spreads` of "
                               "the world's size / choice stratum); floor_w = max(3 SE_w, 0.01 "
                               "bar_w)",
            "own_per_world": "3 * stdev(paired observations) / sqrt(count), paired per episode "
                             "and replicate (same replicates pair by replicate, a deterministic "
                             "arm's one row against each replicate); only with >= 2 paired "
                             "observations on the world (the n = 1 rule), else the atlas floor "
                             "alone",
            "atlas_world_mean": "max(3 sqrt(sum_w SE_w^2) / W, 0.01 * mean_w bar_w)",
            "own_world_mean": "3 sqrt(sum_w s^2 / n_w) / W, s^2 the pooled within-world "
                              "variance of the paired observations over the worlds with >= 2 "
                              "of them (df = sum (n_w - 1)); none when no world has 2 (the "
                              "n = 1 rule)",
            "bar_w": "validation: max of the read's RR, MWW and TR world means; monitor: the "
                     "bar table's B_we",
            "monitor_strata": "a monitor world-episode is its own world (1 episode, R = 2), "
                              "stratum (12, its choice), differing from its pattern / shocks",
            "tie": "a gain at or below its floor is a tie",
        },
        "power_rule": {
            "pairs": [list(p) for p in POWER_PAIRS], "deciding": "max",
            "justification": "success needs the candidate above each of A0, RR, MWW and TR, so "
                             "the read is as weak as its noisiest comparison; MWW - RR stands "
                             "for a stochastic-vs-deterministic pair (candidate vs RR / TR), "
                             "TR - RR for a pair carrying the trained scorer; the larger of the "
                             "two is taken so the choice cannot follow the smaller value",
            "rows": "EXP-0052 confirm episodes.csv on the 9 loaded 12-producer atlas v2 "
                    "validation worlds (rho > 0.28, class not collapse), entity_planner",
            "replicates": STOCHASTIC_R,
            "estimator": "per world, per episode d = mean over the replicates of a - the same "
                         "for b; s = pooled within-world SD of d over the worlds with >= 2 "
                         "episodes (df = sum (n_w - 1)); MDE(n) = 3 s / sqrt(n) / sqrt(W); "
                         "share = MDE / mean bar",
            "mean_bar": "equal-weight mean over the same worlds of the atlas bar "
                        "(reclassified/worlds.csv `bar`)",
            "rule": {"four_episodes_max_share": POWER_FOUR, "else_episodes": 8,
                     "to_user_above_at_eight": POWER_EIGHT},
            "stochastic_arms_R": STOCHASTIC_R,
            "outcome": "computed by task 5.4 (`--mode power`) and recorded in sizing.md with "
                       "this file's SHA-256; never frozen here",
        },
        "validation": {
            "split": VALIDATION, "sizes": list(SIZES), "rho_loaded": RHO_LOADED,
            "gating": "12-producer validation worlds with rho > 0.28 and class not collapse (9)",
            "reported": "42-producer worlds (3 loaded, unretrained), light (rho <= 0.28) and "
                        "collapse worlds, rho quartiles (nearest rank within size)",
            "episode_stage": VALIDATION_STAGE, "episodes": list(EPISODE_CHOICES),
            "arms": {"candidate": "each run's candidate.json, choice-v1, agent history",
                     "ablation": "the candidate with history_mode shared_only (D9), "
                                 "choice-v1, never retrained",
                     A0: "zero residual, choice-v1", RR: "round_robin, logits-v1",
                     MWW: "makeable_weighted_whole, logits-v1",
                     TR: "trained orderer as deployed, logits-v1"},
            "replicates": {"candidate": STOCHASTIC_R, "ablation": STOCHASTIC_R,
                           A0: STOCHASTIC_R, MWW: STOCHASTIC_R, RR: 1, TR: 1},
            "draw_key": "per-order arms: DrawKey(world, episode seed, replicate); MWW: the "
                        "gate's replicate action seeds",
            "producer": PRODUCER,
            "producer_column": "none: the user chose entity_planner (2026-10-30, open "
                               "decision 5), so no EXP-0035-producer column",
            "held_out": "never played or read here (held-out ledger: task 8.1)",
            "nonfinite": "a candidate raising NonFiniteChoice in validation is a contract "
                         "failure (the run is invalid)",
        },
        "selection": "among the seed-0 candidates (A1 / A2 or A2-R2), the highest 12-producer "
                     "loaded validation world mean; A2 - A1 within F_val(A2 vs A1) goes to A1",
        "precedence": list(PRECEDENCE),
        "readings": {
            UNRESOLVED: "g_tr <= F_tr for every candidate read",
            KILL: "the selected candidate has g_tr > F_tr but g_val < g_tr / 3",
            SUCCESS: ["the candidate - each of A0, RR, MWW and TR > F_val(ref) on the world "
                      "mean", "worlds lost to RR by more than floor_w: count * 6 <= worlds",
                      "zero deaths of the candidate", "run-down share (r_end < X r_start) and "
                      "starvation share (a consumer fill < Y) of the candidate's rows <= RR's"],
            PLATEAU: ["(i) every seed-0 arm's last-10-generation monitor improvement < its "
                      "F_tr (seed-1 read: and the seed-1 run's own)",
                      "(ii) |A2 - A1| (seed 0) <= F_val(A2 vs A1) on validation",
                      "(iii) the candidate - MWW <= F_val(MWW) or the candidate - RR <= "
                      "F_val(RR)"],
            BUDGET_LIMITED: "no success, and seed-0 read: any seed-0 arm's improvement >= its "
                            "F_tr; seed-1 read: the seed-1 run's own improvement >= its F_tr",
            OTHER: "anything else",
            "improvement": "center world mean at the last generation - at 10 generations "
                           "before (monitor set; the earlier value from the run dir's rows, "
                           "or from its checkpoint's monitor history when the run was resumed "
                           "past it, which must equal the dir's rows wherever both exist); "
                           "missing -> still improving",
        },
        "kill_ratio": KILL_RATIO, "lost_to_rr_share": LOST_TO_RR_SHARE,
        "next": NEXT,
        "seed_combination": {
            "table": [[SUCCESS, SUCCESS, SUCCESS], [PLATEAU, PLATEAU, PLATEAU],
                      [SUCCESS, PLATEAU, SEED_DISAGREEMENT],
                      [PLATEAU, SUCCESS, SEED_DISAGREEMENT]],
            "carry": "seed 0 success or plateau and seed 1 unresolved / kill / budget_limited / "
                     "other -> the seed-1 reading with its Next",
            "seed_1_rules": "the seed-1 candidate is read with the same table, selection not "
                            "redone; clause (ii) from seed 0; clause (i) = seed 0's (i) and "
                            "the seed-1 run's own improvement < F_tr; budget_limited when the "
                            "seed-1 run's improvement >= F_tr",
            "seed_1_runs_only_after": [SUCCESS, PLATEAU],
        },
        "labels": {
            "table": {NOT_VIA_DECONCENTRATION: [SUCCESS], HISTORY_INERT: [SUCCESS, PLATEAU],
                      HERDING: [SUCCESS, PLATEAU]},
            NOT_VIA_DECONCENTRATION: "candidate - A0 of trace_lead_concurrent_top_share_consumer "
                                     "(rows with >= 1 concurrent group, paired by episode and "
                                     "replicate, 12-producer loaded worlds, world mean) is not "
                                     "below -3 SE (own pooled SE as the floors')",
            HISTORY_INERT: "|candidate - ablation| <= F_val(ablation) on the 12-producer loaded "
                           "validation world mean",
            HERDING: "candidate share_swing_consumer world mean - MWW's > 3 SE_w (the "
                     "candidate's own spread over its rows with swing pairs) on >= 1/3 of the "
                     "loaded validation worlds",
            "label_se": LABEL_SE, "herding_world_share": HERDING_WORLD_SHARE,
            "concentration_measures": [TOP_SHARE, DISPERSION, SWING],
            "concentration_arms": ["candidate", "ablation", A0, MWW, RR],
            "computed": "at the seed-0 read and again at the final read (both seeds' "
                        "candidates; a label on either is recorded)",
            "undefined": "a measure whose count column is 0 on a row is undefined there and "
                         "skipped",
        },
        "normaliser_pins": {
            "rule_sha256": es._sha(ow.NORMALISER_PREREG),
            "fitting_code_sha256": ow.NORMALISER_PREREG["fitting_code"]["sha256"],
            "clip": ow.NORMALISER_PREREG["clip"],
            "zero_spread_rule": ow.NORMALISER_PREREG["zero_spread_rule"],
            "scale_floors": ow.NORMALISER_PREREG["scale_floors"],
            "check": "validation refuses a bar whose normaliser.json `prereg` hashes otherwise, "
                     "or a candidate whose normaliser is not the bar's",
        },
        "probe_interpretation": "the probe starts each arm at its registered initial theta "
                                "(Delta = 0; probe.theta). A2's cosine is structurally lower "
                                "than A1's: at v = 0 its u block (172 of 268 parameters) has no "
                                "first-order effect, so its estimate carries noise there. The "
                                "thresholds read on A1 and are not moved for A2 (A2 < 0.1 with "
                                "A1 >= 0.1 -> A2 uses R = 2)",
    })


V2_CHANGES = {
    "from": {"file": "es_prereg_m4_v1.json", "sha256": PREREG_SHA256,
             "note": "v1 is never edited; v2 is a new file"},
    "why": {"user_decision": "\"Bigger batches + fix rule\" (2026-11-02) and \"Add a "
                             "larger-sigma arm\" (2026-11-03), notes/user-decisions.md",
            "evidence": ["EXP-0058: the v1 gradient-signal probe read no_signal (A1 cosine "
                         "-0.070; 37 / 48 A1 and 42 / 48 A2 member scores disqualified)",
                         "EXP-0068 (DIAGNOSTICS 25): heterogeneous by the registered statistic "
                         "(A1 minibatch pair-difference / re-keyed-noise median ratio 1.76 on "
                         "the 4-df minibatch reference, with chance sign agreement across "
                         "minibatches); not robust: against the 24-df per-world reference the "
                         "same rule reads inconclusive (A1 0.85, A2 0.94); flat at sigma not "
                         "excluded (EXP-0068 review round 1)",
                         "EXP-0068 check 2: the relative rule self-disqualifies re-keyed A0 on "
                         "16 / 48 (33%) episodes"],
            "supervisor_decisions": "2026-11-02 (notes/6.1d.md): c = 2, pooled SDs, target "
                                    "<= 2 / 48, leave-one-world-out reported only"},
    "changed": ["es.minibatch 12 -> 24 (probe and ES estimate; es.weak_minibatch stays 24, so "
                "`weak` and `signal` both train at 24)",
                "probe.minibatch 6 -> 24, drawn from the recorded probe stream at size 24",
                "es.disqualification: the noise margin (noise_margin) on run-down and "
                "starvation; death unchanged",
                "probe.arms + A1-S2 (probe.sigma_arms): A1 at 2 x sigma (0.10) on the same "
                "eps_i, minibatches and draw keys, diagnostic only, with its pre-registered "
                "step reading (flat_at_sigma / heterogeneity / sigma_only / no_signal)"],
    "unchanged": "everything else, the readings section and every threshold (D7 signal 0.3 / "
                 "weak 0.1, the power rule, X / Y, floors) included; training arms and sigma "
                 "(0.05) unchanged",
    "not_tuned": "the margin is fixed from A0's own re-keyed rows; no probe cosine, member "
                 "outcome or training result was read to choose it",
}


V3_CHANGES = {
    "from": {"file": "es_prereg_m4_v2.json", "sha256": PREREG_V2_SHA256,
             "note": "v1 / v2 are never edited; v3 is a new file"},
    "why": {"user_decision": "\"Re-base on MWW\" (2026-11-04, notes/user-decisions.md)",
            "evidence": ["EXP-0060 final reading plateau on both seeds: the A1 residual over "
                         "the TS base generalises (A1 - A0 +68.7 / +45.3 on validation) but "
                         "ends level with RR (+26.0 / +2.5, inside the floor) and below MWW "
                         "(-59.0 / -82.5, beyond the floor); A0 (TS) sits ~128 units below MWW",
                         "supervisor (2026-11-04): herding is not the main gap (validation "
                         "pile-up RR 0.62 < MWW 0.73 < A1 0.80 < A0 0.82 while MWW fills the "
                         "most units); the weak base is"]},
    "changed": ["es.base: the frozen base is MWW's keyed whole-order choice (MWWk); the "
                "residual is a logit add on its log-weights; A0 == MWWk at theta = 0",
                "es.trainable_arms: A1 only (history_inert on both EXP-0060 seeds; A2-R2 - A1 "
                "within its floor)",
                "probe: not run (the re-based pilot is licensed by the user decision)",
                "validation.reported_arms: EXP-0060's A1-s0 theta (TS base), reported only",
                "validation.plateau_ii: clause (ii) carried from EXP-0060 val-seed0 (A2 not "
                "re-run)",
                "the bar is recomputed for the new A0 (order_es_worlds --base mww); the training "
                "worlds follow the registered selection and A0-infeasible rules"],
    "unchanged": "v2's training rule (minibatch 24, the noise margin's constants, sigma, Adam, "
                 "12 pairs, 40 generations), the readings section and every threshold",
    "not_tuned": "no EXP-0060 member outcome or validation row was read to choose the base's "
                 "combination rule (the simplest that makes A0 == MWWk exact)",
}


def frozen_prereg(version: int = 1) -> dict[str, Any]:
    """The frozen pre-registration: order_es's enforced sections (byte-equal to
    `registered_sections(version)`, its `worlds` section to `bar_registration()`) plus
    `readings`; v2 also records what changed and why (`changes`)."""
    if version == 3:
        return {"format": es.PREREG_V3_FORMAT, "status": es.FROZEN,
                "change": "add-m4-orderer-redesign",
                "note": "frozen by task 7.6 after EXP-0060, before any v3 data; enforced "
                        "sections from order_es.registered_sections(3), `readings` from "
                        "order_es_validate.readings_registration() (unchanged from v1)",
                "changes": V3_CHANGES, **es.registered_sections(3),
                "readings": readings_registration()}
    if version == 2:
        return {"format": es.PREREG_V2_FORMAT, "status": es.FROZEN,
                "change": "add-m4-orderer-redesign",
                "note": "frozen by task 6.1d after EXP-0068 (re-frozen 2026-11-03 with the "
                        "A1-S2 probe arm and the qualified EXP-0068 evidence, still before any "
                        "v2 ES data); enforced "
                        "sections from order_es.registered_sections(2), `readings` from "
                        "order_es_validate.readings_registration() (unchanged from v1)",
                "changes": V2_CHANGES, **es.registered_sections(2),
                "readings": readings_registration()}
    return {"format": es.PREREG_FORMAT, "status": es.FROZEN, "change": "add-m4-orderer-redesign",
            "note": "frozen by task 5.3 before any ES data; enforced sections from "
                    "order_es.registered_sections(), `readings` from "
                    "order_es_validate.readings_registration()",
            **es.registered_sections(), "readings": readings_registration()}


def prereg_text(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"


def prereg_read_problems(doc: Any) -> list[str]:
    """Why a validation read may not use `doc`: schema, not frozen, or any section differing
    from this code's (order_es's enforced sections and `readings`)."""
    out = es.prereg_schema_problems(doc)
    if out:
        return out
    if doc["status"] != es.FROZEN:
        out.append(f"pre-registration status is {doc['status']!r}, a real run needs 'frozen'")
    want = {**es.registered_sections(es.prereg_version(doc)), "readings": readings_registration()}
    for sec, w in want.items():
        diff = es._diff_paths(es._json_norm(doc.get(sec)), w)
        if diff:
            out.append(f"pre-registration differs from the registered values at "
                       f"{', '.join(f'{sec}.{d}' if d else sec for d in diff[:6])}")
    return out


# --- floors (pure) ---

Units = dict[int, dict[int, float]]  # episode seed -> replicate -> value


@dataclass(frozen=True)
class Cell:
    """What a floor needs to know about one world (or one monitor world-episode)."""

    world: str
    size: int
    choice: str
    differing: bool
    spread: Spread | None
    rho: float = math.nan
    cls: str = ""

    @property
    def group(self) -> str:
        if self.cls == COLLAPSE:
            return COLLAPSE
        return LIGHT if self.rho <= RHO_LOADED else LOADED


def spread_from_dict(d: Mapping[str, Any]) -> Spread:
    return Spread(**{f.name: d.get(f.name) for f in fields(Spread)})


def atlas_se(cell: Cell, episodes: int, stochastic_reps: Sequence[int]) -> float:
    """The atlas SE of one compared pair's world mean at the read's counts (D12's formula)."""
    return standard_error(episodes, cell.differing,
                          {f"arm{i}": int(r) for i, r in enumerate(stochastic_reps)}, cell.spread)


def own_3se(diffs: Sequence[float]) -> float | None:
    """3 SE of the paired observations; None below 2 (the n = 1 rule)."""
    if len(diffs) < MIN_OWN_PAIRS:
        return None
    return FLOOR_SE * statistics.stdev(diffs) / math.sqrt(len(diffs))


@dataclass(frozen=True)
class WorldComp:
    world: str
    a: float
    b: float
    diff: float
    pairs: int
    atlas_se: float
    atlas_floor: float
    own3se: float | None
    diffs: tuple[float, ...]
    episodes: int

    @property
    def floor(self) -> float:
        return self.atlas_floor if self.own3se is None else max(self.atlas_floor, self.own3se)


def world_comp(a: Units, b: Units, cell: Cell, stochastic_reps: Sequence[int],
               bar: float) -> WorldComp:
    diffs = paired_diffs(a, b)
    se = atlas_se(cell, len(a), stochastic_reps)
    return WorldComp(cell.world, world_mean(a), world_mean(b), world_mean(a) - world_mean(b),
                     len(diffs), se, max(FLOOR_SE * se, FLOOR_BAR_SHARE * bar), own_3se(diffs),
                     tuple(diffs), len(a))


@dataclass(frozen=True)
class MeanComp:
    """One comparison on an equal-weight world mean, with its floor (D7)."""

    a: str
    b: str
    diff: float
    worlds: int
    atlas_floor: float
    own3se: float | None
    df: int
    per_world: tuple[WorldComp, ...]

    @property
    def floor(self) -> float:
        return self.atlas_floor if self.own3se is None else max(self.atlas_floor, self.own3se)

    @property
    def clears(self) -> bool:
        return self.diff > self.floor

    @property
    def within(self) -> bool:
        return abs(self.diff) <= self.floor

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "diff": self.diff, "floor": self.floor,
                "atlas_floor": self.atlas_floor, "own3se": self.own3se, "df": self.df,
                "worlds": self.worlds}


def pooled_own_3se(comps: Sequence[WorldComp]) -> tuple[float | None, int]:
    """(3 SE of the world mean from the pooled within-world variance, df); (None, 0) when no
    world has >= 2 paired observations."""
    ss, df = 0.0, 0
    for c in comps:
        if c.pairs >= MIN_OWN_PAIRS:
            m = statistics.fmean(c.diffs)
            ss += sum((x - m) ** 2 for x in c.diffs)
            df += c.pairs - 1
    if df == 0 or not comps:
        return None, 0
    s2 = ss / df
    return FLOOR_SE * math.sqrt(sum(s2 / c.pairs for c in comps)) / len(comps), df


def mean_comp(units: Mapping[str, Mapping[str, Units]], cells: Mapping[str, Cell], a: str,
              b: str, reps: Mapping[str, int], bars: Mapping[str, float],
              worlds: Sequence[str]) -> MeanComp | None:
    """`a` - `b` over `worlds` (each world needs both arms), with the D7 floor; None when a
    world lacks an arm (never from partial rows)."""
    per = []
    for w in worlds:
        u = units.get(w, {})
        if a not in u or b not in u:
            return None
        stoch = [reps[x] for x in (a, b) if reps.get(x, 1) > 1]
        per.append(world_comp(u[a], u[b], cells[w], stoch, bars[w]))
    if not per:
        return None
    own, df = pooled_own_3se(per)
    se = math.sqrt(sum(c.atlas_se ** 2 for c in per)) / len(per)
    atlas = max(FLOOR_SE * se, FLOOR_BAR_SHARE * statistics.fmean(bars[w] for w in worlds))
    return MeanComp(a, b, statistics.fmean(c.diff for c in per), len(per), atlas, own, df,
                    tuple(per))


# --- runs (the training logs) ---


@dataclass(frozen=True)
class RunStats:
    """What the readings need from one ES run's logs."""

    label: str
    arm: str
    opt_seed: int
    generation: int
    g_tr: float
    f_tr: float
    improvement: float | None  # center(last) - center(last - 10) on the monitor set
    theta_sha256: str = ""

    @property
    def clears(self) -> bool:
        return self.g_tr > self.f_tr

    @property
    def improving(self) -> bool:
        """Still improving: improvement >= F_tr, or unknown (no monitor 10 generations back)."""
        return self.improvement is None or self.improvement >= self.f_tr

    @property
    def stopped(self) -> bool:
        return not self.improving

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "arm": self.arm, "opt_seed": self.opt_seed,
                "generation": self.generation, "g_tr": self.g_tr, "f_tr": self.f_tr,
                "improvement": self.improvement, "theta_sha256": self.theta_sha256}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunStats":
        return cls(str(d["label"]), str(d["arm"]), int(d["opt_seed"]), int(d["generation"]),
                   float(d["g_tr"]), float(d["f_tr"]),
                   None if d["improvement"] is None else float(d["improvement"]),
                   str(d.get("theta_sha256", "")))


def monitor_units(rows: Sequence[Mapping[str, str]], generation: int
                  ) -> dict[str, dict[str, Units]]:
    """monitor key (world#episode) -> {"center", "A0"} -> episode -> replicate -> units, from a
    run's monitor.csv: the center at `generation`, A0's own rows."""
    out: dict[str, dict[str, Units]] = {}
    for d in rows:
        if d["member"] == es.CENTER and d["generation"] == str(generation):
            arm = "center"
        elif d["member"] == es.A0_LABEL:
            arm = A0
        else:
            continue
        key = f"{d['world']}#{d['episode']}"
        out.setdefault(key, {}).setdefault(arm, {}).setdefault(
            int(d["episode"]), {})[int(d["replicate"])] = float(d["units"])
    return out


def center_mean(rows: Sequence[Mapping[str, str]], generation: int) -> float | None:
    u = [float(d["units"]) for d in rows
         if d["member"] == es.CENTER and d["generation"] == str(generation)]
    return statistics.fmean(u) if u else None


def run_stats(label: str, arm: str, opt_seed: int, generation: int,
              rows: Sequence[Mapping[str, str]], cells: Mapping[str, Cell],
              bars: Mapping[str, float], theta_sha: str = "",
              history: Sequence[Mapping[str, Any]] = ()) -> tuple[RunStats, MeanComp]:
    """g_tr, F_tr (center vs A0 on the monitor set, both R = 2) and the last-10-generation
    improvement of one run. The center mean 10 generations back comes from this dir's rows, or,
    for a run resumed past it (the main run continues the pilot), from its checkpoint's monitor
    history; every history entry this dir has rows for must equal them."""
    units = monitor_units(rows, generation)
    keys = sorted(units)
    reps = {"center": ow.MONITOR_REPLICATES, A0: ow.MONITOR_REPLICATES}
    comp = mean_comp(units, cells, "center", A0, reps, bars, keys)
    if comp is None:
        raise ValueError(f"{label}: the monitor set lacks the center at generation {generation} "
                         "or A0 on some world-episode")
    recorded = {int(m["generation"]): m.get("center_units") for m in history}
    for g, units_g in recorded.items():
        mine = center_mean(rows, g)
        if mine is not None and (units_g is None or abs(mine - float(units_g)) > 1e-9):
            raise ValueError(f"{label}: monitor history at generation {g} ({units_g}) differs "
                             f"from monitor.csv ({mine})")
    now = center_mean(rows, generation)
    back = center_mean(rows, generation - IMPROVEMENT_WINDOW)
    if back is None and recorded.get(generation - IMPROVEMENT_WINDOW) is not None:
        back = float(recorded[generation - IMPROVEMENT_WINDOW])  # type: ignore[arg-type]
    imp = None if now is None or back is None else now - back
    return RunStats(label, arm, opt_seed, generation, comp.diff, comp.floor, imp,
                    theta_sha), comp


# --- candidate reads (pure) ---


@dataclass(frozen=True)
class CandidateRead:
    """One candidate on the 12-producer loaded validation worlds."""

    label: str
    mean: float
    comps: dict[str, MeanComp]  # ref (A0 / RR / MWW / TR / ablation) -> comparison
    worlds: int
    lost_to_rr: int
    deaths: int
    rundown_share: float
    starved_share: float
    rr_rundown_share: float
    rr_starved_share: float

    @property
    def g_val(self) -> float:
        return self.comps[A0].diff

    @property
    def success_clauses(self) -> dict[str, bool]:
        return {"beats_refs": all(self.comps[r].clears for r in SUCCESS_REFS),
                "lost_to_rr": self.lost_to_rr * 6 <= self.worlds,
                "no_deaths": self.deaths == 0,
                "flags_no_worse": (self.rundown_share <= self.rr_rundown_share
                                   and self.starved_share <= self.rr_starved_share)}

    @property
    def success(self) -> bool:
        return all(self.success_clauses.values())

    @property
    def plateau_iii(self) -> bool:
        return any(self.comps[r].diff <= self.comps[r].floor for r in (MWW, RR))

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "mean": self.mean, "g_val": self.g_val,
                "worlds": self.worlds, "lost_to_rr": self.lost_to_rr, "deaths": self.deaths,
                "rundown_share": self.rundown_share, "starved_share": self.starved_share,
                "rr_rundown_share": self.rr_rundown_share,
                "rr_starved_share": self.rr_starved_share,
                "success_clauses": self.success_clauses, "plateau_iii": self.plateau_iii,
                "comparisons": {r: c.to_dict() for r, c in self.comps.items()}}


def row_flags(d: Mapping[str, str]) -> tuple[bool, bool, bool]:
    """(dead, run down, starved) of one gate row against the registered X / Y."""
    dead = d["dead"] in ("true", "True")
    rs, re_ = es._opt(d["r_start"]), es._opt(d["r_end"])
    rundown = rs is not None and re_ is not None and re_ < ow.X_RUNDOWN * rs
    fills = json.loads(d["consumer_fill_rate"])
    starved = any(v is not None and v < ow.Y_STARVATION for v in fills.values())
    return dead, rundown, starved


def flag_shares(dicts: Sequence[Mapping[str, str]], arm: str, worlds: Sequence[str]
                ) -> tuple[int, float, float]:
    ws = set(worlds)
    rows = [row_flags(d) for d in dicts if d["arm"] == arm and d["world"] in ws]
    if not rows:
        return 0, math.nan, math.nan
    return (sum(f[0] for f in rows), sum(f[1] for f in rows) / len(rows),
            sum(f[2] for f in rows) / len(rows))


def candidate_read(label: str, units: Mapping[str, Mapping[str, Units]],
                   cells: Mapping[str, Cell], reps: Mapping[str, int],
                   bars: Mapping[str, float], worlds: Sequence[str],
                   dicts: Sequence[Mapping[str, str]]) -> CandidateRead | None:
    comps: dict[str, MeanComp] = {}
    for ref, other in ((A0, A0), (RR, RR), (MWW, MWW), (TR, TR),
                       (ABLATION, ablation_label(label))):
        c = mean_comp(units, cells, label, other, reps, bars, worlds)
        if c is None:
            return None
        comps[ref] = c
    rr = comps[RR]
    lost = sum(1 for w in rr.per_world if -w.diff > w.floor)
    deaths, rd, st = flag_shares(dicts, label, worlds)
    _, rr_rd, rr_st = flag_shares(dicts, RR, worlds)
    return CandidateRead(label, statistics.fmean(world_mean(units[w][label]) for w in worlds),
                         comps, len(worlds), lost, deaths, rd, st, rr_rd, rr_st)


# --- labels (pure) ---


def metric_units(dicts: Sequence[Mapping[str, str]], column: str, count: str
                 ) -> dict[str, dict[str, Units]]:
    """world -> arm -> episode -> replicate -> metric, rows whose count column is 0 skipped
    (undefined there, carry-forward 2.1)."""
    return arm_units([d for d in dicts if float(d.get(count) or 0) > 0], column=column)


def deconcentration(label: str, munits: Mapping[str, Mapping[str, Units]],
                    worlds: Sequence[str]) -> dict[str, Any]:
    """Candidate - A0 of the concurrent top-producer share, paired per (episode, replicate) on
    the rows where both are defined; the own pooled SE as the floors'."""
    per = []
    for w in worlds:
        u = munits.get(w, {})
        a, b = u.get(label, {}), u.get(A0, {})
        keys = [(e, r) for e in sorted(set(a) & set(b)) for r in sorted(set(a[e]) & set(b[e]))]
        if not keys:
            continue
        diffs = [a[e][r] - b[e][r] for e, r in keys]
        per.append((statistics.fmean(diffs), diffs))
    if not per:
        return {"defined_worlds": 0, "diff": None, "se3": None,
                "not_via_deconcentration": True}
    ss = sum(sum((x - m) ** 2 for x in d) for m, d in per if len(d) >= 2)
    df = sum(len(d) - 1 for m, d in per if len(d) >= 2)
    se3 = (LABEL_SE * math.sqrt(sum((ss / df) / len(d) for _, d in per)) / len(per)
           if df else 0.0)
    diff = statistics.fmean(m for m, _ in per)
    return {"defined_worlds": len(per), "diff": diff, "se3": se3, "df": df,
            "not_via_deconcentration": not diff < -se3}


def herding(label: str, munits: Mapping[str, Mapping[str, Units]],
            worlds: Sequence[str]) -> tuple[bool, list[dict[str, Any]]]:
    """Per loaded world: the candidate's share_swing world mean - MWW's against 3 SE_w of the
    candidate's own rows; `herding` when flagged on >= 1/3 of the worlds."""
    rows, flagged = [], 0
    for w in worlds:
        u = munits.get(w, {})
        cand = [x for reps in u.get(label, {}).values() for x in reps.values()]
        mww = [x for reps in u.get(MWW, {}).values() for x in reps.values()]
        defined = len(cand) >= 2 and bool(mww)
        se3 = LABEL_SE * statistics.stdev(cand) / math.sqrt(len(cand)) if defined else None
        diff = statistics.fmean(cand) - statistics.fmean(mww) if defined else None
        flag = bool(defined and diff > se3)
        flagged += flag
        rows.append({"candidate": label, "world": w, "candidate_swing": statistics.fmean(cand)
                     if cand else None, "mww_swing": statistics.fmean(mww) if mww else None,
                     "diff": diff, "se3": se3, "rows": len(cand), "defined": defined,
                     "flagged": flag})
    return bool(worlds) and flagged * 3 >= len(worlds), rows


def candidate_labels(read: CandidateRead, deconc: Mapping[str, Any], herd: bool) -> dict:
    ab = read.comps[ABLATION]
    return {NOT_VIA_DECONCENTRATION: bool(deconc["not_via_deconcentration"]),
            HISTORY_INERT: ab.within, HERDING: herd}


def labels_for(reading: str, flags: Mapping[str, bool]) -> list[str]:
    """The labels D7's table records with this reading (none outside success / plateau)."""
    return [lab for lab in LABELS_WITH.get(reading, ()) if flags.get(lab)]


# --- readings (pure) ---


def first_match(clauses: Mapping[str, bool]) -> str:
    for r in PRECEDENCE[:-1]:
        if clauses[r]:
            return r
    return OTHER


def select(reads: Mapping[str, CandidateRead], a2_vs_a1: MeanComp | None,
           runs: Mapping[str, RunStats]) -> str:
    """D6 selection among seed-0 candidates: the highest world mean; A2 within F_val of A1 goes
    to A1."""
    best = max(sorted(reads), key=lambda k: reads[k].mean)
    a1 = next((k for k in reads if runs[k].arm == "A1"), None)
    if a1 is not None and best != a1 and a2_vs_a1 is not None and a2_vs_a1.within:
        return a1
    return best


def seed0_reading(runs: Mapping[str, RunStats], reads: Mapping[str, CandidateRead],
                  selected: str, a2_vs_a1: MeanComp | None) -> dict[str, Any]:
    sel, run = reads[selected], runs[selected]
    clause_i = all(r.stopped for r in runs.values())
    clause_ii = a2_vs_a1 is not None and a2_vs_a1.within
    success = sel.success
    clauses = {
        UNRESOLVED: all(not r.clears for r in runs.values()),
        KILL: run.clears and sel.g_val < run.g_tr * KILL_RATIO,
        SUCCESS: success,
        PLATEAU: clause_i and clause_ii and sel.plateau_iii,
        BUDGET_LIMITED: not success and any(r.improving for r in runs.values()),
    }
    reading = first_match(clauses)
    return {"reading": reading, "next": NEXT[reading], "selected": selected,
            "clauses": clauses, "plateau": {"i": clause_i, "ii": clause_ii,
                                            "iii": sel.plateau_iii},
            "g_tr": run.g_tr, "f_tr": run.f_tr, "g_val": sel.g_val,
            "a2_vs_a1": None if a2_vs_a1 is None else a2_vs_a1.to_dict()}


def seed1_reading(run1: RunStats, read1: CandidateRead, seed0: Mapping[str, Any]
                  ) -> dict[str, Any]:
    """The seed-1 candidate with the same table; clause (ii) from seed 0, clause (i) = seed 0's
    and the seed-1 run's own stop; budget from the seed-1 run alone."""
    clause_i = bool(seed0["plateau"]["i"]) and run1.stopped
    clause_ii = bool(seed0["plateau"]["ii"])
    success = read1.success
    clauses = {
        UNRESOLVED: not run1.clears,
        KILL: run1.clears and read1.g_val < run1.g_tr * KILL_RATIO,
        SUCCESS: success,
        PLATEAU: clause_i and clause_ii and read1.plateau_iii,
        BUDGET_LIMITED: not success and run1.improving,
    }
    reading = first_match(clauses)
    return {"reading": reading, "next": NEXT[reading], "candidate": run1.label,
            "clauses": clauses, "plateau": {"i": clause_i, "ii": clause_ii,
                                            "iii": read1.plateau_iii},
            "g_tr": run1.g_tr, "f_tr": run1.f_tr, "g_val": read1.g_val}


def combine(seed0: str, seed1: str) -> str:
    """D7's seed combination table."""
    if seed0 not in (SUCCESS, PLATEAU):
        raise ValueError(f"seed 1 runs only after success / plateau, seed 0 read {seed0!r}")
    if seed1 in (SUCCESS, PLATEAU):
        return seed1 if seed1 == seed0 else SEED_DISAGREEMENT
    return seed1


# --- the analysis ---


def world_cells(views: Sequence[Mapping], spreads: Mapping[str, Any]) -> dict[str, Cell]:
    out = {}
    for v in views:
        key = f"{v['size']}/{v['choice']}"
        out[v["world"]] = Cell(v["world"], int(v["size"]), v["choice"],
                               space_info(v["record"]).differing,
                               spread_from_dict(spreads[key]) if key in spreads else None,
                               float(v["rho"]), v["class"])
    return out


def monitor_cells(training: Mapping[str, Any], spreads: Mapping[str, Any]
                  ) -> dict[str, Cell]:
    """Each monitor world-episode as a cell (its world's stratum and differing)."""
    out = {}
    for w in training["worlds"]:
        info = space_info(w["record"])
        key = f"{info.size}/{info.choice}"
        for ep in w["episodes"]:
            k = f"{w['world']}#{int(ep)}"
            out[k] = Cell(k, info.size, info.choice, info.differing,
                          spread_from_dict(spreads[key]) if key in spreads else None)
    return out


def reps_of(labels: Sequence[str]) -> dict[str, int]:
    out = {A0: STOCHASTIC_R, MWW: STOCHASTIC_R, RR: 1, TR: 1}
    for lab in labels:
        out[lab] = STOCHASTIC_R
        out[ablation_label(lab)] = STOCHASTIC_R
    return out


def validation_bars(units: Mapping[str, Mapping[str, Units]], worlds: Sequence[str]
                    ) -> dict[str, float]:
    return {w: max((world_mean(units[w][r]) for r in REFERENCES if r in units.get(w, {})),
                   default=0.0) for w in worlds}


def rho_quartiles(cells: Mapping[str, Cell]) -> dict[str, str]:
    out = {}
    for size in sorted({c.size for c in cells.values()}):
        cs = [c for c in cells.values() if c.size == size]
        cuts = quartile_cuts([c.rho for c in cs])
        for c in cs:
            out[c.world] = f"q{1 + sum(1 for x in cuts if c.rho > x)}"
    return out


def gating_worlds(cells: Mapping[str, Cell]) -> list[str]:
    return sorted(w for w, c in cells.items() if c.size == READING_SIZE and c.group == LOADED)


def analyse(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Cell],
            runs: Mapping[str, RunStats], *, seed_combine: bool = False,
            seed0_runs: Mapping[str, RunStats] | None = None,
            carried_ii: "MeanComp | None" = None) -> dict[str, Any]:
    """The whole reading from the rows and the runs' stats. Without `seed_combine`, `runs` are
    the seed-0 runs; with it, `seed0_runs` are the seed-0 runs (from the seed-0 read) and `runs`
    holds the seed-1 run (and may hold the selected seed-0 run, ignored)."""
    units = arm_units(dicts)
    gate = gating_worlds(cells)
    s0 = dict(seed0_runs) if seed_combine else dict(runs)
    labels_all = sorted(set(s0) | set(runs))
    reps = reps_of(labels_all)
    bars = validation_bars(units, list(cells))
    reads: dict[str, CandidateRead] = {}
    for lab in labels_all:
        r = candidate_read(lab, units, cells, reps, bars, gate, dicts)
        if r is None:
            raise ValueError(f"candidate {lab}: rows missing on a gating world")
        reads[lab] = r
    a1 = next((k for k in sorted(s0) if s0[k].arm == "A1"), None)
    a2 = next((k for k in sorted(s0) if s0[k].arm in ("A2", "A2-R2")), None)
    a2_vs_a1 = (mean_comp(units, cells, a2, a1, reps, bars, gate)
                if a1 is not None and a2 is not None else None)
    if a2_vs_a1 is None and a2 is None and carried_ii is not None:
        a2_vs_a1 = carried_ii  # v3: A2 not re-run, clause (ii) carried (VALIDATION_V3)
    selected = select({k: reads[k] for k in s0}, a2_vs_a1, s0)
    seed0 = seed0_reading(s0, reads, selected, a2_vs_a1)
    munits_top = metric_units(dicts, TOP_SHARE, TOP_SHARE_N)
    munits_swing = metric_units(dicts, SWING, SWING_N)
    flags, herd_rows, deconc = {}, [], {}
    for lab in labels_all:
        deconc[lab] = deconcentration(lab, munits_top, gate)
        herd, rows = herding(lab, munits_swing, gate)
        herd_rows += rows
        flags[lab] = candidate_labels(reads[lab], deconc[lab], herd)
    seed0["labels"] = labels_for(seed0["reading"], flags[selected])
    out: dict[str, Any] = {
        "runs": {k: v.to_dict() for k, v in sorted({**s0, **runs}.items())},
        "candidates": {k: v.to_dict() for k, v in reads.items()},
        "label_measures": {k: {**flags[k], "deconcentration": deconc[k]} for k in labels_all},
        "gating_worlds": gate, "seed0": seed0, "seed_combine": seed_combine,
    }
    if not seed_combine:
        out["final"] = {"reading": seed0["reading"], "next": seed0["next"],
                        "labels": seed0["labels"], "selected": selected}
    else:
        if seed0["reading"] not in (SUCCESS, PLATEAU):
            raise ValueError(f"--seed-combine after a seed-0 {seed0['reading']!r} read")
        s1 = [k for k in runs if runs[k].opt_seed == 1]
        if len(s1) != 1 or runs[s1[0]].arm != s0[selected].arm:
            raise ValueError(f"the seed-1 read needs exactly one seed-1 run of the selected arm "
                             f"{s0[selected].arm}, got {s1}")
        seed1 = seed1_reading(runs[s1[0]], reads[s1[0]], seed0)
        seed1["labels"] = labels_for(seed1["reading"], flags[s1[0]])
        final = combine(seed0["reading"], seed1["reading"])
        labels = sorted({*labels_for(final, flags[selected]), *labels_for(final, flags[s1[0]])})
        out["seed1"] = seed1
        out["final"] = {"reading": final, "labels": labels, "selected": selected,
                        "labels_by_seed": {selected: labels_for(final, flags[selected]),
                                           s1[0]: labels_for(final, flags[s1[0]])},
                        "next": FINAL_NEXT.get(final, NEXT.get(final, ""))}
    out["herding_rows"] = herd_rows
    return out


# --- tables ---

VALIDATION_COLUMNS = ("scope", "size", "group", "rho_quartile", "world", "rho", "arm", "worlds",
                      "episodes", "rows", "units", "deaths", "rundown_share", "starved_share",
                      *(f"d_{r}" for r in (*SUCCESS_REFS, ABLATION)),
                      *(f"floor_{r}" for r in (*SUCCESS_REFS, ABLATION)))
CONCENTRATION_COLUMNS = ("world", "size", "group", "arm", "top_share", "top_share_rows",
                         "dispersion", "dispersion_rows", "share_swing", "share_swing_rows")
HERDING_COLUMNS = ("candidate", "world", "candidate_swing", "mww_swing", "diff", "se3", "rows",
                   "defined", "flagged")


def _ref_of(label: str, ref: str) -> str:
    return ablation_label(label) if ref == ABLATION else ref


def validation_rows(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Cell],
                    candidates: Sequence[str]) -> list[dict[str, Any]]:
    """Per world and arm, then per candidate the summaries by size x group and size x rho
    quartile (D9 tables; only the 12-producer loaded group gates)."""
    units = arm_units(dicts)
    reps = reps_of(candidates)
    bars = validation_bars(units, list(cells))
    q = rho_quartiles(cells)
    out: list[dict[str, Any]] = []
    for w in sorted(cells, key=lambda x: (cells[x].size, x)):
        c = cells[w]
        for arm in sorted(units.get(w, {})):
            u = units[w][arm]
            deaths, rd, st = flag_shares(dicts, arm, [w])
            row: dict[str, Any] = {
                "scope": "world", "size": c.size, "group": c.group, "rho_quartile": q[w],
                "world": w, "rho": c.rho, "arm": arm, "worlds": 1, "episodes": len(u),
                "rows": sum(len(r) for r in u.values()), "units": world_mean(u),
                "deaths": deaths, "rundown_share": rd, "starved_share": st}
            if arm in candidates:
                for ref in (*SUCCESS_REFS, ABLATION):
                    other = _ref_of(arm, ref)
                    if other in units[w]:
                        stoch = [reps[x] for x in (arm, other) if reps.get(x, 1) > 1]
                        wc = world_comp(u, units[w][other], c, stoch, bars[w])
                        row[f"d_{ref}"], row[f"floor_{ref}"] = wc.diff, wc.floor
            out.append(row)
    for lab in candidates:
        groups: dict[tuple[str, int, str], list[str]] = {}
        for w, c in cells.items():
            groups.setdefault(("group", c.size, c.group), []).append(w)
            groups.setdefault(("rho_quartile", c.size, q[w]), []).append(w)
        for (scope, size, key), ws in sorted(groups.items()):
            ws = sorted(w for w in ws if lab in units.get(w, {}))
            if not ws:
                continue
            row = {"scope": scope, "size": size, "arm": lab, "worlds": len(ws),
                   "group" if scope == "group" else "rho_quartile": key,
                   "units": statistics.fmean(world_mean(units[w][lab]) for w in ws)}
            for ref in (*SUCCESS_REFS, ABLATION):
                mc = mean_comp(units, cells, lab, _ref_of(lab, ref), reps, bars, ws)
                if mc is not None:
                    row[f"d_{ref}"], row[f"floor_{ref}"] = mc.diff, mc.floor
            out.append(row)
    return out


def _metric_mean(dicts: Sequence[Mapping[str, str]], w: str, arm: str, col: str,
                 count: str) -> tuple[float | None, int]:
    vals = [float(d[col]) for d in dicts if d["world"] == w and d["arm"] == arm
            and float(d.get(count) or 0) > 0]
    return (statistics.fmean(vals) if vals else None), len(vals)


def concentration_rows(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Cell]
                       ) -> list[dict[str, Any]]:
    out = []
    arms = sorted({d["arm"] for d in dicts} - {TR})  # D7: candidate, ablation, A0, MWW, RR
    for w in sorted(cells, key=lambda x: (cells[x].size, x)):
        for arm in arms:
            top, nt = _metric_mean(dicts, w, arm, TOP_SHARE, TOP_SHARE_N)
            disp, nd = _metric_mean(dicts, w, arm, DISPERSION, DISPERSION_N)
            sw, ns = _metric_mean(dicts, w, arm, SWING, SWING_N)
            if nt or nd or ns:
                out.append({"world": w, "size": cells[w].size, "group": cells[w].group,
                            "arm": arm, "top_share": top, "top_share_rows": nt,
                            "dispersion": disp, "dispersion_rows": nd, "share_swing": sw,
                            "share_swing_rows": ns})
    return out


# --- power rule (pure; task 5.4 applies it) ---


def detectable_effect(diffs_by_world: Mapping[str, Sequence[float]], mean_bar: float,
                      episodes: int) -> dict[str, Any]:
    """MDE(n) = 3 s / sqrt(n) / sqrt(W), s = pooled within-world SD of the per-episode paired
    differences (worlds with >= 2 episodes), as a share of the mean bar."""
    ss, df = 0.0, 0
    for d in diffs_by_world.values():
        if len(d) >= 2:
            m = statistics.fmean(d)
            ss += sum((x - m) ** 2 for x in d)
            df += len(d) - 1
    if df == 0:
        raise ValueError("no world with >= 2 episodes: the pooled SD is undefined")
    s = math.sqrt(ss / df)
    mde = FLOOR_SE * s / math.sqrt(episodes) / math.sqrt(len(diffs_by_world))
    return {"s": s, "df": df, "worlds": len(diffs_by_world), "episodes": episodes, "mde": mde,
            "share": mde / mean_bar}


def power_rule(effects: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    """effects: pair -> {4: detectable_effect, 8: ...}; the registered max over the pairs decides
    (4 episodes when <= 2% at 4, else 8; > 3% at 8 -> the user)."""
    deciding = max(sorted(effects), key=lambda p: effects[p][4]["share"])
    s4, s8 = effects[deciding][4]["share"], effects[deciding][8]["share"]
    return {"deciding_pair": deciding, "share_4": s4, "share_8": s8,
            "episodes": 4 if s4 <= POWER_FOUR else 8, "to_user": s4 > POWER_FOUR and
            s8 > POWER_EIGHT}


def power_from_rows(rows: Sequence[Mapping[str, str]], worlds: Sequence[str],
                    bars: Mapping[str, float]) -> dict[str, Any]:
    """The power rule on EXP-0052-style rows (subjects `POWER_SUBJECTS`) of `worlds`."""
    by_subject = {v: k for k, v in POWER_SUBJECTS.items()}
    units: dict[str, dict[str, Units]] = {}
    for d in rows:
        arm = by_subject.get(d["subject"])
        if arm is None or d["world"] not in worlds:
            continue
        units.setdefault(d["world"], {}).setdefault(arm, {}).setdefault(
            int(d["episode"]), {})[int(d["replicate"])] = float(d["units"])
    mean_bar = statistics.fmean(bars[w] for w in worlds)
    effects: dict[str, dict[int, dict[str, Any]]] = {}
    for a, b in POWER_PAIRS:
        diffs = {}
        for w in worlds:
            ua, ub = units[w][a], units[w][b]
            if set(ua) != set(ub):
                raise ValueError(f"{w}: episodes of {a} and {b} differ")
            diffs[w] = [statistics.fmean(ua[e].values()) - statistics.fmean(ub[e].values())
                        for e in sorted(ua)]
        effects[f"{a}-{b}"] = {n: detectable_effect(diffs, mean_bar, n) for n in (4, 8)}
    return {"mean_bar": mean_bar, "effects": effects, "rule": power_rule(effects)}


# --- sample, arms, jobs ---


def validation_views(atlas: Mapping, worlds: Sequence[str] | None = None,
                     max_worlds: int | None = None) -> list[dict]:
    """Every atlas v2 validation world at 12 and 42 producers (D9), or a dry run's explicit
    validation worlds; any other split is refused, dry runs included (held-out protection)."""
    views = {}
    for entry in atlas["worlds"]:
        v = world_view(entry)
        v["rho"] = float(entry["record"]["coordinates"]["rho"])
        views[v["world"]] = v
    if worlds:
        unknown = sorted(set(worlds) - set(views))
        if unknown:
            raise ValueError(f"worlds {unknown} are not in the atlas")
        chosen = [views[w] for w in worlds]
    else:
        chosen = sorted((v for v in views.values() if v["split"] == VALIDATION),
                        key=lambda v: (v["size"], v["world"]))
    for v in chosen:
        if v["split"] != VALIDATION:
            raise ValueError(f"refused: {v['world']} is in the {v['split']} split; validation "
                             "reads the validation split only")
        if v["size"] > es.MAX_READ_SIZE:
            raise ValueError(f"refused: {v['world']} has {v['size']} producers (> "
                             f"{es.MAX_READ_SIZE}, no override)")
        if v["size"] not in SIZES:
            raise ValueError(f"refused: {v['world']} has {v['size']} producers")
    return chosen[:max_worlds] if max_worlds is not None else chosen


def world_entry(view: Mapping, episodes: int, shock_profile: str | None = None) -> WorldEntry:
    rec = view["record"]
    point = point_from_record(rec)
    if shock_profile is None:
        config_from_record(rec)  # the recorded config hash must rebuild
    else:
        point = replace(point, shock_window=tuple(SMOKE_START_WINDOW))
    seeds = episode_seeds(space_info(rec).id, episodes, VALIDATION_STAGE)
    return space_entry(point, seeds, split=VALIDATION)


@dataclass(frozen=True)
class Candidate:
    """A staged run's candidate (its composite) and the run's stats."""

    label: str
    arm: str
    opt_seed: int
    theta: tuple[float, ...]
    theta_sha256: str
    normaliser: dict
    normaliser_sha256: str
    candidate_sha256: str
    run_dir: str
    stats: RunStats
    training_sha256: str
    prereg_sha256: str | None
    kind: str
    prereg_version: int = 1  # the run's training rule (6.1f); no v2 pin = v1
    base_kind: str = BASE_TS  # its composite's frozen base (task 7.6)


def legacy_table() -> list[dict]:
    """RR, MWW (R = 2) and TR as routing_audit router entries (logits-v1)."""
    table = [
        {"router": LEGACY_ROUTER[RR], "subject": f"{PRODUCER}/{LEGACY_ROUTER[RR]}",
         "producing": PRODUCER, "ordering": LEGACY_ROUTER[RR], "contract": "joint_stack",
         "random": False, "roster": False, "kind": ROUTER_KIND[LEGACY_ROUTER[RR]],
         "trained": False, "arm": RR},
        {"router": LEGACY_ROUTER[MWW], "subject": f"{PRODUCER}/{LEGACY_ROUTER[MWW]}",
         "producing": PRODUCER, "ordering": LEGACY_ROUTER[MWW], "contract": "joint_stack",
         "random": True, "roster": False, "kind": ROUTER_KIND[LEGACY_ROUTER[MWW]],
         "trained": False, "replicates": STOCHASTIC_R, "arm": MWW},
        {**_trained_entry(TRAINED_ORDERER), "arm": TR},
    ]
    if ROUTER_KIND[LEGACY_ROUTER[MWW]] != STOCHASTIC:
        raise AssertionError("MWW is no longer stochastic")
    for a in table:
        a["decode"] = LOGITS_V1
    return table


def _kind(kind: str) -> dict[str, str]:
    """A job / table field for the MWW base only, so TS (v1 / v2) entries stay byte-identical."""
    return {"base_kind": kind} if kind == BASE_MWW else {}


def per_order_table(cands: Sequence[Candidate], base_kind: str = BASE_TS,
                    reported: Sequence["Reported"] = ()) -> list[dict]:
    """A0 (on the read's base), then each candidate and its shared-only ablation, then each
    reported-only arm (task 7.6, no ablation) (choice-v1, R = 2)."""
    out = [{"arm": A0, "subject": ow.A0_SUBJECT, "po_arm": A0, "theta": [],
            "history_mode": AGENT, "decode": CHOICE_V1, **_kind(base_kind)}]
    for c in cands:
        for label, mode in ((c.label, AGENT), (ablation_label(c.label), SHARED_ONLY)):
            out.append({"arm": label, "subject": f"es/{label}", "po_arm": c.arm,
                        "theta": list(c.theta), "theta_sha256": c.theta_sha256,
                        "normaliser": c.normaliser, "normaliser_sha256": c.normaliser_sha256,
                        "history_mode": mode, "decode": CHOICE_V1, **_kind(c.base_kind)})
    for r in reported:
        out.append({"arm": r.label, "subject": f"es/{r.label}", "po_arm": r.arm,
                    "theta": list(r.theta), "theta_sha256": r.theta_sha256,
                    "normaliser": r.normaliser, "normaliser_sha256": r.normaliser_sha256,
                    "history_mode": AGENT, "decode": CHOICE_V1, **_kind(r.base_kind)})
    for a in out:
        a.setdefault("theta_sha256", theta_sha256(np.zeros(0)))
    return out


def build_validation_jobs(entries: Sequence[WorldEntry], legacy: Sequence[dict],
                          per_order: Sequence[dict], horizon: int,
                          checkpoints: Mapping[str, Mapping[str, str]]) -> list[Job]:
    jobs = build_jobs(entries, legacy, 1, horizon, None, checkpoints)
    by_subject = {a["subject"]: a for a in legacy}
    for j in jobs:
        j.update({"arm": by_subject[j["subject"]]["arm"], "per_order": False,
                  "decode": LOGITS_V1})
    for e in entries:
        for ep in e.episode_seeds:
            for a in per_order:
                for rep in range(STOCHASTIC_R):
                    jobs.append({
                        "world": e.id, "size": e.size, "regime": e.regime,
                        "world_seed": e.world_seed, "generator": e.generator, "episode": ep,
                        "subject": a["subject"], "replicate": rep, "producing": PRODUCER,
                        "ordering": "per_order", "contract": "per_order_choice_v1",
                        "horizon": horizon, "action_seed": None,
                        "checkpoints": {"orderer": dict(checkpoints["orderer"])},
                        "per_order": True, **{k: a[k] for k in (
                            "arm", "po_arm", "theta", "theta_sha256", "history_mode",
                            "decode")},
                        "normaliser": a.get("normaliser"),
                        "normaliser_sha256": a.get("normaliser_sha256"),
                        **_kind(a.get("base_kind", BASE_TS))})
    return jobs


def job_cost(job: Job) -> float:
    """Longest-first estimate: size^2, a torch-scored arm 3x, the shared-only ablation's
    per-product history mean another 1.5x (cost ~ agents per product)."""
    return ((job["size"] / READING_SIZE) ** 2 * (3.0 if job.get("checkpoints") else 1.0)
            * (1.5 if job.get("history_mode") == SHARED_ONLY else 1.0))


def longest_first(jobs: Sequence[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: -job_cost(j))


# --- worker ---


def _per_order_episode(job: Job) -> Any:
    from ..eval.baselines import PRODUCING_BASELINES
    from ..eval.runner import run_episode
    from .per_order_scorer import ARMS, per_order_env

    env = per_order_env(job_config(job), job["horizon"])
    set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
    tracer = RoutingTracer()  # no forks in validation (DIAGNOSTICS 22)
    tracer.attach(env)
    monitor = ContractMonitor(env)
    arm = ARMS[job["po_arm"]]
    kind = job.get("base_kind", BASE_TS)
    args = (ow._base_scorer(job), arm, np.asarray(job["theta"], np.float64),
            es._normaliser(job) if arm.residual else None, job["history_mode"])
    policy = PerOrderPolicy(*args) if kind == BASE_TS else make_policy(*args, kind)
    if theta_sha256(policy.theta) != job["theta_sha256"]:
        raise AssertionError("job theta does not match its sha256")
    run_episode(env, PRODUCING_BASELINES[job["producing"]], policy, seed=job["episode"],
                action_seed=job["action_seed"], monitor=monitor)
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
    return env, tracer


def run_validate_job(job: Job) -> dict[str, str]:
    """One traced episode's row, or its failure row (anything raising, NonFiniteChoice too)."""
    try:
        started = time.perf_counter()
        if job["per_order"]:
            env, tracer = _per_order_episode(job)
            secs = time.perf_counter() - started
        else:
            env, tracer, secs, _ = ow._bar_episode(job)
        decode = getattr(env, "ordering_decode", LOGITS_V1)
        if decode != job["decode"]:
            raise AssertionError(f"{job['arm']} ran under {decode}, pinned {job['decode']}")
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         env.episode_stats))
        row.update(trace_cells(tracer))
        row.update({"arm": job["arm"], "ordering_decode": decode,
                    "history_mode": job.get("history_mode") or "",
                    "theta_sha256": job.get("theta_sha256") or "",
                    "episode_seconds": repr(secs),
                    "job_seconds": repr(time.perf_counter() - started)})
        return row
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


# --- inputs: runs, the bar, a seed-0 read ---


def load_candidate(run_dir: str | Path, inp: es.Inputs, cells: Mapping[str, Cell],
                   bars: Mapping[str, float], orderer_path: str) -> Candidate:
    """A complete order_es train run on this bar: its candidate composite and its stats."""
    from .per_order_scorer import load_composite

    rd = Path(run_dir)
    spec = json.loads((rd / "spec.json").read_text())
    if spec.get("driver") != es.DRIVER or spec.get("status") != "complete":
        raise ValueError(f"{rd} is not a complete {es.DRIVER} run")
    tp = spec["pins"]["training"]
    for k, v in inp.hashes.items():
        if tp["inputs"].get(k) != v:
            raise ValueError(f"{rd} was trained on another bar ({k} differs)")
    comp = json.loads((rd / "candidate.json").read_text())
    # the composite's own checks (versions, base hash, theta / normaliser hashes)
    loaded = load_composite(rd / "candidate.json", orderer_path,
                            SimpleNamespace(ordering_decode=CHOICE_V1))
    if comp["theta_sha256"] != spec["result"]["theta_sha256"]:
        raise ValueError(f"{rd}: candidate.json theta differs from the run's final theta")
    if comp["normaliser_sha256"] != inp.normaliser.sha256:
        raise ValueError(f"{rd}: candidate normaliser is not the bar's")
    arm, seed = str(tp["arm"]["name"]), int(tp["opt_seed"])
    gen = int(spec["result"]["generation"])
    label = run_label(arm, seed)
    rows = es.read_csv(rd / "monitor.csv")
    ckpt = es.load_checkpoint(rd)  # the latest; its monitor history spans a resume
    if int(ckpt["generation"]) != gen or ckpt["theta_sha256"] != comp["theta_sha256"]:
        raise ValueError(f"{rd}: the latest checkpoint is not the run's final generation")
    stats, _ = run_stats(label, arm, seed, gen, rows, cells, bars, comp["theta_sha256"],
                         ckpt["monitor"])
    recorded = next((m for m in reversed(spec["result"].get("monitor") or [])), None)
    if recorded is not None and recorded.get("g_tr_raw") is not None and \
            abs(float(recorded["g_tr_raw"]) - stats.g_tr) > 1e-6:
        raise ValueError(f"{rd}: g_tr from monitor.csv {stats.g_tr} differs from the run's "
                         f"monitor summary {recorded['g_tr_raw']}")
    return Candidate(label, arm, seed, tuple(float(x) for x in comp["theta"]),
                     comp["theta_sha256"], comp["normaliser"], comp["normaliser_sha256"],
                     sha256_file(str(rd / "candidate.json")), str(rd), stats,
                     spec["pins"]["training_sha256"], tp["inputs"].get("prereg_sha256"),
                     str(spec.get("kind")), run_prereg_version(tp), base_kind_of(loaded))



@dataclass(frozen=True)
class Reported:
    """A reported-only arm (task 7.6, pre-registration v3 `validation.reported_arms`): a
    composite played on the read's worlds and keys, compared, never selected or read."""

    label: str
    arm: str
    theta: tuple[float, ...]
    theta_sha256: str
    normaliser: dict
    normaliser_sha256: str
    candidate_sha256: str
    base_kind: str


def reported_label(name: str) -> str:
    return "reported:" + name.replace(" ", "-")


def load_reported(name: str, run_dir: str | Path, orderer_path: str) -> Reported:
    """A registered reported arm's composite (its own base and normaliser), by its
    `candidate.json`; no bar, monitor or training-version checks (it is never read)."""
    from .per_order_scorer import load_composite

    path = Path(run_dir) / "candidate.json"
    pol = load_composite(path, orderer_path, SimpleNamespace(ordering_decode=CHOICE_V1))
    comp = json.loads(path.read_text())
    return Reported(reported_label(name), pol.arm.name, tuple(float(x) for x in comp["theta"]),
                    comp["theta_sha256"], comp["normaliser"], comp["normaliser_sha256"],
                    sha256_file(str(path)), base_kind_of(pol))


def reported_problems(reported: Sequence[Reported], version: int) -> list[str]:
    """A real read's reported arms must be exactly the registered ones (v3 only)."""
    reg = (es.registered_sections(version).get("validation") or {}).get("reported_arms") or {}
    want = {reported_label(k): v for k, v in reg.items()}
    got = {r.label: {"theta_sha256": r.theta_sha256, "candidate_sha256": r.candidate_sha256,
                     "base": r.base_kind} for r in reported}
    out = [f"reported arms {sorted(got)} != the registered {sorted(want)}"] \
        if set(got) != set(want) else []
    for k in sorted(set(got) & set(want)):
        for f in ("theta_sha256", "candidate_sha256", "base"):
            if got[k][f] != want[k][f]:
                out.append(f"reported arm {k}: {f} {str(got[k][f])[:12]} is not the registered "
                           f"{str(want[k][f])[:12]}")
    return out


def carried_ii_for(version: int) -> "MeanComp | None":
    """v3's carried plateau clause (ii) (A2 not re-run): EXP-0060 val-seed0's A2-R2 - A1."""
    reg = (es.registered_sections(version).get("validation") or {}).get("plateau_ii")
    if not reg:
        return None
    d = reg["a2_vs_a1"]
    return MeanComp(d["a"], d["b"], float(d["diff"]), int(d["worlds"]), float(d["atlas_floor"]),
                    float(d["own3se"]), int(d["df"]), ())


def reported_comparisons(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Cell],
                         labels: Sequence[str], candidates: Sequence[str]) -> dict[str, Any]:
    """Each reported arm against A0, RR, MWW, TR and every candidate, on the gating worlds with
    D7's floors (reported only)."""
    units = arm_units(dicts)
    gate = gating_worlds(cells)
    bars = validation_bars(units, list(cells))
    reps = reps_of([*labels, *candidates])
    out: dict[str, Any] = {}
    for lab in labels:
        out[lab] = {}
        for ref in (A0, RR, MWW, TR, *candidates):
            comp = mean_comp(units, cells, lab, ref, reps, bars, gate)
            out[lab][ref] = None if comp is None else comp.to_dict()
    return out


@dataclass(frozen=True)
class SeedZero:
    """A complete seed-0 read reused by the seed-1 read (`--reuse-references`)."""

    path: Path
    spec: dict
    reading: dict
    rows: list[dict[str, str]]
    episodes_sha256: str
    reading_sha256: str

    @property
    def runs(self) -> dict[str, RunStats]:
        return {k: RunStats.from_dict(v) for k, v in self.reading["runs"].items()
                if int(v["opt_seed"]) == 0}


REUSE_PINS = ("horizon", "dry_run", "episodes", "shock_profile", "atlas", "bar", "orderer",
              "prereg_sha256", "producer", "sample", "source_tree_sha256", "spreads")


def load_seed_zero(path: str | Path) -> SeedZero:
    p = Path(path)
    spec = json.loads((p / "spec.json").read_text())
    if spec.get("driver") != DRIVER or spec.get("status") != "complete":
        raise ValueError(f"{p} is not a complete {DRIVER} read")
    if spec["pins"].get("seed_combine"):
        raise ValueError(f"{p} is a seed-1 read; --reuse-references takes the seed-0 read")
    reading = json.loads((p / "reading.json").read_text())
    if reading.get("status") != "valid":
        raise ValueError(f"{p}/reading.json is not a valid reading")
    return SeedZero(p, spec, reading, read_episode_dicts(p / "episodes.csv"),
                    sha256_file(str(p / "episodes.csv")), sha256_file(str(p / "reading.json")))


ANALYSIS_KEYS = ("runs", "candidates", "label_measures", "gating_worlds", "seed0", "seed1",
                 "final", "seed_combine", "reported")


def _analysis_view(doc: Mapping[str, Any]) -> dict[str, Any]:
    return es._json_norm({k: doc.get(k) for k in ANALYSIS_KEYS})


def real_run_problems(*, atlas: Mapping, inp: es.Inputs, prereg: Mapping, prereg_sha: str,
                      cands: Sequence[Candidate], horizon: int, episodes: int,
                      source_commit: str, orderer_sha: str, cells: Mapping[str, Cell],
                      filters: Mapping[str, Any]) -> list[str]:
    counts = {s: sum(1 for c in cells.values() if c.size == s and c.group == LOADED)
              for s in SIZES}
    out = [msg for bad, msg in (
        (prereg_sha not in FROZEN_PREREGS,
         f"pre-registration file {prereg_sha[:12]} is not the frozen {PREREG_SHA256[:12]} "
         f"(v1) or {PREREG_V2_SHA256[:12]} (v2)"),
        (atlas.get("content_hash") != REGISTERED_ATLAS_HASH,
         f"atlas {str(atlas.get('content_hash'))[:12]} is not the registered EXP-0052 "
         f"reclassified atlas {REGISTERED_ATLAS_HASH[:12]}"),
        (orderer_sha != ROLE_CHECKPOINTS["orderer"],
         f"orderer checkpoint {orderer_sha[:12]} is not the registered "
         f"{ROLE_CHECKPOINTS['orderer'][:12]}"),
        (inp.dry_run or inp.bar_spec.get("kind") != "bar", "the bar run is a dry run"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (episodes not in EPISODE_CHOICES,
         f"--episodes {episodes} not in the power rule's {EPISODE_CHOICES}"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (counts != EXPECTED_LOADED,
         f"loaded validation worlds {counts} != the registered {EXPECTED_LOADED}"),
        (es._sha(inp.normaliser_doc.get("prereg")) !=
         readings_registration()["normaliser_pins"]["rule_sha256"],
         "the bar's normaliser rule block (fitting code included) is not the frozen one"),
    ) if bad]
    out += prereg_read_problems(prereg)
    for c in cands:
        if c.kind != es.TRAIN:
            out.append(f"run {c.label} is a {c.kind!r} run, a real read needs train runs")
        if c.prereg_sha256 != prereg_sha or prereg_sha not in FROZEN_PREREGS:
            out.append(f"run {c.label} was trained under pre-registration "
                       f"{str(c.prereg_sha256)[:12]}, not the frozen {prereg_sha[:12]}")
        if c.stats.generation != MAIN_GENERATIONS:
            out.append(f"run {c.label} ended at generation {c.stats.generation}, the main runs "
                       f"train {MAIN_GENERATIONS}")
    out += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return out


DRY_BANNER = ("> **DRY RUN — not a validation read.** Filtered worlds, another horizon or "
              "episode count, the smoke shock profile or dry-run inputs.\n\n")
SKELETON_END = "## Conclusions\n\nTODO.\n"
RESULT_SKELETON = """# {title}

{banner}Driver: `python -m {driver}` (add-m4-orderer-redesign D6 / D7 / D9). Status: {status}.

## Question

Does outcome-trained per-order routing beat A0 (its zero residual), `round_robin`,
`makeable_weighted_whole` and the trained orderer on the 12-producer loaded validation worlds,
by more than each comparison's floor, without deaths, run-down or starvation worse than RR's?

## Result

- Rows: {n_rows} over {n_worlds} worlds x {episodes} episodes; arms: {arms}.
- Reading: {reading} (next: {next_}); labels: {labels}.
- Tables: `validation.csv`, `concentration.csv`, `herding.csv`; `reading.json` holds every
  clause and floor (the rules: the pre-registration's `readings` section).

{end}"""


def _write_result(out: Path, *, dry_run: bool, status: str, rows: int, worlds: int,
                  episodes: int, arms: Sequence[str], reading: Mapping[str, Any]) -> None:
    path = out / "RESULT.md"
    if path.exists() and not path.read_text().endswith(SKELETON_END):
        return  # filled records are never overwritten
    final = reading.get("final") or {}
    path.write_text(RESULT_SKELETON.format(
        title=out.name, banner=DRY_BANNER if dry_run else "", driver=DRIVER, status=status,
        n_rows=rows, n_worlds=worlds, episodes=episodes, arms=", ".join(arms),
        reading=final.get("reading", "withheld"), next_=final.get("next", "-"),
        labels=", ".join(final.get("labels") or []) or "none", end=SKELETON_END))


def run_validate(*, runs: Sequence[str], bar_dir: str | Path, atlas_path: str | Path,
                 orderer_ckpt: str, out_dir: str | Path, episodes: int, workers: int = 1,
                 prereg_path: str | Path | None = None, seed_combine: bool = False,
                 reuse_references: str | Path | None = None, horizon: int = HORIZON,
                 worlds: Sequence[str] | None = None, max_worlds: int | None = None,
                 shock_profile: str | None = None, source_commit: str = "",
                 dry_run: bool = False, reported_runs: Mapping[str, str] | None = None,
                 run_job_fn: Callable[[Job], dict[str, str]] = run_validate_job,
                 log: Callable[[str], None] = _log) -> int:
    """Play (or resume) one validation read into `out_dir`; exit 0, or `CONTRACT_INVALID` on
    contract failures, missing rows or digest differences. `reported_runs` (v3: registered name
    -> run dir) adds reported-only arms (task 7.6)."""
    from .per_order_scorer import BaseScorer, per_order_pins

    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    if seed_combine != (reuse_references is not None):
        raise ValueError("--seed-combine and --reuse-references go together")
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")
    out = Path(out_dir)
    for r in runs:
        if Path(r).resolve() == out.resolve():
            raise ValueError("--out-dir must differ from every --runs dir")
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    views = validation_views(atlas, worlds, max_worlds)
    spreads = atlas["spreads"]
    cells = world_cells(views, spreads)
    inp = es.load_inputs(bar_dir)
    mcells = monitor_cells(inp.training, spreads)
    mbars = {f"{w}#{ep}": b.bar for (w, ep), b in inp.bar.items()}
    pin = checked_orderer(out, orderer_ckpt, log)
    cands = [load_candidate(r, inp, mcells, mbars, pin["path"]) for r in runs]
    if len({c.label for c in cands}) != len(cands):
        raise ValueError(f"two runs share a label: {[c.label for c in cands]}")
    prereg_path = Path(prereg_path) if prereg_path else PREREG_PATH
    prereg = es.load_prereg(prereg_path)
    prereg_sha = sha256_file(str(prereg_path))
    if es.prereg_schema_problems(prereg):
        raise ValueError("pre-registration: " + "; ".join(es.prereg_schema_problems(prereg)))
    version = read_prereg_version(prereg_sha, prereg)
    mix = prereg_mix_problems(cands, version)
    if mix:
        raise ValueError("; ".join(mix))
    base_kind = es.base_kind_for(version)
    if es.bar_base_kind(inp) != base_kind:
        raise ValueError(f"the bar's A0 plays the {es.bar_base_kind(inp)!r} base, pre-registration "
                         f"v{version} reads the {base_kind!r} base (task 7.6)")
    wrong = [c.label for c in cands if c.base_kind != base_kind]
    if wrong:
        raise ValueError(f"runs {wrong} correct another base than the read's {base_kind!r}")
    reported = [load_reported(k, d, pin["path"]) for k, d in sorted((reported_runs or {}).items())]
    if reported and version < 3:
        raise ValueError(f"pre-registration v{version} registers no reported arm")
    source = source_fingerprint(source_commit or None)
    pins: dict[str, Any] = {
        "driver": DRIVER, "mode": VALIDATE, "horizon": horizon, "dry_run": dry_run,
        "episodes": episodes, "shock_profile": shock_profile,
        "atlas": {"sha256": sha256_file(str(atlas_path)), "content_hash": atlas["content_hash"]},
        "bar": inp.hashes, "orderer": pin["sha256"], "prereg_sha256": prereg_sha,
        "producer": PRODUCER, "spreads": spreads,
        "sample": [{"world": v["world"], "size": v["size"], "class": v["class"],
                    "rho": v["rho"], "choice": v["choice"], "pattern": v["pattern"],
                    "shocks": v["shocks"], "split": v["split"],
                    "differing": cells[v["world"]].differing} for v in views],
        "source_tree_sha256": source["tree_sha256"], "source_commit": source["commit"],
        "seed_combine": seed_combine,
    }
    seed0 = None
    if seed_combine:
        seed0 = load_seed_zero(reuse_references)  # type: ignore[arg-type]
        diff = [k for k in REUSE_PINS if seed0.spec["pins"].get(k) != es._json_norm(pins[k])]
        if diff:
            raise ValueError(f"--reuse-references {seed0.path} was read with other pins: {diff}")
        if seed0.reading["seed0"]["reading"] not in (SUCCESS, PLATEAU):
            raise ValueError("seed 1 runs only after a seed-0 success / plateau")
        old = seed0.reading["runs"]
        for c in cands:
            if c.opt_seed == 0 and (c.label not in old
                                    or old[c.label]["theta_sha256"] != c.theta_sha256):
                raise ValueError(f"run {c.label} is not the seed-0 read's candidate")
        played = [c for c in cands if c.opt_seed == 1]
        if len(played) != 1:
            raise ValueError("the seed-1 read plays exactly one seed-1 run")
        pins["reuse"] = {"path": str(seed0.path), "episodes_sha256": seed0.episodes_sha256,
                         "reading_sha256": seed0.reading_sha256}
    else:
        if not cands or any(c.opt_seed != 0 for c in cands):
            raise ValueError("the seed-0 read takes seed-0 runs only (seed 1: --seed-combine)")
        if len({c.arm for c in cands}) != len(cands):
            raise ValueError("one seed-0 run per arm")
        played = list(cands)
    if not dry_run:
        problems = real_run_problems(
            atlas=atlas, inp=inp, prereg=prereg, prereg_sha=prereg_sha, cands=cands,
            horizon=horizon, episodes=episodes, source_commit=source_commit,
            orderer_sha=pin["sha256"], cells=cells,
            filters={"worlds": worlds, "max_worlds": max_worlds,
                     "shock_profile": shock_profile})
        problems += reported_problems(reported, version) if version >= 3 else []
        if problems:
            raise ValueError("not a valid validation read (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    entries = [world_entry(v, episodes, shock_profile) for v in views]
    grid = job_grid(entries, None)
    legacy = legacy_table()
    po_all = per_order_table(played, base_kind, reported)
    if seed_combine:
        po_all = [a for a in po_all if a["arm"] != A0]  # A0 rows come from the seed-0 read
        legacy = []
    checkpoints = {"orderer": pin}
    jobs = build_validation_jobs(entries, legacy, po_all, horizon, checkpoints)
    expected: set[Key] = {job_key(j["world"], j["episode"], j["subject"], j["replicate"])
                          for j in jobs}
    reused: list[dict[str, str]] = []
    if seed0 is not None:
        reused = seed0.rows
        expected |= {job_key(d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
                     for d in reused}
    base = BaseScorer.load(pin["path"])
    per_order = {}
    for c in played:
        for label, mode in ((c.label, AGENT), (ablation_label(c.label), SHARED_ONLY)):
            from .per_order_scorer import FeatureNormaliser

            args = (base, c.arm, np.asarray(c.theta, np.float64),
                    FeatureNormaliser.from_dict(c.normaliser), mode)
            pol = PerOrderPolicy(*args) if c.base_kind == BASE_TS else make_policy(*args,
                                                                                c.base_kind)
            per_order[label] = per_order_pins(pol)["per_order"]
    pins.update({
        "candidates": {c.label: {"arm": c.arm, "opt_seed": c.opt_seed,
                                 "theta_sha256": c.theta_sha256,
                                 "candidate_sha256": c.candidate_sha256,
                                 "normaliser_sha256": c.normaliser_sha256,
                                 "training_sha256": c.training_sha256,
                                 "generation": c.stats.generation} for c in cands},
        **decode_pins(CHOICE_V1),
        "decode_by_arm": {a["arm"]: a["decode"] for a in [*legacy_table(),
                                                          *per_order_table(played, base_kind,
                                                                           reported)]},
        "per_order": per_order, "grid": grid,
        **({"reported": {r.label: {"theta_sha256": r.theta_sha256,
                                   "candidate_sha256": r.candidate_sha256,
                                   "base": r.base_kind} for r in reported}}
           if reported else {}),
    })
    pins = es._json_norm(pins)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else VALIDATE,
        "acceptance_read": False, "pins": pins,
        "inputs": {"runs": [str(r) for r in runs], "bar_dir": str(bar_dir),
                   "atlas": str(atlas_path), "orderer": pin, "prereg": str(prereg_path),
                   "reuse_references": None if reuse_references is None
                   else str(reuse_references)},
        "source": source, "decode_pins": decode_pins(CHOICE_V1),
        "pre_registered": {"design": "add-m4-orderer-redesign D6 / D7 / D9 (task 5.3)",
                           "prereg_sha256": prereg_sha, "readings": readings_registration()},
    })
    ensure_failures_file(out)
    episodes_path = out / "episodes.csv"
    if reused:
        have = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
        new = [d for d in reused if job_key(d["world"], int(d["episode"]), d["subject"],
                                            int(d["replicate"])) not in have]
        if new:
            fresh = not episodes_path.exists() or not episodes_path.stat().st_size
            with episodes_path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
                if fresh:
                    w.writeheader()
                for d in new:
                    w.writerow({k: d.get(k, "") for k in COLUMNS})
            log(f"[order_es_validate] reused {len(new)} rows of {seed0.path}")  # type: ignore
    rows_now = read_rows(episodes_path)
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows_now}
    if read_failures(out) and old_spec is not None:
        log("[order_es_validate] contract_failures.csv holds rows of an earlier session")
    try:
        execute(out, spec, done_keys, read_episode_dicts(episodes_path), longest_first(jobs),
                workers, run_job_fn, log, columns=COLUMNS)
    except DigestMismatch as exc:  # recorded as a run row by execute; the run ends invalid
        log(f"[order_es_validate] INVALID: {exc}")
    rows = read_rows(episodes_path)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes_path)
    failures = read_failures(out)
    invalid = bool(failures or missing)
    reading: dict[str, Any] = {"driver": DRIVER, "dry_run": dry_run, "prereg_sha256": prereg_sha,
                               "rules": readings_registration()}
    if invalid:
        reading.update({"status": INVALID, "readings_withheld": True,
                        "contract_failures": len(failures), "missing_rows": len(missing)})
    else:
        runs_now = {c.label: c.stats for c in cands}
        try:
            got = analyse(dicts, cells, runs_now, seed_combine=seed_combine,
                          seed0_runs=seed0.runs if seed0 is not None else None,
                          carried_ii=carried_ii_for(version))
            if reported:
                got["reported"] = reported_comparisons(
                    dicts, cells, [r.label for r in reported], [c.label for c in played])
        except ValueError as exc:
            if not dry_run:
                raise
            got = {"not_read": str(exc)}
        herd_rows = got.pop("herding_rows", [])
        reading.update(got)
        reading.update({"status": "valid", "readings_withheld": False})
        labels = sorted({*reused_labels(seed0), *(c.label for c in played)})
        write_csv(out / "validation.csv", VALIDATION_COLUMNS,
                  validation_rows(dicts, cells, labels))
        write_csv(out / "concentration.csv", CONCENTRATION_COLUMNS,
                  concentration_rows(dicts, cells))
        write_csv(out / "herding.csv", HERDING_COLUMNS, herd_rows)
        if seed0 is not None and "seed0" in got:
            recorded = seed0.reading["seed0"]["reading"]
            if got["seed0"]["reading"] != recorded:
                raise ValueError(f"the seed-0 reading recomputes as {got['seed0']['reading']}, "
                                 f"{seed0.path} recorded {recorded}")
    (out / "reading.json").write_text(json.dumps(reading, indent=2, sort_keys=True,
                                                 default=str) + "\n")
    if invalid:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing)}
        status, code = (f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing "
                        f"row(s))"), CONTRACT_INVALID
    else:
        final = reading.get("final") or {}
        spec["status"] = "complete"
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          "acceptance_read": False, "reading": final.get("reading"),
                          "labels": final.get("labels"),
                          "reading_sha256": sha256_file(str(out / "reading.json"))}
        status, code = "complete", 0
    _write_spec(out, spec)
    _write_result(out, dry_run=dry_run, status=status, rows=len(rows), worlds=len(grid),
                  episodes=episodes, arms=sorted({d["arm"] for d in dicts}), reading=reading)
    final = reading.get("final") or {}
    log(f"[order_es_validate] {'DRY RUN ' if dry_run else ''}{status}"
        + (f", reading {final.get('reading')}, labels {final.get('labels')}" if code == 0
           else ""))
    return code


def reused_labels(seed0: SeedZero | None) -> list[str]:
    return [] if seed0 is None else sorted(seed0.reading["runs"])


def run_recompute(*, out_dir: str | Path, runs: Sequence[str], bar_dir: str | Path,
                  orderer_ckpt: str, log: Callable[[str], None] = _log) -> int:
    """Rebuild the reading from `episodes.csv` and the runs' logs; MATCH -> 0, else 2. Writes
    only `recompute.json`."""
    out = Path(out_dir)
    spec = json.loads((out / "spec.json").read_text())
    if spec.get("driver") != DRIVER:
        raise ValueError(f"{out} is not a {DRIVER} read")
    pins = spec["pins"]
    recorded = json.loads((out / "reading.json").read_text())
    inp = es.load_inputs(bar_dir)
    if es._json_norm(inp.hashes) != pins["bar"]:
        raise ValueError("--bar-dir is not the read's bar")
    spreads = pins["spreads"]
    mcells = monitor_cells(inp.training, spreads)
    mbars = {f"{w}#{ep}": b.bar for (w, ep), b in inp.bar.items()}
    cands = [load_candidate(r, inp, mcells, mbars, orderer_ckpt) for r in runs]
    for c in cands:
        if pins["candidates"].get(c.label, {}).get("theta_sha256") != c.theta_sha256:
            raise ValueError(f"run {c.label} is not a candidate of this read")
    version = FROZEN_PREREGS.get(str(pins.get("prereg_sha256")))
    mix = [] if version is None else prereg_mix_problems(cands, version)
    if mix:
        raise ValueError("; ".join(mix))
    cells = {s["world"]: Cell(s["world"], int(s["size"]), s["choice"], bool(s["differing"]),
                              spread_from_dict(spreads[f"{s['size']}/{s['choice']}"])
                              if f"{s['size']}/{s['choice']}" in spreads else None,
                              float(s["rho"]), s["class"]) for s in pins["sample"]}
    seed0_runs = None
    if pins["seed_combine"]:
        seed0_runs = load_seed_zero(pins["reuse"]["path"]).runs
    dicts = read_episode_dicts(out / "episodes.csv")
    got = analyse(dicts, cells,
                  {c.label: c.stats for c in cands}, seed_combine=bool(pins["seed_combine"]),
                  seed0_runs=seed0_runs, carried_ii=None if version is None
                  else carried_ii_for(version))
    if pins.get("reported"):
        got["reported"] = reported_comparisons(dicts, cells, sorted(pins["reported"]),
                                               [c.label for c in cands if c.opt_seed == (
                                                   1 if pins["seed_combine"] else 0)])
    got.pop("herding_rows", None)
    match = _analysis_view(got) == _analysis_view(recorded)
    doc = {"status": "MATCH" if match else "MISMATCH",
           "reading": (got.get("final") or {}).get("reading"),
           "recorded": (recorded.get("final") or {}).get("reading")}
    (out / "recompute.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    log(f"[order_es_validate] recompute {doc['status']}: {doc['reading']}")
    return 0 if match else CONTRACT_INVALID


def run_power(*, exp0052_episodes: str | Path, atlas_path: str | Path,
              atlas_worlds_csv: str | Path, out: str | Path | None,
              prereg_path: str | Path | None = None,
              log: Callable[[str], None] = _log) -> dict[str, Any]:
    """The frozen power rule on EXP-0052's confirm rows (task 5.4 records the outcome). The rule
    is `readings.power_rule`, the same in v1 and v2; `prereg_path` (default v1) only says which
    frozen file the record pins (task 6.1f)."""
    prereg_path = Path(prereg_path) if prereg_path else PREREG_PATH
    prereg_sha = sha256_file(str(prereg_path))
    if prereg_sha not in FROZEN_PREREGS or prereg_read_problems(es.load_prereg(prereg_path)):
        raise ValueError(f"--prereg {prereg_path}: not a frozen pre-registration "
                         f"({prereg_sha[:12]})")
    atlas = load_atlas(atlas_path)
    cells = world_cells(validation_views(atlas), atlas["spreads"])
    worlds = gating_worlds(cells)
    with Path(atlas_worlds_csv).open(newline="") as fh:
        bars = {d["world"]: float(d["bar"]) for d in csv.DictReader(fh) if d["world"] in worlds}
    doc = power_from_rows(read_episode_dicts(Path(exp0052_episodes)), worlds, bars)
    doc.update({"worlds": worlds, "inputs": {
        "exp0052_episodes_sha256": sha256_file(str(exp0052_episodes)),
        "atlas_content_hash": atlas["content_hash"],
        "atlas_worlds_csv_sha256": sha256_file(str(atlas_worlds_csv))},
        "prereg_sha256": prereg_sha,
        "rule": {**doc["rule"], "registered": readings_registration()["power_rule"]["rule"]}})
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if out is not None:
        Path(out).write_text(text)
    log(text)
    return doc


def run_freeze(path: str | Path = PREREG_PATH, log: Callable[[str], None] = _log,
               version: int = 1) -> int:
    """Write the frozen pre-registration (`version`) if absent; else compare (MATCH -> 0, else
    2). Never overwrites."""
    p = Path(path)
    text = prereg_text(frozen_prereg(version))
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        log(f"[order_es_validate] wrote {p} sha256 {sha256_file(str(p))}")
        return 0
    if p.read_text() == text:
        log(f"[order_es_validate] {p} MATCH sha256 {sha256_file(str(p))}")
        return 0
    diff = es._diff_paths(json.loads(p.read_text()), json.loads(text))
    log(f"[order_es_validate] {p} MISMATCH at {diff[:8]}")
    return CONTRACT_INVALID


def _csv_strs(s: str | None) -> list[str] | None:
    return None if s is None else [x for x in s.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=(VALIDATE, RECOMPUTE, POWER, FREEZE), default=VALIDATE)
    p.add_argument("--runs", help="comma-separated order_es train run dirs")
    p.add_argument("--bar-dir", help="the bar run (EXP-0058 bar/) the runs trained on")
    p.add_argument("--atlas", help="atlas_v2.json (EXP-0052 reclassified)")
    p.add_argument("--orderer-ckpt", help="the trained orderer (EXP-0012)")
    p.add_argument("--episodes", type=int, help="episodes per validation world (4 or 8)")
    p.add_argument("--prereg", help="validate / power: the frozen pre-registration (default: v1; "
                                    "v2-trained runs need es_prereg_m4_v2.json)")
    p.add_argument("--seed-combine", action="store_true")
    p.add_argument("--reuse-references", help="the seed-0 read's dir (with --seed-combine)")
    p.add_argument("--out-dir", help="validate / recompute dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worlds", help="dry run: comma-separated validation world ids")
    p.add_argument("--max-worlds", type=int, help="dry run: the first N validation worlds")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,))
    p.add_argument("--exp0052-episodes", help="power: EXP-0052's episodes.csv")
    p.add_argument("--atlas-worlds", help="power: EXP-0052 reclassified/worlds.csv (bar)")
    p.add_argument("--out", help="power: output JSON; freeze: the file (default packaged)")
    p.add_argument("--reported-runs", help="v3 validate: NAME=DIR,... reported-only arms "
                                           "(registered names, e.g. 'EXP-0060 A1-s0'=<dir>)")
    p.add_argument("--prereg-version", type=int, choices=(1, 2, 3), default=1,
                   help="freeze: which pre-registration (v2: task 6.1d)")
    a = p.parse_args(argv)
    if a.mode == FREEZE:
        return run_freeze(a.out or {1: PREREG_PATH, 2: PREREG_V2_PATH,
                                     3: PREREG_V3_PATH}[a.prereg_version],
                          version=a.prereg_version)
    if a.mode == POWER:
        run_power(exp0052_episodes=a.exp0052_episodes, atlas_path=a.atlas,
                  atlas_worlds_csv=a.atlas_worlds, out=a.out, prereg_path=a.prereg)
        return 0
    runs = _csv_strs(a.runs) or []
    if a.mode == RECOMPUTE:
        return run_recompute(out_dir=a.out_dir, runs=runs, bar_dir=a.bar_dir,
                             orderer_ckpt=a.orderer_ckpt)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    try:
        return run_validate(
            runs=runs, bar_dir=a.bar_dir, atlas_path=a.atlas, orderer_ckpt=a.orderer_ckpt,
            out_dir=a.out_dir, episodes=a.episodes, workers=a.workers, prereg_path=a.prereg,
            seed_combine=a.seed_combine, reuse_references=a.reuse_references,
            reported_runs=dict(x.split("=", 1) for x in _csv_strs(a.reported_runs) or []),
            horizon=a.horizon, worlds=_csv_strs(a.worlds), max_worlds=a.max_worlds,
            shock_profile=a.shock_profile, source_commit=commit, dry_run=a.dry_run)
    except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
