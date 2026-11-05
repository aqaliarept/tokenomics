"""M5 ES driver for the producer residual: `python -m tokenomics.train.producer_es`
(add-m5-potential-game-gnn task 2.2, design D3 / D5 / D6 / D7).

Training set (D7, supervisor decision 2026-11-05): M4's pinned ES training worlds and schedule
(EXP-0058 `bar/`: 64 worlds at 12 producers, 128 world-episodes with fresh episode seeds, the
16-world-episode monitor set, per-seed minibatches, the probe's minibatches), hash-checked. Every
job plays the producer under test with the frozen MWWk orderer (`mww_keyed`) at DrawKey(world,
episode, replicate), joint_stack, horizon 100; replicate 0 for the bar, the members and the monitor
(common random numbers: the key never names the member), 1..4 for the noise floor.

Modes:
- `bar` (L1): the base `entity_planner_buyahead_k4_rf3` on every training world-episode at r0 ->
  `bar.csv` (order_es's bar format: B_we = the base's consumer units, its guards and digests) and
  `base_rows.csv`. With `--atlas` and `--exp0079-episodes`, rf3 is also replayed on 2 EXP-0057
  world-episodes through the supply-trace job path that wrote EXP-0079, and must equal its rows
  (`crosscheck.csv`; design D7).
- `noise-floor` (L1, D5): the base re-keyed at r1..r4 plus r0 again on every world-episode. r0 must
  reproduce the bar's rows (`reproduction.csv`). The margin is order_es's v4 formula (c x sqrt 2 x
  the pooled within-world-episode SD of the run-down ratio and of each consumer's fill), with
  the base's self-flag rate against its own r0 under it (target <= 5%) -> `margins.json`.
- `probe` (L2): the ES estimate at theta = 0 on the two disjoint probe minibatches (24
  world-episodes each, M4's recorded probe stream) with one common noise draw; their cosine reads
  `signal` (>= 0.3) / `weak` (>= 0.1) / `no_signal` -> `probe.json`.
- `train` (L3 / L4): antithetic ES (PAIRS pairs, sigma per parameter), centered ranks over the 2n
  members, a disqualified member ranks worst (pre-registration v3's flag-count rule: a death where
  rf3 dies at no key, or more flagged episodes than rf3's re-keys on the same minibatch by more
  than FLAG_THRESHOLD; `score_member_v3`), Adam ascent. The minibatch
  per generation comes from M4's recorded stream for the optimiser seed. One checkpoint per
  generation; resume needs every pin to match. The monitor (the centre and the bar on the
  16-world-episode monitor set) every MONITOR_EVERY generations.

Fitness (D3, supervisor decision 2026-11-05): consumer units only, relative to the bar,
units / B_we (order_es `we_fitness` with h = 0); no holding term. Holding units per cycle and the
investment rejection rate are reported per member, generation and read as diagnostics, never in the
fitness. The `stock_heavy` sanity check belongs to the reads (task 2.4).

Constants marked "registered by 2.3" are frozen in the pre-registration (task 2.3); a real run
will need it. A real run needs a 40-hex source commit, horizon 100 and no filters; anything else
needs `--dry-run`. Contract failures, missing rows or a digest / reproduction mismatch: exit 2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from ..eval.contract import (ContractMonitor, append_failure, ensure_failures_file,
                             failure_from_dict, is_failure, job_failure, read_failures,
                             run_failure)
from ..eval.gate import CSV_COLUMNS, row_from_stats, row_to_dict
from ..eval.gate_cli import CONTRACT_INVALID, Job
from ..eval.provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .producer_residual import (BASE_PRODUCER, N_IN, N_P, THETA_DIM, Normaliser,
                                producer_residual_policy)

DRIVER = "tokenomics.train.producer_es"
RESIDUAL = "m5_residual"
ORDERING = "mww_keyed"
HORIZON = 100
TRAINING_SIZE = 12
# registered by 2.3 (the pre-registration freezes these values)
PAIRS = 12  # 24 members
MINIBATCH = 24  # DIAGNOSTICS 25
SIGMA = 0.05  # per parameter, on the normalised features (M4's sigma; task 2.3)
SIGMA_ARMS = {"S": 1.0, "S2": 2.0}  # the probe's arms: sigma and 2 sigma (EXP-0069's A1-S2)
MONITOR_REPLICATES = (0, 1)  # M4's monitor: replicates 0 and 1, from generation 0
ADAM_LR = 0.02  # order_es's Adam (beta1 0.9, beta2 0.999, eps 1e-8), weight decay 0.005
GENERATIONS = {"pilot": 20, "main": 40}  # DIAGNOSTICS 27
MONITOR_EVERY = 10
NOISE_REPLICATES = (1, 2, 3, 4)
SIGNAL, WEAK = 0.3, 0.1
SELF_FLAG_TARGET = 0.05
NOISE_STREAM_PURPOSE = {"train": "m5-train", "probe": "m5-probe"}
BAR, NOISE, PROBE, TRAIN = "bar", "noise-floor", "probe", "train"
MODES = (BAR, NOISE, PROBE, TRAIN)
ES_COLUMNS = ("job_id", "kind", "generation", "member", "sign", "pair", "batch", "theta_sha256",
              "producing", "nonfinite", "episode_seconds", "job_seconds")
ROW_COLUMNS = CSV_COLUMNS + ES_COLUMNS
CENTER = "c"
BASE_LABEL = "base"
ROW_FILES = {BAR: "bar_episodes.csv", NOISE: "noise_episodes.csv", PROBE: "probe_episodes.csv",
             TRAIN: "members.csv"}
CHECKPOINT_FORMAT = "m5-es-checkpoint-v1"
# The frozen pre-registration (task 2.3) every real probe / train run needs; None until 2.3 pins it,
# so a real L2 / L3 run is refused (review m5-2-2).
# v1 (task 2.3) is never edited; v2 (task 2.4, before any data) adds L6's candidate selection and
# its gating worlds; v3 (after EXP-0081) replaces the guard. The code is pinned to v3.
PREREG_V1_PATH = Path(__file__).resolve().parent / "registry" / "es_prereg_m5_v1.json"
PREREG_V1_SHA256 = "67721b3828931c7904616368465f03b8febad7362d7e039bf8e5124478c18623"
PREREG_V2_PATH = Path(__file__).resolve().parent / "registry" / "es_prereg_m5_v2.json"
PREREG_V2_SHA256 = "5887b2b272ed5829fa1374166cbdf13fbb533aad9766386a331796e71942dbf0"
# v3 (2026-11-05, after EXP-0081, before EXP-0082): the flag-count guard (user decision).
PREREG_PATH = Path(__file__).resolve().parent / "registry" / "es_prereg_m5_v3.json"
# v3, frozen 2026-11-05 (task 4.2b)
PREREG_SHA256: str | None = "97c3332c3584ce6fd2d84fe40cac5c9e03eb4f8e59fdd76d1d036defef7b9d04"


PREREG_FORMAT = "m5-es-prereg-v1"
PREREG_VERSION = 3
TRAINING_PINS = {  # M4's pinned ES training set (EXP-0058 bar/), design D7
    "training_worlds_sha256": "02abf7f7e247a4e68cc56460cd6ce9a0c335f2fd5af2454dcbfc0049295109d9",
    "schedule_sha256": "122ebe64f117cc67892336f2e9da47819161b741ec4fd9a6ece143b67c72ca38"}


def registered_prereg(version: int = PREREG_VERSION) -> dict[str, Any]:
    """The M5 pre-registration (task 2.3), built from this module's constants: every value cites
    its source; `carried` marks values taken from M4 unchanged and not re-tuned for the producer."""
    from . import order_es as oe
    from . import order_es_worlds as ow
    from .producer_residual import INPUT_FEATURES, K_CLIP, PRODUCER_FEATURES

    doc = {
        "format": PREREG_FORMAT,
        "change": "add-m5-potential-game-gnn",
        "status": "frozen before L2 (design D6); a real probe / train run refuses any other file",
        "decisions": "notes/user-decisions.md (2026-11-05) and notes/2.1-2.3.md (supervisor)",
        "base": {"producer": BASE_PRODUCER, "orderer": ORDERING,
                 "source": "EXP-0079 rf_helps (design D8); theta = 0 equals it action for action "
                           "(tests/test_producer_residual.py)"},
        "policy": {
            "theta_dim": THETA_DIM, "input_features": list(INPUT_FEATURES),
            "producer_features": list(PRODUCER_FEATURES),
            "buffer": f"k_g = 4 exp(clip(theta_k . phi_g, +-{K_CLIP}))",
            "extra": "round(max(0, theta_b . phi_g)) units on top of the planner's want",
            "expansion": "the base's triggers or theta_e . psi > 0 (configured, not running, "
                         "affordable in the planner's spending order)",
            "initial_theta": "0 (the base)", "source": "design D2, notes/2.1.md"},
        "normaliser": {"rule": NORMALISER_RULE, "clip": 5.0,
                       "source": "M4's normaliser (es_prereg_m4_v4 `normaliser`); supervisor "
                                 "decision 2026-11-05 (notes/2.3.md); pinned by the bar run's "
                                 "normaliser.json SHA-256"},
        "worlds": {"training": {**TRAINING_PINS, "world_episodes": 128, "size": TRAINING_SIZE,
                                "source": "design D7 (supervisor decision 2026-11-05)"},
                   "horizon": HORIZON, "order_ttl": 2, "orderer_keys": "DrawKey(world, episode, "
                   "replicate); replicate 0 for bar / members, 0 and 1 for the monitor, 1..4 for "
                   "the noise floor"},
        "es": {
            "pairs": PAIRS, "minibatch": MINIBATCH,
            "minibatch_source": "M4's recorded stream for the optimiser seed "
                                "(order_es_worlds.minibatch)",
            "sigma": SIGMA, "sigma_source": "M4's sigma on its normalised features "
                                            "(es_prereg_m4_v4 `sigma` 0.05)",
            "adam": dict(oe.ADAM), "weight_decay": oe.WEIGHT_DECAY,
            "adam_carried": "carried from M4 (es_prereg_m4_v1 `es.adam`), not re-tuned for the "
                            "producer (flagged)",
            "generations": dict(GENERATIONS), "opt_seeds": [0, 1],
            "pairs_minibatch_source": "DIAGNOSTICS 25 / EXP-0069 (minibatch 24 suffices)",
            "generations_source": "DIAGNOSTICS 27 (under-trained at 20; >= 40)",
            "noise": f"order_es.noise(opt_seed, g, pairs, theta_dim, "
                     f"'{NOISE_STREAM_PURPOSE['train']}') (stream {oe.NOISE_STREAM})",
            "rank_shaping": "order_es.centered_ranks over the 2 * pairs members; disqualified "
                            "members share the worst rank",
            "estimate": "g_j = sum_i (r(+i) - r(-i)) eps_ij / (2 pairs sigma_j)",
            "update": "Adam ascent with L2 weight decay coupled into the gradient (order_es)"},
        "fitness": {"formula": "mean over the minibatch of U / B_we; B_we = rf3's units on the "
                               "world-episode at r0 (the bar)",
                    "holding_term": None,
                    "source": "design D3, supervisor decision 2026-11-05 (units only; notes/2.2.md)",
                    "diagnostics": "holding units per cycle and the investment rejection rate per "
                                   "member, generation and read; never in a fitness or reading"},
        "guard": {"rule": "order_es.disqualifying against rf3: death where rf3 lives; run-down "
                          "(r_end < x r_start) or a consumer fill < y where rf3 has none, beyond "
                          "the margin below rf3's own value",
                  "x_rundown": ow.X_RUNDOWN, "y_starvation": ow.Y_STARVATION,
                  "side_effect_params": "e2aee97ac38524d709862bc956c43940413ac0cb44c846304a4b8c3"
                                        "d324c3a99",
                  "margin": f"c sqrt(2) x pooled within-world-episode SD over rf3's replicates "
                            f"0..4 (run-down ratio; each consumer's fill), c = {oe.NOISE_MARGIN_C}",
                  "self_flag_target": SELF_FLAG_TARGET,
                  "source": "design D5; M4 v4 formula (EXP-0071: 4.7% self-flags on MWWk)"},
        "readings": {
            "L1": {"pass": f"r0 reproduces the bar's rf3 rows; rf3 equals EXP-0079 on 2 "
                           f"world-episodes; rf3's self-flag rate under the margin <= "
                           f"{SELF_FLAG_TARGET}", "else": "stop, the user"},
            "L2": {"arms": {k: SIGMA * v for k, v in SIGMA_ARMS.items()},
                   "same_as_S": "theta 0, eps, both minibatches (the recorded probe stream at "
                                "size 24) and every key; only sigma differs",
                   "readings": {"signal": SIGNAL, "weak": WEAK, "else": "no_signal"},
                   "licence": "the pilot is licensed by arm S's reading (signal or weak)",
                   "step_reading": {oe.STEP_BOTH: "both arms >= weak", oe.STEP_FLAT:
                                    "only S2 >= weak", oe.STEP_SIGMA_ONLY: "only S >= weak",
                                    "no_signal": "neither", "next": dict(oe.STEP_NEXT)},
                   "no_signal": "DIAGNOSTICS entry, stop",
                   "source": "M4 D7 thresholds; EXP-0069's A1-S2 step reading (supervisor "
                             "decision 2026-11-05: probe at sigma and 2 sigma)"},
            "monitor": {"world_episodes": 16, "replicates": list(MONITOR_REPLICATES),
                        "every": MONITOR_EVERY, "at_generation_0": True,
                        "arms": ["center", "base (rf3)"],
                        "source": "M4's monitor (es_prereg_m4_v4 `es.monitor`); supervisor "
                                  "decision 2026-11-05"},
            "floors": {"rule": "per comparison: max(the atlas-spread floor at the read's counts, "
                               "own 3 SE); own SE only with >= 2 paired observations on a world "
                               "(the n = 1 rule); a gain at or below its floor is a tie",
                       "source": "es_prereg_m4_v4 `readings.floors` (applied to rf3 for A0)"},
            "L3": {"g_tr": "center - rf3 on the monitor set at generation 20, equal weight over "
                           "the 16 world-episodes, mean over replicates 0 and 1",
                   "F_tr": "the floors rule on that comparison",
                   "feasible": "g_tr > F_tr", "unresolved": "else (DIAGNOSTICS 27's checks first)"},
            "L4": "seed 0 resumed to 40 generations; seed 1 from theta 0 to 40",
            "L5": {"worlds": "the loaded (rho > 0.28, not collapse) atlas v2 validation worlds at "
                             "12 (deciding) and 42 (reported)",
                   "episodes_per_world": 8, "replicates": 2,
                   "episodes_source": "carried from M4's power-rule outcome (sizing.md §MDE: "
                                      "8-episode reads); not re-derived for the producer pair "
                                      "(flagged)",
                   "success": "both seeds' candidate - rf3 > F_val at 12, no extra deaths, "
                              "run-down / starvation shares <= rf3's",
                   "plateau": "within F_val", "kill": "below -F_val or extra deaths",
                   "transfer_failure": "success at 12 and below -F_val at 42",
                   "precedence": ["kill", "transfer_failure", "success", "plateau"],
                   "reported": "candidate vs entity_planner_buyahead_k4 and entity_planner; loss "
                               "classes (observer on)",
                   "source": "design D6"},
            "L6": {"confirmed": "candidate beats rf3 + MWW beyond the floor on held-out worlds at "
                                "12, not below the floor at 42, no extra deaths, side effects "
                                "within the margin", "else": "not_confirmed",
                   "ledger": "key held-out/m5-producer; pending entry committed before launch",
                   "source": "design D6"},
            "stock_heavy": {"rule": "the selected theta's mean holding units per cycle > 1.5 x "
                                    "rf3's on the read's world-episodes", "role": "reported only, "
                                    "never deciding; to the user",
                            "source": "supervisor decision 2026-11-05 (notes/2.2.md)"}},
        "flags": ["Adam lr 0.02 / weight decay 0.005 carried from M4, not re-tuned",
                  "L5 8 episodes per world carried from M4's power rule, not re-derived"],
    }
    if version >= 2:
        doc["version"] = 2
        doc["changes"] = {
            "from": {"file": PREREG_V1_PATH.name, "sha256": PREREG_V1_SHA256,
                     "note": "v1 is never edited; v2 is a new file"},
            "changed": ["readings.L6.selection", "readings.L6.gating_worlds",
                        "readings.L6.episodes", "readings.L6.confirmed", "readings.L5.arms"],
            "why": "supervisor decisions 2026-11-05 (notes/2.4.md), before any data: L6 reads "
                   "one candidate, chosen on validation; L6's world set, episodes and side-effect "
                   "clause written out as L5's; L5's arms written out (review m5-2-4)",
            "unchanged": "every other section of v1"}
        doc["readings"]["L6"].update({
            "confirmed": "the candidate - rf3 clears its floor on the loaded held-out worlds at "
                         "12 and is not below -floor at 42; at 12 no extra deaths and the "
                         "candidate's run-down and starvation shares (rows with r_end < x r_start; "
                         "rows with a consumer fill < y) <= rf3's (L5's clause); at 42 no extra "
                         "deaths. This replaces v1's 'side effects within the margin' (review "
                         "m5-2-4: the margin is a per-episode guard, not a read clause)",
            "selection": "the seed whose candidate - rf3 is larger on the 12-producer loaded "
                         "validation world mean (L5's comparison); seed 0 on a tie. Selecting on "
                         "validation is fine: L6 reads held-out worlds on fresh seeds",
            "gating_worlds": "the loaded (rho > 0.28, not collapse) atlas v2 held-out worlds at 12 "
                             "(deciding) and 42 (reported), as L5's; the other held-out worlds "
                             "are played and reported",
            "episodes": "8 per world (fresh seeds, stage m5-heldout), replicates 0 and 1, as L5; "
                        "arms: the selected candidate, rf3, entity_planner_buyahead_k4, "
                        "entity_planner, all with MWWk"})
        doc["readings"]["L5"]["arms"] = ("both seeds' candidates, rf3, entity_planner_buyahead_k4, "
                                         "entity_planner, all with MWWk; episode seeds from M4's "
                                         "validation stage (m4-es-validate); loaded worlds played")
    if version >= 3:
        doc["version"] = 3
        doc["changes"] = {
            "from": {"file": PREREG_V2_PATH.name, "sha256": PREREG_V2_SHA256,
                     "note": "v1 and v2 are never edited; v3 is a new file"},
            "changed": ["guard"],
            "why": "EXP-0081 (DIAGNOSTICS 25, disqualification-driven): under v2's any-episode "
                   "rule rf3's own re-keys are disqualified on probe minibatch B 4 / 4 (cosine "
                   "with the rule off 0.840 / 0.902); user decision 2026-11-05 'Compare flag "
                   "rates' (notes/user-decisions.md), before EXP-0082's data",
            "unchanged": "every other section of v2 (L5 / L6 reads included)"}
        doc["guard"] = {
            "rule": FLAG_RULE,
            "death": "a member episode that dies where rf3 dies at no key (r0 and the "
                     "noise floor's r1..r4) disqualifies the member (absolute)",
            "flag": "an episode is flagged when order_es.disqualifying's non-death clauses fire "
                    "against rf3's r0 (run-down r_end < x r_start or a consumer fill < y where "
                    "rf3 has none, beyond the margin below rf3's own value)",
            "count_rule": "disqualified if flags(member, minibatch) - mean over rf3's re-keys "
                          "r1..r4 of flags(re-key, same world-episodes) > threshold",
            "threshold": FLAG_THRESHOLD,
            "calibration": {
                "rule": "leave-one-out on rf3's re-keys: flags(r_k, M) - mean over the other "
                        "re-keys of flags(r_j, M); threshold = the smallest t among 0 and the "
                        "observed excesses with share(excess > t) <= target",
                "minibatches": "the two probe minibatches and opt seeds 0 / 1 generations "
                               "1..40 at size 24 (M4's recorded stream)",
                "target": FLAG_TARGET, **FLAG_CALIBRATION,
                "sources": {"run": "EXP-0080 (noise_floor/noise_episodes.csv, "
                                   "noise_floor/margins.json, bar/bar.csv)", **FLAG_SOURCES}},
            "x_rundown": doc["guard"]["x_rundown"], "y_starvation": doc["guard"]["y_starvation"],
            "side_effect_params": doc["guard"]["side_effect_params"],
            "margin": doc["guard"]["margin"],
            "applies_to": "probe and train member scores; the monitor and the L5 / L6 reads "
                          "are unchanged",
            "source": "user decision 2026-11-05 after EXP-0081 (DIAGNOSTICS 25)"}
    return doc


def freeze_prereg(path: Path = PREREG_PATH) -> tuple[str, str]:
    """Write the registered pre-registration if absent (FROZEN), else compare (MATCH / MISMATCH
    with the differing sections). Returns (status, file sha256)."""
    doc = registered_prereg()
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return "FROZEN", sha256_file(str(path))
    have = json.loads(path.read_text())
    diff = sorted(k for k in set(doc) | set(have) if doc.get(k) != have.get(k))
    return ("MATCH" if not diff else "MISMATCH " + ",".join(diff)), sha256_file(str(path))


def prereg_problems(path: Path = PREREG_PATH, pinned: str | None = None) -> list[str]:
    """A real probe / train run needs the frozen file: its SHA-256 = PREREG_SHA256, and every
    section equal to this code's registration (a changed section or constant is refused)."""
    pinned = PREREG_SHA256 if pinned is None else pinned
    if pinned is None:
        return ["no pinned pre-registration (task 2.3 freezes es_prereg_m5_v1.json)"]
    if not path.exists() or sha256_file(str(path)) != pinned:
        return [f"{path.name} does not match the pinned SHA-256 {pinned[:12]}"]
    have, want = json.loads(path.read_text()), registered_prereg()
    diff = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    return [f"pre-registration sections differ from the code: {diff}"] if diff else []


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def theta_sha256(theta: np.ndarray) -> str:
    return _sha([float(x) for x in np.asarray(theta, np.float64)])


# --- the training set (EXP-0058 `bar/`) ---


@dataclass
class TrainingSet:
    dir: Path
    training: dict
    schedule: dict
    entries: dict
    costs: dict
    world_episodes: list[tuple[str, int]]
    pins: dict


def load_training(training_dir: str | Path) -> TrainingSet:
    """M4's pinned training worlds and schedule, each checked against its content hash and the
    recorded streams (order_es.load_inputs' checks, without M4's orderer bar or normaliser)."""
    from . import order_es_worlds as ow
    from ..eval.world_space import point_from_record

    d = Path(training_dir)
    training = json.loads((d / "training_worlds.json").read_text())
    if ow.content_hash(training) != training.get("content_hash"):
        raise ValueError("training_worlds.json does not match its content hash")
    schedule = json.loads((d / "schedule.json").read_text())
    if ow.content_hash(schedule) != schedule.get("content_hash"):
        raise ValueError("schedule.json does not match its content hash")
    if int(training["size"]) != TRAINING_SIZE:
        raise ValueError(f"training size {training['size']} != {TRAINING_SIZE}")
    entries, costs = {}, {}
    for w in training["worlds"]:
        point = replace(point_from_record(w["record"]), shock_window=tuple(w["shock_window"]))
        if point.world_id != w["world"]:
            raise ValueError(f"record of {w['world']} describes {point.world_id}")
        c = ow.Candidate(int(w[ow.ORDER_INDEX]), str(w["source"]), point,
                         tuple(int(e) for e in w["episodes"]), float(w["rho"]))
        entries[c.world] = c.entry()
        cs = w.get("cost_s") or {}
        costs[c.world] = float(max(cs.values(), default=1.0))
    we = [(w["world"], int(ep)) for w in training["worlds"] for ep in w["episodes"]]
    mon = _fmt_we(schedule["monitor"]["world_episodes"])
    if mon != ow.monitor_set(we):
        raise ValueError("schedule.json monitor set differs from the recorded stream")
    return TrainingSet(d, training, schedule, entries, costs, we, {
        "training_worlds_sha256": training["content_hash"],
        "schedule_sha256": schedule["content_hash"]})


def _fmt_we(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    """schedule.json's world-episode entries ({world, episode}) as tuples (order_es's format)."""
    from .order_es import _fmt_we as fmt

    return fmt(items)


def monitor_set(ts: TrainingSet) -> list[tuple[str, int]]:
    keep = set(ts.world_episodes)
    return [k for k in _fmt_we(ts.schedule["monitor"]["world_episodes"]) if k in keep]


def matched_to_bar(ts: TrainingSet, bar: Mapping, dry_run: bool) -> TrainingSet:
    """A real run's bar covers every training world-episode; a dry run's (fewer) bar restricts the
    training set to its world-episodes, so the streams draw only what the bar holds."""
    if set(bar) == set(ts.world_episodes):
        return ts
    if not dry_run or not set(bar) <= set(ts.world_episodes):
        raise ValueError(f"the bar holds {len(bar)} of the {len(ts.world_episodes)} training "
                         "world-episodes")
    return replace(ts, world_episodes=sorted(bar))


def minibatch(ts: TrainingSet, opt_seed: int, generation: int, size: int) -> list[tuple[str, int]]:
    """M4's recorded CRN stream; checked against schedule.json where it holds this entry."""
    from . import order_es_worlds as ow

    want = ow.minibatch(ts.world_episodes, opt_seed, generation, size)
    mbs = ts.schedule["minibatches"].get(str(opt_seed), [])
    if size == int(ts.schedule["minibatch_size"]) and generation <= len(mbs):
        got = _fmt_we(mbs[generation - 1])
        if got != want:
            raise ValueError(f"schedule.json minibatch {opt_seed}/{generation} differs from the "
                             "recorded stream")
    return want


def probe_batches(ts: TrainingSet, size: int) -> tuple[list, list]:
    from . import order_es_worlds as ow

    a, b = ow.probe_minibatches(ts.world_episodes, size)
    if len(a) != size or len(b) != size or set(a) & set(b):
        raise ValueError(f"no two disjoint probe minibatches of {size}")
    return a, b


# --- jobs ---


def job_id(kind: str, generation: object, member: str, world: str, episode: int,
           replicate: int) -> str:
    return f"{kind}|{generation}|{member}|{world}|{episode}|{replicate}"


def make_job(ts: TrainingSet, *, kind: str, producing: str, theta: np.ndarray | None,
             generation: object, member: str, world: str, episode: int, replicate: int,
             horizon: int, sign: int = 0, pair: int = -1, batch: int = -1,
             **extra: Any) -> Job:
    e = ts.entries[world]
    t = None if theta is None else [float(x) for x in np.asarray(theta, np.float64)]
    return {"kind": kind, "producing": producing, "theta": t,
            "theta_sha256": "" if t is None else theta_sha256(np.asarray(t)),
            "generation": str(generation), "member": member, "sign": sign, "pair": pair,
            "batch": batch, "world": world, "size": e.size, "regime": e.regime,
            "world_seed": e.world_seed, "generator": e.generator, "episode": int(episode),
            "replicate": int(replicate), "horizon": horizon, "ordering": ORDERING,
            "subject": f"{kind}/{member}", "cost": ts.costs[world],
            "job_id": job_id(kind, generation, member, world, episode, replicate), **extra}


def run_producer_job(job: Job) -> dict[str, str]:
    """One keyed episode of the job's producer with MWWk -> its row, or its failure row."""
    from ..eval.baselines import producing_policy
    from ..eval.joint_stack import joint_stack_env
    from ..eval.routing_audit import job_config, ordering_policy
    from ..eval.routing_diagnostics import DrawKey, set_draw_key
    from ..eval.runner import run_episode

    started = time.perf_counter()
    try:
        env = joint_stack_env(job_config(job), job["horizon"])
        set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
        tracer = observer = None
        if job.get("observe"):  # the L5 / L6 reads: loss classes (design D7)
            from ..eval.routing_trace import RoutingTracer
            from ..eval.supply_trace import SupplyObserver

            tracer = RoutingTracer()
            tracer.attach(env)
            observer = SupplyObserver()
            observer.attach(env)  # after the tracer: the outer complete_cycle wrapper
        monitor = ContractMonitor(env)
        recorded: dict | None = None
        if job["producing"] == RESIDUAL:
            nm = None if job.get("normaliser") is None else Normaliser.from_dict(job["normaliser"])
            if nm is not None and nm.sha256 != job.get("normaliser_sha256"):
                raise AssertionError("job normaliser does not match its sha256")
            recorder = None
            if job.get("record_dir"):  # the bar run: rf3's (theta = 0's) raw feature rows
                recorded = {"phi": [], "psi": []}

                def recorder(pid: str, cycle: int, phi: dict, psi: np.ndarray) -> None:
                    recorded["phi"].extend(phi.values())
                    recorded["psi"].append(psi)
            prod = producer_residual_policy(np.asarray(job["theta"], np.float64), nm, recorder)
            if theta_sha256(prod.theta) != job["theta_sha256"]:
                raise AssertionError("job theta does not match its sha256")
        else:
            prod = producing_policy(job["producing"])
        t0 = time.perf_counter()
        run_episode(env, prod, ordering_policy(job["ordering"]), seed=job["episode"],
                    action_seed=None, monitor=monitor)
        secs = time.perf_counter() - t0
        if recorded is not None:
            d = Path(job["record_dir"])
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{_sha(job['job_id'])[:24]}.npz",
                     phi=np.asarray(recorded["phi"], np.float64).reshape(-1, N_IN),
                     psi=np.asarray(recorded["psi"], np.float64).reshape(-1, N_P))
        stats = env.episode_stats
        monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         stats))
        if tracer is not None:
            from ..eval.routing_audit import trace_cells

            row.update(trace_cells(tracer))
            row.update(observer.cells())
        row.update({"job_id": job["job_id"], "kind": job["kind"], "generation": job["generation"],
                    "member": job["member"], "sign": str(job["sign"]), "pair": str(job["pair"]),
                    "batch": str(job["batch"]), "theta_sha256": job["theta_sha256"],
                    "producing": job["producing"], "nonfinite": "false",
                    "episode_seconds": repr(secs),
                    "job_seconds": repr(time.perf_counter() - started)})
        if tracer is not None:
            row["arm"] = str(job.get("arm", ""))
            return row
        return {c: row.get(c, "") for c in ROW_COLUMNS}
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


def _init_worker() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


# --- csv / execution ---


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append(path: Path, row: Mapping[str, str], columns: Sequence[str] = ROW_COLUMNS) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


class DigestMismatch(RuntimeError):
    pass


def play(out: Path, rows_file: str, jobs: Sequence[Job], workers: int,
         run_job_fn: Callable[[Job], dict[str, str]], bar: Mapping | None,
         log: Callable[[str], None], columns: Sequence[str] = ROW_COLUMNS
         ) -> tuple[dict[str, dict[str, str]], int]:
    """Run every job whose row is not yet in `rows_file` (longest first), appending rows as they
    land. A row of an earlier session must carry this job's theta; with a `bar`, every row must see
    the bar's demand / shock digests (CRN). Returns (rows by job id, new contract failures)."""
    path = out / rows_file
    want = {j["job_id"]: j for j in jobs}
    have = {d["job_id"]: d for d in read_csv(path) if d["job_id"] in want}
    for jid, d in have.items():
        if d["theta_sha256"] != want[jid]["theta_sha256"]:
            raise ValueError(f"{rows_file} row {jid} has another theta than this run's job")
    todo = sorted((j for j in jobs if j["job_id"] not in have),
                  key=lambda j: (-float(j["cost"]), j["job_id"]))
    failures = 0

    def finish(d: dict[str, str]) -> None:
        nonlocal failures
        if is_failure(d):
            failures += 1
            append_failure(out, failure_from_dict(d))
            log(f"[producer_es] CONTRACT FAILURE {d.get('world')} ep={d.get('episode')} "
                f"{d.get('subject')} r{d.get('replicate')}: {d.get('check')} {d.get('error_type')}")
            return
        if bar is not None:
            b = bar[(d["world"], int(d["episode"]))]
            for what, ref in (("demand_digest", b.demand_digest),
                              ("shock_digest", b.shock_digest)):
                if d[what] != ref:
                    from ..eval.contract import DIGEST
                    exc = DigestMismatch(f"{d['job_id']} saw {what} {d[what]}, the bar saw {ref}")
                    append_failure(out, run_failure(DIGEST, exc))
                    raise exc
        _append(path, d, columns)
        have[d["job_id"]] = d

    if workers <= 1 or len(todo) <= 1:
        for j in todo:
            finish(run_job_fn(j))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_init_worker) as pool:
            futures = [pool.submit(run_job_fn, j) for j in todo]
            try:
                for fut in as_completed(futures):
                    finish(fut.result())
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    return have, failures


# --- bar rows (order_es's bar format) ---

BAR_COLUMNS = ("world", "episode", "bar", "demand_per_cycle", "a0_units", "a0_dead",
               "a0_rundown", "a0_starved_consumers", "demand_digest", "shock_digest")


def bar_row(d: Mapping[str, str], horizon: int) -> dict[str, str]:
    from . import order_es_worlds as ow

    units = float(d["units"])
    rs = float(d["r_start"]) if d["r_start"] not in ("", "None") else 0.0
    re_ = float(d["r_end"]) if d["r_end"] not in ("", "None") else 0.0
    fills = json.loads(d["consumer_fill_rate"])
    starved = sorted(c for c, v in fills.items() if v is not None and v < ow.Y_STARVATION)
    return {"world": d["world"], "episode": d["episode"], "bar": repr(max(units, 1.0)),
            "demand_per_cycle": repr(max(float(d["ordered_units"]), 1.0) / horizon),
            "a0_units": repr(units), "a0_dead": d["dead"],
            "a0_rundown": "true" if re_ < ow.X_RUNDOWN * rs else "false",
            "a0_starved_consumers": json.dumps(starved), "demand_digest": d["demand_digest"],
            "shock_digest": d["shock_digest"]}


FIT_ROWS = "fit_rows"
NORMALISER_RULE = ("M4's: per non-bias feature, (x - median) / scale clipped to +-5; scale = IQR, "
                   "else the population std, else 1.0; fitted on rf3's (theta = 0's) raw rows of "
                   "every producer-cycle of the bar run's training world-episodes")


def load_normaliser(bar_dir: str | Path) -> tuple[Normaliser, dict]:
    doc = json.loads((Path(bar_dir) / "normaliser.json").read_text())
    nm = Normaliser.from_dict(doc)
    if nm.sha256 != doc.get("sha256"):
        raise ValueError("normaliser.json does not match its sha256")
    return nm, {"normaliser_sha256": nm.sha256}


def load_bar(bar_dir: str | Path) -> tuple[dict, list[dict[str, str]], dict]:
    """(bar rows by (world, episode) as order_es.BarRow, the base's r0 rows, pins)."""
    from .order_es import read_bar

    bd = Path(bar_dir)
    spec = json.loads((bd / "spec.json").read_text())
    if spec.get("mode") != BAR or spec.get("status") != "complete":
        raise ValueError(f"{bd} is not a complete producer_es bar run")
    return (read_bar(bd / "bar.csv"), read_csv(bd / "base_rows.csv"),
            {"bar_csv_sha256": sha256_file(str(bd / "bar.csv")),
             "bar_spec_pins_sha256": _sha(spec.get("pins"))})


# --- spec / pins ---


def _open_run(out: Path, mode: str, pins: dict, workers: int, log: Callable[[str], None]
              ) -> dict:
    """A new spec, or (same pins) the recorded one with a resume noted; other pins are refused
    before any write."""
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    pins = json.loads(json.dumps(pins, sort_keys=True, default=str))
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if old.get("driver") != DRIVER or old.get("mode") != mode:
            raise ValueError(f"{out}/spec.json belongs to another run "
                             f"({old.get('driver')} {old.get('mode')})")
        if old.get("pins") != pins:
            diff = sorted(k for k in set(old.get("pins") or {}) | set(pins)
                          if (old.get("pins") or {}).get(k) != pins.get(k))
            raise ValueError(f"{out} holds a run with different pins; refusing to resume "
                             f"(differs at {diff})")
        old.setdefault("resumes", []).append({"workers": workers, "at": time.time()})
        spec = old
        log(f"[producer_es] resuming {out} ({mode})")
    else:
        spec = {"driver": DRIVER, "mode": mode, "pins": pins, "status": "running",
                "acceptance_read": False, "started": time.time(), "workers": workers}
    ensure_failures_file(out)
    _write_spec(out, spec)
    return spec


def _write_spec(out: Path, spec: dict) -> None:
    tmp = out / "spec.json.tmp"
    tmp.write_text(json.dumps(spec, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out / "spec.json")


def real_run_problems(*, horizon: int, source_commit: str, filters: Mapping[str, object]
                      ) -> list[str]:
    import re

    problems = []
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
        problems.append("no full 40-hex source commit (--source-commit or --verify-source-commit)")
    if horizon != HORIZON:
        problems.append(f"horizon {horizon} != {HORIZON}")
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


def _finish(out: Path, spec: dict, expected: int, rows: int, extra_bad: list[str],
            result: dict, log: Callable[[str], None]) -> int:
    failures = read_failures(out)
    missing = expected - rows
    bad = bool(failures or missing > 0 or extra_bad)
    spec["status"] = "invalid" if bad else "complete"
    spec["result"] = {"status": spec["status"], "rows": rows, "expected_rows": expected,
                      "contract_failures": len(failures), "problems": extra_bad, **result}
    _write_spec(out, spec)
    log(f"[producer_es] {spec['mode']} {spec['status']}"
        + (f" ({len(failures)} failure(s), {missing} missing, {extra_bad})" if bad else ""))
    return CONTRACT_INVALID if bad else 0


def _diag(rows: Sequence[Mapping[str, str]]) -> dict[str, float | None]:
    """Mean holding units per cycle and investment rejection rate (diagnostics, never fitness)."""
    def m(col: str) -> float | None:
        xs = [float(r[col]) for r in rows if r.get(col) not in (None, "", "None")]
        return fmean(xs) if xs else None
    return {"holding_units_per_cycle": m("holding_units_per_cycle"),
            "investment_rejection_rate": m("investment_rejection_rate"),
            "units": m("units")}


# --- modes ---


def _select(ts: TrainingSet, max_world_episodes: int | None) -> list[tuple[str, int]]:
    we = sorted(ts.world_episodes)
    return we if max_world_episodes is None else we[:max_world_episodes]


def run_bar(*, training_dir: str | Path, out_dir: str | Path, workers: int = 1,
            horizon: int = HORIZON, max_world_episodes: int | None = None,
            atlas: str | Path | None = None, exp0079_episodes: str | Path | None = None,
            source_commit: str = "", dry_run: bool = False,
            run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
            log: Callable[[str], None] = _log) -> int:
    ts = load_training(training_dir)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit,
                              filters={"max_world_episodes": max_world_episodes})
        if (atlas is None) != (exp0079_episodes is None) or atlas is None:
            p.append("a real bar run needs --atlas and --exp0079-episodes (the cross-check)")
        if p:
            raise ValueError("not a valid bar run (" + "; ".join(p) + "); pass --dry-run")
    we = _select(ts, max_world_episodes)
    pins = {"driver": DRIVER, "mode": BAR, "base": BASE_PRODUCER, "ordering": ORDERING,
            "horizon": horizon, "dry_run": dry_run, "world_episodes": we, **ts.pins,
            "exp0079_episodes_sha256": (sha256_file(str(exp0079_episodes))
                                        if exp0079_episodes else None),
            "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, BAR, pins, workers, log)
    # rf3 played as the residual at theta = 0 (equal to rf3 action for action, task 2.1), recording
    # its raw feature rows for the normaliser (task 2.3, supervisor decision)
    fit_dir = out / FIT_ROWS
    jobs = [make_job(ts, kind=BAR, producing=RESIDUAL, theta=np.zeros(THETA_DIM), generation=0,
                     member=BASE_LABEL, world=w, episode=e, replicate=0, horizon=horizon,
                     record_dir=str(fit_dir))
            for w, e in we]
    rows, _ = play(out, ROW_FILES[BAR], jobs, workers, run_job_fn, None, log)
    ordered = [rows[j["job_id"]] for j in jobs if j["job_id"] in rows]
    write_csv(out / "base_rows.csv", ROW_COLUMNS, ordered)
    write_csv(out / "bar.csv", BAR_COLUMNS, [bar_row(d, horizon) for d in ordered])
    bad: list[str] = []
    if len(ordered) == len(jobs):
        phi, psi = [], []
        for j in jobs:
            z = np.load(fit_dir / f"{_sha(j['job_id'])[:24]}.npz")
            phi.append(z["phi"])
            psi.append(z["psi"])
        phi_all, psi_all = np.concatenate(phi), np.concatenate(psi)
        nm = Normaliser.fit(phi_all, psi_all)
        doc = {**nm.to_dict(), "sha256": nm.sha256, "rule": NORMALISER_RULE,
               "fit_rows": {"input_rows": int(len(phi_all)), "producer_rows": int(len(psi_all)),
                            "sha256": hashlib.sha256(phi_all.tobytes() + psi_all.tobytes()
                                                     ).hexdigest()}}
        (out / "normaliser.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    cross: dict = {}
    if atlas is not None and exp0079_episodes is not None:
        cross = crosscheck_exp0079(atlas, exp0079_episodes, out, log)
        if not cross["equal"]:
            bad.append(f"rf3 replay differs from EXP-0079 on {cross['differing']}")
    return _finish(out, spec, len(jobs), len(ordered), bad,
                   {"diagnostics": _diag(ordered), "crosscheck": cross}, log)


CROSSCHECK_N = 2


def crosscheck_exp0079(atlas: str | Path, exp0079_episodes: str | Path, out: Path,
                       log: Callable[[str], None]) -> dict:
    """rf3 replayed through the supply-trace job path on the first CROSSCHECK_N MWWk r0
    world-episodes of EXP-0079 (sorted): every cell EXP-0079 wrote must be equal (D7)."""
    from ..eval import supply_trace as st
    from ..eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from ..eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from ..eval.routing_audit import build_jobs
    from ..eval.teacher_audit import load_atlas

    csv.field_size_limit(1 << 30)
    ref = {(r["world"], r["episode"], r["replicate"]): r for r in read_csv(Path(exp0079_episodes))
           if r["subject"].endswith("/mww_keyed") and r["replicate"] == "0"}
    keys = sorted(ref)[:CROSSCHECK_N]
    views = select_views(load_atlas(atlas), sorted({k[0] for k in keys}))
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    arm = {**arm, "producing": BASE_PRODUCER, "subject": member_id(BASE_PRODUCER, arm["ordering"])}
    jobs = [j for j in build_jobs([world_entry(v) for v in views], [arm], 1, HORIZON, None, {})
            if (j["world"], str(j["episode"]), str(j["replicate"])) in set(keys)]
    for j in jobs:
        j["arm"] = MWWK
        j["order_ttl"] = 2
    rows, differing = [], []
    for j in jobs:
        d = st.run_supply_job(j)
        r = ref[(j["world"], str(j["episode"]), str(j["replicate"]))]
        diff = [c for c in r if d.get(c) != r[c]]
        rows.append({"world": j["world"], "episode": j["episode"], "compared": len(r),
                     "differing": ";".join(diff[:8]), "equal": not diff})
        if diff:
            differing.append(f"{j['world']}:{j['episode']}")
    write_csv(out / "crosscheck.csv", ("world", "episode", "compared", "differing", "equal"), rows)
    log(f"[producer_es] EXP-0079 cross-check: {sum(r['equal'] for r in rows)} / {len(rows)} equal")
    return {"rows": len(rows), "equal": len(rows) == CROSSCHECK_N and not differing,
            "differing": differing}


def run_noise_floor(*, training_dir: str | Path, bar_dir: str | Path, out_dir: str | Path,
                    workers: int = 1, horizon: int = HORIZON,
                    replicates: Sequence[int] = NOISE_REPLICATES, source_commit: str = "",
                    dry_run: bool = False,
                    run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
                    log: Callable[[str], None] = _log) -> int:
    from .order_es import NOISE_MARGIN_C, NoiseMargin, noise_margins, self_flags

    ts = load_training(training_dir)
    bar, base_rows, bar_pins = load_bar(bar_dir)
    ts = matched_to_bar(ts, bar, dry_run)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit,
                              filters={"replicates": None if tuple(replicates) == NOISE_REPLICATES
                                       else replicates})
        if p:
            raise ValueError("not a valid noise-floor run (" + "; ".join(p) + "); pass --dry-run")
    we = sorted(bar)
    pins = {"driver": DRIVER, "mode": NOISE, "base": BASE_PRODUCER, "horizon": horizon,
            "dry_run": dry_run, "replicates": list(replicates), **ts.pins, **bar_pins,
            "c": NOISE_MARGIN_C, "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, NOISE, pins, workers, log)
    jobs = [make_job(ts, kind=NOISE, producing=BASE_PRODUCER, theta=None, generation=0,
                     member=f"r{r}", world=w, episode=e, replicate=r, horizon=horizon)
            for w, e in we for r in (0, *replicates)]
    rows, _ = play(out, ROW_FILES[NOISE], jobs, workers, run_job_fn, bar, log)
    got = [rows[j["job_id"]] for j in jobs if j["job_id"] in rows]
    base = {(d["world"], d["episode"]): d for d in base_rows}
    repro, bad = [], []
    for d in got:
        if d["replicate"] != "0":
            continue
        b = base[(d["world"], d["episode"])]
        diff = [c for c in CSV_COLUMNS if c != "subject" and d.get(c) != b.get(c)]
        repro.append({"world": d["world"], "episode": d["episode"], "differing": ";".join(diff),
                      "equal": not diff})
    write_csv(out / "reproduction.csv", ("world", "episode", "differing", "equal"), repro)
    if not all(r["equal"] for r in repro) or len(repro) != len(we):
        bad.append(f"r0 reproduces {sum(r['equal'] for r in repro)} / {len(we)} bar rows")
    margins = {}
    if got and not bad:
        m = noise_margins(got)
        margin = NoiseMargin(m["rundown"], m["starvation"])
        flagged, n = self_flags(got, bar, margin)
        raw, _ = self_flags(got, bar, None)
        margins = {**m, "self_flags": flagged, "self_flags_without_margin": raw, "episodes": n,
                   "self_flag_rate": flagged / n if n else None,
                   "target_rate": SELF_FLAG_TARGET,
                   "target_met": bool(n) and flagged / n <= SELF_FLAG_TARGET}
        doc = {"format": "m5-noise-margin-v1", "base": BASE_PRODUCER, **margins,
               "rows_sha256": _sha(sorted((d["job_id"], d["units"], d["r_end"]) for d in got))}
        (out / "margins.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return _finish(out, spec, len(jobs), len(got), bad,
                   {"margins": margins, "diagnostics": _diag(got)}, log)


def load_margin(margins_path: str | Path) -> tuple[Any, dict]:
    from .order_es import NoiseMargin

    doc = json.loads(Path(margins_path).read_text())
    if doc.get("format") != "m5-noise-margin-v1":
        raise ValueError(f"{margins_path} is not an M5 noise-margin file")
    return NoiseMargin(float(doc["rundown"]), float(doc["starvation"])), {
        "margins_sha256": sha256_file(str(margins_path))}


# --- the flag-count rule (pre-registration v3; user decision 2026-11-05 after EXP-0081) ---
# A member is disqualified if it dies where rf3 dies at no key (r0..r4), or if its count of
# flagged episodes (run-down / starvation under the margin, order_es.disqualifying's non-death
# clauses) on its minibatch exceeds rf3's expected count there (the mean over rf3's re-keys
# r1..r4 of their flagged episodes on the same world-episodes) by more than FLAG_THRESHOLD.
# FLAG_THRESHOLD is calibrated on EXP-0080's re-keys only (`calibrate_flag_threshold`): rf3 itself
# is disqualified on <= FLAG_TARGET of its re-keyed minibatches (each re-key against the mean of
# the other three; the minibatches the probe and both seeds' 40 generations draw at size 24).
FLAG_RULE = "flag-count-v1"
FLAG_TARGET = 0.05
FLAG_THRESHOLD = 5 / 3  # calibrate_flag_threshold on EXP-0080 (2026-11-05, before EXP-0082)
FLAG_SOURCES = {  # EXP-0080's files the threshold and the reference counts come from
    "noise_episodes_sha256": "cc6ef40557672ef9390b7827ae6c287ff0930f523f506673c71237be2f642a63",
    "margins_sha256": "561c94e2afacf7daf53ce6f07fbdef9b3f567b6d9f29114b157129e00686f09b",
    "bar_csv_sha256": "c42b9ecc2bdcd9c4d39454facd79144ba6cb0497c8b167a4a1d358abcd1265bf"}
FLAG_CALIBRATION: dict[str, Any] = {  # calibrate_flag_threshold's output on those files
    "minibatches": 82, "re_keys": 4, "samples": 328, "self_disqualified": 9,
    "rate": 9 / 328, "rate_without_threshold": 135 / 328, "max_excess": 8 / 3}
NOISE_ROWS = "noise_episodes.csv"


@dataclass(frozen=True)
class FlagReference:
    """rf3's re-keyed non-death flags per world-episode (replicates >= 1) and the world-episodes
    where rf3 dies at any key (r0 included)."""

    flags: dict[tuple[str, int], tuple[bool, ...]]
    dead: frozenset[tuple[str, int]]

    def expected(self, wes: Sequence[tuple[str, int]]) -> float:
        missing = [k for k in wes if not self.flags.get(k)]
        if missing:
            raise ValueError(f"no re-keyed rf3 rows for {missing[:3]}")
        return float(sum(fmean(self.flags[k]) for k in wes))


def row_flagged(row: Mapping[str, str], b: Any, margin: Any, ref: Any) -> bool:
    """The episode trips a non-death guard (run-down / starvation beyond the margin)."""
    from .order_es import disqualifying

    return any(x != "death" for x in disqualifying(row, b, margin, ref))


def flag_reference(noise_rows: Sequence[Mapping[str, str]], bar: Mapping, base_rows,
                   margin: Any) -> FlagReference:
    from .order_es import a0_values, _bool

    refs = {k: v[0] for k, v in a0_values(base_rows).items()}
    flags: dict[tuple[str, int], list[bool]] = {}
    dead = {k for k, b in bar.items() if b.a0_dead}
    for d in sorted(noise_rows, key=lambda d: (d["world"], int(d["episode"]), int(d["replicate"]))):
        k = (d["world"], int(d["episode"]))
        if int(d["replicate"]) == 0 or k not in bar:
            continue
        flags.setdefault(k, []).append(row_flagged(d, bar[k], margin, refs[k]))
        if _bool(d["dead"]):
            dead.add(k)
    return FlagReference({k: tuple(v) for k, v in flags.items()}, frozenset(dead))


def load_flag_reference(margins: str | Path, bar: Mapping, base_rows, margin: Any
                        ) -> tuple[FlagReference, dict]:
    """The noise-floor run's re-keyed rows (next to its margins.json)."""
    path = Path(margins).parent / NOISE_ROWS
    return flag_reference(read_csv(path), bar, base_rows, margin), {
        "flag_rule": FLAG_RULE, "flag_threshold": FLAG_THRESHOLD,
        "noise_episodes_sha256": sha256_file(str(path))}


def flag_source_problems(pins: Mapping[str, Any]) -> list[str]:
    """A real probe / train run reads the registered EXP-0080 files."""
    return [f"{k} {str(pins.get(k))[:12]} is not the registered {v[:12]}"
            for k, v in FLAG_SOURCES.items() if pins.get(k) != v]


def flag_minibatches(ts: TrainingSet, size: int = MINIBATCH) -> list[list[tuple[str, int]]]:
    """Every minibatch the registered probe and runs draw: the two probe minibatches and both
    optimiser seeds' generations 1..GENERATIONS['main']."""
    a, b = probe_batches(ts, size)
    return [a, b] + [minibatch(ts, s, g, size) for s in (0, 1)
                     for g in range(1, GENERATIONS["main"] + 1)]


def calibrate_flag_threshold(ts: TrainingSet, ref: FlagReference, target: float = FLAG_TARGET
                             ) -> dict[str, Any]:
    """Leave-one-out on rf3's re-keys: re-key k's flag count on a minibatch minus the mean of the
    other re-keys' counts there. The threshold is the smallest value t among 0 and the observed
    excesses with share(excess > t) <= target."""
    mbs = flag_minibatches(ts)
    n_keys = {len(v) for v in ref.flags.values()}
    if len(n_keys) != 1 or min(n_keys) < 2:
        raise ValueError(f"the calibration needs the same >= 2 re-keys everywhere, got {n_keys}")
    nk = n_keys.pop()
    ex = []
    for mb in mbs:
        counts = [sum(ref.flags[w][k] for w in mb) for k in range(nk)]
        ex += [counts[k] - (sum(counts) - counts[k]) / (nk - 1) for k in range(nk)]
    t = next(c for c in sorted({0.0, *ex}) if sum(e > c + 1e-9 for e in ex) <= target * len(ex))
    hit = sum(e > t + 1e-9 for e in ex)
    return {"threshold": float(t), "target": target, "minibatches": len(mbs), "re_keys": nk,
            "samples": len(ex), "self_disqualified": hit, "rate": hit / len(ex),
            "rate_without_threshold": sum(e > 1e-9 for e in ex) / len(ex),
            "max_excess": max(ex)}


def score_member_v3(rows: Sequence[Mapping[str, str]], bar: Mapping, margin: Any, refs: Mapping,
                    fref: FlagReference, threshold: float = FLAG_THRESHOLD) -> Any:
    """Units-only fitness (h = 0) and the flag-count rule against rf3's re-keys."""
    from .order_es import MemberScore, _bool, we_fitness

    if any(_bool(r["nonfinite"]) for r in rows):
        return MemberScore(None, False, ("nonfinite",), True)
    wes = [(r["world"], int(r["episode"])) for r in rows]
    fs = [we_fitness(r, bar[k], 0.0) for r, k in zip(rows, wes)]
    reasons = [f"{k[0]}:{k[1]}:death" for r, k in zip(rows, wes)
               if _bool(r["dead"]) and k not in fref.dead]
    flags = sum(row_flagged(r, bar[k], margin, refs[k]) for r, k in zip(rows, wes))
    excess = flags - fref.expected(wes)
    if excess > threshold + 1e-9:
        reasons.append(f"flags:{flags}-{excess:+.2f}")
    return MemberScore(fmean(fs), bool(reasons), tuple(reasons), False)


def _scores(rows_by_member: Mapping[str, Sequence[Mapping[str, str]]], bar: Mapping,
            margin: Any, refs: Mapping, fref: FlagReference) -> dict:
    """Units-only fitness; disqualification by the v3 flag-count rule against rf3's re-keys."""
    return {lab: score_member_v3(rs, bar, margin, refs, fref)
            for lab, rs in rows_by_member.items()}


def _member_jobs(ts: TrainingSet, kind: str, thetas: Mapping[str, np.ndarray], world_eps,
                 generation: object, horizon: int, batch: int = -1, *,
                 normaliser: Normaliser, replicates: Sequence[int] = (0,)) -> list[Job]:
    def sign_pair(lab: str) -> tuple[int, int]:
        m = lab.split(":", 1)[-1]
        return (0, -1) if m == CENTER else (1 if m[0] == "+" else -1, int(m[1:]))

    return [make_job(ts, kind=kind, producing=RESIDUAL, theta=th, generation=generation,
                     member=lab, world=w, episode=e, replicate=r, horizon=horizon,
                     sign=sign_pair(lab)[0], pair=sign_pair(lab)[1], batch=batch,
                     normaliser=normaliser.to_dict(), normaliser_sha256=normaliser.sha256)
            for lab, th in thetas.items() for w, e in world_eps for r in replicates]


def _by_member(rows: Mapping[str, dict], jobs: Sequence[Job]) -> dict[str, list]:
    out: dict[str, list] = {}
    for j in jobs:
        if j["job_id"] in rows:
            out.setdefault(j["member"], []).append(rows[j["job_id"]])
    return out


def run_probe(*, training_dir: str | Path, bar_dir: str | Path, margins: str | Path,
              out_dir: str | Path, workers: int = 1, horizon: int = HORIZON,
              size: int = MINIBATCH, pairs: int = PAIRS, sigma: float = SIGMA,
              source_commit: str = "", dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
              log: Callable[[str], None] = _log) -> int:
    """L2: the ES estimate at theta = 0 on the two disjoint probe minibatches for each sigma arm
    (sigma and 2 sigma: the same eps, minibatches and keys; only sigma differs). Each arm's
    cosine reads signal / weak / no_signal; the pilot is licensed by the sigma arm's reading, and
    M4's step reading (`order_es.step_reading`) over the two arms decides the pilot's step."""
    from .order_es import a0_values, cosine, member_thetas, noise, rank_and_estimate, step_reading

    ts = load_training(training_dir)
    bar, base_rows, bar_pins = load_bar(bar_dir)
    nm, n_pins = load_normaliser(bar_dir)
    margin, m_pins = load_margin(margins)
    fref, f_pins = load_flag_reference(margins, bar, base_rows, margin)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit, filters={
            "size": None if size == MINIBATCH else size,
            "pairs": None if pairs == PAIRS else pairs,
            "sigma": None if sigma == SIGMA else sigma}) + prereg_problems() + \
            flag_source_problems({**bar_pins, **m_pins, **f_pins})
        if p:
            raise ValueError("not a valid probe run (" + "; ".join(p) + "); pass --dry-run")
    ts = matched_to_bar(ts, bar, dry_run)  # after the real-run refusals
    a, b = probe_batches(ts, size)
    eps = noise(0, 0, pairs, THETA_DIM, NOISE_STREAM_PURPOSE["probe"])
    pins = {"driver": DRIVER, "mode": PROBE, "horizon": horizon, "dry_run": dry_run,
            "size": size, "pairs": pairs, "sigma": sigma, "sigma_arms": SIGMA_ARMS,
            "theta_dim": THETA_DIM, **ts.pins, **bar_pins, **n_pins, **m_pins, **f_pins,
            "batches": [a, b], "eps_sha256": _sha(eps.tolist()), "prereg": PREREG_SHA256,
            "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, PROBE, pins, workers, log)
    jobs, arms = [], {}
    for arm, scale in SIGMA_ARMS.items():
        sig = np.full(THETA_DIM, sigma * scale)
        thetas = {f"{arm}:{k}": v for k, v in member_thetas(np.zeros(THETA_DIM), eps, sig).items()
                  if k != CENTER}
        arms[arm] = sig
        for bi, batch in enumerate((a, b)):
            jobs += _member_jobs(ts, PROBE, thetas, batch, 0, horizon, bi, normaliser=nm)
    rows, _ = play(out, ROW_FILES[PROBE], jobs, workers, run_job_fn, bar, log)
    refs = {k: v[0] for k, v in a0_values(base_rows).items()}
    per_arm = {}
    for arm, sig in arms.items():
        est, batches = [], []
        for bi in (0, 1):
            bj = [j for j in jobs if j["batch"] == bi and j["member"].startswith(arm + ":")]
            sc = {k.split(":", 1)[1]: v for k, v in _scores(_by_member(rows, bj), bar, margin,
                                                            refs, fref).items()}
            if len(sc) < 2 * pairs:
                est.append(None)
                continue
            ranked = rank_and_estimate(eps, sig, {k: v.fitness for k, v in sc.items()},
                                       {k: v.disqualified for k, v in sc.items()})
            est.append(ranked.estimate)
            batches.append({"batch": bi,
                            "disqualified": sum(v.disqualified for v in sc.values()),
                            "fitness_mean": fmean(v.fitness for v in sc.values()
                                                  if v.fitness is not None),
                            "diagnostics": _diag([r for j in bj
                                                  if (r := rows.get(j["job_id"]))])})
        cos = cosine(est[0], est[1]) if all(e is not None for e in est) else None
        per_arm[arm] = {"sigma": float(sig[0]), "cosine": cos,
                        "reading": (None if cos is None else "signal" if cos >= SIGNAL
                                    else "weak" if cos >= WEAK else "no_signal"),
                        "batches": batches,
                        "estimates": [None if e is None else [float(x) for x in e] for e in est]}
    step = step_reading(per_arm["S"]["cosine"], per_arm["S2"]["cosine"])
    doc = {"reading": per_arm["S"]["reading"], "licence": "the pilot is licensed by arm S's "
           "reading (signal / weak); the step reading decides its sigma",
           "step": step, "arms": per_arm, "thresholds": {"signal": SIGNAL, "weak": WEAK}}
    (out / "probe.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return _finish(out, spec, len(jobs), sum(j["job_id"] in rows for j in jobs), [],
                   {"reading": doc["reading"], "step_reading": step["step_reading"],
                    "cosine": {k: v["cosine"] for k, v in per_arm.items()}}, log)


def checkpoint_path(out: Path, generation: int) -> Path:
    return out / "checkpoints" / f"gen-{generation:03d}.json"


def save_checkpoint(path: Path, doc: dict) -> None:
    doc = dict(doc)
    doc["content_hash"] = _sha({k: v for k, v in doc.items() if k != "content_hash"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if _sha({k: v for k, v in doc.items() if k != "content_hash"}) != doc.get("content_hash"):
        raise ValueError(f"{path} does not match its content hash")
    if theta_sha256(np.asarray(doc["theta"])) != doc["theta_sha256"]:
        raise ValueError(f"{path} theta does not match its sha256")
    return doc


GEN_COLUMNS = ("generation", "theta_sha256", "minibatch_sha256", "fitness_mean", "fitness_best",
               "disqualified", "grad_norm", "theta_norm", "holding_units_per_cycle",
               "investment_rejection_rate", "units")
MON_COLUMNS = ("generation", "theta_sha256", "world_episodes", "g_tr", "center_units",
               "bar_units", "center_holding", "bar_holding", "center_investment_rejection",
               "bar_investment_rejection")


def run_train(*, training_dir: str | Path, bar_dir: str | Path, margins: str | Path,
              out_dir: str | Path, generations: int, opt_seed: int = 0, workers: int = 1,
              horizon: int = HORIZON, size: int = MINIBATCH, pairs: int = PAIRS,
              sigma: float = SIGMA, lr: float = ADAM_LR, monitor_every: int = MONITOR_EVERY,
              source_commit: str = "", dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
              log: Callable[[str], None] = _log) -> int:
    """`generations` = the target generation count; a resume continues from the latest
    checkpoint of the same pins (the target may grow: pilot 20 -> main 40)."""
    from .order_es import AdamState, a0_values, adam_ascend, member_thetas, noise, \
        rank_and_estimate

    ts = load_training(training_dir)
    bar, base_rows, bar_pins = load_bar(bar_dir)
    nm, n_pins = load_normaliser(bar_dir)
    margin, m_pins = load_margin(margins)
    fref, f_pins = load_flag_reference(margins, bar, base_rows, margin)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit, filters={
            "size": None if size == MINIBATCH else size,
            "pairs": None if pairs == PAIRS else pairs,
            "sigma": None if sigma == SIGMA else sigma, "lr": None if lr == ADAM_LR else lr,
            "monitor_every": None if monitor_every == MONITOR_EVERY else monitor_every}
        ) + prereg_problems() + flag_source_problems({**bar_pins, **m_pins, **f_pins})
        if p:
            raise ValueError("not a valid train run (" + "; ".join(p) + "); pass --dry-run")
    ts = matched_to_bar(ts, bar, dry_run)  # after the real-run refusals
    pins = {"driver": DRIVER, "mode": TRAIN, "horizon": horizon, "dry_run": dry_run,
            "opt_seed": opt_seed, "size": size, "pairs": pairs, "sigma": sigma, "lr": lr,
            "monitor_every": monitor_every, "monitor_replicates": list(MONITOR_REPLICATES),
            "theta_dim": THETA_DIM, **ts.pins, **bar_pins, **n_pins, **m_pins, **f_pins,
            "prereg": PREREG_SHA256, "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, TRAIN, pins, workers, log)
    refs = {k: v[0] for k, v in a0_values(base_rows).items()}
    sig = np.full(THETA_DIM, sigma)
    done = sorted((out / "checkpoints").glob("gen-*.json")) if (out / "checkpoints").exists() else []
    if done:
        ck = load_checkpoint(done[-1])
        if ck["pins_sha256"] != _sha(spec["pins"]):
            raise ValueError(f"{done[-1]} was written under other pins")
        theta, adam, start = np.asarray(ck["theta"]), AdamState.from_dict(ck["adam"]), ck["generation"]
    else:
        theta, adam, start = np.zeros(THETA_DIM), AdamState.zeros(THETA_DIM), 0
    # rows beyond the checkpoint (a crash between generations.csv and the checkpoint) are dropped,
    # one row per generation (review m5-2-2)
    gens = list({int(d["generation"]): d for d in read_csv(out / "generations.csv")
                 if int(d["generation"]) <= start}.values())
    gens.sort(key=lambda d: int(d["generation"]))
    mon = list({int(d["generation"]): d for d in read_csv(out / "monitor.csv")}.values())
    mon.sort(key=lambda d: int(d["generation"]))
    if len(gens) != start:
        raise ValueError(f"generations.csv holds {len(gens)} rows for checkpoint generation {start}")
    write_csv(out / "generations.csv", GEN_COLUMNS, gens)

    def monitor(g: int, th: np.ndarray) -> None:
        """M4's monitor: the centre at replicates 0 and 1 on the 16 monitor world-episodes and the
        base (rf3) at the same keys (run once; rows cached by job id); g_tr = the equal-weight mean
        over world-episodes of mean_r (centre - base). F_tr is the read's (task 2.4: M4's rule,
        max(the atlas-spread floor at these counts, own 3 SE), the n = 1 rule)."""
        mws = monitor_set(ts)
        mjobs = _member_jobs(ts, "monitor", {CENTER: th}, mws, g, horizon, normaliser=nm,
                             replicates=MONITOR_REPLICATES)
        bjobs = [make_job(ts, kind="monitor_base", producing=BASE_PRODUCER, theta=None,
                          generation="base", member=BASE_LABEL, world=w, episode=e, replicate=r,
                          horizon=horizon) for w, e in mws for r in MONITOR_REPLICATES]
        mrows, _ = play(out, "monitor_episodes.csv", mjobs + bjobs, workers, run_job_fn, bar, log)
        if any(j["job_id"] not in mrows for j in mjobs + bjobs):
            raise RuntimeError(f"monitor at generation {g}: missing rows")
        key = lambda j: (j["world"], j["episode"], j["replicate"])  # noqa: E731
        cen = {key(j): mrows[j["job_id"]] for j in mjobs}
        bas = {key(j): mrows[j["job_id"]] for j in bjobs}
        per_we = [fmean(float(cen[(w, e, r)]["units"]) - float(bas[(w, e, r)]["units"])
                        for r in MONITOR_REPLICATES) for w, e in mws]
        dc, db = _diag(list(cen.values())), _diag(list(bas.values()))
        mon[:] = [m for m in mon if int(m["generation"]) != g]
        mon.append({"generation": g, "theta_sha256": theta_sha256(th),
                    "world_episodes": len(mws), "g_tr": fmean(per_we) if per_we else None,
                    "center_units": dc["units"], "bar_units": db["units"],
                    "center_holding": dc["holding_units_per_cycle"],
                    "bar_holding": db["holding_units_per_cycle"],
                    "center_investment_rejection": dc["investment_rejection_rate"],
                    "bar_investment_rejection": db["investment_rejection_rate"]})
        mon.sort(key=lambda d: int(d["generation"]))
        write_csv(out / "monitor.csv", MON_COLUMNS, mon)

    # M4's monitor runs at generation 0 too; a monitor that died after its generation's checkpoint
    # is run on resume (review m5-2-2)
    if start % monitor_every == 0 and not any(int(m["generation"]) == start for m in mon):
        if start:
            log(f"[producer_es] resume: running the missing monitor of generation {start}")
        monitor(start, theta)
    expected = rows_seen = 0
    for g in range(start + 1, generations + 1):
        mb = minibatch(ts, opt_seed, g, size)
        eps = noise(opt_seed, g, pairs, THETA_DIM, NOISE_STREAM_PURPOSE["train"])
        thetas = {k: v for k, v in member_thetas(theta, eps, sig).items() if k != CENTER}
        jobs = _member_jobs(ts, TRAIN, thetas, mb, g, horizon, normaliser=nm)
        rows, _ = play(out, ROW_FILES[TRAIN], jobs, workers, run_job_fn, bar, log)
        expected += len(jobs)
        rows_seen += sum(j["job_id"] in rows for j in jobs)
        sc = _scores(_by_member(rows, jobs), bar, margin, refs, fref)
        if len(sc) < 2 * pairs or any(len(v) != len(mb) for v in _by_member(rows, jobs).values()):
            log(f"[producer_es] generation {g}: missing member rows; stopping")
            break
        ranked = rank_and_estimate(eps, sig, {k: s.fitness for k, s in sc.items()},
                                   {k: s.disqualified for k, s in sc.items()})
        theta, adam = adam_ascend(theta, ranked.estimate, adam, lr=lr)
        member_rows = [r for j in jobs if (r := rows.get(j["job_id"]))]
        fits = [s.fitness for s in sc.values() if s.fitness is not None]
        gens.append({"generation": g, "theta_sha256": theta_sha256(theta),
                     "minibatch_sha256": _sha(mb), "fitness_mean": fmean(fits),
                     "fitness_best": max(fits),
                     "disqualified": sum(s.disqualified for s in sc.values()),
                     "grad_norm": float(np.linalg.norm(ranked.estimate)),
                     "theta_norm": float(np.linalg.norm(theta)), **_diag(member_rows)})
        write_csv(out / "generations.csv", GEN_COLUMNS, gens)
        save_checkpoint(checkpoint_path(out, g), {
            "format": CHECKPOINT_FORMAT, "generation": g, "theta": [float(x) for x in theta],
            "theta_sha256": theta_sha256(theta), "adam": adam.to_dict(),
            "pins_sha256": _sha(spec["pins"])})
        if g % monitor_every == 0:
            monitor(g, theta)
        log(f"[producer_es] generation {g}: fitness {fmean(fits):.4f}, "
            f"disqualified {gens[-1]['disqualified']}, |grad| {gens[-1]['grad_norm']:.4f}")
    last = int(gens[-1]["generation"]) if gens else start
    due = [g for g in range(0, last + 1, monitor_every)
           if not any(int(m["generation"]) == g for m in mon)]
    return _finish(out, spec, expected, rows_seen,
                   ([] if last >= generations else [f"stopped at generation {last}"])
                   + ([f"monitor missing at generations {due}"] if due else []),
                   {"generation": last, "theta_sha256": theta_sha256(theta)}, log)


# --- cli ---


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", required=True, choices=(*MODES, "freeze", "flag-calibrate"))
    p.add_argument("--training-dir", help="EXP-0058 bar/ (training set, schedule)")
    p.add_argument("--bar-dir", help="noise-floor / probe / train: this driver's bar run")
    p.add_argument("--margins", help="probe / train: the noise-floor run's margins.json")
    p.add_argument("--out-dir")
    p.add_argument("--prereg", default=str(PREREG_PATH), help="freeze: the file to write / compare")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-world-episodes", type=int, help="bar dry run: the first N")
    p.add_argument("--atlas", help="bar: atlas_v2.json (the EXP-0079 cross-check)")
    p.add_argument("--exp0079-episodes", help="bar: EXP-0079's rf3 episodes.csv")
    p.add_argument("--replicates", help="noise-floor dry run: comma-separated (default 1,2,3,4)")
    p.add_argument("--generations", type=int, help="train: target generation count")
    p.add_argument("--opt-seed", type=int, default=0)
    p.add_argument("--size", type=int, default=MINIBATCH, help="dry run: minibatch size")
    p.add_argument("--pairs", type=int, default=PAIRS, help="dry run: antithetic pairs")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--lr", type=float, default=ADAM_LR)
    p.add_argument("--monitor-every", type=int, default=MONITOR_EVERY)
    a = p.parse_args(argv)
    if a.mode == "flag-calibrate":
        bar, base_rows, _ = load_bar(a.bar_dir)
        margin, _ = load_margin(a.margins)
        fref, _ = load_flag_reference(a.margins, bar, base_rows, margin)
        print(json.dumps(calibrate_flag_threshold(load_training(a.training_dir), fref),
                         indent=1, sort_keys=True))
        return 0
    if a.mode == "freeze":
        status, sha = freeze_prereg(Path(a.prereg))
        print(f"[producer_es] pre-registration {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.training_dir is None or a.out_dir is None:
        p.error(f"--mode {a.mode} needs --training-dir and --out-dir")
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    common = dict(training_dir=a.training_dir, out_dir=a.out_dir, workers=a.workers,
                  horizon=a.horizon, source_commit=commit, dry_run=a.dry_run)
    if a.mode == BAR:
        return run_bar(**common, max_world_episodes=a.max_world_episodes, atlas=a.atlas,
                       exp0079_episodes=a.exp0079_episodes)
    if a.bar_dir is None:
        p.error(f"--mode {a.mode} needs --bar-dir")
    if a.mode == NOISE:
        reps = NOISE_REPLICATES if a.replicates is None else tuple(
            int(x) for x in a.replicates.split(","))
        return run_noise_floor(**common, bar_dir=a.bar_dir, replicates=reps)
    if a.margins is None:
        p.error(f"--mode {a.mode} needs --margins")
    if a.mode == PROBE:
        return run_probe(**common, bar_dir=a.bar_dir, margins=a.margins, size=a.size,
                         pairs=a.pairs, sigma=a.sigma)
    if a.generations is None:
        p.error("--mode train needs --generations")
    return run_train(**common, bar_dir=a.bar_dir, margins=a.margins, generations=a.generations,
                     opt_seed=a.opt_seed, size=a.size, pairs=a.pairs, sigma=a.sigma, lr=a.lr,
                     monitor_every=a.monitor_every)


if __name__ == "__main__":
    sys.exit(main())
