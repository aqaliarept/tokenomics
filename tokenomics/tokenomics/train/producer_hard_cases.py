"""Reachable producer hard-case mining (ADR-0049 Phase 3).

The decode-aware clone (EXP-0002) passes `d3_p00` and holds across trained topologies,
but unseen topologies vary. Phase 3 mines REACHABLE producer states — visited by the
current learner under the base orderer — where it diverges from `obs_stock_aware` and
the rollout underperforms, tagged by the EXP-0001 failure surface, to retrain on
(Phase 4). v1 only uses states from real rollouts; it never synthesizes raw domain
state.

This module's pure core is the failure tagger: given one producer decision's
diagnostics it returns the accepted failure tags. The mining loop computes those
diagnostics live from the env."""

from __future__ import annotations

from dataclasses import dataclass

# Accepted failure tags (ADR-0049 Phase 3) — the EXP-0001 producer-trace diagnosis menu.
MIDDLE_CAPACITY_LOW = "middle_capacity_low"
LAYER_CAPACITY_IMBALANCE = "layer_capacity_imbalance"
BLOCKED_RESERVATION_HIGH = "blocked_reservation_high"
MISSING_INPUT_INVENTORY = "missing_input_inventory"
INBOUND_MISSING_OR_STALLED = "inbound_missing_or_stalled"
DOWNSTREAM_DEMAND_UNSERVED = "downstream_demand_unserved"

ACCEPTED_TAGS = frozenset({
    MIDDLE_CAPACITY_LOW, LAYER_CAPACITY_IMBALANCE, BLOCKED_RESERVATION_HIGH,
    MISSING_INPUT_INVENTORY, INBOUND_MISSING_OR_STALLED, DOWNSTREAM_DEMAND_UNSERVED,
})


@dataclass
class Thresholds:
    """Tag thresholds. Heuristic, tuned to the EXP-0001 trace (L1 cap collapse, blocked
    reservations piling up, inputs at 0 stock). Exposed so a mining run can sweep them."""

    cap_low: float = 0.5        # capacity ratio below this = starved
    blocked_high: int = 5       # reservations waiting on inputs above this = blocked-high
    imbalance: float = 0.3      # another layer this much healthier = imbalance
    demand_min: int = 5         # open consumer demand units above this = real demand


@dataclass
class DecisionDiag:
    """One producer decision's diagnostics, computed live during a rollout. Cross-layer
    fields (`layer_cap_ratio`, `other_layer_cap_ratio`, `open_demand_units`) come from
    the per-cycle system snapshot; the rest are the acting producer's own state."""

    layer: int
    depth: int
    cap_ratio: float            # acting producer's weakest current/max capacity ratio
    blocked: int                # reservations waiting on missing inputs
    reserved_cap: int           # capacity held by those reservations
    min_input_inv: int          # lowest stock over the producer's relevant inputs
    total_inbound: int          # outstanding replenishment units in flight
    has_inputs: bool            # producer assembles composites (has relevant inputs)
    layer_cap_ratio: float      # mean cap ratio over this producer's layer
    other_layer_cap_ratio: float  # best mean cap ratio among the OTHER layers
    open_demand_units: int      # open consumer demand this cycle (units)


def classify_failures(diag: DecisionDiag, thresholds: Thresholds | None = None) -> set[str]:
    """Map a producer decision to the accepted failure tags it matches (zero or more).
    A clean decision returns an empty set; the miner keeps only tagged decisions."""
    th = thresholds or Thresholds()
    tags: set[str] = set()
    is_middle = 0 < diag.layer < diag.depth - 1

    if is_middle and diag.cap_ratio < th.cap_low:
        tags.add(MIDDLE_CAPACITY_LOW)
    if diag.other_layer_cap_ratio - diag.layer_cap_ratio >= th.imbalance:
        tags.add(LAYER_CAPACITY_IMBALANCE)
    if diag.blocked >= th.blocked_high:
        tags.add(BLOCKED_RESERVATION_HIGH)
    if diag.has_inputs and diag.min_input_inv == 0:
        tags.add(MISSING_INPUT_INVENTORY)
        if diag.total_inbound == 0:  # nothing on the way -> not just timing
            tags.add(INBOUND_MISSING_OR_STALLED)
    if diag.open_demand_units >= th.demand_min and diag.cap_ratio < th.cap_low:
        tags.add(DOWNSTREAM_DEMAND_UNSERVED)
    return tags
