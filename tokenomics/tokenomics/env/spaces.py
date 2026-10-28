"""Observation/action space construction + normalization constants (ADR-0025/0026).

Track B redesign. Producers own *multiple* capacity types and many producers may
own the same type, so the old one-producer-per-type assumption is gone. Spaces are
derived from the config so they scale with the economy:

  producing obs = 2T (per-type cap+max) + 1 (requested) + 3 (invest-running flags)
                  + G (per-input stock) + 3 (inv units/limit/headroom) + 2P
                  + 2*Q (open consumer demand: quantity + tokens per product, #2) + 1
  ordering  obs = K*3 (order slots: active, tokens, quantity — NO capacity, ADR-0029)
                  + P (producer cut-off prices) + 1 (cycle progress)
  producing action = Box(G + 3)  -> G replenish quantities + 3 investment toggles
                     (restoration, upgrade, inventory expansion), raw reals
  ordering  action = Box(K*P)    -> per slot, raw logits over *all* producers;
                     decode restricts to the order's eligible producers

P = num producers, T = num capacity types, K = max order slots, G = global input-
product vocabulary (sorted union over all composite recipes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from gymnasium import spaces

from ..domain.common import CapacityType, Product, ProducerId
from ..domain.config import Configuration

# An order lives up to ORDER_LIFETIME cycles (TTL=2 -> counters {0,1,2}) so up to
# that many orders of one stream coexist before the oldest times out.
ORDER_LIFETIME = 3
# Investment streams a producer can have in flight (restoration/upgrade/inventory
# expansion). Each is gated to one at a time, so they do NOT stack across cycles.
INVESTMENT_STREAMS = 3
OBS_CLIP = 10.0
ACTION_LOGIT_BOUND = 10.0

# Enriched orderer observation block widths (EXP-0006A, orderer_obs_version 2). Each
# mirrors a block the encoder emits from the domain OrdererEconomyView; the dim and
# the encoder are kept in lockstep by test D2. Per order slot: scalar header, then
# the slot product one-hot, then the product-level features, then the producer-
# product grid; plus one global producer block per producer and cycle progress.
ORDERER_SLOT_SCALARS = 10  # active, tokens, qty, tokens/unit, age, ttl, type(3), elig#
ORDERER_PRODUCT_FEATURES = 9  # open demand q/tok, demand EMA, rej EMA, depth, recipe(3), elig#
ORDERER_PP_FEATURES = 15  # per (slot, producer): eligibility, makeability, bottleneck, EMAs
ORDERER_PRODUCER_FEATURES = 16  # per producer: capacity/load/cut-off/in-flight/rates
ORDERER_PRODUCER_CUTOFF_OFFSET = 8  # cut-off price's index within that block
# (_enriched_producer_block: total_cur, total_max, headroom, min/mean ratio, requested,
#  reserved, total_load, [8] cut_off, ...). Guarded by the obs_price v1==v2 test.

# Product-stream orderer observation block widths (EXP-0007A, orderer_obs_version 3,
# ADR-0052). A stream agent holds a single fixed product, so the obs drops the product
# one-hot, carries the product features once (not per slot), and collapses the
# producer-product grid from K*P to P. The width scales with producers (P rows) and
# backlog (K), never with the product vocabulary Q — the key simplification.
STREAM_PRODUCT_FEATURES = ORDERER_PRODUCT_FEATURES  # once per stream (fixed product)
STREAM_SLOT_SCALARS = 9  # active, tokens, qty, tokens/unit, age, ttl, type(3) — NO product
# per producer row: global producer block + producer-product block for the fixed product.
STREAM_PRODUCER_FEATURES = ORDERER_PRODUCER_FEATURES + ORDERER_PP_FEATURES
# cut-off price index within a producer row (it lives in the producer-global sub-block).
STREAM_PRODUCER_CUTOFF_OFFSET = ORDERER_PRODUCER_CUTOFF_OFFSET

# Entity-factored producer observation block widths (ADR-0053, producer_obs_version 2,
# EXP-0010A). The encoder emits the domain ProducerEntityView as rows: producer-wide
# globals, one row per owned capacity type, one per declared product, one per input
# product, then the recipe-edge relation planes, then cycle progress. Each row leads
# with an active-mask column (mask, not slot index, defines a real row — ADR-0053);
# padded rows are all-zero with mask 0. Dim and encoder are tied by the dim-tie test.
# cut-off, inv units/limit/headroom, outstanding, invest(3), then the 4 producer-global
# action economics the cap rows don't carry per-type (ADR-0044): max-replenish,
# replenish cost, inventory-expansion increases + cost.
PRODUCER_GLOBAL_FEATURES = 12
PRODUCER_CAP_ROW_FEATURES = 16  # mask + owned/cur/max/headroom/ratio/degr/reserved/
#                                 prev_req/free/reserved_ratio/restore(2)/upgrade(2)/demand
PRODUCER_PRODUCT_ROW_FEATURES = 19  # mask + eligible/demand(2)/recipe(4)/EMA(4)/resv/
#                                     makeable(3)/bottleneck(3)
PRODUCER_INPUT_ROW_FEATURES = 5  # mask + inventory/outstanding/blocked/downstream


@dataclass
class ProducerMechanics:
    """Per-producer investment economics exposed in the producing obs (ADR-0044).
    Static config values that vary per scale seed; `None` token costs (unbounded
    funding target) and a producer with no inventory expansion read 0."""

    replenish_cost_per_unit: float
    restoration_cost: float
    upgrade_cost: float
    inventory_expansion_increases: int
    inventory_expansion_cost: float


@dataclass
class EnvSpec:
    producer_ids: list[ProducerId]
    capacity_types: list[CapacityType]
    type_index: dict[CapacityType, int]
    # global input-product vocabulary: sorted union over all composite recipes.
    input_products: list[Product]
    input_index: dict[Product, int]
    # sorted vocabulary of every product that can be the subject of an order
    # (= all process-sheet products). The ordering obs one-hots each slot's product
    # over this so a seed-randomized demand product is observable (B9, not hidden).
    order_products: list[Product]
    order_product_index: dict[Product, int]
    # per producer: capacity types it owns, and the input products it can
    # replenish (inputs of composite products it declares it produces, ADR-0028).
    producer_owns: dict[ProducerId, set[CapacityType]]
    producer_inputs: dict[ProducerId, list[Product]]
    # per producer replenishment ceiling (config max_replenish, default limit).
    max_replenish: dict[ProducerId, int]
    # per producer investment economics for the obs mechanics block (ADR-0044).
    producer_mechanics: dict[ProducerId, ProducerMechanics]
    cap_ref: float
    token_ref: float
    price_ref: float
    inv_ref: float
    k: int = ORDER_LIFETIME
    # 1 = legacy capacity-blind ordering obs (ADR-0029); 2 = enriched producer/
    # product/order obs (EXP-0006A). Opt-in so v1 checkpoints stay valid. Hashed into
    # config_fingerprint (which takes an explicit orderer_obs_version) so a v2 model
    # can't be mis-scored against a v1 economy.
    orderer_obs_version: int = 1
    # 1 = legacy flat producer obs (encode_producing); 2 = entity-factored producer
    # obs (ADR-0053, EXP-0010A). Opt-in so flat_v1 producer checkpoints stay valid;
    # hashed into config_fingerprint so an entity checkpoint can't collide with a flat.
    producer_obs_version: int = 1

    @property
    def num_producers(self) -> int:
        return len(self.producer_ids)

    @property
    def k_stream(self) -> int:
        # Corrected per-stream backlog bound (PRODUCT_AGNOSTIC O0). A product-stream
        # agent (ADR-0052) serves ONE (source, product) stream. It is keyed by
        # (source, product), NOT by order type, so the stream holds up to ORDER_LIFETIME
        # replenishment orders (the TTL window) PLUS the producer's gated investment
        # orders whose means-of-production product is that same product — and there are
        # INVESTMENT_STREAMS independently-gated investment types. So the bound is
        # ORDER_LIFETIME + INVESTMENT_STREAMS: enough to give every order a controllable
        # decode slot (no uniform-routing fallback for overflow), yet still a CONSTANT —
        # it does NOT inherit the global legacy `k`'s ORDER_LIFETIME*max_rel term, so it
        # never leaks producer input-product width into the orderer contract (the
        # ADR-0052 goal). Exposed as a corrected versioned contract (orderer_obs_version
        # 4); v3 keeps `k` so its checkpoints stay loadable.
        return ORDER_LIFETIME + INVESTMENT_STREAMS if self.orderer_obs_version >= 4 else self.k

    @property
    def num_types(self) -> int:
        return len(self.capacity_types)

    @property
    def num_inputs(self) -> int:
        return len(self.input_products)

    @property
    def num_order_products(self) -> int:
        return len(self.order_products)

    @property
    def producing_mechanics_len(self) -> int:
        # Per-agent producer-mechanics block (ADR-0044): the action semantics that
        # vary per scale seed so a trained policy can adapt instead of memorizing a
        # fixed operating point — per-type degradation units (num_types) +
        # restoration amount + upgrade amount + max-replenish + replenish cost +
        # restoration cost + upgrade cost + inventory-expansion increases + its cost.
        return self.num_types + 8

    @property
    def producing_mechanics_start(self) -> int:
        # The mechanics block is appended just before the trailing cycle-progress
        # scalar, so the front offsets used by the obs baselines stay valid.
        return self.producing_obs_dim - self.producing_mechanics_len - 1

    @property
    def producing_obs_dim(self) -> int:
        if self.producer_obs_version >= 2:
            return self.producer_entity_obs_dim
        # ... 2*num_inputs (per-input inventory stock + outstanding inbound, ADR-0041)
        # ... + 2P (per-producer summary) + 2*num_order_products (open consumer
        # demand: quantity + tokens per product, #2) + producing_mechanics_len
        # (per-agent mechanics, ADR-0044) + 1 (cycle progress).
        return (
            2 * self.num_types + 1 + 3 + 2 * self.num_inputs + 3
            + 2 * self.num_producers + 2 * self.num_order_products
            + self.producing_mechanics_len + 1
        )

    @property
    def producer_entity_obs_dim(self) -> int:
        # ADR-0053 entity producer obs: globals + per-type capacity rows (padded to
        # num_types) + per-product rows (padded to num_order_products, the upper bound
        # on any producer's `produces`) + per-input rows (padded to num_inputs) + the
        # two recipe-edge relation planes (product x capacity, product x input) + the
        # trailing cycle-progress scalar. Mirror of encode_producing_entity (dim-tie).
        Q, T, G = self.num_order_products, self.num_types, self.num_inputs
        return (
            PRODUCER_GLOBAL_FEATURES
            + T * PRODUCER_CAP_ROW_FEATURES
            + Q * PRODUCER_PRODUCT_ROW_FEATURES
            + G * PRODUCER_INPUT_ROW_FEATURES
            + Q * T  # product->capacity relation plane
            + Q * G  # product->input relation plane
            + 1
        )

    @property
    def ordering_obs_dim(self) -> int:
        if self.orderer_obs_version >= 3:
            return self._ordering_obs_dim_stream
        if self.orderer_obs_version >= 2:
            return self._ordering_obs_dim_enriched
        # v1: per slot: active, tokens, quantity (no capacity, ADR-0029) + a product
        # one-hot (B9 observability); plus one cut-off price per producer and the
        # cycle-progress scalar.
        return self.k * (3 + self.num_order_products) + self.num_producers + 1

    @property
    def _ordering_obs_dim_stream(self) -> int:
        # EXP-0007A product-stream obs (ADR-0052): product features once + K backlog
        # slots (no product one-hot) + one producer row per producer + cycle progress.
        # Mirror of encode_ordering_stream (test ties them). No Q dependence.
        P, K = self.num_producers, self.k_stream
        return STREAM_PRODUCT_FEATURES + K * STREAM_SLOT_SCALARS + P * STREAM_PRODUCER_FEATURES + 1

    def ordering_price_indices(self) -> list[int]:
        """Indices of the P producer cut-off prices within the ordering obs, for the
        active obs version — the only producer signal `obs_price` routes by (ADR-0029).
        Single source of truth so `decide_ordering_obs` reads the *same* price regardless
        of version: v1 stores them contiguously after the K slots; v2 (enriched) strides
        them through the per-producer block. Keeps the search candidate fixed across the
        EXP-0006A/B v1->v2 A/B."""
        P, K, Q = self.num_producers, self.k_stream, self.num_order_products
        if self.orderer_obs_version >= 3:
            head = STREAM_PRODUCT_FEATURES + K * STREAM_SLOT_SCALARS  # before producer rows
            return [head + j * STREAM_PRODUCER_FEATURES + STREAM_PRODUCER_CUTOFF_OFFSET
                    for j in range(P)]
        if self.orderer_obs_version >= 2:
            slot_block = K * (ORDERER_SLOT_SCALARS + Q + ORDERER_PRODUCT_FEATURES
                              + P * ORDERER_PP_FEATURES)
            return [slot_block + j * ORDERER_PRODUCER_FEATURES + ORDERER_PRODUCER_CUTOFF_OFFSET
                    for j in range(P)]
        start = K * (3 + Q)  # after the K order slots, before cycle progress
        return [start + j for j in range(P)]

    @property
    def _ordering_obs_dim_enriched(self) -> int:
        # EXP-0006A: per slot = scalar header + product one-hot + product features +
        # producer-product grid; plus one global producer block per producer + cycle
        # progress. Mirror of the encode_ordering_enriched layout (test D2).
        P, K, Q = self.num_producers, self.k, self.num_order_products
        return (
            K * (ORDERER_SLOT_SCALARS + Q + ORDERER_PRODUCT_FEATURES + P * ORDERER_PP_FEATURES)
            + P * ORDERER_PRODUCER_FEATURES
            + 1
        )

    @property
    def ordering_action_dim(self) -> int:
        # k_stream == k for v1/v2/v3 (back-compat); only the corrected v4 stream
        # contract narrows it to the per-stream bound (PRODUCT_AGNOSTIC O0).
        return self.k_stream * self.num_producers

    @property
    def producing_action_dim(self) -> int:
        return self.num_inputs + 3

    @property
    def global_mechanics_len(self) -> int:
        # Per-producer producer-mechanics rolled up for the critic (plan item 1):
        # total degradation this cycle (sum over types) + restoration amount +
        # upgrade amount + max-replenish + replenish cost + restoration cost +
        # upgrade cost + inventory-expansion increases + its cost = 9 fields. The
        # critic counterpart of the per-agent obs block (ADR-0044): without it two
        # scale seeds give the critic the same global vector but a different return
        # distribution, so value targets are noisy under topology DR.
        return 9

    @property
    def global_mechanics_start(self) -> int:
        # Mechanics loop follows the 5-field per-producer state block.
        return 5 * self.num_producers

    @property
    def global_state_dim(self) -> int:
        # Centralized-critic input (ADR-0004): per-producer rolled-up state
        # (total cap, max total, requested, inventory units, cut-off) + per-producer
        # mechanics block (plan item 1) + open consumer demand (quantity + tokens per
        # product, #2) + progress.
        return (
            5 * self.num_producers
            + self.global_mechanics_len * self.num_producers
            + 2 * self.num_order_products + 1
        )

    def producing_obs_space(self) -> spaces.Box:
        return spaces.Box(0.0, OBS_CLIP, shape=(self.producing_obs_dim,), dtype=np.float32)

    def ordering_obs_space(self) -> spaces.Box:
        return spaces.Box(0.0, OBS_CLIP, shape=(self.ordering_obs_dim,), dtype=np.float32)

    def producing_action_space(self) -> spaces.Box:
        return spaces.Box(
            -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND, shape=(self.producing_action_dim,), dtype=np.float32
        )

    def ordering_action_space(self) -> spaces.Box:
        return spaces.Box(
            -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND, shape=(self.ordering_action_dim,), dtype=np.float32
        )


# Optional economy fields added after checkpoints were stamped (add-shocks-i-drift D5): when
# unset they are left out, so every config without them keeps its fingerprint. The consumer
# `schedule` (add-measurement-gate-harness 2.2) is included: hashing its empty default had
# silently moved every fingerprint (EXP-0006B pins 548120bd72921234); omitting it restores the
# pre-M1 values.
_OMIT_WHEN_UNSET = ("shocks", "demand_drift", "demand_pattern")
_OMIT_CONSUMER_WHEN_EMPTY = ("schedule",)


def _canonical_economy(config: Configuration) -> str:
    """Deterministic JSON of every *economy-determining* field of a config — recipes,
    capacities, degradation, investment/replenish costs, emission, consumer demand
    distribution and shocks. `dataclasses.asdict` recurses the frozen config; `sort_keys` + a
    frozenset->sorted-list default make it order-independent and stable across loads.
    Unset optional fields (`_OMIT_WHEN_UNSET`, an empty consumer `schedule`) are dropped so
    legacy fingerprints hold."""
    economy = asdict(config)
    for key in _OMIT_WHEN_UNSET:
        if key in economy and economy[key] is None:
            del economy[key]
    for consumer in economy.get("consumers", ()):
        for key in _OMIT_CONSUMER_WHEN_EMPTY:
            if key in consumer and not consumer[key]:
                del consumer[key]
    return json.dumps(
        economy,
        sort_keys=True,
        default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else repr(o),
    )


def config_fingerprint(
    config: Configuration, orderer_obs_version: int = 1, producer_obs_version: int = 1
) -> str:
    """Stable short hash identifying a config's *economy* (review P1, #7).

    Stamped into a checkpoint at save time so an eval can detect it is scoring a policy
    on a different economy. Covers two things: (1) the shape-determining structure
    (producer ids, capacity types, product vocabularies, derived dims) so a dim change
    is caught, AND (2) the full economic content (recipes, capacities, degradation,
    costs, emission, consumer demand distribution) so configs that share *shape* but
    differ in *dynamics* — e.g. competition.json vs complex.json (fixed vs randomized
    demand), random_demand.json vs composite.json (different recipes) — no longer
    collide and get silently reported as a valid match (the P1 review defect).
    Two configs with identical fingerprints are interchangeable as a trained-policy
    comparison; a mismatch is at best a transfer test, at worst an off-config eval.

    `orderer_obs_version` (EXP-0006A) is part of the contract: a v2 (enriched) orderer
    checkpoint and a v1 one on the *same* economy must not collide, so the spec is
    built at that version (changing ordering_obs_dim) and the version is hashed in."""
    sp = build_spec(
        config, orderer_obs_version=orderer_obs_version, producer_obs_version=producer_obs_version
    )
    fields = [
        sorted(sp.producer_ids),
        sorted(sp.capacity_types),
        sorted(sp.order_products),
        sorted(sp.input_products),
        sp.producing_obs_dim,
        sp.ordering_obs_dim,
        sp.global_state_dim,
        sp.producing_action_dim,
        sp.ordering_action_dim,
        sp.k,
        sp.orderer_obs_version,  # EXP-0006A: v1/v2 orderer obs contract
    ]
    # ADR-0053 entity producer obs (v2) already changes producing_obs_dim above, so v1
    # vs v2 fingerprints differ regardless. Only fold the version itself in for v2+ so
    # legacy flat_v1 checkpoints keep their historical fingerprint (back-compat).
    if sp.producer_obs_version >= 2:
        fields.append(("producer_obs_version", sp.producer_obs_version))
    fields.append(_canonical_economy(config))  # P1: same shape, different economy -> diff fp
    canonical = repr(tuple(fields))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_spec(
    config: Configuration, orderer_obs_version: int = 1, producer_obs_version: int = 1
) -> EnvSpec:
    producer_ids = sorted(p.id for p in config.producer_configs)
    capacity_types = sorted({t for p in config.producer_configs for t in p.capacities})
    type_index = {t: i for i, t in enumerate(capacity_types)}

    # global input-product vocabulary: every product consumed as an input by some
    # composite recipe (ADR-0025 B5). Sorted for a stable action layout.
    input_products = sorted({p for ps in config.process_sheets for p in ps.inputs})
    input_index = {p: i for i, p in enumerate(input_products)}

    # every product that can be ordered = every product with a process sheet.
    order_products = sorted(ps.product for ps in config.process_sheets)
    order_product_index = {p: i for i, p in enumerate(order_products)}

    sheet_inputs = {ps.product: ps.inputs for ps in config.process_sheets}

    producer_owns: dict[ProducerId, set[CapacityType]] = {
        p.id: set(p.capacities) for p in config.producer_configs
    }
    # a producer can replenish an input g iff g is an input of some composite
    # product it declares it produces (ADR-0028).
    producer_inputs: dict[ProducerId, list[Product]] = {}
    for p in config.producer_configs:
        relevant: set[Product] = set()
        for product in p.produces:
            if sheet_inputs.get(product):
                relevant.update(sheet_inputs[product])
        producer_inputs[p.id] = sorted(relevant)

    max_replenish: dict[ProducerId, int] = {
        p.id: (p.max_replenish if p.max_replenish is not None else p.inventory_limit)
        for p in config.producer_configs
    }

    # investment economics for the obs mechanics block (ADR-0044). None costs and a
    # producer with no inventory expansion read 0 — an unbounded/absent knob.
    producer_mechanics: dict[ProducerId, ProducerMechanics] = {
        p.id: ProducerMechanics(
            replenish_cost_per_unit=float(p.replenish_cost_per_unit or 0),
            restoration_cost=float(p.restoration.cost or 0),
            upgrade_cost=float(p.upgrade.cost or 0),
            inventory_expansion_increases=(
                p.inventory_expansion.increases if p.inventory_expansion else 0
            ),
            inventory_expansion_cost=float(
                p.inventory_expansion.cost if p.inventory_expansion and p.inventory_expansion.cost else 0
            ),
        )
        for p in config.producer_configs
    }

    cap_ref = float(max(sum(p.capacities.values()) for p in config.producer_configs))
    token_ref = float(config.cycle_emission)
    price_ref = max(1.0, token_ref / cap_ref)
    inv_ref = float(max(1, max(p.inventory_limit for p in config.producer_configs)))

    # Order slots must cover the most orders any one ordering agent can hold in a
    # cycle (ADR-0025). A consumer agent has a single order stream (-> ORDER_LIFETIME
    # concurrent). A producer agent has INVESTMENT_STREAMS gated investment orders
    # (one each, no stacking) plus one replenishment stream per relevant input,
    # each ungated so it stacks up to ORDER_LIFETIME. Otherwise orders beyond K are
    # invisible in the obs and fall back to an uncontrollable even split in decode.
    max_rel = max((len(v) for v in producer_inputs.values()), default=0)
    k = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)
    return EnvSpec(
        producer_ids=producer_ids,
        capacity_types=capacity_types,
        type_index=type_index,
        input_products=input_products,
        input_index=input_index,
        order_products=order_products,
        order_product_index=order_product_index,
        producer_owns=producer_owns,
        producer_inputs=producer_inputs,
        max_replenish=max_replenish,
        producer_mechanics=producer_mechanics,
        cap_ref=cap_ref,
        token_ref=token_ref,
        price_ref=price_ref,
        inv_ref=inv_ref,
        k=k,
        orderer_obs_version=orderer_obs_version,
        producer_obs_version=producer_obs_version,
    )
