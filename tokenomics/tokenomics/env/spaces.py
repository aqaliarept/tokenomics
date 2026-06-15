"""Observation/action space construction + normalization constants (ADR-0025/0026).

Track B redesign. Producers own *multiple* capacity types and many producers may
own the same type, so the old one-producer-per-type assumption is gone. Spaces are
derived from the config so they scale with the economy:

  producing obs = 2T (per-type cap+max) + 1 (requested) + 3 (invest-running flags)
                  + G (per-input stock) + 3 (inv units/limit/headroom) + 2P + 1
  ordering  obs = K*(3+T) (order slots: active, tokens, quantity, per-type cap)
                  + 2P (producer summary) + 1 (cycle progress)
  producing action = Box(G + 3)  -> G replenish quantities + 3 investment toggles
                     (restoration, upgrade, inventory expansion), raw reals
  ordering  action = Box(K*P)    -> per slot, raw logits over *all* producers;
                     decode restricts to the order's eligible producers

P = num producers, T = num capacity types, K = max order slots, G = global input-
product vocabulary (sorted union over all composite recipes).
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class EnvSpec:
    producer_ids: list[ProducerId]
    capacity_types: list[CapacityType]
    type_index: dict[CapacityType, int]
    # global input-product vocabulary: sorted union over all composite recipes.
    input_products: list[Product]
    input_index: dict[Product, int]
    # per producer: capacity types it owns, and the input products it can
    # replenish (inputs of composite products it is eligible to assemble).
    producer_owns: dict[ProducerId, set[CapacityType]]
    producer_inputs: dict[ProducerId, list[Product]]
    # per producer replenishment ceiling (config max_replenish, default limit).
    max_replenish: dict[ProducerId, int]
    cap_ref: float
    token_ref: float
    price_ref: float
    inv_ref: float
    k: int = ORDER_LIFETIME

    @property
    def num_producers(self) -> int:
        return len(self.producer_ids)

    @property
    def num_types(self) -> int:
        return len(self.capacity_types)

    @property
    def num_inputs(self) -> int:
        return len(self.input_products)

    @property
    def producing_obs_dim(self) -> int:
        return 2 * self.num_types + 1 + 3 + self.num_inputs + 3 + 2 * self.num_producers + 1

    @property
    def ordering_obs_dim(self) -> int:
        return self.k * (3 + self.num_types) + 2 * self.num_producers + 1

    @property
    def ordering_action_dim(self) -> int:
        return self.k * self.num_producers

    @property
    def producing_action_dim(self) -> int:
        return self.num_inputs + 3

    @property
    def global_state_dim(self) -> int:
        # Centralized-critic input (ADR-0004): per-producer rolled-up state
        # (total cap, max total, requested, inventory units, cut-off) + progress.
        return 5 * self.num_producers + 1

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


def build_spec(config: Configuration) -> EnvSpec:
    producer_ids = sorted(p.id for p in config.producer_configs)
    capacity_types = sorted({t for p in config.producer_configs for t in p.capacities})
    type_index = {t: i for i, t in enumerate(capacity_types)}

    # global input-product vocabulary: every product consumed as an input by some
    # composite recipe (ADR-0025 B5). Sorted for a stable action layout.
    input_products = sorted({p for ps in config.process_sheets for p in ps.inputs})
    input_index = {p: i for i, p in enumerate(input_products)}

    sheet_require = {ps.product: ps.require for ps in config.process_sheets}
    sheet_inputs = {ps.product: ps.inputs for ps in config.process_sheets}

    producer_owns: dict[ProducerId, set[CapacityType]] = {
        p.id: set(p.capacities) for p in config.producer_configs
    }
    # a producer can replenish an input g iff g is an input of some composite
    # product the producer is *eligible* to assemble (owns every required type).
    producer_inputs: dict[ProducerId, list[Product]] = {}
    for p in config.producer_configs:
        owned = producer_owns[p.id]
        relevant: set[Product] = set()
        for product, require in sheet_require.items():
            if sheet_inputs[product] and all(t in owned for t in require):
                relevant.update(sheet_inputs[product])
        producer_inputs[p.id] = sorted(relevant)

    max_replenish: dict[ProducerId, int] = {
        p.id: (p.max_replenish if p.max_replenish is not None else p.inventory_limit)
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
        producer_owns=producer_owns,
        producer_inputs=producer_inputs,
        max_replenish=max_replenish,
        cap_ref=cap_ref,
        token_ref=token_ref,
        price_ref=price_ref,
        inv_ref=inv_ref,
        k=k,
    )
