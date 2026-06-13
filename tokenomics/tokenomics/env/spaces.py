"""Observation/action space construction + normalization constants.

Dimensions follow ADR-0002 but are derived from the config so they scale with
the network size:
  producing obs = 6 (own) + 2*P (producer summary) + 1 (cycle progress)
  ordering  obs = K*(2+T) (order slots) + 2*P + 1
  producing action = Discrete(4)  -> (do_restoration, do_upgrade) bits
  ordering  action = Box(K*T)     -> raw logits, simplex-normalized at decode

P = num producers, T = num capacity types, K = max order slots (=2, ADR-0002).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces

from ..domain.config import Configuration

# K=3: a single ordering agent can hold up to 3 orders at once because
# still-processing orders carry over and are re-placed next to a freshly placed
# order. TTL=2 lets an order live across cycles with counters {0,1,2} before
# timeout, so up to 3 coexist for a consumer agent. (ADR-0002 originally said
# K=2; corrected here — it had not accounted for carry-over.)
K_ORDER_SLOTS = 3
OBS_CLIP = 10.0
ACTION_LOGIT_BOUND = 10.0


@dataclass
class EnvSpec:
    producer_ids: list[str]
    capacity_types: list[str]
    type_index: dict[str, int]
    type_to_producer: dict[str, str]
    cap_ref: float
    token_ref: float
    price_ref: float
    k: int = K_ORDER_SLOTS

    @property
    def num_producers(self) -> int:
        return len(self.producer_ids)

    @property
    def num_types(self) -> int:
        return len(self.capacity_types)

    @property
    def producing_obs_dim(self) -> int:
        return 6 + 2 * self.num_producers + 1

    @property
    def ordering_obs_dim(self) -> int:
        return self.k * (2 + self.num_types) + 2 * self.num_producers + 1

    @property
    def ordering_action_dim(self) -> int:
        return self.k * self.num_types

    def producing_obs_space(self) -> spaces.Box:
        return spaces.Box(0.0, OBS_CLIP, shape=(self.producing_obs_dim,), dtype=np.float32)

    def ordering_obs_space(self) -> spaces.Box:
        return spaces.Box(0.0, OBS_CLIP, shape=(self.ordering_obs_dim,), dtype=np.float32)

    def producing_action_space(self) -> spaces.Discrete:
        # 0:noop 1:restoration 2:upgrade 3:both  (bit0=restoration, bit1=upgrade)
        return spaces.Discrete(4)

    def ordering_action_space(self) -> spaces.Box:
        return spaces.Box(
            -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND, shape=(self.ordering_action_dim,), dtype=np.float32
        )


def build_spec(config: Configuration) -> EnvSpec:
    producer_ids = sorted(p.id for p in config.producer_configs)
    capacity_types = sorted({p.type for p in config.producer_configs})
    type_index = {t: i for i, t in enumerate(capacity_types)}

    # Phase 1 assumes exactly one producer per capacity type (single product per
    # firm type — deep-research Phase 1). Enforce it so decode can map a required
    # capacity type straight to its producer.
    type_to_producer: dict[str, str] = {}
    for p in config.producer_configs:
        if p.type in type_to_producer:
            raise ValueError(
                f"Phase 1 requires one producer per capacity type; {p.type} has multiple"
            )
        type_to_producer[p.type] = p.id

    cap_ref = float(max(p.capacity for p in config.producer_configs))
    token_ref = float(config.cycle_emission)
    price_ref = max(1.0, token_ref / cap_ref)
    return EnvSpec(
        producer_ids=producer_ids,
        capacity_types=capacity_types,
        type_index=type_index,
        type_to_producer=type_to_producer,
        cap_ref=cap_ref,
        token_ref=token_ref,
        price_ref=price_ref,
    )
