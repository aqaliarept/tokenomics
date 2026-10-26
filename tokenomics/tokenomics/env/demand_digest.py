"""Per-episode demand-schedule digest (design D5, add-measurement-gate-harness;
gate-harness "Identical demand across observation contracts").

The gate pairs a candidate with roster members that may run under other observation or
ordering contracts. The pairing is fair only if every run of a (world, episode seed)
faces the same demand. Demand depends only on config + episode seed, so the env can
digest it at `reset` and the gate can compare digests per episode (task 8.1) without
trusting that claim.

What is digested: every consumer's id and the full schedule it replays, as built at
reset, in sorted consumer-id order. A `ScriptedConsumer` (fixed product, quantity 1 per
cycle) is recorded as the one-entry schedule `[(product, 1)]`, which is the demand it
presents. The digest is fixed at reset, so policy actions, death and the cycle count do
not move it. Not digested: token amounts. Each consumer's emission share is a function
of the config alone (`cycleEmission`, consumer count), so it is the same for every run
of one world.

How: sha256 over a canonical compact JSON encoding, as a hex string. It does not use
`hash()`, so it does not depend on `PYTHONHASHSEED` and digests from separate worker
processes compare equal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from ..domain.common import ConsumerId
from ..domain.consumer import ScheduleEntry


def demand_digest(schedules: Mapping[ConsumerId, Sequence[ScheduleEntry]]) -> str:
    """sha256 hex digest of the canonical encoding
    `[[consumer_id, [[product, quantity], ...]], ...]`, consumers sorted by id."""
    canonical = [
        [cid, [[int(product), int(quantity)] for product, quantity in schedules[cid]]]
        for cid in sorted(schedules)
    ]
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
