# Go → Python port notes

The Python domain (`tokenomics/domain/`) is a faithful port of `emulation/domain/`
with a few deliberate divergences. All are latent Go bugs that never fired because
the Go emulator was driven over HTTP for single cycles with healthy capacity.

| # | Where | Go behavior | Python behavior | Why |
|---|-------|-------------|-----------------|-----|
| 1 | `System.producer_infos` | Computed once in `NewSystem`, never refreshed → ordering agents saw the *initial* producer state forever | Refreshed each cycle in `start_ordering` + fresh per-observation in the env | RL needs live capacity / cut-off price (ADR-0002) |
| 2 | `OrderingAgent.complete_cycle` | Sets `cmdHandled = true` → `handle_cmd` would raise on every cycle after the first | Resets to `false` | Multi-cycle episodes; mirrors `ProducingAgent.produce` reset |
| 3 | `System.complete_cycle` | A timed-out/rejected order is deleted while a producer still holds it as an `in_progress` booking → next `produce` emits a bid for a missing order and crashes | Dangling bookings are released when their order is removed | Capacity can collapse over a long episode |
| 4 | `distribute_investment_fund` | Raw `float32` division by `totalCutOffSum`; if every priced order has a zero cut-off price the total is 0 → Go yields inf/garbage (no panic) | Guarded: zero total → those orders funded 0, NaN orders split the remainder | Python raises `ZeroDivisionError`; zero price = no deficit |

## Float parity

Go computes bid prices and fund proportions in `float32`. The port routes those
spots through `numpy.float32` (`domain.common.f32`) so prices and allocations
match Go. Map iteration order differs (Go randomizes, Python dicts preserve
insertion order) — the port is therefore *more* deterministic than Go for the
NaN-remainder split.

## Not ported

`logEvent`/`slog` logging, the HTTP `application/` layer, and `swagger.yaml`. The
Go emulator remains the human-facing reference; this package is the RL training
environment.
