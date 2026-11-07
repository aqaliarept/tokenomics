# EXP-0124: the EXP-0121 probe with the order-aware accept pass (hard-stopped, no data)

- Pre-registration note `notes/exp0124-prereg.md` (SHA `7e7ecb4f…`). Code 264c857.
- Pod 8tcohug99rbfsy (secure 8 vCPU, EUR-IS-1), 16:35–16:58 UTC ≈ **3.1 vCPU-h** (stop 3, M6-A).
- **Outcome: the watchdog's hard stop fired before the slowest settings finished.** The driver wrote its results
  only after all 8 settings, so nothing was pulled. The pod was removed.
- **Cause (my sizing error):** the timing came from a 3,000-turn local sample. Measured afterwards on the real half
  (35,183 accept turns, mean 4.6 items, p99 20, max 42), the slowest setting (w64 L3 e90, ordered) costs ≈ 20 s per
  epoch locally, ≈ 11 s on the pod. That is ≈ 17 min of fitting, plus the held-out scoring and ≈ 4 min of pod setup:
  ≈ 24 min, past the 23-min stop.
- **Fix:** the probe now appends each setting's result to `probe_partial.jsonl` as it finishes.
- **No reading.** A re-run is the supervisor's call.
