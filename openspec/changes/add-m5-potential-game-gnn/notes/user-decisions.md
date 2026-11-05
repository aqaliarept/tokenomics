# User decisions — add-m5-potential-game-gnn (producer learning)

- 2026-11-05: core design accepted as proposed: learn the producer alone first (learned correction on entity_planner_buyahead_k4, MWW frozen as orderer), ES as the learner (policy gradient only as a diagnosed fallback), shared per-producer features; the joint potential game and the GNN are a gated phase 2 (after a confirmed held-out producer result and evidence of orderer headroom).
- 2026-11-05: pre-step: run "expand inventory earlier" alone on k = 4 first; if it helps, it becomes the base. The k = 5–6 sweep is not run.
- 2026-11-05: tranche M5-P = 80 vCPU-h released; the supervisor decides spend within it (per-run launch-fit, larger-pod-first, pinned data center).
- 2026-11-05: the training guard is measured relative to buy-ahead k = 4, with its noise margin re-measured on k = 4.
