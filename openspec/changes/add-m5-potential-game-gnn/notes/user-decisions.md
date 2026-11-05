# User decisions — add-m5-potential-game-gnn (producer learning)

- 2026-11-05: core design accepted as proposed: learn the producer alone first (learned correction on entity_planner_buyahead_k4, MWW frozen as orderer), ES as the learner (policy gradient only as a diagnosed fallback), shared per-producer features; the joint potential game and the GNN are a gated phase 2 (after a confirmed held-out producer result and evidence of orderer headroom).
- 2026-11-05: pre-step: run "expand inventory earlier" alone on k = 4 first; if it helps, it becomes the base. The k = 5–6 sweep is not run.
- 2026-11-05: tranche M5-P = 80 vCPU-h released; the supervisor decides spend within it (per-run launch-fit, larger-pod-first, pinned data center).
- 2026-11-05: the training guard is measured relative to buy-ahead k = 4, with its noise margin re-measured on k = 4.
- 2026-11-05: after EXP-0081 (probe signal 0.84–0.90 with the rule off; the any-episode disqualification rule threw out every member of minibatch B because 12-0044 / 12-0007 flag even rf3 on most re-draws), user chose "Compare flag rates": disqualify a member only if it is flagged on clearly more of its episodes than rf3 is on the same episodes under re-drawn randomness (threshold registered from EXP-0080 re-keys); deaths stay an absolute disqualifier. New pre-registration version (v3; v1 / v2 untouched), then re-run the L2 probe.
