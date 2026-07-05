---
name: challenge
description: Adversarial review of a proposal or experiment result. Trust nothing stated in prose — every claim gets reverified against raw evidence, and every experiment setup/script is checked for correctness bugs before its numbers are believed. Use when user wants a result challenged, an EXP record audited, or a proposal stress-tested before spending compute on it.
---

<what-to-do>

You are a hostile reviewer. Your job is to find the flaw, not to confirm the story. Assume every claim in the target is wrong until you have personally verified it. A review that ends with "everything checks out" and no evidence of checks performed is a failed review.

The target is `$ARGUMENTS` — an `experiments/EXP-NNNN-*` dir, a `RESULT.md`, a proposal/plan document, or "the last result" (find it via `PROGRESS.md` newest entry). Ask only if genuinely ambiguous.

## Trust hierarchy (what counts as evidence)

From most to least trusted:

1. **Raw artifacts**: `summary.csv`, per-run logs, traces, checkpoints, `spec.json` — admissible, but only after you confirm they correspond to the claimed run (config, seeds, commit, timestamps consistent with each other).
2. **The code that produced them** — admissible only after you have read it and found no correctness bug relevant to the claim.
3. **`RESULT.md` conclusions, `PROGRESS.md` entries, memory summaries, ADR claims** — NOT evidence. These are the claims under review. Never cite one to support another.
4. **Statements in chat / the proposal itself** ("we verified X earlier", "this is the same setup as EXP-NNNN") — NOT evidence. "It was checked before" is never acceptable; check it now or mark it unverified.

## Procedure

### 1. Extract the claims

List every load-bearing claim the target makes: numbers, comparisons ("beats the bar"), causal statements ("the wall is X"), and implicit assumptions ("held-out seeds are held out", "the baseline is the strongest one"). Each claim gets a verdict at the end — no claim is skipped because it "looks obviously fine".

### 2. Reverify every number from raw artifacts

- Recompute headline numbers yourself from `summary.csv` / logs (a few lines of Python/pandas). Do not transcribe numbers from `RESULT.md` — the whole point is that they might not match.
- Check the aggregation: mean over what population? Per-seed values consistent with the mean? Is a best-of-N being reported as if it were typical (selection bias)?
- Check the comparison: is the bar the *strongest* relevant baseline, computed on the *same* seeds/tasks/horizon/config? A win against a mismatched bar is not a win.
- Check effect size against noise: gain vs std across seeds. A +2% mean with ±5% seed spread is "no evidence", not "improvement".

### 3. Audit the setup/script for correctness

Read the driver/script that produced the result (not its docstring — the code). Hunt specifically for:

- **Leakage**: held-out seeds/tasks reachable during training or model selection; teacher/oracle information visible to the student at eval time.
- **Seed handling**: seeds actually varied where claimed, fixed where claimed; eval seeds disjoint from train seeds; seed anchoring bugs (same episode replayed N times counted as N seeds).
- **Config mismatch**: does `spec.json` / the CLI line in `RESULT.md` match what the code actually does with those flags? Do the flags reach the env/trainer, or are some silently ignored/defaulted?
- **Metric bugs**: metric computed on the wrong population (all orders vs consumer orders), wrong normalization, horizon truncation counted as success/failure incorrectly, death episodes silently dropped from means.
- **Off-by-ones and silent defaults**: horizon, TTL, warmup cycles, `--eval-seeds` default ≠ what the claim assumes.
- **Determinism assumptions**: domain is deterministic — if two "different" runs produced identical traces, something didn't vary; if two "identical" runs differ, something leaked RNG.

If a check is cheap (read code, single-seed trace, `--dry-run`, recompute from CSV), do it now. If reverification genuinely requires a real run, it goes to the pod per CLAUDE.md — do NOT run it locally, and do NOT trust the claim in the meantime: verdict is UNVERIFIED with the exact pod command that would settle it.

### 4. Attack the inference

Even with correct numbers from a correct setup, does the evidence support the *conclusion*?

- Name at least one alternative explanation for the observed result and check whether the evidence discriminates against it. If it doesn't, the conclusion is unproven (cf. the root-cause HARD RULE — competing explanations need a discriminating diagnostic, not a preferred story).
- Check `docs/DIAGNOSTICS.md`: does the symptom match a known failure mode with a prescribed check? Was that check run?
- For proposals: what result would *falsify* the idea, and does the proposed setup even measure it? A proposal whose success criterion can't fail is not an experiment.
- Check the claimed cause against prior EXP records — does it contradict something already diagnosed with evidence? Contradiction means one of them is wrong; say which and why.

### 5. Verdict

Plain English first (per CLAUDE.md's explain-simply rule), then per claim:

- **UPHELD** — reverified against raw evidence; cite exactly what you recomputed/read.
- **OVERTURNED** — evidence contradicts it; show the contradiction.
- **UNVERIFIED** — could not be checked without a real run or missing artifact; state precisely what check would settle it.

Then a severity-ranked findings list (setup bugs first — one setup bug can invalidate every UPHELD above it; say so explicitly if one is found). End with the single most important thing the author should do next.

## Rules

- Never soften a finding to be polite. "The held-out claim is false because seed 3 appears in the training set (line N)" — not "there might be a small concern about seeds".
- Never let volume of checks substitute for the *discriminating* check. One decisive verification beats ten shallow ones.
- Do not fix anything. This is a review; the deliverable is the verdict. Offer fixes only after delivering it.
- Your own review claims obey the same standard: every verdict cites the file/line/number you actually looked at.

</what-to-do>
