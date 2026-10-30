"""Task 12.1 verify: the M3 exit summary in experiments/HISTORY.md against its records.

Stdlib only, from the repo root:
  python3 openspec/changes/add-m3-development-audit/probes/m3_exit_check.py
1. Verdict lines: each quote appears verbatim in its RESULT.md and in the summary.
2. Claims: each CLAIMS entry binds a summary phrase (the number in the words of its claim) to
   one or more phrases of the named record (RESULT.md or a CSV of the EXP dir) that state it.
   The summary phrase must occur in the §13 summary or the §0 paragraph, every record phrase in
   the record. The binding (that the record phrase says the same thing as the summary phrase)
   is made by hand in this table; the probe checks that both sides exist verbatim.
3. Coverage: every number in the §13 summary and the §0 paragraph lies inside a matched claim
   phrase or a REFERENCE (ids, dates, design / task numbers, world sizes, the budget). An
   unbound number fails.
4. Spend: the first billed vCPU-h of each RESULT.md, summed, equals the summary's total; the
   per-record figures, the upper bound and the reset point are recomputed.
Text is compared after dropping `**` and collapsing whitespace.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "experiments"
SIZING = ROOT / "openspec/changes/add-m3-development-audit/sizing.md"

EXPS = ["0048", "0049", "0050", "0051", "0052", "0053", "0054", "0055", "0056"]

VERDICTS: dict[str, list[str]] = {
    "0048": ["Verdict: diagnosed, two factors."],
    "0049": ["Not the three rules tested, on these worlds, and the reason is diagnosed.",
             "Partly, and without randomness."],
    "0050": ["Yes, routing matters much more with more makers, mostly because the hand-written "
             "rules break down."],
    "0051": ["Measured headroom is narrower than the screen's labels"],
    "0052": ["Confirmed headroom is a 42-producer, wide-choice (f14) phenomenon"],
    "0053": ["Not an acceptance read", "open — search headroom on 2 of 4"],
    "0054": ["D6 reading: orderer"],
    "0055": ["Not an acceptance read; no stage flagged."],
    "0056": ["Reading: `joint` (3 of 3 worlds)."],
}

# (record, summary phrase, record phrases). Record = "NNNN" (RESULT.md) or "NNNN/<file>".
CLAIMS: list[tuple[str, str, list[str]]] = [
    # answer 1 (EXP-0048-0050)
    ("0048", "recovers at least 1.6× random's edge in every cell random dominated",
     ["recovers the whole edge in all three random-dominated cells: at least 1.6× of it"]),
    ("0048", "beats it on all 104 of those episodes",
     ["beats `random` on all 104 of their episodes"]),
    ("0050", "at 42 producers `round_robin`'s lead over the best roster rule grows from +116 / +150 "
             "units per episode (calm / shocked) at f2 to +335 / +421 at f7",
     ["At 42 producers, `round_robin`",
      "+116 / +150 units per episode at fanout 2 (calm / shocked)", "+335 / +421 at fanout 7"]),
    ("0050", "(+153 / +151 over `round_robin` at f14)", ["14 (+153 / +151, t ≥ 8.9)"]),
    ("0049", "trails `round_robin` at f2 (−28.3 and −14.2 units on registry v2's shocked worlds)",
     ["In registry v2 every product has exactly two possible producers",
      "loses to `round_robin` on shocked worlds (−28.3 and −14.2 units)"]),
    # answer 2 (EXP-0051, 4.7 addendum, EXP-0052, 5.3 addendum)
    ("0051", "Saturation is rare: 21 of 640 screened worlds, 10 in the low-tightness, "
             "small-order corner",
     ["(21 of 640 worlds)", "the low-tightness, small-order corner (10 worlds"]),
    ("0051", "11 over-demanded worlds (tightness ρ = demand against starting throughput; ρ up to "
             "14) where the bar already delivers ≥ 97% of what the ceiling allows",
     ["expected demand per cycle against the top layer's starting throughput",
      "11 over-demanded worlds (ρ up to 14)",
      "the bar already delivers ≥ 97% of what the ceiling allows"]),
    ("0051", "replicate spread is 18–35 times smaller", ["18–35 times larger"]),
    ("0052", "kept 4 of 48 selected hard worlds at 12 producers and 10 of 24 at 42",
     ["4 of 48 selected hard worlds stay hard at 12 producers and 10 of 24 at 42"]),
    ("0052", "split train 60 / validation 16 / held-out 16 / held-out-region 4",
     ["split train 60 / validation 16 / held-out 16 / held-out-region 4"]),
    ("0052", "from 397 to 145 units at 12 producers and from 577 to 349 at 42",
     ["The median floor falls from 397 to 145 units at 12 producers and from 577 to 349 at 42"]),
    ("0052", "hard 11 / open 49 / collapse 4 at 12 producers and hard 13 / open 17 / collapse 2 "
             "at 42",
     ["hard 11 / open 49 / collapse 4 at > 12 producers and hard 13 / open 17 / collapse 2 at "
      "42"]),
    ("0052", "denser and larger at 42 producers (13 of 32 selected worlds; 11 f14, 1 f7, 1 dense; "
             "median +16% of the bar under atlas v1) than at 12 (11 of 64; 6 dense and 5 f4, "
             "none at f2; gains 5–22% of the bar, one at 83%, 12-0273)",
     ["hard 13 / open 17 / collapse 2", "32 at 42", "42 producers: 11 f14, 1 f7, 1 dense",
      "median +16% of the bar", "hard 11 / open 49 / collapse 4", "64 at 12 producers",
      "12 producers: 6 dense and 5 f4 worlds, none at f2",
      "gains of 5–22% of the bar (12-0273, held-out, 83%)"]),
    ("0052", "the earlier \"isolated at 12\" reading came from the floor",
     ["at 12 producers the earlier \"isolated\" reading came from the floor"]),
    ("0052", "the bar fills about 46% of orders and the ceiling leaves about half the room",
     ["bar ≈ 46% fill, room ≈ 0.5 under the input-output ceiling"]),
    ("0051", "it equals scheduled demand on 601 of 648 world-episodes and is never below 0.913 "
             "of it, while no played row exceeds it (0 of 7,776)",
     ["on 601 of 648 world-episodes it still equals the scheduled demand",
      "(min B_lp / demand 0.913)", "(0 of 7,776 rows)"]),
    # answer 3 (deciding check)
    ("0052", "Read on the hard worlds with 4 or more makers", ["(24 hard worlds with E ≥ 4)"]),
    ("0052", "none of its pre-registered branches reaches 2/3",
     ["no pre-registered branch holds", "still no branch at 2/3"]),
    ("0052", "on 4 / 8 / 2 of 14 worlds under atlas v1 and 5 / 24, 15 / 24, 7 / 24 under atlas v2",
     ["(4 / 8 / 2 of 14 against 10 needed)",
      "input-aware weighting 5 / 24, splitting costs 15 / 24, scorer beyond target 7 / 24"]),
    ("0052", "from ≈ 0.89 to 0.71–0.73", ["on consumer orders from ≈ 0.89 to 0.71–0.73"]),
    # answer 4 (EXP-0053)
    ("0053", "(+7 to +203 units) on its mined sample, 8 world-episodes at 12 producers",
     ["the bar on all 8 (+7 to +203)", "8 world-episodes at 12 producers"]),
    ("0053", "by more than the noise floor on 2 of 4 (12-0344, 12-0308)",
     ["open — search headroom on 2 of 4 (12-0344, 12-0308"]),
    ("0053", "the margins over the matching one-episode floor are +38.7 and +16.7 units, 2 of 4 "
             "is the minimum",
     ["Under it the margins are +38.7 (12-0344) and +16.7 (12-0308) units",
      "Two of four is the minimum for \"at least half\""]),
    ("0053", "It closes only 0.1–3.3% of the open worlds' room",
     ["the teacher closes very little of the room: 0.1–3.3%"]),
    ("0053", "scores 1–99 units *below* it on all 5 worlds where it ran",
     ["`ub` 1–99 units below `d2` on 5 / 5"]),
    # answer 5 (EXP-0054)
    ("0054", "by 2,440 units", ["G = 2,440"]),
    ("0054", "c_O = 1.023 of that gap", ["c_O = 1.023"]),
    ("0054", "c_P = 0.118", ["c_P = 0.118"]),
    ("0054", "c_O above 1 is not an overshoot (A − T is inside the floor on every world, so "
             "c_O ≈ 1)",
     ["c_O above 1 is not an overshoot",
      "is inside the floor on every world, so the reading is c_O ≈ 1"]),
    ("0054", "clear the floor only on 12-0111",
     ["not distinguishable from zero except on 12-0111"]),
    ("0054", "72% of the gap is the trained orderer trailing plain `round_robin` (by 274–891 "
             "units per world), 16% the search's gain over that floor and 12% the producer",
     ["(72%)", "+398 (16%)", "B − S = +288 (12%)", "by 274–891 units"]),
    ("0054", "three shocked 12-producer worlds, one episode each",
     ["Three shocked 12-producer worlds (one hard, two open), all burst demand, one episode"]),
    # answer 6 (EXP-0056)
    ("0056", "(TW) recovers −0.48 / 0.61 / 0.34 of the per-world gap",
     ["(TW) recovers −0.48 / 0.61 / 0.34 of the gap"]),
    ("0056", "(RS) −0.02 / 0.23 / 0.09", ["(RS) recovers −0.02 / 0.23 / 0.09"]),
    ("0056", "reaches the pre-registered 2/3", ["that gain is at least 2/3 of the gap"]),
    ("0056", "Reading `joint` on 3 of 3", ["Reading: `joint` (3 of 3 worlds)"]),
    ("0056", "(on 12-0111 TW +544 and RS +202 units, both above the 164.3 floor)",
     ["| 12-0111 | 2,364 | 2,908 | 2,566 | 3,255 | 3,038.5 | 891 ✓ | +544 ✓ (0.61) | +202 ✓ "
      "(0.23)", "3 σ_ep = 164.3 units"]),
    ("0056", "(positive on 3 worlds, above the floor on 2, clean of TW's investment failure only "
             "on 12-0344, +335)",
     ["(positive on 3 worlds, above the floor on 2", "is 12-0344 alone (+335"]),
    ("0056", "concurrent top share 0.98–0.99", ["(TR 0.98–0.99)"]),
    ("0056", "three shocked burst-demand 12-producer worlds, one episode each",
     ["Small sample: three shocked 12-producer worlds with burst demand, one episode each"]),
    ("0056", "are a description, not a causal test", ["(description, not a causal test)"]),
    ("0056", "route an order whole and vary its receivers",
     ["route an order whole, and vary its receivers"]),
    # answer 7 (EXP-0055)
    ("0055", "(slope 0.03, 268 → 285 µs)", ["it is flat, 0.03 (268 → 285 µs)"]),
    ("0055", "the producer decision grows with slope 0.48",
     ["Per producer decision it grows with slope 0.48"]),
    ("0055", "every per-decision consumer-axis slope is ≤ 0.25",
     ["Consumers (load), the flagged axis. Every per-decision slope is ≤ 0.25"]),
    ("0055", "against the 1.1 flag",
     ["a per-decision slope (pooled or per role) above 1.1 on the fanout or th"]),
    ("0055", "Per cycle, cost grows with slope 1.24 on producers",
     ["Per cycle the slope is 1.24 (10.3 → 49.3 ms for 3.5× the producers)"]),
    ("0055", "Up to 42 producers no stage of a deployed decision is superlinear",
     ["up to 42 producers", "No stage is superlinear"]),
    ("0055", "A 100-cycle episode at 42 producers costs 5–19 pod-seconds",
     ["a horizon-100 episode at 42 producers costs 5–19 pod s"]),
    # roster v3 cost (EXP-0048)
    ("0048", "baseline-only re-screen (≈ 4 vCPU-h)", ["baseline-only re-screen (≈ 4 vCPU-h)"]),
    # verdict table, 0052 row
    ("0052", "hard 11 / open 49 / collapse 4 (12), hard 13 / open 17 / collapse 2 (42)",
     ["hard 11 / open 49 / collapse 4 at > 12 producers and hard 13 / open 17 / collapse 2"]),
    # decisions still with the user (atlas regions)
    ("0052", "the evidence points at the 42-producer wide-choice hard worlds (f14); the 12-producer "
             "dense / f4 hard worlds are a smaller second region",
     ["42 producers: 11 f14, 1 f7, 1 dense", "12 producers: 6 dense and 5 f4 worlds"]),
    # §0 paragraph
    ("0052", "on 13 of 32 selected worlds at 42 producers (11 of them with 14 possible makers per "
             "product) and on 11 of 64 at 12 (dense and 4-maker worlds), with larger gains at 42",
     ["hard 13 / open 17 / collapse 2", "32 at 42", "11 f14", "hard 11 / open 49 / collapse 4",
      "64 at 12 producers", "6 dense and 5 f4", "median +16% of the bar", "5–22% of the bar"]),
    ("0055", "No stage of a deployed decision is superlinear up to 42 producers",
     ["up to 42 producers", "No stage is superlinear"]),
    ("0053", "beats the bar beyond noise on 2 of 4 open 12-producer worlds",
     ["open — search headroom on 2 of 4"]),
    ("0053", "(both margins under 40 units over the one-episode floor)",
     ["Under it the margins are +38.7 (12-0344) and +16.7 (12-0308) units"]),
    ("0053", "closes only 0.1–3.3% of their room",
     ["the teacher closes very little of the room: 0.1–3.3%"]),
    ("0054", "(c_O = 1.023 against c_P = 0.118, on 3 worlds)", ["c_O = 1.023, c_P = 0.118"]),
    # spend: per-record figures (also recomputed in check 4)
    ("0048", "EXP-0048 4.4", ["4.4 vCPU-h billed"]),
    ("0049", "EXP-0049 2.5", ["2.5 vCPU-h billed"]),
    ("0050", "EXP-0050 3.9", ["3.9 vCPU-h billed"]),
    ("0051", "EXP-0051 7.3", ["7.3 vCPU-h billed"]),
    ("0052", "EXP-0052 3.8", ["3.8 vCPU-h billed"]),
    ("0053", "EXP-0053 3.1", ["3.1 vCPU-h billed"]),
    ("0054", "EXP-0054 2.6", ["2.6 vCPU-h billed"]),
    ("0055", "EXP-0055 0.23", ["0.23 billed vCPU-h"]),
    ("0056", "EXP-0056 0.1", ["0.1 vCPU-h billed"]),
    ("0052", "failed 16 vCPU pod before EXP-0052 (≈ 2.8 vCPU-h-equivalent)",
     ["2.8 vCPU-h-equivalent"]),
    ("0056", "after ≈ 27.8 vCPU-h (≈ 28, before EXP-0056)", ["≈ 27.8 through EXP-0055"]),
    ("0056", "against it M3 used ≈ 0.1", ["against the fresh 40 vCPU-h counter"]),
]

# Spend totals are recomputed in check 4; these are references, not measured values.
REFERENCE = [
    r"EXP-00\d\dA?", r"EXP-0048–0050", r"EXP-0048–0056", r"\| 00(4[89]|5\d) \|",
    r"2026-10-2[789]", r"[Tt]asks? \d+\.\d+( and \d+\.\d+)?", r"tasks 11\.2 / 11\.3",
    r"DIAGNOSTICS \d+", r"D\d+(–D\d+)?", r"M[1-7]", r"M4/M5", r"PoC-[ABC]", r"§\d+",
    r"(?:^|\s)[1-7]\. (?=[A-Z])", r"answer [36]",
    r"of the 40 vCPU-h", r"billed of 40", r"fresh 40", r"≤ 40 vCPU-h", r"≈ 27\.9 vCPU-h",
    r"≈ 30\.7", r"\*fanout\* fN = N", r"so f14 = 14 makers",
    r"as the economy grows to 42 producers\?", r"every selected 42-producer regime-switch world",
    r"PoC-B capped at 42 producers", r"A 42-producer teacher read \(PoC-A / PoC-C at 42 producers",
    r"a 42-producer teacher read, and how M4", r"wide-choice \(f14\) phenomenon",
    r"([Aa]tlas|[Rr]egistry|[Rr]oster) v[123]\b", r"atlas v[12]\)",
    r"depth-2", r"1200", r"2 factories", r"2 × 2",
    r"the 3 shocked mined worlds", r"16 vCPU", r"12-0\d{3}",
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def record(key: str) -> str:
    exp, _, name = key.partition("/")
    (path,) = EXP.glob(f"EXP-{exp}-*/{name or 'RESULT.md'}")
    return norm(path.read_text())


def blocks(history: str) -> tuple[str, str]:
    s13 = history[history.index("### M3 exit summary"):]
    p0 = re.search(r"\*\*Where M3 left it.*?\n\n", history, re.S)
    assert p0, "§0 paragraph missing"
    return norm(s13), norm(p0.group(0))


def main() -> int:
    fails: list[str] = []
    summary, para0 = blocks((EXP / "HISTORY.md").read_text())
    texts = {"§13 summary": summary, "§0 paragraph": para0}
    cache: dict[str, str] = {}

    def rec(key: str) -> str:
        if key not in cache:
            cache[key] = record(key)
        return cache[key]

    # 1. verdict lines, and every EXP cited
    for exp, quotes in VERDICTS.items():
        for q in quotes:
            if norm(q) not in rec(exp):
                fails.append(f"verdict {exp}: not in RESULT.md: {q!r}")
            if norm(q) not in summary:
                fails.append(f"verdict {exp}: not quoted in summary: {q!r}")
        if f"EXP-{exp}" not in summary:
            fails.append(f"EXP-{exp} not cited in summary")

    # 2. claims: both sides exist verbatim
    for key, phrase, sources in CLAIMS:
        if not any(norm(phrase) in t for t in texts.values()):
            fails.append(f"claim not in summary or §0: {phrase!r}")
        for src in sources:
            if norm(src) not in rec(key):
                fails.append(f"claim {phrase!r}: EXP-{key} lacks {src!r}")

    # 3. coverage: every number sits in a claim phrase or a reference
    quotes = [q for qs in VERDICTS.values() for q in qs]
    pats = [re.escape(norm(p)) for _, p, _ in CLAIMS] + [re.escape(norm(q)) for q in quotes]
    pats += REFERENCE
    for name, text in texts.items():
        covered = [False] * len(text)
        for pat in pats:
            for m in re.finditer(pat, text):
                covered[m.start():m.end()] = [True] * (m.end() - m.start())
        for m in re.finditer(r"(?<![\w-])(?:[fv](?=\d))?[+−]?\d+(?:,\d{3})*(?:\.\d+)?", text):
            if not all(covered[m.start():m.end()]):
                ctx = text[max(0, m.start() - 40): m.end() + 40]
                fails.append(f"{name}: unbound number {m.group()!r} in …{ctx}…")

    # 4. spend recomputed from the records
    billed: dict[str, float] = {}
    for e in EXPS:
        m = re.search(r"(\d+(?:\.\d+)?) (?:vCPU-h billed|billed vCPU-h)", rec(e))
        if not m:
            fails.append(f"no billed vCPU-h in EXP-{e}")
            continue
        billed[e] = float(m.group(1))
    failed_pod = re.search(r"(\d+\.\d+) vCPU-h-equivalent", rec("0052"))
    total = sum(billed.values())
    upper = total + float(failed_pod.group(1)) if failed_pod else float("nan")
    print("billed:", billed, f"total {total:.2f}, upper {upper:.2f}")
    if f"≈ {total:.1f} vCPU-h of the 40 vCPU-h M3 budget" not in summary:
        fails.append(f"summary total ≠ ≈ {total:.1f}")
    if f"≈ {upper:.1f} if the failed" not in summary:
        fails.append(f"summary upper ≠ ≈ {upper:.1f}")
    for e, v in billed.items():
        if f"EXP-{e} {v:g}" not in summary:
            fails.append(f"summary spend for EXP-{e} ≠ {v:g}")
    if f"≈ {total:.1f} vCPU-h billed of 40" not in para0:
        fails.append("§0 spend ≠ total")
    if f"after ≈ {total - billed['0056']:.1f} vCPU-h" not in summary:
        fails.append("reset point ≠ total before EXP-0056")
    if "Budget counter reset (2026-10-29)" not in SIZING.read_text():
        fails.append("sizing.md lacks the dated counter-reset line")

    for f in fails:
        print("FAIL", f)
    print("ALL CHECKS PASS" if not fails else f"{len(fails)} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
