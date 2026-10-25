# Tokenomics

A cooperative multi-agent **reinforcement-learning environment** for a tokenomics economy
game, plus the policies trained on it. Consumers demand products; producing agents own
manufacturing capacity and assemble products (possibly from other products, forming supply
chains); ordering agents split each order across eligible producers. Agents learn to fulfil
consumer demand by routing tokens and managing production capacity.

Python only. All code lives in [`tokenomics/`](tokenomics/).

## Start here

| Document | What it is |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Working rules, project map, and the authoritative **domain vocabulary** (token, cycle, capacity, process sheet, system death, …) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the project trains, measures, and improves its policies — written for readers new to ML |
| [GLOSSARY.md](GLOSSARY.md) | Training/evaluation terms used across the experiment records |
| [experiments/HISTORY.md](experiments/HISTORY.md) | The compacted empirical record: 49 experiments in nine arcs, and why each closed branch closed |
| [docs/adr/](docs/adr/) | 54 numbered design decisions |
| [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) | Symptom → discriminating check registry |

Founding requirements and design rationale (in Russian):
[docs/requirements.md](docs/requirements.md),
[docs/desicion-records.md](docs/desicion-records.md),
[docs/deep-research-report.md](docs/deep-research-report.md).

## Layout

```
tokenomics/tokenomics/
  domain/   pure simulation logic — no RL, no numpy, no IO, no logging
  env/      PettingZoo AEC adapter: spaces, encode, decode, aec_env
  eval/     baselines, metrics harness, transfer matrix, curriculum/mining tooling
  train/    scorer distillation, DAgger, champion iteration, graph probes
  config/   phase1.json
tokenomics/tests/   one file per layer
experiments/        one directory per experiment — see experiments/README.md
docs/               ADRs, diagnostics, requirements
run_pod_training.py Runpod driver: start pod → sync → install → train → pull → stop
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e 'tokenomics[dev,train]'  # tests + training/scorer dependencies
```

## Common commands

OpenSpec 1.15.0 is initialized at the repository root for Codex and Claude Code.
Specs and changes live in `openspec/`; workflow configuration is in
[`openspec/config.yaml`](openspec/config.yaml). Install the CLI using the
[OpenSpec installation guide](https://openspec.dev/docs/installation), then run
`openspec list` from the root to check the setup. Start a change with
`$openspec-propose "your idea"` in Codex or `/opsx:propose "your idea"` in Claude Code.
Restart your coding agent to load the generated skills and commands.

Run from `tokenomics/`:

```bash
../.venv/bin/python -m pytest -q                        # test suite
../.venv/bin/python -m tokenomics.eval.signal_check     # baseline comparison table
```

`signal_check` is the guardrail: it prints a matrix of producing × ordering policies and
proves the environment poses a real learning problem — a sensible policy must clearly beat
do-nothing, and the strongest baseline is the bar a learned policy has to clear.

## The champion stack

`tokenomics/champion.json` pins the current policies. The primary metric is
`consumer_completed_units` against a strict best-of-baseline envelope on held-out topology
seeds, with a hard 0% death-rate requirement. See
[experiments/HISTORY.md](experiments/HISTORY.md) §0 for the checkpoint table.

## Where things run

Real training and every experiment run on Runpod, never locally — wrap the command with
`run_pod_training.py` (`--cpu` for CPU pods). Only fast throwaway smoke checks and quick
one-shot diagnostics stay local. The full rule, including the compute-profile test for
classifying a run, is in [CLAUDE.md](CLAUDE.md).
