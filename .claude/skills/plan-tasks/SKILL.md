---
name: plan-tasks
description: Write or re-slice an OpenSpec change's tasks.md so the `supervise` skill can execute it. Every task is exactly one kind (code or experiment), carries a resource class sized for the small shared local machine, and is explicitly marked when it must run on a separate, dynamically provisioned pod/VM. Use when the user says "plan tasks for <change>", "slice tasks for supervise", "/plan-tasks <change>", or asks to make an OpenSpec task list supervisor-ready.
---

<what-to-do>

You produce the `tasks.md` of an OpenSpec change in the format the `supervise` skill consumes (`/srv/research/tokenomics/.claude/skills/supervise/SKILL.md`). This is planning only. Do not edit project code, run experiments or start the apply workflow.

`$ARGUMENTS` = `<change-name> [--reslice]`. Without `--reslice`, write a missing `tasks.md`. With it, rewrite an existing one, keeping the ticked state of any task that survives unchanged. If the change name is missing, run `openspec list --json` and ask which change.

## Phase 0: Inputs

1. Run `~/.local/bin/openspec status --change <name> --json` and `openspec instructions tasks --change <name> --json`. Use the returned template, rules and output path. Proposal, specs and design must exist; if any is missing, stop and tell the user to finish them first (`/opsx:propose` or `/opsx:update`).
2. Re-read from disk the change's `proposal.md`, `design.md` and every delta spec. Read the project rules that constrain tasks: `/srv/research/tokenomics/CLAUDE.md` (compute rules, record rules, root-cause rule) and `/srv/research/tokenomics/openspec/config.yaml`.
3. Inspect the code the tasks will touch, read-only, so every task names real files and real verification commands.
4. **Probe the local machine** and record the result in the tasks.md preamble (see Format):
   ```
   nproc; free -m | head -2; cat /proc/loadavg; df -h /srv | tail -1
   ```
   This box is shared: the supervisor, the review orchestrator (another harness), one implementer and one reviewer agent can run commands on it at the same time. Plan for that concurrency, not for an idle machine. When this skill was written the box had 2 cores, about 3 GB RAM (about 2 GB available) and 26 GB free disk.

## Phase 1: Slice into kinds

Every tracked task is **exactly one kind**.

- **`code`**: local edits to code, specs or docs, plus their tests. Smoke checks are allowed only with every knob tiny: horizon ≤ ~12, 1–2 tasks, ≤ 1 seed, seconds of wall time. A code task never produces an `EXP-NNNN` result.
- **`experiment`**: runs already-committed code to produce or extend an `EXP-NNNN` record, then writes its records (`RESULT.md` plain English first, README index row, HISTORY paragraph if an arc opens or closes). It never edits `tokenomics/tokenomics/` or `tokenomics/tests/`.

Slicing rules:
1. **Never mix.** A source item that both builds and runs something becomes two tasks: `code` (build + tests + tiny local dry-run) and then a dependent `experiment` (the real run + records).
2. **Records ride with their work.** Tests and docs go in the code task that needs them. EXP records go in the experiment task that produced the data. Never leave a final "write tests" or "write docs" group.
3. **Small and verifiable.** One task is one reviewable session: one cohesive diff for code, or one run family writing one EXP record for experiments. Each task states its verification.
4. **Unknown cost → measure first.** If you cannot estimate a run's cost, insert a `code`/`local` timing probe before it: single episode, horizon ≤ 12, one seed, timed. The probe's output sizes the remote task. The remote task's resource line then says "sized by task <id>".
5. **Failure paths are not pre-planned as strategy.** Do not add speculative "if it fails, try X" tasks. The supervisor inserts diagnosis tasks when a failure actually happens (root-cause rule).
6. **Dependencies are explicit.** Every task that depends on something other than the task above it says `after <ids>`. An experiment task always depends on the code tasks whose code it runs, and on a commit of that code.

## Phase 2: Resource class

Every tracked task gets exactly one resource class. Classify by the **heaviest command the task needs**, including its verification and any command the reviewer will rerun.

| Class | When | Limits on this box |
|---|---|---|
| `local` | edits, targeted tests, tiny smokes, quick one-shot diagnostics | single process, ~1 core, ≤ ~2 min wall, ≤ ~1 GB RAM per command |
| `local-exclusive` | needs most of the box briefly: the full test suite, a short multi-process smoke | ≤ ~5 min wall, ≤ ~half of available RAM; the supervisor runs at most one at a time and no other heavy local job alongside it |
| `remote:<profile>` | anything above those limits, and every `experiment` | runs on a **separate, dynamically provisioned pod/VM**, never on the local box |

Rules:
- **Every `experiment` is `remote`.** This follows the project's rule that all experiments run on the cloud; the local box never materializes an EXP result.
- A `code` task is `remote` only when its *verification* is too heavy for the box: a slow-marked test suite, a large-fixture invariance check, or memory above the local limit. The code is still edited locally; only the heavy check runs remotely, and it must not produce an EXP result. If you can split that check into its own task, do.
- **Classify by compute profile, not by name.** A command called a "smoke" with horizon 100, several seeds or multiple tasks is remote. When in doubt, choose the heavier class.
- Known heavy local commands in this repo (update when they change):
  - the full `pytest -q` suite is about 2.5 min on 2 cores → `local-exclusive`;
  - `pytest -m champion` is about 13 min → remote;
  - any full-horizon search, rollout or distillation run → remote.

Remote profiles (passed to `run_pod_training.py`):

| Profile | Flags | Use for |
|---|---|---|
| `cpu16` | `--create-pod --cpu --python python3.12 --vcpu 16 --mem 32 --disk 30` (default flavor `cpu5c`) | CPU-bound rollouts, search, distillation, gate runs; fan out ~16 workers |
| `cpu32` | `--create-pod --cpu --python python3.12 --vcpu 32 --mem 64 --disk 30` | when a sized estimate exceeds ~4 h wall on `cpu16` |
| `gpu-<type>x<n>` | `--create-pod --compute-type GPU --gpu-type <type> --gpu-count <n>` with a torch image | only GPU-bound training (rare here: batch-1 inference gains nothing) |

**Dynamic provisioning is the default.** A remote task creates its own fresh pod with `--create-pod`; the driver stops and terminates it after `--pull`. A named existing pod (`--pod-id`) is used only if the user says so, and then the task must say which pod.

Every `remote` task carries a detail line (see Format) with:
- the profile;
- estimated wall time;
- estimated compute (vCPU-hours or GPU-hours);
- the commit it must run (always "the commit after task <id>");
- inputs to stage under `tokenomics/_pod_ckpts/` or `tokenomics/_pod_inputs/`;
- the outputs it pulls back;
- whether `--verify-source-commit` applies.

## Format

Keep the template's `## N. Group` headings and `- [ ] N.M` checkboxes; apply-phase parsing depends on them.

Put a preamble under the title, as untracked lines:
```
Local budget (probed <date>): <nproc> cores, <total>/<available> MB RAM, <free> disk.
Shared by supervisor + implementer + reviewer. Classes: local | local-exclusive | remote:<profile>.
```

Each tracked task:
```
- [ ] <N.M> [<kind>][<resource>] <what to do>. After <ids>. Verify: <check>.
```
Remote tasks add one indented untracked detail line directly under the checkbox:
```
  - Resources: <profile> fresh pod (--create-pod), ~<wall>, ~<vCPU-h>; runs commit after <id>; stage <inputs>; pull <outputs>; --verify-source-commit <yes|n/a>.
```
- `<kind>` is `code` or `experiment`. `<resource>` is `local`, `local-exclusive` or `remote:<profile>`.
- Group by dependency: code groups first, then the experiment groups that consume them. A group may hold only one kind, or the supervisor cannot run it in order.
- Use absolute paths for any file named in a task.
- Keep a `## Workflow follow-up` section of plain bullets (review, sync, archive) where the template allows it.

## Phase 3: Check, then present

Before finishing, verify mechanically:
```
grep -nE '^- \[[ xX]\] [0-9]+\.[0-9]+ ' tasks.md                         # all tracked tasks
grep -nE '^- \[[ xX]\] [0-9]+\.[0-9]+ \[(code|experiment)\]\[(local|local-exclusive|remote:[a-z0-9x-]+)\] ' tasks.md
```
The two counts must match. Then check by reading:
- no `experiment` task is `local` or `local-exclusive`;
- every `remote` task has a `Resources:` line;
- every experiment depends on the code it runs;
- no group mixes kinds;
- every task has a `Verify:` clause.

Then run `openspec validate <change> --strict`.

Present a short table to the user: id, kind, resource class, summary, and depends-on. Then list the remote tasks with their profile and estimates, and their total estimated compute, since pod spend needs the user's budget before `supervise` will launch them. Do not start implementation.

## Rules

- Planning artifacts only. Never edit code, never launch pods.
- One kind and one resource class per tracked task. Ambiguity resolves to the heavier class.
- Estimates must be grounded in a measurement, a prior EXP record's wall time, or an explicit probe task, never a bare guess. Cite the source in the `Resources:` line.
- Do not change requirements or design while slicing. If slicing exposes a gap, stop and tell the user; fix it through `/opsx:update`.

</what-to-do>
