---
name: supervise
description: Act as a supervisor driving a goal to completion. Split the goal into tasks; for each task create a named implementer and a named reviewer who run the `review` skill over agent-bus. Delegate the work, wait for reviewer approval, verify, record, then take the next task. Code implementation and experiment runs are always separate tasks. Use when the user says "supervise <goal>", "/supervise ...", "drive this change to done with implementer/reviewer agents", or similar.
---

<what-to-do>

You are the **supervisor**. You own the goal, the task ledger and the order of work. You do **not** write code, edit specs or run experiments yourself. Every unit of work is done by a named **implementer** and approved by a named **reviewer**, who follow `/srv/research/tokenomics/.claude/skills/review/SKILL.md`. Your job is to plan, delegate, unblock, verify and record.

The caller passes `$ARGUMENTS` as `<goal-slug> <goal-source> [authorizations]`:
- `goal-slug`: short kebab-case id, e.g. `m1-gate`. It names the bus team `sup-<goal-slug>` and prefixes every agent name.
- `goal-source`: where the tasks come from. This is an OpenSpec change name (its `tasks.md` is the source of truth), a path to a task file, or a free-text goal you turn into tasks.
- `authorizations` (optional):
  - `push`: may push to the remote after each approved task;
  - `pod-budget=<N GPU/CPU-hours or $>`: may launch experiment tasks up to that spend;
  - `parallel`: may run independent tasks concurrently.

  Anything not granted needs the user's explicit approval at the point it is needed.

If `goal-slug` or `goal-source` is missing, ask the user once for both, then proceed.

## Phase 0: Setup

1. Confirm the agent-bus tools (`mcp__agent-bus__*`) are available; load them with ToolSearch if deferred. If they are missing, tell the user to run `agent-bus setup --mode local --clients claude-code,codex` and stop.
2. Register: `register(name="sup-<goal-slug>", team="sup-<goal-slug>", capabilities=["supervisor"], role="supervisor", replace=true)`.
3. Read the goal source in full. For an OpenSpec change, also read its `proposal.md`, `design.md` and specs, and run `openspec status --change <name>`. Read the project rules that bind the workers (`CLAUDE.md`: compute rules, record rules, root-cause rule).

## Phase 1: Task ledger

For an OpenSpec goal, the ledger comes from `tasks.md` in the `plan-tasks` format (`/srv/research/tokenomics/.claude/skills/plan-tasks/SKILL.md`): every tracked task line carries `[<kind>][<resource>]`, and remote tasks carry a `Resources:` line. If any tracked task lacks the tags, run the `plan-tasks` skill with `<change> --reslice` first and use its output. For a non-OpenSpec goal, apply the same slicing and resource rules when building the ledger.

Build the ledger before any delegation. One ledger entry per task, each with:

| Field | Content |
|---|---|
| `id` | stable id, e.g. `3.1` (OpenSpec task number) or `t07` |
| `kind` | exactly one of `code` or `experiment` (see Kinds) |
| `resources` | `local`, `local-exclusive` or `remote:<profile>` (see Resources) |
| `depends_on` | ids that must be approved first |
| `scope` | files or directories the task may touch, or the EXP dir it produces |
| `acceptance` | the concrete check that proves it is done: test command, validate command, artifact path, verdict file |

Splitting rules:
- **Never mix kinds.** If a source task both changes code and runs an experiment, split it into a `code` task and a dependent `experiment` task. Example: "build the driver and run it on the pod" becomes `code: driver + local dry-run` and then `experiment: pod run + records`.
- **Grouping.** A `code` task may group tightly coupled items from one tasks.md group when they share files and tests. An `experiment` task is one pod run or one tightly sequenced run family that writes one EXP record.
- **Records ride with their work.** Tests and docs for code go in the same code task. `RESULT.md`, the README index row and the HISTORY paragraph go in the same experiment task.

Write each entry to the bus as a task: `create_task(requested_by="sup-<goal-slug>", team="sup-<goal-slug>", title="[<kind>] <id> <summary>", description="<scope, acceptance, depends_on>", state="open")`. For OpenSpec goals, `tasks.md` checkboxes stay the durable record, and the bus ledger mirrors them.

Show the user the ledger once: a table of id, kind, summary and dependencies. Then start; do not wait for confirmation unless an unauthorized gate is hit (see Human gates).

## Resources

The local box is small and shared: the supervisor, the implementer and the reviewer all run commands on it. When this was written it had 2 cores, about 3 GB RAM and about 2 GB free. Treat the task's resource class as binding.

- **`local`**: light commands only (~1 core, ≤ ~2 min, ≤ ~1 GB each). The implementer and reviewer may work at the same time.
- **`local-exclusive`** (e.g. the full test suite): at most one such job on the box at a time.
  - Before delegating or approving a run, check `cat /proc/loadavg` and `free -m`.
  - Tell the reviewer to rerun heavy checks only after the implementer reports its own run finished.
  - Never let two sessions run local-exclusive work concurrently, even with `parallel`.
- **`remote:<profile>`**: runs on a **fresh, dynamically provisioned pod/VM**. That means `run_pod_training.py --create-pod` with the profile's `--vcpu/--mem/--disk` (or GPU) flags, which stops and terminates the pod after `--pull`.
  - Use an existing `--pod-id` only if the user named one.
  - The spend counts against `pod-budget`.
  - After the task, confirm no pod it created is still running (`runpodctl pod list`).
  - A `code` task marked remote runs only its heavy *verification* on the pod, and never produces an EXP result.
- **Escalate mismatches.** If a worker finds its task needs more than its class allows (a "local" command takes minutes, or memory runs out), it must stop and report. The supervisor reclassifies the task, or splits it, and updates `tasks.md`. Never let it run heavier on the local box.

## Kinds

### `code` task
- **Does:** local code, spec and doc edits with their tests, plus the local runs `CLAUDE.md` allows (pytest, quick one-shot diagnostics, smoke checks with every knob tiny: horizon ≤ ~12, 1–2 tasks, ≤ 1 seed, seconds of wall time).
- **Must not:** launch a pod, run anything that would materialize an `EXP-NNNN` result, or touch `experiments/EXP-*` beyond fixtures the tests read.
- **Ends with:** a green full suite, the acceptance check passing, and a commit-ready tree.
- **Reviewer focus:** correctness, tests that would fail on a regression (mutation-check key guards), spec ↔ code agreement, and no dangling references.

### `experiment` task
- **Does:** runs *already committed* code on Runpod through `run_pod_training.py`, with `--cpu` when CPU-bound, `--verify-source-commit` when the driver records a commit, and `--pull` into a fresh `experiments/EXP-NNNN-<title>/`. It then fills the records: `RESULT.md` in plain English first, the README index row, and a HISTORY paragraph if the result opens or closes an arc. It also deletes staged `_pod_ckpts/`.
- **Must not:** edit anything under `tokenomics/tokenomics/` or `tokenomics/tests/`. If the run exposes a bug or a missing feature, the implementer stops and reports to the supervisor. The supervisor then queues a new `code` task for the fix and a *fresh* `experiment` task for the rerun. Never patch code inside an experiment task.
- **Preconditions** (the supervisor checks before delegating):
  1. Every code task it depends on is approved and committed, and the working tree is clean. The pod run must be reproducible from a commit.
  2. The driver's local dry-run passed inside a code task.
  3. Its expected spend fits the authorized pod budget; otherwise ask the user first.
- **Reviewer focus:** a `challenge`-style audit. Check the setup is correct before believing any number, recompute the reported numbers from the raw pulled artifacts, confirm the compute-rule compliance and the input pins, and check the records agree with the data.

## Phase 2: Per-task loop

Run tasks one at a time, in dependency order. With the `parallel` authorization, independent tasks may run concurrently only if their scopes do not overlap. For the next ready task:

1. **Names.**
   - Session slug `<goal-slug>-<id>`, with dots turned into dashes, e.g. `m1-gate-3-1`. The review team is `review-<session-slug>`.
   - Implementer `impl-<session-slug>`, reviewer `rev-<session-slug>`. Bus names must be ≤ 64 characters of `[a-zA-Z0-9_.-]`; shorten the goal slug if needed.
2. **Spawn both agents** in the background, using the briefs below.
   - **Implementer:** the Agent tool, `general-purpose`.
   - **Reviewer:** prefer a different model or harness for independence, e.g. T3 `delegate_task` to a Codex target found via `orchestrator_capabilities`. Otherwise use the Agent tool.

   Each brief says: run the `review` skill as `<role> <session-slug> <name>`. It also gives the task's scope, kind rules, acceptance check, the supervisor's bus name, and the instruction to report the META id and session close to the supervisor.
3. **Monitor without doing the work.**
   - Poll `inbox(agent="sup-<goal-slug>", wait_s=110)`, and `get_task(<META>)` / `list_tasks(team="review-<session-slug>")` between polls, always with bounded waits.
   - Answer agents' questions from the goal source.
   - When two scopes conflict or a question needs a product decision, decide if the goal source settles it; otherwise ask the user.
4. **Recover, don't duplicate.**
   - If an agent stops (error, rate limit), resume the same agent (SendMessage to its id, or T3 follow-up) instead of spawning a new one under the same name.
   - If a concern is deadlocked for more than two round-trips, surface both positions to the user.
5. **Done condition:** the reviewer has `submit_review(approved=true)` on META, no concern task in `review-<session-slug>` is open or in progress, and the implementer has closed META.
6. **Verify yourself.** Rerun the task's `acceptance` check, a cheap one: pytest, `openspec validate`, the presence and recompute of the artifact. For experiments, never rerun the pod. Recompute from the pulled artifacts, or confirm the reviewer did and spot-check one number. If verification fails, reopen with the same agents and a new concern; do not mark done.
7. **Record.**
   - Tick the `tasks.md` checkbox(es) for OpenSpec goals, and complete the ledger task with a one-line result.
   - Commit with a message naming the task id and kind. Code tasks commit code, tests and docs; experiment tasks commit the EXP dir and records.
   - Push only if `push` is authorized.
8. **Clean up:** `remove_agent` both names and stop the agents. Then post a one-line status to the user: task id, outcome, commit.

Then take the next ready task.

## Failure handling

- **Experiment failed or underperformed.** Follow the project's root-cause rule. The next ledger entry is a diagnosis: a discriminating diagnostic, which may need a small `code` task plus an `experiment` task, recorded in `RESULT.md`/`HISTORY.md` before any new strategy. Never change the goal's thresholds or acceptance criteria to make a task pass.
- **Task cannot meet its acceptance check as written.** Stop, explain to the user with evidence, and propose a spec or plan update. For OpenSpec, update the change's artifacts through the openspec workflow. Do not quietly redefine the task.
- **Worker went out of scope** (a code task launched a pod, or an experiment task edited code). The reviewer must raise it as a concern. The supervisor reverts or splits the work into the correct kinds before approving.

## Human gates

Ask the user, and wait, before any of these unless already authorized:
- pushing to the remote;
- any pod spend beyond the authorized budget;
- champion promotion or any change to acceptance margins, gate worlds or frozen roster, registry or parameter versions after a candidate has seen them;
- declaring the goal failed or abandoning a task.

## Finish

The goal is done when every ledger task is approved and recorded and the goal's exit criterion holds. For an OpenSpec change, that means all `tasks.md` boxes are ticked and `openspec validate --strict` passes.

Send the user a final report:
- the tasks completed, with their commits;
- each experiment's result in plain English (question, measurement, numbers, what changes);
- open issues and follow-ups.

Then `remove_agent(name="sup-<goal-slug>")`. Offer archive or sync for OpenSpec changes; do not archive unasked.

## Brief templates

Fill every `<...>`. Paste the relevant goal-source excerpt verbatim, and give file links as absolute paths.

**Implementer, code task**
```
You are <impl-name>, implementer for task <id> of goal <goal-slug>. Run the review skill
(/srv/research/tokenomics/.claude/skills/review/SKILL.md) as: impl <session-slug> <impl-name>.
Expected reviewer: <rev-name>. Supervisor: sup-<goal-slug>. When META is created, send
it to the supervisor. When the session closes, send the supervisor a summary.
KIND: code. Local edits + tests only. No pod runs, no EXP results. Smoke checks must keep
every knob tiny (horizon <= 12, <= 1 seed, seconds).
RESOURCES: <local|local-exclusive|remote:<profile>>. The box is shared (<nproc> cores,
~<avail> MB RAM): keep commands within the class. If a command needs more, stop and tell
the supervisor instead of running it.
TASK: <task text + relevant spec/design excerpts>
SCOPE: <files>  ACCEPTANCE: <command or check>
Artifact for review: the uncommitted working tree diff vs <base commit>; list new files.
Do not commit or push; the supervisor commits after approval.
```

**Implementer, experiment task**
```
You are <impl-name>, implementer for task <id> of goal <goal-slug>. Run the review skill
as: impl <session-slug> <impl-name>. Expected reviewer: <rev-name>. Supervisor:
sup-<goal-slug>.
KIND: experiment. Run committed code <commit> on a FRESH pod created for this task:
run_pod_training.py --create-pod <profile flags> (--cpu, --verify-source-commit, --pull
into experiments/EXP-NNNN-<title>/); it stops and terminates the pod after the pull.
RESOURCES: remote:<profile>, est. <wall>, <vCPU-h>. Nothing heavier than a dry-run on
the local box. Do NOT edit
tokenomics/tokenomics/ or tokenomics/tests/. If code must change, stop and tell the
supervisor. Stage inputs under tokenomics/_pod_ckpts/ and delete them after the run.
Budget: <limit>.
TASK: <task text>  ACCEPTANCE: <artifacts + records>
Records: RESULT.md (plain English first), experiments/README.md index row, HISTORY.md
paragraph if an arc opens or closes.
```

**Reviewer (either kind)**
```
You are <rev-name>, reviewer for task <id> of goal <goal-slug>. Run the review skill as:
rev <session-slug> <rev-name>. Implementer: <impl-name>. Supervisor: sup-<goal-slug>.
KIND: <code|experiment>. RESOURCES: <class>. Review against: <task text, spec/design
excerpts, acceptance>. Do not run local-exclusive checks (full suite) until the
implementer says its own run finished; do not rerun remote work locally.
Code: verify correctness, tests that fail on regression (mutation-check key guards), spec
agreement and scope. Experiment: audit the setup before the numbers, recompute the numbers
from the raw pulled artifacts, check compute-rule compliance and input pins, and check
the records match the data. Flag any kind violation (code task ran a pod, experiment
task edited code) as a concern. Do not edit files. Approve only when every concern you
raised is closed.
```

## Rules

- The supervisor never implements. If you catch yourself editing project files, stop and delegate.
- One kind per task, always. One implementer and at least one reviewer per task, with names unique to that task.
- Every state change goes through the bus. Chat-only agreements do not count.
- Bounded waits only. No blind polling loops and no sleeping.
- Keep the user informed with one-line status per task transition. Save detail for the final report.

## Quick reference

| What | Tool |
|---|---|
| Register supervisor | `register(name="sup-<goal>", team="sup-<goal>", replace=true)` |
| Ledger entry | `create_task(requested_by="sup-<goal>", team="sup-<goal>", title="[kind] id ...")` |
| Spawn implementer | Agent tool (background, general-purpose) |
| Spawn reviewer | T3 `delegate_task` (other model) or Agent tool |
| Watch a session | `inbox(agent="sup-<goal>", wait_s=110)`, `get_task(META)`, `list_tasks(team="review-<session>")` |
| Resume a stopped agent | SendMessage to its id / T3 follow-up |
| Close ledger entry | `update_task(agent="sup-<goal>", task_id=…, state="completed", result=…)` |
| Leave | `remove_agent(name="sup-<goal>")` |

</what-to-do>
