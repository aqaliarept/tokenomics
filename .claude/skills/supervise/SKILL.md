---
name: supervise
description: Act as a supervisor driving a goal to completion. Split the goal into tasks; run them with one long-lived implementer spawned here (more only when the user grants it) and delegate every review to the `review-orchestrator` session in another harness, which runs one long-lived reviewer. Both sides follow the `review` skill over agent-bus at fixed addresses, so the user only starts the orchestrator skill. Wait for approval, verify, record, then take the next task. Code implementation and experiment runs are always separate tasks. Use when the user says "supervise <goal>", "/supervise ...", "drive this change to done with implementer/reviewer agents", or similar.
---

<what-to-do>

You are the **supervisor**. You own the goal, the task ledger and the order of work. You do **not** write code, edit specs or run experiments yourself. Every unit of work is done by the **implementer** you spawn here and approved by the **reviewer** that the **review orchestrator** spawns in another harness. By default there is exactly one of each for the whole goal: the same implementer run does every task, one after another (PROTOCOL "Workers per goal"). Both follow `/srv/research/tokenomics/.claude/skills/review/SKILL.md`. Your job is to plan, delegate, unblock, verify and record.

The supervisor ↔ orchestrator wire protocol (fixed addresses, message formats, review lifecycle, bus facts, timeouts) is `/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md`. Read it first; it is binding. Below, `orch` = `review-orchestrator`.

The caller passes `$ARGUMENTS` as `<goal-slug> <goal-source> [authorizations]`:
- `goal-slug`: short kebab-case id, e.g. `m1-gate`. It prefixes ledger titles and every per-task agent name.
- `goal-source`: where the tasks come from. This is an OpenSpec change name (its `tasks.md` is the source of truth), a path to a task file, or a free-text goal you turn into tasks.
- `authorizations` (optional):
  - `push`: may push to the remote after each approved task;
  - `pod-budget=<N GPU/CPU-hours or $>`: may launch experiment tasks up to that spend;
  - `implementers=<N>`: may run up to N implementer runs, so independent tasks can overlap (default 1);
  - `reviewers=<N>`: the orchestrator may run up to N reviewer runs (default 1);
  - `parallel`: shorthand for `implementers=2 reviewers=2`.

  Never raise the implementer or reviewer count without one of these.

  Anything not granted needs the user's explicit approval at the point it is needed.

If `goal-slug` or `goal-source` is missing, ask the user once for both, then proceed.

**Mode.** When every worker is a Claude agent (no Codex/other-harness reviewer), use **Claude-native mode** (section below): no agent-bus, no inbox loops, no 10-min heartbeats; the review orchestrator stays and owns every review. The bus mode in Phases 0–2 is for cross-harness review only. The kinds, resources, human gates, records and brief contents below apply in both modes.

## Claude-native mode

Workers talk through the harness, not the bus. In a T3 thread: `delegate_task` to start a worker, the automatic completion notification to hear back, `t3_thread_send` to the worker's `childThreadId` to continue it. Standalone Claude Code: the Agent tool in the background, its completion notification, and `SendMessage`.

- **Ledger.** The goal source's `tasks.md` is the ledger; no bus tasks. Per task the implementer writes `<change dir>/notes/<id>.md`: what changed, how it was verified, measured numbers, gotchas the next task needs. That file replaces handoff messages.
- **One fresh implementer per task.** `delegate_task(mode="async", isolation="shared", role="implementation", title="impl-<goal> <id>", clientRequestId="impl-<goal>-<id>")`. Brief = the implementer brief template minus every bus instruction, plus: read the previous tasks' `notes/*.md` you need; write `notes/<id>.md`; do not commit; end your run with a final report (`ARTIFACT <pointer>` + verification + numbers). No inbox loop, no heartbeat, no handoff: a task too big for one context ends with `PARTIAL` + its notes file, and you start a continuation run.
- **Review orchestrator (kept).** One long-lived orchestrator run per goal, started by you with `delegate_task(title="orch-<goal>", role="review", clientRequestId="orch-<goal>-r<k>")` and the `review-orchestrator` skill's "Claude-native mode". You hand it each review with `t3_thread_send` to its thread: `REVIEW <session>` + the reviewer brief (minus bus instructions) + the artifact pointer + the implementer's `childThreadId`. It runs the review rounds itself (one reviewer per task, an Agent-tool subagent continued across rounds with `SendMessage` so it keeps its context), relays concerns to the implementer's thread and back, and ends its run with `APPROVED <session> concerns=<n>` (or `DEADLOCK`/`STALLED` with the dispute), which wakes you. Between reviews it is idle; no heartbeat, no inbox loop.
- **Concerns.** The orchestrator sends them verbatim to the implementer's thread; the implementer fixes or argues each, re-runs its checks and ends with a report. Done = the orchestrator's `APPROVED`, its last reviewer round having no open concern.
- **Then as before:** verify yourself (step 7), commit + tick `tasks.md` (step 8), post the one-line status.
- **Wake-ups.** Completion notifications wake you; do not poll. Keep at most one slow fallback heartbeat (60 min) in case a run dies silently. A run that ended in an error or a usage limit is restarted with a new `delegate_task` that points at the task's notes file (usage-limit resumes are often automatic: wait for them first).
- **Model per task.** Default to the parent's model. Use a smaller model only when the user granted it, for small mechanical tasks (flags, probes, record-only work); never for a reviewer of a design or experiment task.
- **Independence.** Claude reviewing Claude is not cross-model review; say so once to the user when entering this mode.

## Phase 0: Setup

1. Confirm the agent-bus tools (`mcp__agent-bus__*`) are available; load them with ToolSearch if deferred. If they are missing, tell the user to run `agent-bus setup --mode local --clients claude-code,codex` and stop.
2. Register at the fixed address: `register(name="supervisor", team="supervision", capabilities=["supervisor"], role="supervisor", replace=true)`. There is one supervisor per bus; this takes over the name from any earlier session.
3. **Find the orchestrator** (PROTOCOL "Messages"):
   - If `whois(team="supervision")` lists `review-orchestrator`, send it `PING`.
   - Otherwise tell the user, once, in one line: *start the review orchestrator in another harness, from `/srv/research/tokenomics`: in Codex type `$review-orchestrator`, in Claude Code `/review-orchestrator`. No prompt or arguments needed.* Prefer a harness other than this one, so the review is independent.
   - Whenever `HELLO` arrives (now or later), reply `WELCOME goal=<goal-slug> source=<goal-source> max_reviewers=<N from reviewers=/parallel, else 1>`. If `HELLO` names the same harness as yours, warn the user once that the review is no longer cross-model.
   - Do not block on the handshake here. Continue with steps 4–5 and Phase 1; you need the orchestrator only before the first review request.
4. Pick the subagent mechanism for implementers (PROTOCOL "Subagents"): inside a T3 Code thread (the `t3-code` tools `delegate_task` / `task_status` / `task_cancel` exist), T3 `delegate_task`; otherwise the Agent tool in the background. In a T3 thread keep your own turn alive with the inbox loop; bus messages do not wake an idle thread.
4a. **Schedule your heartbeat** (PROTOCOL "Heartbeat"): T3 `schedule_task` every 10 min, or `CronCreate` in standalone Claude Code, with the heartbeat prompt for `supervisor` / `supervise`. Your heartbeat also covers your implementer(s) and the orchestrator. Delete it in Finish.
5. Read the goal source in full. For an OpenSpec change, also read its `proposal.md`, `design.md` and specs, and run `openspec status --change <name>`. Read the project rules that bind the workers (`CLAUDE.md`: compute rules, record rules, root-cause rule).

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

Write each entry to the bus as a task: `create_task(requested_by="supervisor", team="supervision", title="[<goal-slug>][<kind>] <id> <summary>", description="<scope, acceptance, depends_on>")`. Set it `claimed` when the task starts and `completed` when it is recorded. If a previous supervisor session left ledger tasks with the same goal prefix, reuse them instead of creating duplicates. For OpenSpec goals, `tasks.md` checkboxes stay the durable record, and the bus ledger mirrors them.

Show the user the ledger once: a table of id, kind, summary and dependencies. Then start; do not wait for confirmation unless an unauthorized gate is hit (see Human gates).

## Resources

The local box is small and shared: the supervisor, the implementer, the orchestrator and its reviewer all run commands on it. When this was written it had 2 cores, about 3 GB RAM and about 2 GB free. Treat the task's resource class as binding.

- **`local`**: light commands only (~1 core, ≤ ~2 min, ≤ ~1 GB each). The implementer and reviewer may work at the same time.
- **`local-exclusive`** (e.g. the full test suite): at most one such job on the box at a time.
  - Before delegating or approving a run, check `cat /proc/loadavg` and `free -m`.
  - The reviewer reruns heavy checks only after the implementer reports its own run finished (this is in the reviewer brief).
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

Run tasks one at a time, in dependency order, on the single implementer. Only with `implementers=<N>`/`parallel` may independent tasks run concurrently, one per implementer run, and only if their scopes do not overlap. For the next ready task:

1. **Names** (PROTOCOL "Fixed addresses"): session `<goal-slug>-<id>` with dots turned into dashes, e.g. `m1-gate-3-1`; review team `review-<session>`; implementer `impl-<session>`; reviewer `rev-<session>`. Shorten the goal slug if a name would pass 64 characters.
2. **Hand the task to the implementer** with the brief below.
   - **First task of the goal, or no implementer run alive:** spawn it. T3: `delegate_task(task=<brief>, mode="async", isolation="shared", role="implementation", title="impl-<goal-slug> r<k>", clientRequestId="impl-<goal-slug>-r<k>")`, with `target` and `runtimeMode` unset. Standalone: the Agent tool in the background (`general-purpose`). A replacement for a run that ended gets the successor additions from step 5.
   - **Every later task:** do not spawn. Send the live implementer, at its current per-task name, `TASK <session>` followed by a newline and the brief. It re-registers as `impl-<session>` (PROTOCOL "Workers per goal"). Check it is alive first (T3 `task_status`, or the background-agent status); if its run ended, resume it (standalone `SendMessage`) or start a successor round, with the `TASK` brief included.
   - You never spawn the reviewer; the orchestrator does, and it reuses one reviewer the same way.
3. **Request the review** when the implementer sends `ARTIFACT <session> <pointer>`:
   - If no `HELLO`/`WELCOME` exchange has happened yet, repeat the start instruction to the user and wait for `HELLO` in bounded rounds. Do not fall back to a reviewer of your own; ask the user if they want that.
   - Send the `REVIEW <session>` request exactly as PROTOCOL "Review request" specifies, with the reviewer brief below filled in. Record the returned task id as `REQ`.
4. **Monitor without doing the work.**
   - Poll `inbox(agent="supervisor", team="*", wait_s=110)`. Expect `REVIEWER-UP`, then `META <session> <id>` from the implementer, then `APPROVED`. `PARKED` from the implementer goes to step 5. Between polls use `get_task(META)` and `list_tasks(team="review-<session>")`, always with bounded waits.
   - Answer agents' questions from the goal source. `STALLED` and `DEADLOCK` from the orchestrator are questions for you: decide if the goal source settles it, otherwise ask the user, then reply to the orchestrator.
   - Apply the PROTOCOL timeouts. If `REVIEWER-UP` is late, `PING`; if no `HELLO` comes back, ask the user to restart the orchestrator. It rebuilds its sessions from the bus.
5. **Recover, don't duplicate.** If the implementer stops before RELEASE (`PARKED`, error, rate limit), continue it only when there is something to act on (unread messages, changed META or concerns); your heartbeat checks this while you are idle. Standalone: resume the same agent with SendMessage to its id. T3: start a successor round of the same run, as PROTOCOL "Subagents" says: the current task's brief plus its handoff note and the prior bus state, `title="impl-<goal-slug> r<k+1>"`, `clientRequestId="impl-<goal-slug>-r<k+1>"`, never while the previous run is still going. The same applies to `HANDOFF none` between tasks: the successor starts with the next `TASK` brief plus the handoff note. Record each round as an event on the ledger task. Reviewer recovery belongs to the orchestrator.
6. **Done condition:** `APPROVED <session>` arrived and you confirmed it yourself: `get_task(META)` shows `review_state=approved` by `rev-<session>`, and no task in `review-<session>` other than META is `open`, `claimed`, `working` or `blocked`. META is still `claimed` at this point; that is expected.
7. **Verify yourself.** Rerun the task's `acceptance` check, a cheap one: pytest, `openspec validate`, the presence and recompute of the artifact. For experiments, never rerun the pod. Recompute from the pulled artifacts, or confirm the reviewer did and spot-check one number. If verification fails, send the orchestrator `REOPEN <session> <evidence>` and go back to step 4. The same reviewer reopens the review; do not mark done.
8. **Record.**
   - Tick the `tasks.md` checkbox(es) for OpenSpec goals, and complete the ledger task with a one-line result.
   - Commit with a message naming the task id and kind. Code tasks commit code, tests and docs; experiment tasks commit the EXP dir and records.
   - Push only if `push` is authorized.
9. **Release.**
   - Tell the implementer: `send(from="supervisor", to="impl-<session>", message="RELEASE <session> commit=<sha>")`. It completes META and closes the session, but keeps running and waits for the next task.
   - Send the orchestrator `RELEASE <session> commit=<sha>` and wait for `RELEASED <session>`.
   - Do **not** stop the implementer's run. Post a one-line status to the user: task id, outcome, commit.

Then take the next ready task and send it as `TASK` (step 2). If none is ready yet, leave the implementer waiting in its inbox loop.

## Failure handling

- **Experiment failed or underperformed.** Follow the project's root-cause rule. The next ledger entry is a diagnosis: a discriminating diagnostic, which may need a small `code` task plus an `experiment` task, recorded in `RESULT.md`/`HISTORY.md` before any new strategy. Never change the goal's thresholds or acceptance criteria to make a task pass.
- **Task cannot meet its acceptance check as written.** Stop, explain to the user with evidence, and propose a spec or plan update. For OpenSpec, update the change's artifacts through the openspec workflow. Do not quietly redefine the task.
- **Worker went out of scope** (a code task launched a pod, or an experiment task edited code). The reviewer must raise it as a concern. The supervisor reverts or splits the work into the correct kinds before approving.
- **Orchestrator gone mid-review.** Reviews pause; the implementer stays parked. Ask the user to restart `review-orchestrator`; never approve a task without its reviewer's approval on META.

## Human gates

Ask the user, and wait, before any of these unless already authorized:
- pushing to the remote;
- any pod spend beyond the authorized budget;
- champion promotion or any change to acceptance margins, gate worlds or frozen roster, registry or parameter versions after a candidate has seen them;
- declaring the goal failed or abandoning a task;
- running a review without the orchestrator.

## Finish

The goal is done when every ledger task is approved and recorded and the goal's exit criterion holds. For an OpenSpec change, that means all `tasks.md` boxes are ticked and `openspec validate --strict` passes.

Send the user a final report:
- the tasks completed, with their commits;
- each experiment's result in plain English (question, measurement, numbers, what changes);
- open issues and follow-ups.

Then send each implementer `STOP` (stop its run after one bounded round if it is still going), send the orchestrator `SHUTDOWN`, wait one bounded round for `BYE`, delete your heartbeat (T3 `delete_scheduled_task`, or `CronDelete`), and `remove_agent(name="supervisor")`. Offer archive or sync for OpenSpec changes; do not archive unasked.

## Brief templates

Fill every `<...>`. Paste the relevant goal-source excerpt verbatim, and give file links as absolute paths.

**Implementer, code task**
```
You are <impl-name>, implementer for task <id> of goal <goal-slug>. Implement the task, then
run the review skill (/srv/research/tokenomics/.claude/skills/review/SKILL.md) as:
impl <session> <impl-name>. Its "Supervised sessions" section applies. You are the goal's only
implementer: after RELEASE, keep running and wait for the next TASK (PROTOCOL "Workers per goal",
/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md). Schedule your heartbeat
there ("Heartbeat") if your harness lets you; delete it on STOP.
Supervisor: supervisor. Reviewer: <rev-name>, spawned by review-orchestrator in another
harness after you send the supervisor ARTIFACT, so it may take minutes to appear.
When the artifact is ready: send(to="supervisor", message="ARTIFACT <session> <pointer>"),
then the handshake. After META exists: send the supervisor "META <session> <id>".
At the done condition do NOT complete META or leave: wait for the supervisor's RELEASE.
On RELEASE: complete META, close the session, keep your name, and loop on your inbox for
"TASK <next-session>" (re-register, then work it) or "STOP" (remove_agent, end).
Stay in one turn: loop on inbox(agent="<impl-name>", team="*", wait_s=110). If your turn must end earlier, first write
a handoff note (record_task_event on META: done, open concerns, left), then send the
supervisor "PARKED <session> waiting=<what> handoff=<event id>".
KIND: code. Local edits + tests only. No pod runs, no EXP results. Smoke checks must keep
every knob tiny (horizon <= 12, <= 1 seed, seconds).
RESOURCES: <local|local-exclusive|remote:<profile>>. The box is shared (<nproc> cores,
~<avail> MB RAM): keep commands within the class. If a command needs more, stop and tell
the supervisor instead of running it. Tell the reviewer when your own full-suite run is done.
TASK: <task text + relevant spec/design excerpts>
SCOPE: <files>  ACCEPTANCE: <command or check>
Artifact for review: the uncommitted working tree diff vs <base commit>; list new files.
Do not commit or push; the supervisor commits after approval.
```

**Implementer, experiment task**
```
You are <impl-name>, implementer for task <id> of goal <goal-slug>. Do the run, then run the
review skill as: impl <session> <impl-name>. Its "Supervised sessions" section applies.
Supervisor: supervisor. Reviewer: <rev-name>, spawned by review-orchestrator after your
ARTIFACT message. When the artifact is ready: send(to="supervisor", message="ARTIFACT
<session> <pointer>"), then the handshake; after META exists send "META <session> <id>".
Do not complete META until the supervisor's RELEASE. You are the goal's only implementer:
after RELEASE keep running and wait for "TASK <next-session>" or "STOP" (PROTOCOL "Workers per
goal"); schedule your heartbeat per PROTOCOL "Heartbeat" if your harness allows. Stay in one
turn: loop on inbox(agent="<impl-name>", team="*", wait_s=110). If your turn
must end earlier, write a handoff note on META first, then send the supervisor
"PARKED <session> waiting=<what> handoff=<event id>".
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

**Reviewer brief (goes after `brief:` in the REVIEW request; the orchestrator passes it to the reviewer verbatim)**
```
Task <id> of goal <goal-slug>. Implementer: <impl-name>. Supervisor: supervisor.
KIND: <code|experiment>. RESOURCES: <class>. Review against: <task text, spec/design
excerpts, acceptance>. Base commit: <sha>. Do not run local-exclusive checks (full suite)
until the implementer says its own run finished; do not rerun remote work locally.
Code: verify correctness, tests that fail on regression (mutation-check key guards), spec
agreement and scope. Experiment: audit the setup before the numbers, recompute the numbers
from the raw pulled artifacts, check compute-rule compliance and input pins, and check
the records match the data. Flag any kind violation (code task ran a pod, experiment
task edited code) as a concern. Do not edit files. Approve only when every concern you
raised is completed.
```

## Rules

- The supervisor never implements and never reviews. If you catch yourself editing project files, stop and delegate.
- One kind per task, always. One implementer run and one reviewer run for the whole goal unless the user granted more; bus names are unique per task. Reviews always go through the orchestrator.
- Every state change goes through the bus. Chat-only agreements do not count.
- Bounded waits only. No blind polling loops and no sleeping. The heartbeat is the only scheduled wake-up.
- Keep the user informed with one-line status per task transition. Save detail for the final report.

## Quick reference

| What | Tool |
|---|---|
| Register | `register(name="supervisor", team="supervision", replace=true)` |
| Find orchestrator | `whois(team="supervision")`, `send(to="review-orchestrator", message="PING")` |
| Bind orchestrator | on `HELLO`: `send(… "WELCOME goal=… source=… max_reviewers=…")` |
| Ledger entry | `create_task(requested_by="supervisor", team="supervision", title="[goal][kind] id …")` |
| Spawn implementer (first task only) | T3 `delegate_task(mode="async", isolation="shared")`; standalone Agent tool (background, general-purpose) |
| Next task, same implementer | `send(… to=<its current name>, "TASK <session>\n<brief>")` |
| Heartbeat | T3 `schedule_task({type:"interval", everyMs:600000})` / `delete_scheduled_task`; standalone `CronCreate("3-59/10 * * * *")` / `CronDelete` |
| Request review | `delegate(from="supervisor", to_agent="review-orchestrator", team="supervision", title="REVIEW <session>", …)` |
| Watch | `inbox(agent="supervisor", team="*", wait_s=110)`, `get_task(META)`, `list_tasks(team="review-<session>")` |
| Reopen after failed verification | `send(… "REOPEN <session> <evidence>")` |
| End a session | `send(… to="impl-<session>", "RELEASE <session> commit=<sha>")`, then the same to `review-orchestrator` |
| Continue a stopped implementer | standalone: SendMessage to its id; T3: successor round `delegate_task` |
| Close ledger entry | `update_task(agent="supervisor", task_id=…, state="completed", result=…)` |
| Leave | `STOP` to implementer(s), `send(… "SHUTDOWN")`, delete heartbeat, `remove_agent(name="supervisor")` |

</what-to-do>
