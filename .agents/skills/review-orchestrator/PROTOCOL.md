# Supervisor ↔ review-orchestrator protocol

Shared by `/srv/research/tokenomics/.claude/skills/supervise/SKILL.md` and
`/srv/research/tokenomics/.agents/skills/review-orchestrator/SKILL.md`. Change it here, never in a
copy, and keep both skills in agreement with it.

## Fixed addresses

| What | Value |
|---|---|
| Control team | `supervision` |
| Supervisor | `supervisor` |
| Orchestrator | `review-orchestrator` |
| Working directory of both sessions | `/srv/research/tokenomics` (the bus derives the project scope from it) |

The addresses are hardcoded so neither side needs a pasted prompt: the user starts the
orchestrator skill with no arguments and it finds the supervisor on the bus. One supervisor and
one orchestrator per bus at a time; a newly started one takes over its name (`replace=true`).

Per-task names come from the session slug `<goal-slug>-<task-id>` (dots → dashes, ≤ 64 chars of
`[a-zA-Z0-9_.-]`): review team `review-<session>`, implementer `impl-<session>`, reviewer
`rev-<session>`. These are **bus names**, one per task. They are not separate agents: by default
one implementer run and one reviewer run carry every task of the goal (see "Workers per goal") and
re-register under each new task's names.

## Workers per goal

**Default: one implementer and one reviewer for the whole goal.** The supervisor spawns one
implementer run and the orchestrator spawns one reviewer run. Each run does task after task:
after `RELEASE` it stays alive and takes the next task from a `TASK` (implementer) or `REVIEW`
(reviewer) message. Tasks therefore run one at a time. A worker is replaced only when its run
ends and cannot be resumed (successor round, see "Subagents"), never just because a task ended.

**Overrides are explicit only.** The user grants them on the supervisor's arguments:
`implementers=<N>` (concurrent implementer runs) and `reviewers=<N>` (concurrent reviewer runs,
sent as `max_reviewers=<N>` in `WELCOME`); `parallel` is shorthand for `implementers=2
reviewers=2`. With N > 1 each run is still long-lived and reused; extra runs exist only so that
independent tasks can overlap. The user may also tell the orchestrator directly; that overrides
`max_reviewers` for its session. Nobody raises the count on their own judgement.

**Moving a worker to the next task.** On `TASK <session>` / `REVIEW <session>` the worker first
`register`s the new per-task name on `review-<session>`, then `remove_agent`s its previous
per-task name, then works the new task. Between tasks it keeps its last name registered and loops
on its inbox, so the owner can always reach it. On `STOP` it removes its name, deletes its
heartbeat and ends its run.

**Context budget.** A worker whose context is nearly full between tasks sends its owner
`HANDOFF none <note>` (what it learned that the next task needs) and stops; the owner starts a
successor round for the next task. Mid-task it uses the normal handoff note on META.

## Messages

Plain `send(from=…, to=…, message=…)`. The first token is the type; parse by it. Fields are
`key=value` with no spaces inside a value, except the free text at the end of `REOPEN`,
`STALLED` and `DEADLOCK`.

| From → to | Message | Meaning / required reply |
|---|---|---|
| sup → orch | `PING` | Supervisor (re)started. Reply `HELLO`. |
| orch → sup | `HELLO harness=<claude-code\|codex\|…> model=<id> active=<session:rev,…\|none>` | Orchestrator is up. Sent on start once `supervisor` is registered, and on every `PING`. Reply `WELCOME`. |
| sup → orch | `WELCOME goal=<slug> source=<goal-source> max_reviewers=<N>` | Binds the orchestrator to the goal. `max_reviewers` is 1 unless the user granted `reviewers=<N>` or `parallel` ("Workers per goal"). |
| sup → orch | **task** `REVIEW <session>` (see below) | Review request. Orchestrator acknowledges it and hands it to its live reviewer, spawning one only if none is running. |
| orch → sup | `REVIEWER-UP <session> <rev-name>` | The reviewer is registered on `review-<session>`. |
| orch → sup | `APPROVED <session> META=<id> concerns=<n>` | Done condition holds (see Lifecycle 6). |
| orch → sup | `STALLED <session> <reason>` | The review cannot progress without the supervisor (missing artifact, implementer silent, reviewer cannot start). |
| orch → sup | `DEADLOCK <session> task=<id> <impl position> \| <rev position>` | More than two round trips on one concern. The supervisor decides or asks the user. |
| sup → orch | `REOPEN <session> <evidence>` | The supervisor's own verification failed after approval. |
| sup → orch | `RELEASE <session> commit=<sha>` | The task is recorded. End the review session. |
| orch → sup | `RELEASED <session>` | Reviewer stopped and removed; request task completed. |
| sup → orch | `SHUTDOWN` | Goal finished or abandoned. Release every session, then leave. |
| orch → sup | `BYE` | Orchestrator leaving the bus. |

Owner → worker messages (supervisor → its implementer, orchestrator → its reviewer), sent to
the worker's current per-task name:

| Message | Meaning |
|---|---|
| `TASK <session>` + newline + brief | Supervisor → implementer: next task. Re-register as `impl-<session>` and start. |
| `REVIEW <session>` + newline + brief | Orchestrator → reviewer: next review. Re-register as `rev-<session>` and start. |
| `RELEASE <session> commit=<sha>` | Task recorded. Close the session per the `review` skill, keep the name, wait for the next task. |
| `STOP` | Goal over. Remove your name, delete your heartbeat, end your run. |

## Review request

```
delegate(from="supervisor", to_agent="review-orchestrator", team="supervision",
         title="REVIEW <session>", mode="investigate_only",
         expected_output="APPROVED <session>", description=<block>)
```

The description is this block, with the reviewer brief from the supervise skill filled in and
pasted verbatim after `brief:`:

```
session: <session>
team: review-<session>
reviewer: rev-<session>
implementer: impl-<session>
kind: code|experiment
resources: local|local-exclusive|remote:<profile>
repo: /srv/research/tokenomics
artifact: <git ref / diff base + file list / EXP dir>
brief:
<filled reviewer brief>
```

The returned task id is `REQ`. `delegate` leaves it `claimed` by the orchestrator.

## Lifecycle of one review session

1. The implementer finishes its artifact, sends the supervisor `ARTIFACT <session> <pointer>`,
   registers on `review-<session>`, posts the handshake to the team and waits for the reviewer.
2. The supervisor sends the `REVIEW` request.
3. The orchestrator acknowledges `REQ`, hands the request to its live reviewer run (`REVIEW
   <session>`) or spawns one if none is running, and sends `REVIEWER-UP` once `wait_for_agents`
   sees `rev-<session>` on the team.
4. The implementer sees the reviewer, creates META (`review_required=true`), sets it `claimed`,
   and sends the supervisor `META <session> <id>`. The concern cycle of the `review` skill runs.
5. The reviewer approves META with `submit_review(approved=true)` once its own concerns are all
   `completed`, and tells the orchestrator.
6. The orchestrator checks the done condition itself: `get_task(META)` shows
   `review_state=approved` with `reviewed_by=rev-<session>`, and
   `list_tasks(team="review-<session>", state=["open","claimed","working","blocked"])` holds
   nothing but META. Then it sends `APPROVED`.
7. META stays `claimed`. The implementer does **not** complete it and nobody leaves the team,
   because the supervisor may still reopen.
8. The supervisor verifies.
   - **Fail:** it sends `REOPEN`. The orchestrator forwards the evidence to the same reviewer,
     which revokes with `submit_review(approved=false, notes="reopened: …")` (this sets
     `changes_requested`) and raises the failure as a concern. Back to 5.
   - **Pass:** it records and commits, sends the implementer `RELEASE` (implementer completes
     META and closes the session, but stays alive for the next task), then sends the
     orchestrator `RELEASE`. The orchestrator relays `RELEASE` to the reviewer (which also stays
     alive), completes `REQ` with a one-line result, and sends `RELEASED`.

## Subagents

Applies to both sides: the supervisor's implementers and the orchestrator's reviewers.

- **One run holds the goal.** A worker works in one turn and loops on its inbox (`wait_s=110`)
  through every task it is given, until `STOP`. `RELEASE`, `TASK`, `REVIEW` and `STOP` always
  arrive as bus messages: from `supervisor` to the implementer, from `review-orchestrator` to the
  reviewer.
- **Early end.** A worker whose turn must end before `RELEASE` first writes a handoff note
  (`record_task_event` on META: open concerns with both positions, what was verified and how,
  what is left), then tells its owner: `PARKED <session> waiting=<what> handoff=<event id>`.
- **Continue.** Where the harness can resume a subagent (standalone Codex multi-agent tools,
  standalone Claude Code background Agent + `SendMessage`), the owner resumes the same one.
  Where it cannot, the owner starts a **successor round** under the same name: the original
  brief plus the handoff and the prior bus state (META, concern tasks and their states). Never
  two live runs under one name.
- **T3 Code threads** use T3 `delegate_task`; T3 requires it when the child inherits model
  options, and a finished task is not continued. Spawn with `mode="async"`,
  `isolation="shared"` (the reviewer must see the implementer's uncommitted edits),
  `target`/`runtimeMode` unset (inherit), `title="<run-name> r<k>"`,
  `clientRequestId="<run-name>-r<k>"`, where the run name is per goal, not per task:
  `impl-<goal-slug>` / `rev-<goal-slug>` (with N > 1 runs, `impl-<goal-slug>-<j>`). Round k+1 is a new `delegate_task`; never `t3_thread_send`
  to the old `childThreadId`. The owner keeps its own turn alive with its inbox loop, since bus
  messages do not wake an idle T3 thread.
- Replacements are recorded as task events (on `REQ` for reviewers, on the ledger task for
  implementers), not as messages, unless the successor fails to start.

## Heartbeat

Bus messages do not wake an idle session, so a participant whose turn ended (rate limit,
`PARKED`, a crash of its loop) can sit forever with unread messages. Every participant
therefore schedules a **heartbeat** in its own session when it registers and deletes it when
it leaves (`SHUTDOWN`/`BYE`/`STOP`).

- **Interval:** every 10 min. The heartbeat is a stuck-guard, not the main loop; the inbox loop
  stays the normal way to wait.
- **Mechanism, by harness:**
  - T3 Code thread (including a delegated child): `schedule_task(schedule={"type":"interval",
    "everyMs":600000}, title="heartbeat <name>", clientRequestId="heartbeat-<run-name>-r<k>",
    prompt=<below>)`, bound to its own thread (the default). It is persistent, so delete it with
    `delete_scheduled_task` on leave. On start, `list_scheduled_tasks` and delete any earlier
    `heartbeat <name>` left by a previous run of the same role.
  - Standalone Claude Code: `CronCreate(cron="3-59/10 * * * *", recurring=true, prompt=<below>)`.
    It fires only while the session is idle, lives only as long as the session and expires after
    7 days. Delete it with `CronDelete` on leave.
  - Codex or another harness: its own scheduler if it has one. If it has none, tell the owner
    (workers) or the user (supervisor, orchestrator) once, and rely on the inbox loop plus the
    owner's heartbeat.
  - A worker that is a background subagent inside its owner's Claude Code session cannot schedule
    into its own context; the owner's heartbeat covers it (below).
- **Prompt** (fill `<name>` and `<skill>`):
  ```
  HEARTBEAT <name>. You are <name> under the <skill> skill. Check inbox_status for <name>.
  If messages are unread, or a worker you own has stopped while its bus state changed, handle
  them per your skill and go back to your inbox loop. If nothing is pending, answer "idle"
  and end the turn. Do not start new work the bus did not ask for.
  ```
- **What a heartbeat checks.** Everyone: its own unread messages. Owners additionally: each
  worker's `inbox_status` and run status; a worker with unread messages whose run has ended is
  resumed or gets a successor round ("Subagents"). The supervisor also checks the orchestrator
  (`PING` if a review is waiting and nothing arrived for ~30 min).
- **Cost.** A heartbeat that finds nothing must end its turn at once: one `inbox_status` call, no
  reading of files.

## Bus facts (verified on agent-bus 0.43.0)

- `delegate` creates the task already `claimed` by the assignee; `acknowledge_task(agent=…,
  task_id=…, response="claimed")` records the receipt.
- A `claimed` task can be completed by its requester or claimer. An `open` task cannot be
  completed directly: claim it first.
- A `review_required` task refuses `completed` until `review_state=approved`.
  `submit_review` takes `notes`, not `comment`; `approved=false` revokes an earlier approval.
- `review_state` and `reviewed_by` hold a single value, the latest review. One reviewer name per
  session keeps that unambiguous.
- Valid task states: `backlog`, `open`, `claimed`, `working`, `blocked`, `completed`, `failed`,
  `canceled`. There is no `done` or `in_progress`.

## Timeouts

All waits are bounded: `inbox(wait_s≤110)`, `wait_for_agents(timeout_s≤110)`. No sleeping.

| Wait | Limit | Then |
|---|---|---|
| Supervisor waiting for `HELLO` after `PING` | 2 rounds | Treat the orchestrator as absent; ask the user to start it. |
| Supervisor waiting for `REVIEWER-UP` after `REVIEW` | ~10 min | `PING`; if no `HELLO`, ask the user to restart the orchestrator. |
| Implementer waiting for its reviewer | ~20 min | Tell the supervisor, keep waiting. |
| Orchestrator waiting for the reviewer to register | ~10 min | `STALLED`. |
| No activity on `review-<session>` | ~30 min | Orchestrator checks both sides; `STALLED` if the implementer is the silent one. |
