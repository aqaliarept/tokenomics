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
`rev-<session>`.

## Messages

Plain `send(from=…, to=…, message=…)`. The first token is the type; parse by it. Fields are
`key=value` with no spaces inside a value, except the free text at the end of `REOPEN`,
`STALLED` and `DEADLOCK`.

| From → to | Message | Meaning / required reply |
|---|---|---|
| sup → orch | `PING` | Supervisor (re)started. Reply `HELLO`. |
| orch → sup | `HELLO harness=<claude-code\|codex\|…> model=<id> active=<session:rev,…\|none>` | Orchestrator is up. Sent on start once `supervisor` is registered, and on every `PING`. Reply `WELCOME`. |
| sup → orch | `WELCOME goal=<slug> source=<goal-source> max_reviewers=<1\|2>` | Binds the orchestrator to the goal. `max_reviewers` is 2 only with the `parallel` authorization. |
| sup → orch | **task** `REVIEW <session>` (see below) | Review request. Orchestrator acknowledges it and spawns the reviewer. |
| orch → sup | `REVIEWER-UP <session> <rev-name>` | The reviewer is registered on `review-<session>`. |
| orch → sup | `APPROVED <session> META=<id> concerns=<n>` | Done condition holds (see Lifecycle 6). |
| orch → sup | `STALLED <session> <reason>` | The review cannot progress without the supervisor (missing artifact, implementer silent, reviewer cannot start). |
| orch → sup | `DEADLOCK <session> task=<id> <impl position> \| <rev position>` | More than two round trips on one concern. The supervisor decides or asks the user. |
| sup → orch | `REOPEN <session> <evidence>` | The supervisor's own verification failed after approval. |
| sup → orch | `RELEASE <session> commit=<sha>` | The task is recorded. End the review session. |
| orch → sup | `RELEASED <session>` | Reviewer stopped and removed; request task completed. |
| sup → orch | `SHUTDOWN` | Goal finished or abandoned. Release every session, then leave. |
| orch → sup | `BYE` | Orchestrator leaving the bus. |

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
3. The orchestrator acknowledges `REQ`, spawns one long-lived reviewer subagent named
   `rev-<session>`, and sends `REVIEWER-UP` once `wait_for_agents` sees it on the team.
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
   - **Pass:** it records and commits, tells the implementer to close (implementer completes
     META and leaves the team), then sends `RELEASE`. The orchestrator tells the reviewer to
     leave, closes the subagent, completes `REQ` with a one-line result, and sends `RELEASED`.

## Subagents

Applies to both sides: the supervisor's implementers and the orchestrator's reviewers.

- **One run holds the session.** A worker works in one turn and loops on its inbox
  (`wait_s=110`) until `RELEASE`, which always arrives as a bus message: from `supervisor` to
  `impl-<session>`, from `review-orchestrator` to `rev-<session>`.
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
  `target`/`runtimeMode` unset (inherit), `title="<name> r<k>"`,
  `clientRequestId="<name>-r<k>"`. Round k+1 is a new `delegate_task`; never `t3_thread_send`
  to the old `childThreadId`. The owner keeps its own turn alive with its inbox loop, since bus
  messages do not wake an idle T3 thread.
- Replacements are recorded as task events (on `REQ` for reviewers, on the ledger task for
  implementers), not as messages, unless the successor fails to start.

## Bus facts (verified on agent-bus 0.43.0)

- `delegate` creates the task already `claimed` by the assignee; `acknowledge_task(agent=…,
  task_id=…, response="claimed")` records the receipt.
- A `claimed` task can be completed by its requester or claimer. An `open` task cannot be
  completed directly: claim it first.
- A `review_required` task refuses `completed` until `review_state=approved`.
  `submit_review` takes `notes`, not `comment`; `approved=false` revokes an earlier approval.
- `review_state` and `reviewed_by` hold a single value, the latest review. One reviewer per
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
