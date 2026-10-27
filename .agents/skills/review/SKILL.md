---
name: review
description: Multi-agent code/spec review across harnesses (Claude ↔ Codex) using agent-bus. One implementer, one or more reviewers. Reviewers raise concerns as tasks; the implementer addresses each one by either fixing or arguing; the session closes when every reviewer has approved and no concern task is still open. Use when the user says "run a cross-harness review", "review this change with another model", "/review impl <slug>", "/review rev <slug>", or similar.
---

<what-to-do>

You are one participant in a many-sided review. Agent-bus is the shared ledger — every state change (concern, response, approval, veto) MUST go through its MCP tools so the other side sees it. Do not simulate a review by talking only in your own thread.

The caller passes `$ARGUMENTS` as `<role> <slug> [name]`:
- `role` — `impl` for the author of the change, `rev` for a reviewer.
- `slug` — short kebab-case identifier for the review session (e.g. `add-stochastic-env`, `fix-leaky-eval`). Picks the agent-bus team: **`review-<slug>`**.
- `name` — optional agent-bus name to register under. The `supervise` and `review-orchestrator` skills pass per-task names such as `impl-m1-gate-3-1` / `rev-m1-gate-3-1`. Default: `impl` for the implementer, `rev<N>` for a reviewer.

If `role` or `slug` is missing or malformed, stop and ask the user for them.

Below, `impl` and `rev<N>` stand for the **resolved** names. The implementer states its name in the handshake and on the meta-task; reviewers address concerns to that name.

Bus facts this skill relies on (task states, `submit_review`, completion rules) are listed in `/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md` under "Bus facts". Valid task states are `open`, `claimed`, `working`, `blocked`, `completed`, `failed`, `canceled`; there is no `done` or `in_progress`.

## Agent-bus setup check

Before anything else, confirm agent-bus MCP is wired for this session (the `mcp__agent-bus__*` tools must be available). If they are not, tell the user to run `agent-bus setup --mode local --clients claude-code,codex` and stop. Do not try to proceed without the bus — this skill has no local fallback.

## Phase 1 — Register with your role

### If `role == impl`

Call `register(name=<name or "impl">, team="review-<slug>", capabilities=["review:implementer"], replace=true)`.
`replace=true` is intentional — if a previous implementer with that name crashed, reclaim the name.

### If `role == rev`

If a `name` was passed, call `register(name=<name>, team="review-<slug>", capabilities=["review:reviewer"], replace=false)` and skip the steps below. Otherwise:

1. Call `whois(team="review-<slug>")` and list existing `rev<N>` names.
2. Pick the lowest unused `N` starting at 1 (`rev1`, `rev2`, …).
3. Call `register(name="rev<N>", team="review-<slug>", capabilities=["review:reviewer"], replace=false)`.
4. If registration fails because the name is now taken (a sibling reviewer raced you), bump `N` and retry. Register under exactly one name per session.

Record your resolved name once; use it in every subsequent call as `from=` / `by_agent=` / `agent=`.

## Phase 2 — Handshake

The handshake must be visible on the bus so late joiners (and the human) can catch up by reading the team log.

### Implementer

1. `send_team(team="review-<slug>", from="impl", message="<impl name> ready for review. Artifact: <concrete pointer>")`.
   The pointer must be verifiable by the reviewers without asking you — a git ref (branch/sha), a diff path, a PR link, or an explicit file list. Do not be vague.
2. `wait_for_agents(names=[<expected reviewer names, default "rev1">], team="review-<slug>", timeout_s=60)` to confirm at least one reviewer is on the bus. If no reviewer appears, `send_team` the handshake again and tell the user. Do not open the session unilaterally.
3. Create the session meta-task, then claim it so it can later be completed:
   `create_task(requested_by="impl", team="review-<slug>", title="Review session: <slug>", description="implementer: <impl name>. <artifact pointer + scope>", review_required=true)`, then `update_task(agent="impl", task_id=<META>, state="claimed")`.
   Record the returned `task_id` as **`META`**.
4. `send_team(team="review-<slug>", from="impl", message="meta-task=<META>. Reviewers: submit_review on <META> when you approve.")`.

### Reviewer

1. On registering, call `inbox(agent="rev<N>", team="review-<slug>", wait_s=30)` and read the implementer's handshake. If you joined after it was sent and it is not in your inbox, read it from `activity(team="review-<slug>")`. Extract the implementer's name, the artifact pointer and `META` (META may follow a little later).
2. `send_team(team="review-<slug>", from="rev<N>", message="rev<N> joined, reviewing <artifact pointer>")`.
3. Read the artifact. Do a real review — the challenge/tdd/openspec skills' trust rules apply here too: numbers get recomputed from raw artifacts, specs get checked against code, no "looks fine" without evidence.

## Phase 3 — Concern cycle

Every concern is a task. Chat-only messages do not count — they do not appear on the Kanban and have no state.

### Reviewer: raise a concern

Use `delegate` (it bundles `create_task` + `assign_task`; the task starts `claimed` by the implementer):

```
delegate(
  from="rev<N>",
  to_agent="impl",
  team="review-<slug>",
  title="<concern summary, one line>",
  description="<concrete finding: file:line, numbers, quote from spec, etc.>",
  mode="edit_files",
  expected_output="code change OR a written justification for why no change is needed",
  edit_scope=["<file paths you expect to be touched>"]
)
```

Repeat for each distinct concern. Group them if genuinely related; split them if resolution is independent.

Do not create a concern for stylistic nits unless the user explicitly asked for style review. The point is correctness, missing evidence, and spec/code drift.

### Implementer: pick up and act

Pull work with `inbox(agent="impl", team="review-<slug>", wait_s=110)`. For each delegated task:

1. `acknowledge_task(agent="impl", task_id=<id>, response="claimed")` — within the first poll. Reviewers get `AGENT_HAS_ACTIVE_TASKS` surfaced if you don't.
2. Decide: **fix** or **argue**. There is no third option.
   - **Fix:** make the actual change (edit code / spec). Then:
     - `record_task_event(by_agent="impl", task_id=<id>, event_type="progress", message="fixed in <file>:<line>, commit <sha-or-pending>", phase="fixed")`
     - `send(from="impl", to="rev<N>", message="task <id>: fixed in <file>:<line>. Please verify.")` — directed, not team-broadcast.
   - **Argue:** write a specific counter-argument, not a dismissal. Then:
     - `record_task_event(by_agent="impl", task_id=<id>, event_type="progress", message="disagree: <rationale with evidence>", phase="argued")`
     - `send(from="impl", to="rev<N>", message="task <id>: disagree because <rationale>. Open to clarification or additional evidence.")`
   - Leave the task `claimed`; the reviewer owns the close.
3. Never close a concern task yourself. Impl marks progress; reviewer decides whether it's resolved.

### Reviewer: verify a response

For each task where impl has posted an event, re-check the artifact.

- If the fix actually resolves the concern (verify by reading the diff / rerunning the check — don't trust the event message alone):
  - `update_task(agent="rev<N>", task_id=<id>, state="completed", result="<what you verified>")`.
- If the fix is incomplete or the argument fails on the merits:
  - `record_task_event(by_agent="rev<N>", task_id=<id>, event_type="progress", message="still open because <specific reason>", phase="reopened")` and `send` the implementer the reason.
  - Leave the task `claimed`.

Debate goes in the task's event timeline and in `send`/`reply` pairs — that keeps the Kanban honest (non-terminal tasks = open concerns).

## Phase 4 — Approval

Only when every one of your own concern tasks is `completed`:

- To approve: `submit_review(reviewer="rev<N>", task_id=<META>, approved=true, notes="LGTM — concerns N, N, N addressed")`.
- To veto: `submit_review(reviewer="rev<N>", task_id=<META>, approved=false, notes="<the one unfixed thing>")`. A veto blocks completion; raise it only if an open task genuinely cannot be closed.

Reviewers may approve independently. Implementers do not approve. `review_state` on META holds only the **latest** review; with several reviewers, the implementer confirms each one's latest review in `activity(team="review-<slug>")`.

## Phase 5 — Done condition

The implementer polls — reviewers do not need to do anything after their approval.

Implementer, after every inbox round:

1. `get_task(task_id=<META>)` and inspect `review_state` and `reviewed_by` (and, with several reviewers, the review events in `activity`).
2. `list_tasks(team="review-<slug>", state=["open","claimed","working","blocked"])`.

Session is **complete** when:
- Every registered reviewer's latest review on `META` is `approved=true`, AND
- No task on the team other than `META` is in `open`, `claimed`, `working` or `blocked`.

When complete (unsupervised sessions; supervised sessions wait for RELEASE, see below):
- `update_task(agent="impl", task_id=<META>, state="completed", result="<n> concerns closed")` (the bus refuses this until the review is approved)
- `send_team(team="review-<slug>", from="impl", message="review session <slug> closed. Thanks rev1..revN.")`
- `remove_agent(name="impl")` to leave the team cleanly. (Reviewers may do the same after seeing the close message, or just stop checking.)

If complete but a reviewer vetoed: do NOT close. Fix the vetoed concern (open a fresh task if needed) and go back to Phase 3.

## Supervised sessions

When the session runs under the `supervise` skill (the implementer was briefed by `supervisor`, the reviewer by `review-orchestrator`), the protocol in `/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md` adds these rules:

- **Implementer:** when the artifact is ready, first `send(to="supervisor", message="ARTIFACT <slug> <pointer>")`, then do Phase 2. The reviewer is spawned in another harness only after that message, so in step 2 keep waiting in bounded rounds for up to ~20 min, and tell the supervisor, not the user, if it still has not appeared. After creating META, `send(to="supervisor", message="META <slug> <id>")`.
- **Reviewer:** tell `review-orchestrator` `JOINED` after the handshake and `APPROVED` after `submit_review(approved=true)`, as your brief says.
- **Nobody closes on their own.** At the done condition the implementer does **not** complete META and nobody leaves the team: the supervisor verifies first and may reopen. The implementer completes META only when the supervisor says RELEASE; the reviewer stops reviewing only when the orchestrator says RELEASE.
- **RELEASE ends the task, not the worker.** By default you are the goal's only implementer (or reviewer) and you carry every task. After RELEASE keep your name and keep looping on your inbox. On `TASK <next-session>` (implementer) or `REVIEW <next-session>` (reviewer): register the new per-task name on `review-<next-session>`, then `remove_agent` the old one, then run this skill again for the new session. On `STOP`: `remove_agent`, delete your heartbeat, end. Details: PROTOCOL "Workers per goal".
- **Heartbeat.** When you register for your first session, schedule a heartbeat in your own session if your harness can (PROTOCOL "Heartbeat"; every 10 min, it only checks your inbox and ends at once if nothing is pending). Delete it on `STOP`. If your harness cannot, your owner's heartbeat covers you.
- **One turn until STOP.** Loop on your inbox (`wait_s=110`) through RELEASE and the next tasks until STOP arrives over the bus. If your turn must end earlier, write a handoff note first (`record_task_event` on META: open concerns with both positions, what was verified, what is left) and send your owner `PARKED <slug> waiting=<what> handoff=<event id>`; you may be replaced by a successor that reads it (PROTOCOL "Subagents").
- **Reopen:** when the supervisor's verification fails, the reviewer gets the evidence from the orchestrator. It revokes with `submit_review(approved=false, notes="reopened: <evidence>")`, raises the evidence as a normal concern, and the cycle continues from Phase 3.

## Rules

- One name per agent per session. Do not re-register under a different name mid-stream. (A supervised worker moving to its next task re-registers under that task's name; see Supervised sessions.)
- Every concern is a task. Chatter without a task has no state and does not block completion.
- Impl never closes a concern. Reviewer never approves `META` while any of their concerns is still open.
- "Fixed" means verifiable: cite file:line, commit sha, or the exact diff hunk — not just "done".
- "Disagree" means specific: cite the counter-evidence or the spec clause — not "I don't think so".
- If the review is deadlocked for more than two round-trips on the same concern, surface the impasse: in a supervised session send `DEADLOCK` (reviewer → orchestrator); otherwise tell the human via `remember(kind="handoff", content="deadlock on task <id>: <impl position> vs <rev position>", pinned=true)` and `record_decision`. Do not loop forever.
- Do not enter a blind polling loop. Each inbox call uses a bounded `wait_s`; between calls, consult `inbox_status`, `project_board(team="review-<slug>")`, or `activity(team="review-<slug>")` for a non-consuming view.
- When the user opens this skill as a one-shot (e.g. `/review impl add-foo` then exits the thread), register and run through Phases 1–3, then stop with a clear hand-off note. Do not expect to be alive for Phase 5 unless the user keeps the session running.

## Quick reference — the tools you actually call

| What | Tool |
|---|---|
| Register | `register(name=…, team=review-<slug>, capabilities=[…], replace=…)` |
| See who's here | `whois(team=review-<slug>)` / `directory()` |
| Wait for someone | `wait_for_agents(names=[…], team=…, timeout_s=…)` |
| Announce / chatter | `send_team(team=…, from=…, message=…)` |
| Direct message | `send(from=…, to=…, message=…)` |
| Blocking question | `ask(from=…, to=…, question=…)` |
| Raise concern | `delegate(from=rev<N>, to_agent=impl, team=…, title=…, description=…, mode="edit_files", expected_output=…, edit_scope=[…])` |
| Accept assignment | `acknowledge_task(agent=impl, task_id=…, response="claimed")` |
| Progress / argument | `record_task_event(by_agent=…, task_id=…, event_type="progress", message=…)` |
| Reviewer closes concern | `update_task(agent=rev<N>, task_id=…, state="completed", result=…)` |
| Approve/veto session | `submit_review(reviewer=rev<N>, task_id=<META>, approved=…, notes=…)` |
| Inspect state | `get_task(task_id=…)`, `list_tasks(team=…, state=[…])`, `project_board(team=…)`, `activity(team=…)` |
| Pull inbox | `inbox(agent=…, team=review-<slug>, wait_s=…)` |
| Leave | `remove_agent(name=…)` |

</what-to-do>
