---
name: review
description: Multi-agent code/spec review across harnesses (Claude ↔ Codex) using agent-bus. One implementer, one or more reviewers. Reviewers raise concerns as tasks; the implementer addresses each one by either fixing or arguing; the session closes when every reviewer has approved and no concern task is still open. Use when the user says "run a cross-harness review", "review this change with another model", "/review impl <slug>", "/review rev <slug>", or similar.
---

<what-to-do>

You are one participant in a many-sided review. Agent-bus is the shared ledger — every state change (concern, response, approval, veto) MUST go through its MCP tools so the other side sees it. Do not simulate a review by talking only in your own thread.

The caller passes `$ARGUMENTS` as `<role> <slug>`:
- `role` — `impl` for the author of the change, `rev` for a reviewer.
- `slug` — short kebab-case identifier for the review session (e.g. `add-stochastic-env`, `fix-leaky-eval`). Picks the agent-bus team: **`review-<slug>`**.

If `$ARGUMENTS` is empty or malformed, stop and ask the user for both.

## Agent-bus setup check

Before anything else, confirm agent-bus MCP is wired for this session (the `mcp__agent-bus__*` tools must be available). If they are not, tell the user to run `agent-bus setup --mode local --clients claude-code,codex` and stop. Do not try to proceed without the bus — this skill has no local fallback.

## Phase 1 — Register with your role

### If `role == impl`

Call `register(name="impl", team="review-<slug>", capabilities=["review:implementer"], replace=true)`.
`replace=true` is intentional — if a previous `impl` crashed, reclaim the name.

### If `role == rev`

1. Call `whois(team="review-<slug>")` and list existing `rev<N>` names.
2. Pick the lowest unused `N` starting at 1 (`rev1`, `rev2`, …).
3. Call `register(name="rev<N>", team="review-<slug>", capabilities=["review:reviewer"], replace=false)`.
4. If registration fails because the name is now taken (a sibling reviewer raced you), bump `N` and retry. Register under exactly one name per session.

Record your resolved name (`impl` or `rev<N>`) once; use it in every subsequent call as `from=` / `by_agent=` / `agent=`.

## Phase 2 — Handshake

The handshake must be visible on the bus so late joiners (and the human) can catch up by reading the team log.

### Implementer

1. `send_team(team="review-<slug>", from="impl", message="impl ready for review. Artifact: <concrete pointer>")`.
   The pointer must be verifiable by the reviewers without asking you — a git ref (branch/sha), a diff path, a PR link, or an explicit file list. Do not be vague.
2. `wait_for_agents(names=["rev1"], wait_s=60)` to confirm at least one reviewer is on the bus. If no reviewer appears, `send_team` the handshake again and tell the user. Do not open the session unilaterally.
3. Create the session meta-task:
   `create_task(requested_by="impl", title="Review session: <slug>", description="<artifact pointer + scope>", review_required=true, state="open")`.
   Record the returned `task_id` as **`META`**.
4. `send_team(team="review-<slug>", from="impl", message="meta-task=<META>. Reviewers: submit_review on <META> when you approve.")`.

### Reviewer

1. On registering, call `inbox(agent="rev<N>", team="review-<slug>", wait_s=30)` and read the implementer's handshake. Extract the artifact pointer and `META`.
2. `send_team(team="review-<slug>", from="rev<N>", message="rev<N> joined, reviewing <artifact pointer>")`.
3. Read the artifact. Do a real review — the challenge/tdd/openspec skills' trust rules apply here too: numbers get recomputed from raw artifacts, specs get checked against code, no "looks fine" without evidence.

## Phase 3 — Concern cycle

Every concern is a task. Chat-only messages do not count — they do not appear on the Kanban and have no state.

### Reviewer: raise a concern

Use `delegate` (it bundles `create_task` + `assign_task`):

```
delegate(
  from="rev<N>",
  to_agent="impl",
  title="<concern summary, one line>",
  description="<concrete finding: file:line, numbers, quote from spec, etc.>",
  mode="fix",
  expected_output="code change OR a written justification for why no change is needed",
  edit_scope=["<file paths you expect to be touched>"]
)
```

Repeat for each distinct concern. Group them if genuinely related; split them if resolution is independent.

Do not create a concern for stylistic nits unless the user explicitly asked for style review. The point is correctness, missing evidence, and spec/code drift.

### Implementer: pick up and act

Pull work with `inbox(agent="impl", team="review-<slug>", wait_s=110)`. For each delegated task:

1. `acknowledge_task(by_agent="impl", task_id=<id>)` — within the first poll. Reviewers get `AGENT_HAS_ACTIVE_TASKS` surfaced if you don't.
2. Decide: **fix** or **argue**. There is no third option.
   - **Fix:** make the actual change (edit code / spec). Then:
     - `record_task_event(by_agent="impl", task_id=<id>, event_type="progress", message="fixed in <file>:<line>, commit <sha-or-pending>", phase="done")`
     - `send(from="impl", to="rev<N>", message="task <id>: fixed in <file>:<line>. Please verify.")` — directed, not team-broadcast.
   - **Argue:** write a specific counter-argument, not a dismissal. Then:
     - `record_task_event(by_agent="impl", task_id=<id>, event_type="progress", message="disagree: <rationale with evidence>")`
     - `send(from="impl", to="rev<N>", message="task <id>: disagree because <rationale>. Open to clarification or additional evidence.")`
   - Leave the task state as `in_progress`; the reviewer owns the close.
3. Never close a concern task yourself. Impl marks progress; reviewer decides whether it's resolved.

### Reviewer: verify a response

For each task where impl has posted an event, re-check the artifact.

- If the fix actually resolves the concern (verify by reading the diff / rerunning the check — don't trust the event message alone):
  - `update_task(agent="rev<N>", task_id=<id>, state="done")`.
- If the fix is incomplete or the argument fails on the merits:
  - `record_task_event(by_agent="rev<N>", task_id=<id>, event_type="progress", message="still open because <specific reason>")`.
  - Keep state as `in_progress` or push back to `open`.

Debate goes in the task's event timeline and in `send`/`reply` pairs — that keeps the Kanban honest (open tasks = open concerns).

## Phase 4 — Approval

Only when every one of your own concern tasks is `done`:

- To approve: `submit_review(reviewer="rev<N>", task_id=<META>, approved=true, comment="LGTM — concerns N, N, N addressed")`.
- To veto: `submit_review(reviewer="rev<N>", task_id=<META>, approved=false, comment="<the one unfixed thing>")`. A veto blocks completion; raise it only if an open task genuinely cannot be closed.

Reviewers may approve independently. Implementers do not approve.

## Phase 5 — Done condition

The implementer polls — reviewers do not need to do anything after their approval.

Implementer, after every inbox round:

1. `get_task(task_id=<META>)` and inspect `approvals` and `required_reviewers`.
2. `list_tasks(team="review-<slug>", state="open")` and `list_tasks(team="review-<slug>", state="in_progress")`.

Session is **complete** when:
- Every registered `rev<N>` has submitted `approved=true` on `META`, AND
- No tasks on the team remain in `open` or `in_progress` state (excluding `META` itself).

When complete:
- `update_task(agent="impl", task_id=<META>, state="done")`
- `send_team(team="review-<slug>", from="impl", message="review session <slug> closed. Thanks rev1..revN.")`
- `remove_agent(name="impl")` to leave the team cleanly. (Reviewers may do the same after seeing the close message, or just stop checking.)

If complete but a reviewer vetoed: do NOT close. Fix the vetoed concern (open a fresh task if needed) and go back to Phase 3.

## Rules

- One name per agent per session. Do not re-register under a different name mid-stream.
- Every concern is a task. Chatter without a task has no state and does not block completion.
- Impl never closes a concern. Reviewer never approves `META` while any of their concerns is still open.
- "Fixed" means verifiable: cite file:line, commit sha, or the exact diff hunk — not just "done".
- "Disagree" means specific: cite the counter-evidence or the spec clause — not "I don't think so".
- If the review is deadlocked for more than two round-trips on the same concern, surface the impasse to the human via `remember(kind="handoff", content="deadlock on task <id>: <impl position> vs <rev position>", pinned=true)` and `record_decision`. Do not loop forever.
- Do not enter a blind polling loop. Each inbox call uses a bounded `wait_s`; between calls, consult `inbox_status`, `project_board(team="review-<slug>")`, or `activity(team="review-<slug>")` for a non-consuming view.
- When the user opens this skill as a one-shot (e.g. `/review impl add-foo` then exits the thread), register and run through Phases 1–3, then stop with a clear hand-off note. Do not expect to be alive for Phase 5 unless the user keeps the session running.

## Quick reference — the tools you actually call

| What | Tool |
|---|---|
| Register | `register(name=…, team=review-<slug>, capabilities=[…], replace=…)` |
| See who's here | `whois(team=review-<slug>)` / `directory()` |
| Wait for someone | `wait_for_agents(names=[…], wait_s=…)` |
| Announce / chatter | `send_team(team=…, from=…, message=…)` |
| Direct message | `send(from=…, to=…, message=…)` |
| Blocking question | `ask(from=…, to=…, question=…)` |
| Raise concern | `delegate(from=rev<N>, to_agent=impl, title=…, description=…, mode="fix", expected_output=…, edit_scope=[…])` |
| Accept assignment | `acknowledge_task(by_agent=impl, task_id=…)` |
| Progress / argument | `record_task_event(by_agent=…, task_id=…, event_type="progress", message=…)` |
| Reviewer closes concern | `update_task(agent=rev<N>, task_id=…, state="done")` |
| Approve/veto session | `submit_review(reviewer=rev<N>, task_id=<META>, approved=…, comment=…)` |
| Inspect state | `get_task(task_id=…)`, `list_tasks(team=…, state=…)`, `project_board(team=…)` |
| Pull inbox | `inbox(agent=…, team=review-<slug>, wait_s=…)` |
| Leave | `remove_agent(name=…)` |

</what-to-do>
