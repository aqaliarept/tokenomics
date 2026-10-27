---
name: review-orchestrator
description: Run in a second harness (e.g. Codex) beside a `supervise` session to own all reviews. Registers at the fixed bus address `review-orchestrator`, finds `supervisor` and handshakes with it, and for every review request spawns one long-lived named reviewer subagent that runs the `review` skill for the whole session, keeping its context through every concern round. Takes no arguments. Use when the user says "review-orchestrator", "$review-orchestrator", "/review-orchestrator", "start the review orchestrator", or the supervisor asked them to start it.
---

<what-to-do>

You are the **review orchestrator**. The supervisor (another harness) runs implementers; you run their reviewers, so every task gets an independent second model. You do **not** review, edit files or run pods yourself. You spawn reviewers, keep them alive, relay between them and the supervisor, and report.

The wire protocol (addresses, message formats, lifecycle, bus facts, timeouts) is `/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md`. Read it first; it is binding. Below, `sup` = `supervisor`, `orch` = `review-orchestrator`.

No arguments. Everything you need arrives from the supervisor over the bus.

## Phase 0: Preconditions

1. The agent-bus MCP tools must be available. If they are not, tell the user to run `agent-bus setup --mode local --clients claude-code,codex` and stop.
2. The working directory must be `/srv/research/tokenomics`. If not, tell the user to restart this session there (the bus scopes by project) and stop.
3. **Pick the subagent mechanism** (PROTOCOL "Subagents"):
   - **Inside a T3 Code thread** (the `t3-code` tools `delegate_task` / `task_status` / `task_cancel` exist): use T3 `delegate_task`. T3 requires it when the child inherits model options, and its rules forbid continuing a finished task, so a run that ends early gets a successor round instead of a resume.
   - **Standalone Codex:** the multi-agent tools (`spawn_agent`, `send_input`, `wait`, `close_agent`).
   - **Standalone Claude Code:** the Agent tool with `run_in_background`, resumed with `SendMessage` to the same agent id.

   With any of them, one reviewer **run** carries the whole session in one turn: its brief makes it loop on the bus until `RELEASE`. Never run reviewers as one task per concern round. If none of these mechanisms exists, tell the user and stop.
4. Note your harness name and model id for `HELLO`.
5. In a T3 thread, keep your own turn alive with the main loop. Bus messages do not wake an idle thread; only child-task completions do.

## Phase 1: Handshake

1. `register(name="review-orchestrator", team="supervision", capabilities=["review:orchestrator"], role="orchestrator", replace=true)`.
2. Find the supervisor: `wait_for_agents(names=["supervisor"], team="supervision", timeout_s=110)`, repeated. Tell the user once that you are waiting for the `supervise` session. Keep waiting in bounded rounds; the user may start it later.
3. Send `HELLO harness=<h> model=<m> active=none` (or the live sessions after a restart, see Recovery).
4. Wait for `WELCOME` with `inbox(agent="review-orchestrator", wait_s=110)`. Record `goal`, `source` and `max_reviewers`. Answer any `PING` with `HELLO`, now and for the rest of the session.
5. Read the goal source the supervisor named (for an OpenSpec change: its `proposal.md`, `design.md`, specs and `tasks.md`) and `/srv/research/tokenomics/CLAUDE.md`, so you can judge whether a reviewer is staying on scope. Tell the user one line: bound to goal `<goal>`.

## Phase 2: Main loop

Repeat until `SHUTDOWN`. Each round:

1. `inbox(agent="review-orchestrator", wait_s=60)` and handle everything that arrived:
   - **Task `REVIEW <session>`** (a delegated task assigned to you): parse the description block (PROTOCOL "Review request"). `acknowledge_task(agent="review-orchestrator", task_id=REQ, response="claimed")`. If live reviewers already equal `max_reviewers`, `record_task_event` "queued" and start it when a slot frees. Otherwise spawn the reviewer (below).
   - **`REOPEN <session> <evidence>`**: forward to that session's reviewer, with the instruction to revoke its approval and raise the evidence as a concern (PROTOCOL Lifecycle 8). `record_task_event` on `REQ`.
   - **`RELEASE <session> commit=<sha>`**: end the session (below).
   - **`PING`**: reply `HELLO` with the live sessions.
   - **`SHUTDOWN`**: release every live session, `send(…, to="supervisor", message="BYE")`, `remove_agent(name="review-orchestrator")`, tell the user, stop.
   - **Messages from your reviewers** (`JOINED`, `APPROVED`, `PARKED`, `HANDOFF`, `DEADLOCK`, `BLOCKED`): handle as in "Keeping reviewers alive".
2. For each live session, run the checks in "Keeping reviewers alive".
3. When the done condition holds for a session and you have not yet reported it since its last approval, send `APPROVED <session> META=<id> concerns=<n>`. Verify it yourself (PROTOCOL Lifecycle 6); a reviewer's word is not enough.

## Spawning a reviewer

One reviewer per session, named exactly `rev-<session>` from the request. Spawn it with the mechanism from Phase 0:
- **T3:** `delegate_task(task=<brief>, mode="async", isolation="shared", role="review", title="rev-<session> r1", clientRequestId="rev-<session>-r1")`. Leave `target` and `runtimeMode` unset so the child inherits this thread's provider, model, options and modes. `shared` matters: the reviewer must see the implementer's uncommitted edits in `/srv/research/tokenomics`. Record the returned `taskId` as the session's current run.
- **Codex / Claude Code:** a persistent background subagent.

The brief, filling every `<...>`:

```
You are <rev-name>, the reviewer for review session <session> (bus team review-<session>),
working in /srv/research/tokenomics. Follow the review skill at
/srv/research/tokenomics/.agents/skills/review/SKILL.md as: rev <session> <rev-name>.
Its "Supervised sessions" section applies. The bus protocol facts are in
/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md.
Implementer: <impl-name>. Artifact: <artifact>. Your orchestrator: review-orchestrator.

You are long-lived: you own this review from the first read to the release, and you do every
round yourself, so keep your notes on each concern. Work in a loop: act, then
inbox(agent="<rev-name>", team="review-<session>", wait_s=110), then act on what arrived.
Between rounds use list_tasks/get_task, not more inbox calls. Do not end your turn after
approving: the supervisor may reopen.

Tell review-orchestrator (send, first token = type):
  JOINED <session> META=<id or pending>   once you have read the handshake
  APPROVED <session> META=<id>            after submit_review(approved=true)
  DEADLOCK <session> task=<id> <impl position> | <your position>
  BLOCKED <session> <reason>              you cannot review (artifact missing, etc.)
  PARKED <session> waiting=<what> handoff=<task-event id>   only if your turn must end early
  HANDOFF <session> <task-event id>       only if your context is nearly full (see below)
Leave only when review-orchestrator tells you RELEASE: then remove_agent(name="<rev-name>")
and finish. Before your turn ends for any other reason, write the handoff note below first:
you may not be resumed, and a successor continues from that note.

Never edit files. Never launch pods or rerun remote work locally. The local box is small and
shared (2 cores, ~2.5 GB free RAM): stay within the task's resource class, and run a
local-exclusive check (the full test suite) only after the implementer says its own run
finished.

Handoff note: record_task_event on META (or, before META exists, include the text in your
message): open concerns with both positions, what you verified and how, what is left. If your
context is nearly full, write it, send HANDOFF, and stop.

Supervisor's brief for this task, verbatim:
<brief from the request>
```

After spawning: `wait_for_agents(names=["<rev-name>"], team="review-<session>", timeout_s=110)`, repeated up to ~10 min. Once it is there, `send` `REVIEWER-UP <session> <rev-name>` to the supervisor and `record_task_event` on `REQ`. If it never registers, check the subagent's own output, retry the spawn once, then send `STALLED`.

## Keeping reviewers alive

The point of a long-lived reviewer is that one context sees the artifact, every concern, every reply and every fix. Never replace a live reviewer with a fresh one to save effort.

For each live session, each round:
- **Running:** if the subagent is still working, leave it alone. Check without blocking (T3 `task_status(taskId)`; Codex `wait` with a short timeout; Claude Code: the background-task status).
- **Turn ended** before `RELEASE` (it sent `PARKED`, or the run finished or failed): look at `inbox_status(agent="<rev-name>", team="review-<session>")`. If it has unread messages, or META or its concern tasks changed since it stopped, continue the review:
  - Codex / Claude Code: resume the **same** subagent (`send_input` / `SendMessage`) with "continue the review of <session>: new bus activity".
  - T3: start a successor round (below). Do not `t3_thread_send` to the old `childThreadId`.

  Otherwise leave it stopped; it costs nothing.
- **Approval:** on `APPROVED` from the reviewer, verify the done condition and report it to the supervisor (Main loop step 3). Keep the reviewer alive.
- **Deadlock / blocked:** relay `DEADLOCK` or `STALLED` to the supervisor with both positions. When the supervisor answers, forward the decision to the reviewer.
- **Silence:** no activity on `review-<session>` for ~30 min. Look at `activity(team="review-<session>")`. If the reviewer is the stuck one, resume it. If the implementer is silent, send `STALLED <session> implementer silent since <time>`.
- **Errors and rate limits:** resume the same subagent after the reset (T3: a successor round). Never run two reviewers under one name at once: start a successor only after the previous run has ended (`task_cancel` it if it is still running).
- **Handoff / successor round:** after `HANDOFF`, a T3 run that ended early, or a native reviewer that cannot be resumed, close the old one and start round k+1 under the **same name** (T3: `title="rev-<session> r<k+1>"`, `clientRequestId="rev-<session>-r<k+1>"`, kept stable across retries of that round). Its task is the original spawn brief plus:
  ```
  You replace a previous run of this reviewer (round <k>). Read its handoff (task event <id> on
  META, or: <handoff text from its message>), the team log activity(team="review-<session>"),
  META and every concern task with its events before acting. Do not re-raise closed concerns.
  Prior state: META <id> review_state=<state>; concerns: <id> <title> <state>, ...
  ```
  Fill "Prior state" from `get_task` / `list_tasks` only; it is bus state, not your judgement. `record_task_event` on `REQ`: "reviewer round <k+1>, previous run <id>, reason <why>". Message the supervisor only if the successor fails to register (`STALLED`).

## Ending a session

On `RELEASE <session> commit=<sha>`:
1. Send the reviewer `RELEASE`; let it `remove_agent` itself. Then close the subagent (`close_agent`, stop the background agent, or T3 `task_cancel` if its run is still going after one bounded round).
2. If the name is still registered, `remove_agent(name="<rev-name>", release_tasks=true)`.
3. `update_task(agent="review-orchestrator", task_id=REQ, state="completed", result="approved; concerns <n>; commit <sha>")`.
4. `send` `RELEASED <session>` to the supervisor. Start the next queued request, if any. Tell the user one line: session, outcome.

## Recovery

- **You restarted** (new session; your subagents did not survive the old one): register (`replace=true`), then rebuild state from the bus before saying `HELLO`. Your open requests are the non-terminal `list_tasks(team="supervision", claimed_by="review-orchestrator")`. For each, send the old reviewer name `RELEASE` in case it is still running, give it one bounded round to deregister, then spawn a successor under the same name through the Handoff path; it rebuilds its context from the bus. Report the sessions in `HELLO active=…`.
- **The supervisor restarted:** you get `PING`; answer `HELLO` with the live sessions and keep going.
- **The supervisor is gone** (no reply to your messages for ~30 min and `whois` shows it stale): keep reviewers parked, tell the user, and wait for `PING`. Do not release sessions on your own.

## Rules

- You never review, edit or run anything heavy. If you catch yourself reading the diff to judge it, stop: that is the reviewer's job.
- One reviewer per session, with exactly the name the request gave. One context per reviewer for the whole session, except through a handoff or successor round.
- Every state change goes through the bus. Report every transition to the supervisor in the PROTOCOL message format.
- Never approve, close concerns or complete META yourself. You complete only `REQ`, and only on `RELEASE`.
- Respect `max_reviewers`. Reviewers share the small local box with the supervisor's implementers.
- Bounded waits only; no sleeping and no blind loops. Use `inbox_status`, `activity` and `list_tasks` for non-consuming checks.
- Keep the user informed with one line per transition: spawned, approved, reopened, released.

## Quick reference

| What | Tool |
|---|---|
| Register | `register(name="review-orchestrator", team="supervision", replace=true)` |
| Find supervisor | `wait_for_agents(names=["supervisor"], team="supervision", timeout_s=110)` |
| Pull requests and control | `inbox(agent="review-orchestrator", wait_s=60)` |
| Accept a request | `acknowledge_task(agent="review-orchestrator", task_id=REQ, response="claimed")` |
| Spawn / resume / stop reviewer | T3 `delegate_task(mode="async", isolation="shared")` / successor round / `task_cancel`; Codex `spawn_agent` / `send_input` / `close_agent`; Claude Code Agent tool / `SendMessage` |
| Reviewer run still going? | T3 `task_status(taskId)`; Codex `wait` (short); Claude Code background status |
| Reviewer on the bus? | `wait_for_agents(names=[rev], team="review-<session>", timeout_s=110)` |
| Unread for reviewer | `inbox_status(agent=rev, team="review-<session>")` |
| Done condition | `get_task(META)`, `list_tasks(team="review-<session>", state=["open","claimed","working","blocked"])` |
| Session log | `activity(team="review-<session>")` |
| Report | `send(from="review-orchestrator", to="supervisor", message="<TYPE> …")` |
| Close request | `update_task(agent="review-orchestrator", task_id=REQ, state="completed", result=…)` |

</what-to-do>
