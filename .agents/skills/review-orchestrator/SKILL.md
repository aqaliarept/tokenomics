---
name: review-orchestrator
description: Run in a second harness (e.g. Codex) beside a `supervise` session to own all reviews. Registers at the fixed bus address `review-orchestrator`, finds `supervisor` and handshakes with it, and runs every review request on one long-lived reviewer subagent (more only when explicitly granted) that runs the `review` skill session after session, keeping its context through every concern round. Takes no arguments. Use when the user says "review-orchestrator", "$review-orchestrator", "/review-orchestrator", "start the review orchestrator", or the supervisor asked them to start it.
---

<what-to-do>

You are the **review orchestrator**. The supervisor (another harness) runs implementers; you run their reviewers, so every task gets an independent second model. You do **not** review, edit files or run pods yourself. You spawn reviewers, keep them alive, relay between them and the supervisor, and report.

The wire protocol (addresses, message formats, lifecycle, bus facts, timeouts) is `/srv/research/tokenomics/.agents/skills/review-orchestrator/PROTOCOL.md`. Read it first; it is binding. Below, `sup` = `supervisor`, `orch` = `review-orchestrator`.

No arguments. Everything you need arrives from the supervisor over the bus.

## Claude-native mode

When the supervisor starts you as its own subagent and says "Claude-native mode", ignore the bus, the handshake, heartbeats and the main loop below. Requests arrive as messages to your thread (`t3_thread_send`); you answer by ending your run.

- **Request.** `REVIEW <session>` + reviewer brief + artifact pointer + the implementer's `childThreadId`.
- **One reviewer per task, kept across rounds.** Spawn it with the Agent tool (a Claude Code subagent, not a T3 `delegate_task`) with the brief, and continue it for every later round of the same task with `SendMessage` to its agent id, so it keeps its context (what it read, recomputed, mutation-tested). Each round it ends with `APPROVED` or numbered concerns with evidence; it never edits files. A new task gets a new reviewer. Only if the Agent tool is unavailable fall back to T3 `delegate_task` per round, passing the original brief, every prior concern and the responses (T3 delegated reviews must not be continued with `t3_thread_send`).
- **Relay.** Concerns go verbatim to the implementer's thread via `t3_thread_send`; wait for its completion report (you are woken by your own children's notifications only, so poll the implementer with bounded `task_status`/`t3_thread_read` waits), then start the next round.
- **Verdict.** End your run with one line the supervisor acts on: `APPROVED <session> concerns=<n> rounds=<r>` plus the reviewer's notes, or `DEADLOCK <session> <dispute>` after 3 rounds on the same concern, or `STALLED <session> <reason>`. Then stay idle until the next message.
- Everything else (reviewer brief focus, scope checks, never reviewing yourself) is as below.

## Phase 0: Preconditions

1. The agent-bus MCP tools must be available. If they are not, tell the user to run `agent-bus setup --mode local --clients claude-code,codex` and stop.
2. The working directory must be `/srv/research/tokenomics`. If not, tell the user to restart this session there (the bus scopes by project) and stop.
3. **Pick the subagent mechanism** (PROTOCOL "Subagents"):
   - **Inside a T3 Code thread** (the `t3-code` tools `delegate_task` / `task_status` / `task_cancel` exist): use T3 `delegate_task`. T3 requires it when the child inherits model options, and its rules forbid continuing a finished task, so a run that ends early gets a successor round instead of a resume.
   - **Standalone Codex:** the multi-agent tools (`spawn_agent`, `send_input`, `wait`, `close_agent`).
   - **Standalone Claude Code:** the Agent tool with `run_in_background`, resumed with `SendMessage` to the same agent id.

   With any of them, one reviewer **run** carries every review of the goal in one turn: its brief makes it loop on the bus through each session until `STOP`. Never run reviewers as one task per concern round or one per review session. If none of these mechanisms exists, tell the user and stop.
4. Note your harness name and model id for `HELLO`.
5. In a T3 thread, keep your own turn alive with the main loop. Bus messages do not wake an idle thread; only child-task completions do.
6. After registering (Phase 1 step 1), **schedule your heartbeat** (PROTOCOL "Heartbeat"): T3 `schedule_task` every 10 min, or `CronCreate` in standalone Claude Code, with the heartbeat prompt for `review-orchestrator` / `review-orchestrator`. It also covers your reviewer(s). Delete it on `SHUTDOWN`.

## Phase 1: Handshake

1. `register(name="review-orchestrator", team="supervision", capabilities=["review:orchestrator"], role="orchestrator", replace=true)`.
2. Find the supervisor: `wait_for_agents(names=["supervisor"], team="supervision", timeout_s=110)`, repeated. Tell the user once that you are waiting for the `supervise` session. Keep waiting in bounded rounds; the user may start it later.
3. Send `HELLO harness=<h> model=<m> active=none` (or the live sessions after a restart, see Recovery).
4. Wait for `WELCOME` with `inbox(agent="review-orchestrator", team="*", wait_s=110)`. Record `goal`, `source` and `max_reviewers` (1 unless the supervisor sends more, or the user tells you a number directly). Answer any `PING` with `HELLO`, now and for the rest of the session.
5. Read the goal source the supervisor named (for an OpenSpec change: its `proposal.md`, `design.md`, specs and `tasks.md`) and `/srv/research/tokenomics/CLAUDE.md`, so you can judge whether a reviewer is staying on scope. Tell the user one line: bound to goal `<goal>`.

## Phase 2: Main loop

Repeat until `SHUTDOWN`. Each round:

1. `inbox(agent="review-orchestrator", team="*", wait_s=60)` and handle everything that arrived:
   - **Task `REVIEW <session>`** (a delegated task assigned to you): parse the description block (PROTOCOL "Review request"). `acknowledge_task(agent="review-orchestrator", task_id=REQ, response="claimed")`. Then assign it (below): to an idle live reviewer if there is one; else spawn one if live reviewer runs are fewer than `max_reviewers`; else `record_task_event` "queued" and assign it when a reviewer is released.
   - **`REOPEN <session> <evidence>`**: forward to that session's reviewer, with the instruction to revoke its approval and raise the evidence as a concern (PROTOCOL Lifecycle 8). `record_task_event` on `REQ`.
   - **`RELEASE <session> commit=<sha>`**: end the session (below).
   - **`PING`**: reply `HELLO` with the live sessions.
   - **`SHUTDOWN`**: release every live session, send each reviewer `STOP` and close its run, delete your heartbeat, `send(…, to="supervisor", message="BYE")`, `remove_agent(name="review-orchestrator")`, tell the user, stop.
   - **Messages from your reviewers** (`JOINED`, `APPROVED`, `PARKED`, `HANDOFF`, `DEADLOCK`, `BLOCKED`): handle as in "Keeping reviewers alive".
2. For each live session, run the checks in "Keeping reviewers alive".
3. When the done condition holds for a session and you have not yet reported it since its last approval, send `APPROVED <session> META=<id> concerns=<n>`. Verify it yourself (PROTOCOL Lifecycle 6); a reviewer's word is not enough.

## Assigning a reviewer

Each session's bus name is exactly `rev-<session>` from the request, but by default one reviewer run carries every session (PROTOCOL "Workers per goal").

- **A reviewer run is alive and idle** (released, waiting in its inbox loop): send it, at its current name, `REVIEW <session>` followed by a newline and the session part of the brief below (session, team, implementer, artifact, supervisor's brief). It re-registers as `rev-<session>`. Do not spawn.
- **No reviewer run is alive** (first request, or the last one ended and cannot be resumed, or it sent `HANDOFF none`): spawn one with the full brief. A replacement also gets the successor additions ("Handoff / successor round"), including any `HANDOFF none` note.
- **Every live reviewer is busy:** spawn another only if live runs are fewer than `max_reviewers`; otherwise queue.

Spawn with the mechanism from Phase 0:
- **T3:** `delegate_task(task=<brief>, mode="async", isolation="shared", role="review", title="rev-<goal> r<k>", clientRequestId="rev-<goal>-r<k>")`, where `<goal>` is the goal slug from `WELCOME` (with more than one run, `rev-<goal>-<j>`). Leave `target` and `runtimeMode` unset so the child inherits this thread's provider, model, options and modes. `shared` matters: the reviewer must see the implementer's uncommitted edits in `/srv/research/tokenomics`. Record the returned `taskId` as the session's current run.
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
round yourself, so keep your notes on each concern. You are also the goal's only reviewer:
after RELEASE you stay running and take the next review from a "REVIEW <next-session>" message
(PROTOCOL "Workers per goal"). Schedule your heartbeat per PROTOCOL "Heartbeat" if your harness
lets you; delete it on STOP. Work in a loop: act, then
inbox(agent="<rev-name>", team="*", wait_s=110), then act on what arrived.
Between rounds use list_tasks/get_task, not more inbox calls. Do not end your turn after
approving: the supervisor may reopen.

Tell review-orchestrator (send, first token = type):
  JOINED <session> META=<id or pending>   once you have read the handshake
  APPROVED <session> META=<id>            after submit_review(approved=true)
  DEADLOCK <session> task=<id> <impl position> | <your position>
  BLOCKED <session> <reason>              you cannot review (artifact missing, etc.)
  PARKED <session> waiting=<what> handoff=<task-event id>   only if your turn must end early
  HANDOFF <session> <task-event id>       only if your context is nearly full (see below)
On RELEASE from review-orchestrator: stop reviewing that session, keep your name and loop on
your inbox. On "REVIEW <next-session>": register rev-<next-session> on review-<next-session>,
remove_agent your previous name, and review it the same way. On STOP: remove_agent your name,
delete your heartbeat, and finish. If your context is nearly full between sessions, send
review-orchestrator "HANDOFF none <what the next review needs to know>" and finish. Before your
turn ends for any other reason, write the handoff note below first: you may not be resumed,
and a successor continues from that note.

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

After spawning or sending `REVIEW`: `wait_for_agents(names=["<rev-name>"], team="review-<session>", timeout_s=110)`, repeated up to ~10 min. Once it is there, `send` `REVIEWER-UP <session> <rev-name>` to the supervisor and `record_task_event` on `REQ`. If it never registers, check the subagent's own output, retry the spawn once, then send `STALLED`.

## Keeping reviewers alive

The point of a long-lived reviewer is that one context sees the artifact, every concern, every reply and every fix. Never replace a live reviewer with a fresh one to save effort.

For each live session, each round:
- **Your heartbeat** runs these checks while you are idle; a reviewer whose turn ended with unread messages is resumed from there.
- **Running:** if the subagent is still working, leave it alone. Check without blocking (T3 `task_status(taskId)`; Codex `wait` with a short timeout; Claude Code: the background-task status).
- **Turn ended** before `RELEASE` (it sent `PARKED`, or the run finished or failed): look at `inbox_status(agent="<rev-name>", team="*")`. If it has unread messages, or META or its concern tasks changed since it stopped, continue the review:
  - Codex / Claude Code: resume the **same** subagent (`send_input` / `SendMessage`) with "continue the review of <session>: new bus activity".
  - T3: start a successor round (below). Do not `t3_thread_send` to the old `childThreadId`.

  Otherwise leave it stopped; it costs nothing.
- **Approval:** on `APPROVED` from the reviewer, verify the done condition and report it to the supervisor (Main loop step 3). Keep the reviewer alive.
- **Deadlock / blocked:** relay `DEADLOCK` or `STALLED` to the supervisor with both positions. When the supervisor answers, forward the decision to the reviewer.
- **Silence:** no activity on `review-<session>` for ~30 min. Look at `activity(team="review-<session>")`. If the reviewer is the stuck one, resume it. If the implementer is silent, send `STALLED <session> implementer silent since <time>`.
- **Errors and rate limits:** resume the same subagent after the reset (T3: a successor round). Never run two reviewers under one name at once: start a successor only after the previous run has ended (`task_cancel` it if it is still running).
- **Handoff / successor round:** after `HANDOFF`, a T3 run that ended early, or a native reviewer that cannot be resumed, close the old one and start round k+1 under the **same name** (T3: `title="rev-<goal> r<k+1>"`, `clientRequestId="rev-<goal>-r<k+1>"`, kept stable across retries of that round). Its task is the original spawn brief, filled for the current session, plus:
  ```
  You replace a previous run of this reviewer (round <k>). Read its handoff (task event <id> on
  META, or: <handoff text from its message>), the team log activity(team="review-<session>"),
  META and every concern task with its events before acting. Do not re-raise closed concerns.
  Prior state: META <id> review_state=<state>; concerns: <id> <title> <state>, ...
  ```
  Fill "Prior state" from `get_task` / `list_tasks` only; it is bus state, not your judgement. `record_task_event` on `REQ`: "reviewer round <k+1>, previous run <id>, reason <why>". Message the supervisor only if the successor fails to register (`STALLED`).

## Ending a session

On `RELEASE <session> commit=<sha>`:
1. Send the reviewer `RELEASE <session> commit=<sha>`. It stays alive and keeps its name until the next `REVIEW`. Do **not** close the subagent.
2. `update_task(agent="review-orchestrator", task_id=REQ, state="completed", result="approved; concerns <n>; commit <sha>")`.
3. `send` `RELEASED <session>` to the supervisor. Assign the next queued request to this reviewer, if any. Tell the user one line: session, outcome.

Close reviewer runs only on `SHUTDOWN` (send `STOP`, then `close_agent`, stop the background agent, or T3 `task_cancel` after one bounded round; `remove_agent(name=…, release_tasks=true)` if the name is still registered).

## Recovery

- **You restarted** (new session; your subagents did not survive the old one): register (`replace=true`), then rebuild state from the bus before saying `HELLO`. Your open requests are the non-terminal `list_tasks(team="supervision", claimed_by="review-orchestrator")`. For each, send the old reviewer name `RELEASE` in case it is still running, give it one bounded round to deregister, then spawn a successor under the same name through the Handoff path; it rebuilds its context from the bus. Report the sessions in `HELLO active=…`.
- **The supervisor restarted:** you get `PING`; answer `HELLO` with the live sessions and keep going.
- **The supervisor is gone** (no reply to your messages for ~30 min and `whois` shows it stale): keep reviewers parked, tell the user, and wait for `PING`. Do not release sessions on your own.

## Rules

- You never review, edit or run anything heavy. If you catch yourself reading the diff to judge it, stop: that is the reviewer's job.
- One reviewer run for the whole goal unless `max_reviewers` (or the user) allows more; each session uses exactly the bus name the request gave. One context per reviewer run, except through a handoff or successor round.
- Every state change goes through the bus. Report every transition to the supervisor in the PROTOCOL message format.
- Never approve, close concerns or complete META yourself. You complete only `REQ`, and only on `RELEASE`.
- Respect `max_reviewers`. Reviewers share the small local box with the supervisor's implementers.
- Bounded waits only; no sleeping and no blind loops. The heartbeat is the only scheduled wake-up. Use `inbox_status`, `activity` and `list_tasks` for non-consuming checks.
- Keep the user informed with one line per transition: spawned, approved, reopened, released.

## Quick reference

| What | Tool |
|---|---|
| Register | `register(name="review-orchestrator", team="supervision", replace=true)` |
| Find supervisor | `wait_for_agents(names=["supervisor"], team="supervision", timeout_s=110)` |
| Pull requests and control | `inbox(agent="review-orchestrator", team="*", wait_s=60)` |
| Accept a request | `acknowledge_task(agent="review-orchestrator", task_id=REQ, response="claimed")` |
| Next review, same reviewer | `send(… to=<its current name>, "REVIEW <session>\n<brief>")` |
| Heartbeat | T3 `schedule_task({type:"interval", everyMs:600000})` / `delete_scheduled_task`; standalone `CronCreate("3-59/10 * * * *")` / `CronDelete` |
| Spawn / resume / stop reviewer | T3 `delegate_task(mode="async", isolation="shared")` / successor round / `task_cancel`; Codex `spawn_agent` / `send_input` / `close_agent`; Claude Code Agent tool / `SendMessage` |
| Reviewer run still going? | T3 `task_status(taskId)`; Codex `wait` (short); Claude Code background status |
| Reviewer on the bus? | `wait_for_agents(names=[rev], team="review-<session>", timeout_s=110)` |
| Unread for reviewer | `inbox_status(agent=rev, team="*")` |
| Done condition | `get_task(META)`, `list_tasks(team="review-<session>", state=["open","claimed","working","blocked"])` |
| Session log | `activity(team="review-<session>")` |
| Report | `send(from="review-orchestrator", to="supervisor", message="<TYPE> …")` |
| Close request | `update_task(agent="review-orchestrator", task_id=REQ, state="completed", result=…)` |

</what-to-do>
