# Changelog — 2026-05-19

## Summary

Fixed a production bug where user `wangl7101@gmail.com` consistently received **0 RAG results**. Root cause analysis traced the issue to three interconnected failures: premature file-watcher abandonment, a missing empty-result guard in the query rewriter, and an overly aggressive retry strategy.

---

## Bug Report (from production logs)

```
May 18 12:03:53  [MeetingWatcher] Detected new meeting directory:
                 wangl7101@gmail.com/con17791057696360000000003029389598
May 18 12:03:59  [MeetingWatcher] ⚠️ Giving up on ... after 6 retries

May 19 02:44:13  Query rewrite → "What occurred during wangl7101@gmail.com meetings on ?"
                 meeting_ids: []
                 Retrieved 0 chunks
```

**Failure chain:**

1. New meeting directory was detected, but files were still being uploaded.
2. The old watcher checked 6 times over ~6 seconds, then permanently gave up.
3. Meeting was never indexed → metadata stayed empty → RAG returned nothing.
4. When the user asked "how about my last meeting", the query rewriter matched
   `"last meeting"` via regex, called `get_last_n_meetings()` which returned `[]`,
   and produced the malformed query `"meetings on ?"` (empty date string).

---

## Files Changed

### 1. `src/watcher/meeting_watcher.py`

**Why:** The original watcher gave up after only ~6 seconds (6 retries × 1s interval). On a slow network or during large file transfers, this is far too aggressive. Additionally, the watcher only checked whether files *existed* — it never verified that files had finished being *written*.

#### Changes

**a) Replaced timing constants**

| Old | New | Purpose |
|-----|-----|---------|
| `_DEBOUNCE_SECONDS = 10` | `_MAX_WAIT_SECONDS = 60` | Total time budget per directory |
| `_MAX_RETRIES = 6` | `_CHECK_INTERVAL = 2` | Seconds between polls |
| — | `_STABLE_SECONDS = 5` | Files must be unchanged for 5s |

**b) Added `files_stable()` function**

```python
def files_stable(files: list[Path], stable_seconds: float = _STABLE_SECONDS) -> bool:
```

- Takes two snapshots of file `(size, mtime)` separated by `stable_seconds`.
- If both snapshots match, files are considered fully written.
- Wraps each `stat()` call in `try/except OSError` so a file deleted mid-check
  returns `False` instead of crashing.

**c) Rewrote `check_completeness()`**

The old implementation was a single-pass boolean check (files exist → True/False).
The new implementation is a **deadline-based polling loop**:

1. Polls every `_CHECK_INTERVAL` seconds until `_MAX_WAIT_SECONDS` is reached.
2. On each poll: checks that all 4 required file patterns exist.
3. Once all files are found, calls `files_stable()` to confirm they stopped changing.
4. Returns `True` immediately on success, or `False` when the deadline expires.

**d) Simplified `_checker_loop()`**

- **Removed** the redundant outer retry mechanism (`retry_counts`, `still_pending`,
  `_MAX_RETRIES`). Since `check_completeness()` now handles all retrying internally
  (up to 60 seconds), the loop only needs to drain the pending set and call
  `check_completeness()` once per directory.
- **Added** `_stop_event` check between directories so the thread can shut down
  promptly during application exit.
- **Removed** `debounce_seconds` parameter from `MeetingWatcher.__init__()` as it
  is no longer referenced.

---

### 2. `sub_agents/query_rewriter_agent.py`

**Why:** When `resolve_by_regex()` matched a pattern like `"last meeting"`, it called
`get_last_n_meetings()` and blindly used the result to build a rewritten query — even
when the result was an empty list. This produced the malformed output
`"meetings on "` (empty date string) instead of gracefully falling back.

#### Changes

**Added empty-result guard in `resolve_by_regex()`** (line 523):

```python
meetings = mdf.get_last_n_meetings(n=n, person_name=person_name)

if not meetings:
    logger.warning(
        f"[Stage3-Layer1-Regex] No meetings found for n={n}, person={person_name}"
    )
    return None
```

When `meetings` is empty, the function now returns `None` (same as "no regex match"),
which lets the downstream pipeline fall through to the embedding layer or return the
original query unchanged — instead of generating garbage.

---

## Additional Observations (not fixed in this patch)

| Issue | Detail | Suggested Fix |
|-------|--------|---------------|
| **409 Conflict on session** | Frontend POSTs to an existing session ID 12 hours later | Frontend should check session existence before creating |
| **"Unknown agent" warning** | `AnswerSynthesisAgent` not in ADK `sub_agents` list (this I will fix) | Register it alongside `planner_agent` in `super().__init__()` |
