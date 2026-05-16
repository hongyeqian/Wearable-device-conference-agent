# Changelog — 2026-05-16

## Overview

This release introduces **per-user meeting data isolation**, **Google ADK-native userId routing**, and **recursive file-system watching** for automatic hot-reload of new users and meetings.

---

## 1. User Meeting Data Privatization

### What changed

- `datademo/` is now organized by **person name**: `datademo/<PersonName>/con*/`
- Each user's meetings are stored exclusively under their own named directory.

### How isolation works

The core implementation lives in `sub_agents/metadata_manager.py`:

```python
class MeetingsMetadataManager:
    def __init__(self, user_name: str):
        # Only scans DATA_DIR/<user_name>/con*/summary_metadata.json
        user_dir = Path(DATA_DIR) / self.user_name
        ...
```

- On initialization, `MeetingsMetadataManager(user_name)` only loads `summary_metadata.json` files from that specific user's directory.
- A global `_manager_cache: Dict[str, MeetingsMetadataManager]` ensures each user's metadata is loaded once and cached for the lifetime of the process.
- All downstream filtering (by person, date range, year/month) operates exclusively on the loaded user's records — no cross-user data leakage is possible.

---

## 2. Google ADK userId Routing

### What changed

`web_app/agent.py` now accepts the standard ADK `userId` field from API requests and maps it to the corresponding user data directory.

### How it works

1. The ADK framework passes `ctx.user_id` (from the API request's `userId` field) into `FullRAGSystemAgent._run_async_impl()`.
2. `_match_user_folder(user_id)` performs **case-insensitive matching** against folder names in `datademo/`:
   - If `userId` is empty or `"user"` (ADK web UI default), falls back to `ACTIVE_USER`.
   - Otherwise, scans `datademo/` for a directory whose name matches (case-insensitive).
3. The resolved user name is stored in `ctx.session.state["CURRENT_USER"]` for downstream components (QueryRewriter, Retrieval, MetadataManager).

### API Usage Example

**Step 1: Create a session for a specific user**

```powershell
curl.exe -X POST "http://localhost:8000/apps/web_app/users/Hongye%20Qian/sessions" `
  -H "Content-Type: application/json" `
  -d '{"sessionId": "my_test_001"}'
```

Response:
```json
{
  "id": "my_test_001",
  "appName": "web_app",
  "userId": "Hongye Qian",
  "state": {},
  "events": [],
  "lastUpdateTime": 1778925392.0554678
}
```

**Step 2: Send a query as that user**

```powershell
curl.exe -X POST http://localhost:8000/run `
  -H "Content-Type: application/json" `
  -d '{"appName": "web_app", "userId": "Hongye Qian", "sessionId": "my_test_001", "newMessage": {"role": "user", "parts": [{"text": "Who are you and what is your first meeting?"}]}}'
```

The agent will only retrieve and respond based on meetings stored under `datademo/Hongye Qian/con*/`.

### Key points for teammates

- `userId` in the request **must match** a folder name in `datademo/` (case-insensitive).
- To add a new user, simply create a new directory under `datademo/` with their name — no code changes required.
- The server must be started via `python run_server.py` (or `adk web .`) from the project root.

---

## 3. Recursive Watchdog for New Users & Meetings

### What changed

`src/watcher/meeting_watcher.py` now monitors the entire `datademo/` directory tree recursively.

### Capabilities

| Event | Detection | Action |
|-------|-----------|--------|
| New `con*` folder (new meeting) | Watchdog `DirCreatedEvent` | Waits for 4 required files, then triggers incremental indexing |
| New person-name folder (new user) | Watchdog `DirCreatedEvent` at top level | Recognized as a new user directory |
| Folder deletion / rename | `DirDeletedEvent` / `DirMovedEvent` | Triggers cleanup or re-index callback |

### Required files per meeting (`con*` folder)

- `metaData*.json` — meeting metadata
- `data*.md` — transcript
- `meetLevel*.md` — meeting-level summary
- `summary*.md` — summary for embeddings

### How it works

- Observer runs with `recursive=True` on `datademo/`.
- A debounce timer (10s) prevents premature processing while files are still being copied.
- Up to 6 retries if files are incomplete after the initial debounce.
- Once all 4 files are present, the configured callback is invoked for incremental vector-store indexing — **no server restart needed**.

---

## Files Changed (excluding adapter_training)

| File | Change |
|------|--------|
| `web_app/agent.py` | Added `_match_user_folder()`, ADK userId routing, `MemoryMonitorPlugin` |
| `sub_agents/metadata_manager.py` | New — per-user metadata loading with cache |
| `src/watcher/meeting_watcher.py` | Recursive watching, person-name detection, deletion handling |
| `config/settings.py` | Added `DEFAULT_USER` config |
| `run_server.py` | Custom server launcher with NLP component warm-up |
| `datademo/` | Restructured to `<PersonName>/con*/` layout, added new meeting data |
| `vector_store/` | Rebuilt indices for new data structure |
