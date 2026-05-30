# Changelog-2026-05-30

## Long-Term Memory Integration (Phase 1)

### Added

- **`src/memory/mem0_service.py`** — New module wrapping Mem0 initialization and background write logic.
  - Configures Mem0 with a local ChromaDB vector store (`.mem0_store`) for persistent fact storage.
  - Uses `gpt-4o-mini` as the extraction LLM via the OpenAI provider.
  - Exposes `add_session_memories(user_id, summary_text)` for async background invocation.
  - Instantiates a global `mem0_service` singleton for use across the application.

### Changed

- **`web_app/agent.py` — `MemoryMonitorPlugin.after_run_callback`**
  - Extended the callback to detect newly generated Compaction Summaries from `session.events`.
  - On detection, fires a `asyncio.create_task(asyncio.to_thread(...))` call to push the summary into Mem0 in the background.
  - Main thread returns immediately with zero added latency to the user-facing response.
  - Resolves `user_id` from `invocation_context.session.state` (falls back to `"default_user"`).

### Notes

- Mem0 uses its **default extraction prompt** in this phase — no custom prompt engineering applied. Baseline extraction quality will be reviewed before further tuning.
- The 7 default extraction dimensions covered by Mem0 out of the box: Personal Preferences, Personal Details, Plans & Intentions, Activity & Service Preferences, Health & Wellness, Professional Details, and Miscellaneous.

---

### TODO (Phase 2)

- **`web_app/agent.py` — `FullRAGSystemAgent._run_async_impl`**
  - Add retrieval call: `mem0_service.search_memories(user_id)` when `plan.need_memory` is true.
  - Inject retrieved facts into `ctx.session.state["user_memory"]` for downstream agents.

- **`sub_agents/planner_agent.py`**
  - Extend `Plan` schema with a `need_memory: bool` field so the planner can signal when long-term context is relevant to the query.

- **`sub_agents/answer_agent.py`**
  - Inject `{user_memory}` into the `AnswerSynthesisAgent` instruction prompt so responses can be personalized based on retrieved long-term preferences.
