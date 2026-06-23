# Meeting Intelligence RAG System

This repository contains a meeting-centric Retrieval-Augmented Generation (RAG) system built on Google ADK. It loads meeting records from a local data directory, builds a searchable hybrid index, rewrites ambiguous user queries, retrieves relevant meeting context, and answers through an ADK web app.

The current production path is centered on:

- `run_server.py`
- `web_app/agent.py`
- `web_app/tools.py`

This README documents the system as it exists now.

## What the system does

- Answers questions about past meetings
- Rewrites ambiguous meeting queries into retrievable forms
- Enforces per-user meeting visibility through metadata filtering
- Supports hot reload when new meeting folders are added
- Stores long-term user memory for personalization
- Exposes the system through a Google ADK web server

## Current architecture

### Runtime entrypoint

The recommended server entrypoint is:

- `run_server.py`

It does three things before serving traffic:

1. Imports `web_app.agent`
2. Forces RAG initialization and NLP warmup
3. Starts the file-system watcher for incremental indexing

### Main runtime modules

- `web_app/agent.py`  
  Owns the root ADK agent, RAG initialization, vector-store loading, compaction plugin setup, and hot-reload watcher startup.

- `web_app/tools.py`  
  Defines the actual user-facing runtime behavior:
  - QA sub-agent
  - intent router
  - email draft/send tools
  - meeting draft/schedule tools
  - long-term memory lookup

- `src/data_loader/loader.py`  
  Loads meeting folders from `datademo/<user>/con*` and generates `summary_metadata.json`.

- `src/chunking/chunker.py`  
  Converts loaded meetings into chunk objects for indexing.

- `src/retrieval/vector_store.py`  
  Provides hybrid retrieval using FAISS plus BM25.

- `src/retrieval/vector_store_utils.py`  
  Persists and reloads the local vector store.

- `sub_agents/query_rewriter_agent.py`  
  Rewrites ambiguous queries by resolving people, dates, and meeting references.

- `sub_agents/metadata_manager.py`  
  Maintains per-user meeting metadata and enforces visibility filters.

- `src/watcher/meeting_watcher.py`  
  Watches the data directory and triggers incremental indexing for new meetings.

- `src/memory/mem0_service.py`  
  Stores and retrieves long-term user memory.

## High-level architecture

```text
User
  ↓
ADK Web UI
  ↓
run_server.py
  ↓
web_app/agent.py
  ├─ initialize DataLoader + Chunker + Vector Store + Retriever
  ├─ warm up query rewriting and NLP components
  ├─ start MeetingWatcher
  └─ delegate each request to intent_router in web_app/tools.py
       ├─ QA path
       │   ├─ rewrite_query_async(...)
       │   ├─ metadata-based meeting filtering
       │   ├─ hybrid retrieval from vector store
       │   └─ answer synthesis with citations
       ├─ memory lookup path
       ├─ email draft/send path
       └─ meeting draft/schedule path
```

## Data model and on-disk layout

Meeting data is expected under:

```text
datademo/
  <user_name>/
    con1/
    con2/
    ...
```

Each `con*` folder must contain the four source files the watcher expects:

- `metaData*.json`
- `data*.md`
- `meetLevel*.md`
- `summary*.md`

During loading, the system also generates:

- `summary_metadata.json`

That generated file is used later for user-scoped metadata filtering and query rewriting.

## Data flow

### 1. Startup indexing flow

On startup, `web_app/agent.py` initializes the RAG stack:

1. `DataLoader` loads all meetings from `datademo`
2. `HierarchicalChunker` builds chunks
3. Existing vector store is loaded from `vector_store/` if present
4. Only new meetings are chunked and indexed
5. `HierarchicalRetriever` is created on top of the vector store
6. Query rewriting dependencies are warmed up
7. `MeetingWatcher` starts in the background

### 2. Request flow

For a typical meeting question:

1. The root agent receives the user message
2. The current user is resolved from ADK session context
3. The request is delegated to the intent router in `web_app/tools.py`
4. The QA agent callback runs `rewrite_query_async(...)`
5. Metadata filtering computes which meetings the user may access
6. Hybrid retrieval searches the indexed summaries
7. Retrieved chunks are formatted and injected into the QA prompt
8. The QA agent answers with inline chunk citations

### 3. Hot-reload flow

When a new `con*` folder appears:

1. `MeetingWatcher` detects the new directory
2. It waits until all required files exist and become stable
3. `web_app/agent.py` performs incremental sync
4. New chunks are added to the existing vector store
5. User metadata caches are refreshed
6. New meetings become queryable without rebuilding everything manually

### 4. Long-term memory flow

After compaction summaries are produced:

1. Session summaries are passed to `mem0_service`
2. High-value user facts are extracted
3. Those facts are stored in the local Mem0-backed store
4. Future requests can query those memories through a tool

## Important implementation notes

- The production QA path currently retrieves from the summary level by default.
- `config/settings.py` sets `ONLY_SUMMARY=True`, so the active retrieval path is optimized around summary chunks.
- `vector_store/` is a local persisted artifact, not source code.
- `long_term_memory/`, `logs/`, `eval/logs/`, `__pycache__/`, and `.adk/` contain runtime artifacts.

## Repository structure

This is the part of the tree that matters most for the active system:

```text
.
├─ run_server.py
├─ requirements.txt
├─ config/
│  ├─ settings.py
│  └─ meeting_patterns.py
├─ datademo/
├─ web_app/
│  ├─ agent.py
│  └─ tools.py
├─ sub_agents/
│  ├─ query_rewriter_agent.py
│  ├─ metadata_manager.py
│  ├─ person_matcher.py
│  ├─ date_resolver.py
│  └─ constants/
├─ src/
│  ├─ data_loader/
│  ├─ chunking/
│  ├─ embeddings/
│  ├─ retrieval/
│  ├─ watcher/
│  └─ memory/
└─ eval/
```

## Quick start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install the spaCy model used at runtime

```powershell
python -m spacy download en_core_web_sm
```

### 4. Configure environment variables

Create `config/.env` with at least:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Optional settings:

```env
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=
ELASTICSEARCH_PASSWORD=
CURRENT_USER=
```

Notes:

- The current default user fallback is defined in `config/settings.py`.
- Elasticsearch is optional; the active local path uses FAISS.

### 5. Prepare meeting data

Place data under:

```text
datademo/<user_name>/con*/
```

Each meeting folder should contain:

- `metaData*.json`
- `data*.md`
- `meetLevel*.md`
- `summary*.md`

### 6. Start the server

Recommended:

```powershell
python run_server.py
```

Optional custom host/port:

```powershell
python run_server.py --host 127.0.0.1 --port 8000
```

Then open the ADK web UI in your browser and interact with the app.

## Configuration

Key configuration is defined in `config/settings.py`.

Important values include:

- `DATA_DIR`
- `VECTOR_STORE_DIR`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `ONLY_SUMMARY`
- `DEFAULT_USER`

## Evaluation

The repository also includes an evaluation pipeline under `eval/`.

This is not required to run the app, but it is useful for:

- QA dataset generation
- model comparison
- judgment and reporting

The eval path uses `web_app/tools.py:create_eval_app(...)` to build production-like apps for model testing.

## Known scope of the current system

- The active production path is meeting QA plus a small tool layer.
- Some directories contain experiments or runtime artifacts and are not required for the core server path.
- The repository still includes non-core material such as evaluation assets, audio experiments, and generated runtime files.

## Recommended startup command

Use:

```powershell
python run_server.py
```

instead of relying on `adk web .` directly, because `run_server.py` performs warmup and watcher startup before serving requests.
