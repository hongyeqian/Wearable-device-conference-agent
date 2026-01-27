# Meeting Transcript RAG System (English)

This repository is a modular Retrieval-Augmented Generation (RAG) template focused on meeting transcripts. It provides hierarchical chunking, hybrid retrieval (vector + BM25), and modular agent components for query rewriting and answer generation. Below is a concise guide focused on the `datademo`, `src`, `sub_agents`, and `web_app` folders.

## Key directories (focused)

- `datademo/`
  - Sample meeting data. Each meeting lives in a `conXXX/` subfolder and typically contains:
    - `metaDataXXX.json` — metadata (participants, datetime, topics, meeting_id)
    - `dataXXX.md` — raw transcript (timestamped lines or paragraphs)
    - `meetLevelXXX.md` — topic-level structured content
    - `summaryXXX.md` — high-level meeting summary
  - To use your own data, mirror this structure under `datademo/`.

- `src/` — core implementation
  - `data_loader/loader.py` — loads and normalizes files from `datademo/`
  - `chunking/chunker.py` — hierarchical chunking (metadata / summary / meeting)
  - `embeddings/generator.py` — generates embeddings (OpenAI or other providers)
  - `retrieval/` — retrieval implementations and utilities
    - `hierarchical_retriever.py` — main hierarchical retrieval logic
    - `vector_store.py`, `vector_store_utils.py` — vector store and helpers
    - `vector_store_es.py` — optional Elasticsearch-backed store

- `sub_agents/` — modular sub-agents
  - `query_rewriter_agent.py` — rewrites user queries to produce paraphrases/candidates
  - `answer_agent.py` — merges retrieved contexts and generates final answers with citations
  - `planner_agent.py` — optional task planner that splits complex tasks into subtasks
  - These agents are decoupled for reuse in different orchestrations (CLI, web, ADK).

- `web_app/` — agent/web integration
  - `web_app/agent.py` — example integration for Google ADK / web-based agent interface
  - The file is a minimal demo; production use requires auth, rate-limiting and hardened error handling.

## Data format

Place each meeting in `datademo/conXXX/` with the four files listed above. The `loader` standardizes these into a document structure used by the chunker and retriever.

## Quick start (developer)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare data under `datademo/` following the examples.
3. Run embedding/indexing scripts (see `src/embeddings/generator.py` and `generate/` scripts).

4. Run the example web/agent demo:

```bash
adk web ., and then choose the web_app file.
```

## Configuration notes

- Key settings live in `config/settings.py`. By default `DATA_DIR` points to `datademo`.
- Set `OPENAI_API_KEY` (or alternate provider credentials) in env or `.env`.
- FAISS is used by default for development; Elasticsearch is available as an alternative backend.

## Tips and cautions

- Test with a small subset of data before generating embeddings for the entire dataset.
- `web_app/agent.py` is a demo integration — add authentication and rate limiting before production use.
- The sub-agents are intentionally low-coupling; you can replace or extend them for specialized workflows.

## Contribution & next steps

1. Will find out solution for query rewrite agent, maybe a reward model will help.

