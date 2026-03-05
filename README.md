# Meeting Transcript RAG System

A Retrieval-Augmented Generation (RAG) system for meeting transcript question answering.

## Project Overview

This system is designed specifically for processing meeting transcript data, providing the following core features:

- **Hierarchical Document Chunking**: Multi-level chunking for metadata, summaries, and meeting content
- **Hybrid Retrieval**: Combines vector search (FAISS) and BM25 keyword search
- **Modular Agents**: Decoupled sub-agent design for query rewriting, answer generation, and task planning
- **Web Interface**: Interactive web interface based on Google ADK

## Directory Structure

```
DevelopmentRAG/
├── datademo/                    # Meeting data directory
│   └── conXXX/                  # Each meeting folder (con1, con2, ...)
│       ├── metaDataXXX.json     # Meeting metadata
│       ├── dataXXX.md          # Raw meeting transcript
│       ├── meetLevelXXX.md     # Topic-level structured content
│       └── summaryXXX.md       # Meeting summary
│
├── src/                        # Core implementation
│   ├── data_loader/
│   │   └── loader.py           # Load and normalize data from datademo
│   ├── chunking/
│   │   └── chunker.py          # Hierarchical document chunking
│   ├── embeddings/
│   │   └── generator.py        # Generate vector embeddings
│   └── retrieval/
│       ├── hierarchical_retriever.py    # Main retrieval logic
│       ├── vector_store.py              # Hybrid vector store (FAISS + BM25)
│       └── vector_store_utils.py        # Vector store utilities
│
├── sub_agents/                 # Sub-agent modules
│   ├── query_rewriter_agent.py # Query rewriting agent (three-stage process)
│   ├── answer_agent.py         # Answer generation agent
│   ├── planner_agent.py        # Task planning agent
│   ├── pandas_utils.py         # Data processing utilities
│   ├── date_resolver.py        # Date resolution
│   └── person_matcher.py       # Person name matching
│
├── web_app/                    # Web interface
│   └── agent.py               # Google ADK Web Agent
│
├── config/                     # Configuration
│   ├── settings.py            # System configuration
│   └── .env                   # Environment variables (API keys, etc.)
│
├── generate/                   # Data generation scripts
├── test_set/                  # Test cases
└── requirements.txt            # Dependencies list
```

## Quick Start

### 1. Environment Setup

#### 1.1 Create Virtual Environment (Windows)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
./.venv/Scripts/Activate.ps1    // conda environment cannot link openai use venv!!!

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (for Presidio privacy processing)， both need to download, if you do not want to download the large one, go to query_rewriter_agent.py change the  AnalyzerEngine()
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

#### 1.2 Configure Environment Variables

Create a `.env` file in the `config/` directory, do not forgot to build this, if do not build, system cannot run.

```bash
# OpenAI API configuration (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional configuration
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Elasticsearch configuration (optional, FAISS is used by default)
# ELASTICSEARCH_URL=http://localhost:9200
# ELASTICSEARCH_USERNAME=your_username
# ELASTICSEARCH_PASSWORD=your_password
```

> **Note**: Please visit [OpenAI Platform](https://platform.openai.com/) to get your API key.

### 2. Data Preparation

Place your meeting data in the `datademo/conXXX/` directory. Each meeting requires the following four files:

| File | Description | Example |
|-----|-------------|---------|
| `metaDataXXX.json` | Meeting metadata | participants, datetime, topics, meeting_id |
| `dataXXX.md` | Raw meeting transcript | Timestamped dialogues or paragraphs |
| `meetLevelXXX.md` | Topic-level structured content | Meeting content organized by topic |
| `summaryXXX.md` | Meeting summary | High-level summary |

Refer to the sample data in the `datademo/con1/` directory.

### 3. Generate Vector Index

Run the embedding generation script (refer to `src/embeddings/generator.py`):

```bash
# Example: Run data generation script
python generate/your_embedding_script.py
```

### 4. Start Web Service

```bash
# Start web interface using Google ADK
adk web .
```

After starting, access the displayed address in your browser (typically `http://localhost:8000`), select the Agent from the `web_app` directory, and you can start using the system.

## Core Modules

### 1. Data Loader (`src/data_loader/loader.py`)

Responsible for loading meeting data from the `datademo/` directory and normalizing it into a unified document structure.

**Core Classes**:
- `Meeting`: Meeting data model
- `DataLoader`: Data loader

### 2. Hierarchical Chunker (`src/chunking/chunker.py`)

Chunks meeting documents hierarchically:
- **Metadata Chunk**: Meeting participants, time, topics
- **Summary Chunk**: High-level meeting summary
- **Meeting Chunk**: Detailed meeting content

**Configuration Parameters** (in `config/settings.py`):
- `CHUNK_SIZE=800`: Chunk size
- `CHUNK_OVERLAP=120`: Chunk overlap
- `ONLY_SUMMARY=True`: Whether to process summary level only

### 3. Vector Store (`src/retrieval/vector_store.py`)

Hybrid retrieval implementation combining two methods:
- **FAISS Vector Search**: Semantic similarity-based
- **BM25 Keyword Search**: Term frequency-based

### 4. Query Rewriter Agent (`sub_agents/query_rewriter_agent.py`)

**Three-stage Query Rewriting Process**:

```
User's Original Query
    │
    ├─► Stage 1: Ambiguity Check (check_ambiguity)
    │       Detects whether names, dates, etc. in the query are explicit
    │       Example: "that meeting" → needs to determine which specific meeting
    │
    ├─► Stage 2: Replace ambiguity into true information
    │       do simple replacement
    │       Example: hongye -> hongye qian
    │
    └─► Stage 3: react llm to search pandas
            Resolves meeting ambiguous information to specific entities
            Example: "last three meetings" → "2025-11.30", "2025-11.29", "2025-11.21"
```

**Core Functions**:
- `check_ambiguity()`: Detect query ambiguity
- `pandas_query()`: Generate Pandas query to get candidate meetings
- `rewrite_query_async()`: Execute complete rewriting process asynchronously

### 5. Answer Agent (`sub_agents/answer_agent.py`)

Merges retrieved contexts to generate final answers with citation sources.

### 6. Planner Agent (`sub_agents/planner_agent.py`)

Splits complex tasks into subtasks and coordinates multiple agents to complete complex queries.

## Configuration Reference

### Complete `.env` Example

```bash
# ========== OpenAI Configuration ==========
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# ========== Elasticsearch Configuration (Optional) ==========
# ELASTICSEARCH_URL=http://localhost:9200
# ELASTICSEARCH_USERNAME=elastic
# ELASTICSEARCH_PASSWORD=your_password
# ELASTICSEARCH_INDEX_PREFIX=meeting_rag
```

### Core Configuration Items (`config/settings.py`)

| Configuration | Default | Description |
|--------------|---------|-------------|
| `CHUNK_SIZE` | 800 | Chunk size |
| `CHUNK_OVERLAP` | 120 | Chunk overlap |
| `ONLY_SUMMARY` | True | Process summary only |
| `TOP_K_SUMMARY` | 3 | Number of summary retrievals |
| `TOP_K_MEETING` | 3 | Number of meeting retrievals |
| `TOP_K_CHUNK` | 5 | Number of content chunk retrievals |



## Future Improvements

1. **Query Rewriter Agent**: Introduce reward model to improve rewriting quality
2. **Evaluation Framework**: Integrate RAGAs and other evaluation tools
3. **Multi-modal Support**: Support audio and video meeting recordings
4. **Enterprise Deployment**: Add authentication, rate limiting, and other production features
5. conside to use mapping knowledge domain

## LicenseMIT License
