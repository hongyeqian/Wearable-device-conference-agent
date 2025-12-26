# Meeting Transcript RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed for querying and analyzing meeting transcripts. The system implements hierarchical chunking, multi-agent query processing, and hybrid retrieval to provide accurate, context-aware answers to questions about meeting content.

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ Retrieval Layer │    │   Query Layer   │
│                 │    │                 │    │                 │
│ • DataLoader    │    │ • Vector Store  │    │ • Query Rewriter│
│ • Hierarchical  │    │ • Hybrid Search │    │ • Answer Gen.   │
│   Chunker       │    │ • Hierarchical  │    │ • Orchestrator  │
│ • Embeddings    │    │   Retriever     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Web Interface │
                    │                 │
                    │ • Flask App     │
                    │ • REST API      │
                    │ • Session Mgmt  │
                    └─────────────────┘
```

### Data Processing Pipeline

1. **Data Loading**: Loads meeting transcripts, metadata, summaries, and structured content from `datademo/` directory
2. **Hierarchical Chunking**: Processes content at three levels (metadata, summary, meeting) with semantic-aware splitting
3. **Embedding Generation**: Converts text chunks to vector embeddings using OpenAI's embedding models
4. **Vector Storage**: Stores embeddings in FAISS with hybrid search capabilities (vector + BM25)

### Query Processing Flow

```
User Query → Query Rewriter → Hierarchical Retrieval → Answer Generation → Response
     ↓              ↓              ↓                      ↓
  Raw Text    Multi-Agent       Hybrid Search       Multi-Agent
             Processing       (Vector + BM25)       Generation
```

## 🚀 Key Features

### Multi-Level Hierarchical Retrieval
- **Metadata Level**: Meeting overview, participants, topics, and keywords
- **Summary Level**: High-level meeting summaries with topic/action item references
- **Meeting Level**: Detailed topic and action item content with paragraph references

### Advanced Query Processing
- **Multi-Agent Query Rewriter**: Uses Google ADK agents for intelligent query normalization and paraphrasing
- **Session Management**: Maintains conversation context across multiple queries
- **Meeting Catalog Integration**: Provides temporal and participant context for query understanding

### Hybrid Search Technology
- **Vector Search**: Semantic similarity using OpenAI embeddings
- **BM25 Search**: Keyword-based retrieval for precise matching
- **Score Fusion**: Combines vector and BM25 scores for optimal ranking

### Web Interface
- **Gemini-like UI**: Clean, intuitive interface for querying
- **REST API**: Programmatic access via `/api/query` endpoint
- **Health Monitoring**: System status and pipeline initialization checks

## 📁 Project Structure

```
├── config/
│   ├── settings.py          # Configuration and API keys
│   └── requirements.txt     # Python dependencies
├── src/
│   ├── data_loader/
│   │   └── loader.py        # Meeting data loading and parsing
│   ├── chunking/
│   │   └── chunker.py       # Hierarchical text chunking
│   ├── embeddings/
│   │   └── generator.py     # OpenAI embedding generation
│   └── retrieval/
│       ├── vector_store.py      # FAISS + BM25 hybrid search
│       ├── hierarchical_retriever.py
│       ├── rag_pipeline.py      # Main RAG pipeline orchestration
│       ├── orchestrator.py      # Query handling and session management
│       ├── query_rewriter_muti_agent.py  # Multi-agent query rewriting
│       └── answer_generator_muti_agent.py # Multi-agent answer generation
├── datademo/                # Meeting data (con*/ directories)
├── vector_store/            # FAISS indices and BM25 data
├── outputs/                 # Query results and evaluation data
├── templates/
│   └── index.html          # Web interface template
├── tests/                   # Unit tests and debugging tools
├── rag_main.py             # Command-line interface
├── web_app.py              # Flask web application
└── requirements.txt        # Project dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Elasticsearch (optional, for alternative vector storage)

### Environment Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/your/project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the `config/` directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
   EMBEDDING_MODEL=text-embedding-3-small

   # Optional Elasticsearch configuration
   ELASTICSEARCH_URL=http://localhost:9200
   ELASTICSEARCH_USERNAME=your_username
   ELASTICSEARCH_PASSWORD=your_password
   ```

## 🚀 Usage

### Command Line Interface

**Initialize and run interactive mode:**
```bash
python rag_main.py
```

**Process a single query:**
```bash
python rag_main.py --query "What were the main action items from last week's meeting?"
```

**Advanced options:**
```bash
python rag_main.py --query "Who discussed the budget?" \
  --top-k-vector 10 \
  --top-k-bm25 30 \
  --verbose \
  --save-original-rank
```

### Web Application

**Start the web server:**
```bash
python web_app.py
```

Access the interface at `http://localhost:5000`

### API Usage

**Query endpoint:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What action items were assigned?"}'
```

**Health check:**
```bash
curl http://localhost:5000/api/health
```

## 🔧 Configuration

Key settings in `config/settings.py`:

```python
# Data and model configuration
DATA_DIR = PROJECT_ROOT / "datademo"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
ONLY_SUMMARY = True  # Focus on summary and meeting levels

# Retrieval parameters
TOP_K_VECTOR = 5    # Vector search results per query
TOP_K_BM25 = 20     # BM25 search results per query

# Session management
APP_NAME = "agents"
USER_ID = "u-main"
```

## 📊 Data Format

### Meeting Data Structure

Each meeting resides in a `datademo/conXXX/` directory with:

- **`metaDataXXX.json`**: Meeting metadata (participants, datetime, topics, etc.)
- **`dataXXX.md`**: Raw transcript with timestamped conversation
- **`meetLevelXXX.md`**: Topic-level structured summaries
- **`summaryXXX.md`**: High-level meeting summaries

### Hierarchical Content Levels

1. **Metadata Level**: Structured meeting information for quick filtering
2. **Summary Level**: Condensed meeting overviews with key takeaways
3. **Meeting Level**: Detailed topic and action item breakdowns

## 🧪 Testing & Evaluation

### Running Tests

```bash
# Export chunking results for analysis
python tests/test_chunking_export.py

# Test retrieval components
python tests/test_retrieve.py

# Debug timestamp parsing
python tests/debug_timestamps.py
```

### Evaluation Scripts

```bash
# Generate evaluation reports
python generate/evaluation_script.py

# Results saved to outputs/evaluation_results_*.txt
```

## 🔍 Advanced Features

### Multi-Agent Architecture

- **Query Rewriter**: Normalizes queries, extracts entities, generates paraphrases
- **Answer Generator**: Produces context-aware responses with citations
- **Session Service**: Maintains conversation state using Google ADK

### Hybrid Retrieval Strategy

The system combines multiple retrieval techniques:

1. **Query Rewriting**: Transforms user queries for better retrieval
2. **Multi-Query Search**: Searches with original query + paraphrases
3. **Hybrid Scoring**: Fuses vector similarity and BM25 keyword scores
4. **Hierarchical Ranking**: Prioritizes more relevant content levels

### Session Management

- **Turn-based Sessions**: Each query gets isolated session context
- **Memory Persistence**: Maintains conversation history across turns
- **Meeting Catalog**: Provides temporal context for query understanding

## 🚢 Deployment

### Production Deployment

**Using Gunicorn:**
```bash
gunicorn --config gunicorn_config.py web_app:app
```

**Docker deployment:**
```bash
# Build container
docker build -t rag-system .

# Run container
docker run -p 5000:5000 rag-system
```

### Scaling Considerations

- **Vector Store**: FAISS for development, Elasticsearch for production
- **Session Storage**: In-memory for development, Redis/persistent storage for production
- **API Rate Limiting**: Implement request throttling for OpenAI API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Development Guidelines

- Follow the existing code structure and naming conventions
- Add unit tests for new components
- Update documentation for API changes
- Test with both command-line and web interfaces

## 📝 License

[Specify your license here]

## 📧 Contact

[Your contact information]

---

## 🔄 Recent Updates

- **Multi-Agent Architecture**: Implemented Google ADK agents for query rewriting and answer generation
- **Session Management**: Added persistent conversation context across queries
- **Hybrid Search**: Combined vector and BM25 retrieval for improved accuracy
- **Web Interface**: Clean, responsive UI for easy querying
- **Hierarchical Chunking**: Semantic-aware text splitting with overlap preservation

For detailed implementation notes, see the component-specific documentation in each module's docstrings.
