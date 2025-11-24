import os
from pathlib import Path
from dotenv import load_dotenv



#load_dotenv()

# config path, may be need
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent
env_file_config = CONFIG_DIR / ".env"

if env_file_config.exists():
    load_dotenv(env_file_config)
else:
    # Fallback to project root or default behavior
    load_dotenv()

# config path, may be need
DATA_DIR = PROJECT_ROOT / "datademo"
# write future path here
# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#print(OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
#print(OPENAI_MODEL, EMBEDDING_MODEL)

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# retrieval settings
TOP_K_SUMMARY = 3
TOP_K_MEETING = 3
TOP_K_CHUNK = 5


# Elasticsearch settings
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", None)
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", None)
ELASTICSEARCH_INDEX_PREFIX = os.getenv("ELASTICSEARCH_INDEX_PREFIX", "meeting_rag")
ELASTICSEARCH_VERIFY_CERTS = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
ELASTICSEARCH_CA_CERTS = os.getenv("ELASTICSEARCH_CA_CERTS", None)  # Path to CA certificate if needed
ELASTICSEARCH_TIMEOUT = int(os.getenv("ELASTICSEARCH_TIMEOUT", "30"))  # Request timeout in seconds


# following is the necessary settings for the project
# will write future
