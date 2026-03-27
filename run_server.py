"""
Custom ADK Web Server Launcher

Replaces `adk web .` to ensure all components (RAG vector store,
spaCy, Presidio, EmbeddingGenerator, etc.) are fully loaded
BEFORE the web server starts accepting requests.

Usage:
    python run_server.py [--port PORT] [--host HOST]

The server is functionally identical to `adk web .` but all NLP
components are warm when the first user query arrives.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so module imports work correctly.
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Launch ADK web server with pre-loaded components"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Force-import web_app.agent
    #
    # This triggers ALL module-level code in agent.py:
    #   - FullRAGSystemAgent.__init__  →  loads RAG vector store
    #   - warmup_all()                →  loads spaCy, Presidio, pandas, embeddings
    #
    # Python caches the module in sys.modules, so when ADK's AgentLoader
    # later calls importlib.import_module("web_app.agent"), it returns
    # the already-loaded module instantly — no re-initialization.
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pre-loading agent module and all NLP components...")
    logger.info("=" * 60)

    import web_app.agent  # noqa: F401  (import triggers side effects)

    logger.info("=" * 60)
    logger.info("All components loaded. Starting ADK web server...")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 2: Start the ADK web server (same as `adk web .`)
    # ------------------------------------------------------------------
    import uvicorn
    from google.adk.cli.fast_api import get_fast_api_app

    app = get_fast_api_app(
        agents_dir=".",
        web=True,
        host=args.host,
        port=args.port,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
