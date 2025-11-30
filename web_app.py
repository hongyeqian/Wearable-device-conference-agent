"""
Web Application for RAG System
Provides a Gemini-like interface for querying the RAG system.
"""
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_main import initialize_rag_system
from config.settings import DATA_DIR

app = Flask(__name__)
CORS(app)

# Global pipeline instance (initialized on startup)
pipeline = None


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests"""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Process query using RAG pipeline
        result = pipeline.query(
            user_query=query_text,
            top_k_vector=5,
            top_k_bm25=20,
            num_paraphrases=2,
            max_chunks_for_answer=15
        )
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'original_query': result['original_query'],
            'rewritten_query': result.get('rewritten_query'),
            'num_chunks': result['num_chunks'],
            'query_rewrite_data': result.get('query_rewrite_data')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pipeline_initialized': pipeline is not None
    })


def init_pipeline():
    """Initialize the RAG pipeline on startup"""
    global pipeline
    try:
        print("Initializing RAG pipeline for web application...")
        pipeline = initialize_rag_system(data_dir=DATA_DIR)
        print("✅ RAG pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # Initialize pipeline before starting server
    init_pipeline()
    
    # Development mode
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # Production mode with Gunicorn - initialize on import
    init_pipeline()
