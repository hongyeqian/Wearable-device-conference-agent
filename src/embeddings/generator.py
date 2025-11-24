"""
This module is embedding generator for the project
Generates embeddings for different levels of chunks (metadata, summary, meeting)
"""

from typing import List, Union, Dict, Any
import numpy as np
from langchain_openai import OpenAIEmbeddings

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chunking.chunker import ChunkMetadata
from config.settings import EMBEDDING_MODEL


class EmbeddingGenerator:
    """
    Embedding generator for hierarchical RAG system
    
    Supports:
    - Single text embedding (for queries)
    - Batch text embeddings (for indexing)
    - Different chunk levels (metadata, summary, meeting)
    """
    
    def __init__(self, model_name: str | None = None , **kwargs):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the embedding model (default: from settings)
            **kwargs: Additional arguments for OpenAIEmbeddings
        """
        self.model_name = model_name or EMBEDDING_MODEL
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            **kwargs
        )
        
        # Cache for embeddings (optional, for development)
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def generate_embedding(self, text: str, use_cache: bool = False) -> np.ndarray:
        """
        Generate embedding for a single text (for queries)
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings (default: False)
            
        Returns:
            numpy array of embedding vector
        """
        # Check cache if enabled
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Generate embedding
        embedding = self.embeddings.embed_query(text)
        embedding_array = np.array(embedding)
        
        # Cache if enabled
        if use_cache:
            self._embedding_cache[text] = embedding_array
        
        return embedding_array
    
    def generate_embeddings_batch(self, 
                                  texts: List[str], 
                                  batch_size: int = 100,
                                  use_cache: bool = False) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts (for indexing)
        
        Args:
            texts: List of input texts to embed
            batch_size: Batch size for embedding generation (default: 100)
            use_cache: Whether to use cached embeddings (default: False)
            
        Returns:
            List of numpy arrays of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            
            for idx, text in enumerate(batch_texts):
                if use_cache and text in self._embedding_cache:
                    batch_embeddings.append((idx, self._embedding_cache[text]))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(idx)
            
            # Generate embeddings for uncached texts
            if texts_to_embed:
                new_embeddings = self.embeddings.embed_documents(texts_to_embed)
                
                # Convert to numpy arrays and cache
                for idx, embedding in zip(indices_to_embed, new_embeddings):
                    embedding_array = np.array(embedding)
                    if use_cache:
                        self._embedding_cache[batch_texts[idx]] = embedding_array
                    batch_embeddings.append((idx, embedding_array))
            
            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])
        
        return all_embeddings
    
    def generate_chunk_embedding(self, chunk: ChunkMetadata) -> np.ndarray:
        """
        Generate embedding for a single chunk
        
        Args:
            chunk: ChunkMetadata object
            
        Returns:
            numpy array of embedding vector
        """
        return self.generate_embedding(chunk.text)
    
    def generate_chunks_embeddings(self, 
                                   chunks: List[ChunkMetadata],
                                   batch_size: int = 100,
                                   use_cache: bool = False) -> List[np.ndarray]:
        """
        Generate embeddings for a list of chunks (for indexing)
        
        Args:
            chunks: List of ChunkMetadata objects
            batch_size: Batch size for embedding generation (default: 100)
            use_cache: Whether to use cached embeddings (default: False)
            
        Returns:
            List of numpy arrays of embedding vectors
        """
        texts = [chunk.text for chunk in chunks]
        return self.generate_embeddings_batch(texts, batch_size=batch_size, use_cache=use_cache)
    
    def generate_embeddings_by_level(self, 
                                     chunks_by_level: Dict[str, List[ChunkMetadata]],
                                     batch_size: int = 100,
                                     use_cache: bool = False) -> Dict[str, List[np.ndarray]]:
        """
        Generate embeddings for chunks organized by level
        
        Args:
            chunks_by_level: Dictionary with keys 'metadata', 'summary', 'meeting'
                            and values as lists of ChunkMetadata
            batch_size: Batch size for embedding generation (default: 100)
            use_cache: Whether to use cached embeddings (default: False)
            
        Returns:
            Dictionary with same keys and values as lists of embedding vectors
        """
        result = {}
        
        for level, chunks in chunks_by_level.items():
            if chunks:
                print(f"Generating embeddings for {level} level: {len(chunks)} chunks...")
                result[level] = self.generate_chunks_embeddings(
                    chunks, 
                    batch_size=batch_size,
                    use_cache=use_cache
                )
            else:
                result[level] = []
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings generated by this model
        
        Returns:
            Dimension of embedding vectors
        """
        # OpenAI embedding dimensions
        # text-embedding-ada-002: 1536
        # text-embedding-3-small: 1536
        # text-embedding-3-large: 3072
        
        model_dims = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
        }
        
        # Try to get from model name
        for model_name, dim in model_dims.items():
            if model_name in self.model_name.lower():
                return dim
        
        # Default: generate a test embedding to get dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings"""
        return len(self._embedding_cache)


# Testing code
if __name__ == "__main__":
    from src.data_loader.loader import DataLoader
    from src.chunking.chunker import HierarchicalChunker
    from config.settings import DATA_DIR
    
    # Load data
    print("Loading meetings...")
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    
    if not meetings:
        print("No meetings found!")
        exit(1)
    
    # Chunk data
    print("\nChunking meetings...")
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_all_levels(meetings)
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    generator = EmbeddingGenerator()
    
    # Test single embedding
    test_chunk = all_chunks['metadata'][0] if all_chunks['metadata'] else None
    if test_chunk:
        print(f"\nTesting single embedding for chunk: {test_chunk.chunk_id}")
        embedding = generator.generate_chunk_embedding(test_chunk)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    
    # Test batch embeddings
    print(f"\nGenerating batch embeddings...")
    embeddings_by_level = generator.generate_embeddings_by_level(all_chunks)
    
    # Print statistics
    print("\n" + "="*80)
    print("EMBEDDING GENERATION STATISTICS")
    print("="*80)
    for level, embeddings in embeddings_by_level.items():
        print(f"  {level.capitalize()} level: {len(embeddings)} embeddings")
        if embeddings:
            print(f"    - Embedding dimension: {embeddings[0].shape[0]}")
    print("="*80)
