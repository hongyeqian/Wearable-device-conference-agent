"""
Elasticsearch vector store for hierarchical RAG system
Supports dense vector search for embeddings with large-scale text datasets
"""

from typing import List, Dict, Any, Optional
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
import time

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chunking.chunker import ChunkMetadata
from src.embeddings.generator import EmbeddingGenerator
from config.settings import (
    ELASTICSEARCH_URL,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    ELASTICSEARCH_INDEX_PREFIX,
    ELASTICSEARCH_VERIFY_CERTS,
    ELASTICSEARCH_CA_CERTS,
    ELASTICSEARCH_TIMEOUT,
    EMBEDDING_MODEL
)

class ElasticsearchVectorStore:
    """
    Elasticsearch vector store for storing and retrieving chunks with embeddings
    
    Supports three index levels:
    - metadata: Meeting metadata level
    - summary: Summary level chunks
    - meeting: Meeting level chunks
    
    Designed for large-scale text datasets with efficient vector search
    """
    
    def __init__(self, 
                 es_client: Optional[Elasticsearch] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize Elasticsearch vector store
        
        Args:
            es_client: Optional Elasticsearch client (default: create new)
            embedding_generator: Optional EmbeddingGenerator (default: create new)
        """
        # Initialize Elasticsearch client
        if es_client:
            self.es = es_client
        else:
            es_config = {
                'hosts': [ELASTICSEARCH_URL],
                'verify_certs': ELASTICSEARCH_VERIFY_CERTS,
                'request_timeout': ELASTICSEARCH_TIMEOUT,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            # For Elasticsearch 8.x, disable SSL warnings when cert verification is disabled
            if not ELASTICSEARCH_VERIFY_CERTS:
                es_config['ssl_show_warn'] = False
            
            if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
                es_config['basic_auth'] = (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
            
            if ELASTICSEARCH_CA_CERTS:
                es_config['ca_certs'] = ELASTICSEARCH_CA_CERTS
            
            self.es = Elasticsearch(**es_config)
        
        # Initialize embedding generator
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Get embedding dimension
        self.embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        # Index names for different levels
        self.index_names = {
            'metadata': f"{ELASTICSEARCH_INDEX_PREFIX}_metadata",
            'summary': f"{ELASTICSEARCH_INDEX_PREFIX}_summary",
            'meeting': f"{ELASTICSEARCH_INDEX_PREFIX}_meeting"
        }
        
        
    def create_indices(self, force_recreate: bool = False):
        
        for level, index_name in self.index_names.items():
            if force_recreate and self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                
                print(f"Deleted the existing index name{index_name}")
                
                
                
            if not self.es.indices.exists(index=index_name):
                # Define index mapping with dense vector field
                mapping = {
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "meeting_id": {"type": "keyword"},
                            "level": {"type": "keyword"},
                            "text": {
                                "type": "text",
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {"type": "keyword", "ignore_above": 256}
                                }
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                            # Metadata fields for filtering
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "text"},
                                    "datetime": {"type": "date"},
                                    "duration_sec": {"type": "integer"},
                                    "language": {"type": "keyword"},
                                    "meeting_type": {"type": "keyword"},
                                    "participants": {
                                        "type": "nested",  # Changed from keyword to nested (object array)
                                        "properties": {
                                            "name": {"type": "keyword"},
                                            "role": {"type": "keyword"},
                                            "aliases": {"type": "keyword"}  # Array of keywords
                                        }
                                    },
                                    "organizations": {"type": "keyword"},  # Array of keywords (automatically supported)
                                    "topics": {"type": "keyword"},  # Array of keywords (automatically supported)
                                    "keywords": {"type": "keyword"},  # Array of keywords (automatically supported)
                                    "summary_ids": {"type": "keyword"},
                                    "references": {"type": "keyword"},
                                    "topic_id": {"type": "keyword"},
                                    "paragraph_indices": {"type": "keyword"},
                                    "actions": {
                                        "type": "nested",
                                        "properties": {
                                            "action_id": {"type": "keyword"},
                                            "task": {"type": "text"},
                                            "assignee": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "keyword"},
                                                    "role": {"type": "keyword"},
                                                    "aliases": {"type": "keyword"}
                                                }
                                            },
                                            "due_date": {"type": "keyword"},
                                            "status": {"type": "keyword"},
                                            "priority": {"type": "keyword"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,  # Adjust based on data size
                        "number_of_replicas": 0,  # Set to 1+ for production
                        "refresh_interval": "30s"  # Balance between indexing speed and search freshness
                        # Note: In Elasticsearch 8.x, k-NN is built-in when dense_vector has index: true
                    }
                }
                
                self.es.indices.create(index=index_name, body=mapping)
                print(f"Created index: {index_name}")
            else:
                print(f"Index already exists: {index_name}")
                
                
                
                
    def add_chunks(self, 
                   chunks: List[ChunkMetadata],
                   level: str,
                   generate_embeddings: bool = True,
                   batch_size: int=100,
                   show_progress: bool=True):
        """
        Add chunks to Elasticsearch index
        
        Args:
            chunks: List of ChunkMetadata objects
            level: Level name ('metadata', 'summary', 'meeting')
            generate_embeddings: Whether to generate embeddings (default: True)
            batch_size: Batch size for bulk indexing (default: 100)
            show_progress: Whether to show progress (default: True)
            
        Returns:
            Number of chunks successfully added
        """
        if level not in self.index_names:
            raise ValueError(f"Invalid level: {level}, must be in the {list(self.index_names.keys())}")
        
        index_name = self.index_names[level]
        
        if not chunks:
            return 0
        
        if generate_embeddings:
            if show_progress:
                print(f"Generating embedding for {len(chunks)}{level} ...")
                
            embeddings = self.embedding_generator.generate_chunks_embeddings(
                chunks = chunks,
                batch_size= batch_size
            )
            
        else:
            embeddings = [None]* len(chunks)
            
            
            
            
        # Prepare documents for bulk indexing
        def generate_actions():
            for chunk, embedding in zip(chunks, embeddings):
                
                doc = {
                    "_index": index_name,
                    "_id": chunk.chunk_id,  # Use chunk_id as document ID
                    "_source": {
                        "chunk_id": chunk.chunk_id,
                        "meeting_id": chunk.meeting_id,
                        "level": level,
                        "text": chunk.text,
                        "embedding": embedding.tolist() if embedding is not None else None,
                        "metadata": chunk.metadata if chunk.metadata else {}
                    }
                }
                yield doc
                
        # Bulk index with progress tracking
        success_count = 0
        failed_count = 0
        
        if show_progress:
            print(f"Indexing {len(chunks)} {level} chunks to Elasticsearch...")
        
        for ok, response in streaming_bulk(
            self.es, 
            generate_actions(), 
            chunk_size=batch_size,
            raise_on_error=False,
            max_retries=3
        ):
            if ok:
                success_count += 1
            else:
                failed_count += 1
                if show_progress and failed_count <= 5:  # Show first 5 errors
                    print(f"  Error indexing document: {response}")
        
        if show_progress:
            print(f"Successfully indexed {success_count} {level} chunks")
            if failed_count > 0:
                print(f"Failed to index {failed_count} {level} chunks")
        
        # Refresh index to make documents searchable immediately
        self.es.indices.refresh(index=index_name)
        
        return success_count
    
    
    
    def search(self,
               query_text: str,
               level: str,
               top_k: int =5,
               filters: Optional[Dict[str, Any]] =None,
               meeting_ids: Optional[List[str]] =None) -> List[Dict[str,Any]]:
        """
        Search chunks using vector similarity
        
        Args:
            query_text: Query text to search for
            level: Level to search ('metadata', 'summary', 'meeting')
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {'participants': ['Alice']})
            meeting_ids: Optional list of meeting_ids to filter by
            
        Returns:
            List of search results with chunks and scores
        """
        
        if level not in self.index_names:
            raise ValueError(f"Invalid level: {level}. Must be one of {list(self.index_names.keys())}")
        
        index_name = self.index_names[level]
        
        # Check if index exists
        if not self.es.indices.exists(index=index_name):
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        # Build k-NN query
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding.tolist(),
            "k": top_k * 2 if (filters or meeting_ids) else top_k,  # Get more candidates if filtering
            "num_candidates": top_k * 10  # Number of candidates to consider
        }
        
        # Build filter query
        filter_clauses = []
        
        # Filter by meeting_ids
        if meeting_ids:
            filter_clauses.append({
                "terms": {"meeting_id": meeting_ids}
            })
        
        # Add metadata filters
        if filters:
            for key, value in filters.items():
                # Handle nested actions fields
                if key.startswith('actions.'):
                    # Extract the nested field name (e.g., 'actions.status' -> 'status')
                    nested_field = key.split('.', 1)[1]  # 'status', 'priority', etc.
                    
                    if nested_field.startswith('assignee.'):
                        # Handle assignee sub-fields (e.g., 'actions.assignee.name')
                        assignee_field = nested_field.split('.', 1)[1]  # 'name', 'role', etc.
                        filter_clauses.append({
                            "nested": {
                                "path": "metadata.actions",
                                "query": {
                                    "term" if not isinstance(value, list) else "terms": {
                                        f"metadata.actions.assignee.{assignee_field}": value
                                    }
                                }
                            }
                        })
                    elif nested_field == 'task':
                        # Handle task field (text type) - use match query
                        filter_clauses.append({
                            "nested": {
                                "path": "metadata.actions",
                                "query": {
                                    "match": {
                                        f"metadata.actions.{nested_field}": value
                                    }
                                }
                            }
                        })
                    else:
                        # Handle direct action fields (keyword type) - use term/terms query
                        # e.g., 'actions.status', 'actions.priority', 'actions.action_id', 'actions.due_date'
                        filter_clauses.append({
                            "nested": {
                                "path": "metadata.actions",
                                "query": {
                                    "term" if not isinstance(value, list) else "terms": {
                                        f"metadata.actions.{nested_field}": value
                                    }
                                }
                            }
                        })
                # Handle nested participants fields
                elif key.startswith('participants.'):
                    # Extract the nested field name (e.g., 'participants.name' -> 'name')
                    nested_field = key.split('.', 1)[1]  # 'name', 'role', 'aliases'
                    
                    # All participants fields are keyword type
                    filter_clauses.append({
                        "nested": {
                            "path": "metadata.participants",
                            "query": {
                                "term" if not isinstance(value, list) else "terms": {
                                    f"metadata.participants.{nested_field}": value
                                }
                            }
                        }
                    })
                else:
                    # Handle regular metadata fields (non-nested)
                    # Note: metadata.title is text type, but typically we filter by exact match
                    # If you need full-text search on title, use match query instead
                    if isinstance(value, list):
                        filter_clauses.append({
                            "terms": {f"metadata.{key}": value}
                        })
                    else:
                        filter_clauses.append({
                            "term": {f"metadata.{key}": value}
                        })
        
        # Combine k-NN with filters
        if filter_clauses:
            # In Elasticsearch 8.x, filters should be inside knn query
            # This ensures filters are applied BEFORE k-NN search
            if len(filter_clauses) == 1:
                # Single filter clause
                knn_query["filter"] = filter_clauses[0]
            else:
                # Multiple filter clauses - combine with bool filter
                knn_query["filter"] = {
                    "bool": {
                        "filter": filter_clauses
                    }
                }
            
            query = {"knn": knn_query}
        else:
            query = {"knn": knn_query}
        
        # Execute search
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "size": top_k,
                    "_source": ["chunk_id", "meeting_id", "level", "text", "metadata"],
                    **query
                }
            )
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        # Format results
        results = []
        for hit in response['hits']['hits']:
            
            results.append({
                'chunk_id': hit['_source']['chunk_id'],
                'meeting_id': hit['_source']['meeting_id'],
                'level': hit['_source']['level'],
                'text': hit['_source']['text'],
                'metadata': hit['_source'].get('metadata', {}),
                'score': hit['_score']
            })
        
        return results
    
    
    
    def delete_index(self, level: str):
        
        if level not in self.index_names:
            raise ValueError(f"Invalid level: {level}. Must be one of {list(self.index_names.keys())}")
        
        index_name = self.index_names[level]
        
        if self.es.indices.exists(index= index_name):
            self.es.indices.delete(index=index_name)
            print(f"Delete index: {index_name}")
            
            
            
    def delete_all_indices(self):
        for level in self.index_names.keys():
            self.delete_index(level)
            
            
            
            
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for all indices"""
        stats = {}
        for level, index_name in self.index_names.items():
            if self.es.indices.exists(index=index_name):
                count_response = self.es.count(index=index_name)
                stats[level] = count_response['count']
            else:
                stats[level] = 0
        return stats
    
    
    def health_check(self) -> bool:
        """Check if Elasticsearch is accessible"""
        try:
            return self.es.ping()
        except Exception as e:
            print(f"Elasticsearch health check failed: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Elasticsearch cluster information"""
        try:
            info = self.es.info()
            return {
                'cluster_name': info.get('cluster_name'),
                'version': info.get('version', {}).get('number'),
                'status': 'healthy'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

            
            