"""
Hybrid Search Vector Store combining FAISS vector search with BM25 keyword search.
Supports hierarchical retrieval across metadata, summary, and meeting levels.
"""
from typing import List, Dict, Any, Optional, Set
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import re
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chunking.chunker import ChunkMetadata
from src.embeddings.generator import EmbeddingGenerator
from config.settings import EMBEDDING_MODEL


class HybridSearchVectorStore:
    """
    Hybrid vector store combining FAISS (vector search) with BM25 (keyword search).
    
    Supports three index levels:
    - metadata: Meeting metadata level
    - summary: Summary level chunks
    - meeting: Meeting level chunks
    
    Uses alpha parameter to weight vector vs BM25 scores in final ranking.
    """
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None, alpha: float = 0.6):
        
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        self.alpha = alpha
        
        # FAISS indices for each level
        self.faiss_indices: Dict[str, faiss.Index] = {}
        
        # Number of clusters (nlist) for each level's IVF index
        self.faiss_nlist: Dict[str, int] = {}
        
        # Number of clusters to probe (nprobe) during search - controls speed vs accuracy
        self.faiss_nprobe: Dict[str, int] = {}
        
        # BM25 indices for each level
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        
        
        # Mapping back to the chunk metadata
        self.doc_mapping: Dict[str, Dict[int, ChunkMetadata]] = {
            'metadata': {},
            'summary': {},
            'meeting': {}
        }
        
        self.doc_text_tokenized: Dict[str, List[List[str]]] = {
            'metadata': [],
            'summary': [],
            'meeting': []
        }
        
        # Inverted indices for filtering
        self.meeting_id_to_summary_indices: Dict[str, Set[int]] = {}
        self.entry_id_to_meeting_indices: Dict[str, Set[int]] = {}
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search.
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase tokens
        """
        # Lowercase, split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
        
        
    def add_chunks(self, chunks: List[ChunkMetadata],
                   level: str,
                   generate_embedding: bool):
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of ChunkMetadata objects
            level: Index level ('metadata', 'summary', 'meeting')
            generate_embedding: Whether to generate embeddings
        """
        if level not in ['metadata', 'summary', 'meeting']:
            raise ValueError(f"Invalid level: {level}")
        
        if not chunks:
            print(f"No chunks in the {level}")
            return
            
            
        print(f"Adding {len(chunks)} chunks to {level} level")
        
        
        if generate_embedding:
            print(f"Generating embeddings...")
            embeddings = self.embedding_generator.generate_chunks_embeddings(chunks)
        else:
            raise ValueError("Embedding must be generated for FAISS index")
        
        embedding_array = np.array([emb.tolist() for emb in embeddings]).astype('float32')
        
        
        if level not in self.faiss_indices:
            faiss.normalize_L2(embedding_array)
            
            # Choose index type based on data size
            # - Flat index: exact search, better for small datasets (< 1000 vectors)
            # - IVF index: approximate search, better for large datasets
            data_size = len(embedding_array)
            
            # Todo: here maybe some problems, I have not sure about this.
            if data_size >= 1000:
                # Use IVF index for large datasets
                nlist = min(100, max(1, data_size // 10))
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train the index before adding vectors
                index.train(embedding_array)  # type: ignore            
                index.add(embedding_array)  # type: ignore
                
                self.faiss_indices[level] = index
                self.faiss_nlist[level] = nlist
                self.faiss_nprobe[level] = 1  # Default: probe 1 cluster for fast search
                
                print(f"Created FAISS IVF index for {level} level (nlist={nlist}, nprobe={self.faiss_nprobe[level]})")
            else:
                # Use Flat index for small datasets (exact search, no quantization loss)
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(embedding_array)  # type: ignore
                
                self.faiss_indices[level] = index
                self.faiss_nlist[level] = 0  # 0 indicates flat index
                self.faiss_nprobe[level] = 0  # Not applicable for flat index
                
                print(f"Created FAISS Flat index for {level} level (exact search)")
            
            
        else:
            faiss.normalize_L2(embedding_array)
            self.faiss_indices[level].add(embedding_array)  # type: ignore
            print(f"Updated FAISS index for {level} level")
            
        # Store FAISS mapping
        current_size = len(self.doc_mapping[level])
        for idx, chunk in enumerate(chunks):
            global_idx = current_size + idx
            self.doc_mapping[level][current_size + idx] = chunk
            
            
        # Building meeting_id -> summary_indices mapping
            if level == 'summary':
                meeting_id = chunk.meeting_id
                if meeting_id:
                    if meeting_id not in self.meeting_id_to_summary_indices:
                        self.meeting_id_to_summary_indices[meeting_id] = set()
                    self.meeting_id_to_summary_indices[meeting_id].add(global_idx)
                    
                    
            elif level == 'meeting':
                entry_id = chunk.metadata.get('entry_id')
                if entry_id:
                    if entry_id not in self.entry_id_to_meeting_indices:
                        self.entry_id_to_meeting_indices[entry_id] = set()
                    self.entry_id_to_meeting_indices[entry_id].add(global_idx)
            
        # Build BM25 mapping
        tokenized_docs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tokenized_docs.append(tokens)
            self.doc_text_tokenized[level].append(tokens)
            
        if tokenized_docs:
            if level not in self.bm25_indices:
                self.bm25_indices[level] = BM25Okapi(tokenized_docs)
                print(f"Created BM25 index for {level} level")
            else:
                # BM25 doesn't support incremental updates easily, rebuild
                # For now, we'll rebuild the entire index
                all_tokenized = self.doc_text_tokenized[level]
                self.bm25_indices[level] = BM25Okapi(all_tokenized)
                print(f"Rebuilt BM25 index for {level} level")
        
        print(f"Successfully indexed {len(chunks)} {level} chunks")
        
        
        
    def _vector_search(self,
                       query_text: str,
                       level: str,
                       top_k: int,
                       allowed_indices: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search using FAISS.
        
        Args:
            query_text: Query text
            level: Level to search
            top_k: Number of results to return
            allowed_indices: Optional set of allowed FAISS indices to search in
            
        Returns:
            List of results with chunk info and vector scores
        """
        if level not in self.faiss_indices:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        query_vector = np.array([query_embedding.tolist()]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS
        index = self.faiss_indices[level]
        
        # Check if this is an IVF index or Flat index
        is_ivf = self.faiss_nlist.get(level, 0) > 0
        
        # Todo: here may be some problems, I have not consider it.
        # Use SearchParametersIVF with IDSelectorArray for filtering (IVF only)
        if is_ivf and allowed_indices is not None and len(allowed_indices) > 0:
            allowed_ids = np.array(sorted(allowed_indices), dtype=np.int64)
            id_selector = faiss.IDSelectorArray(len(allowed_ids), faiss.swig_ptr(allowed_ids))
            
            # Extract IVF index and use SearchParametersIVF
            ivf_index = faiss.extract_index_ivf(index)  # type: ignore
            ivf_index.nprobe = self.faiss_nprobe.get(level, 1)
            search_params = faiss.SearchParametersIVF()  # type: ignore
            search_params.sel = id_selector  # type: ignore
            
            search_k = min(top_k, len(allowed_ids), index.ntotal)
            print(f"search_k: {search_k} numbers of search k")
            scores, indices = ivf_index.search(query_vector, search_k, params=search_params)  # type: ignore
        elif is_ivf:
            # IVF index without filtering
            ivf_index = faiss.extract_index_ivf(index)  # type: ignore
            ivf_index.nprobe = self.faiss_nprobe.get(level, 1)
            scores, indices = index.search(query_vector, min(top_k, index.ntotal))  # type: ignore
        else:
            # Flat index - simple search (no IVF parameters)
            if allowed_indices is not None and len(allowed_indices) > 0:
                # For flat index with filtering, search all and filter results
                scores, indices = index.search(query_vector, index.ntotal)  # type: ignore
                
                # Filter results to only include allowed indices
                filtered_scores = []
                filtered_indices = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx in allowed_indices:
                        filtered_scores.append(score)
                        filtered_indices.append(idx)
                
                scores = [filtered_scores[:top_k]]
                indices = [filtered_indices[:top_k]]
            else:
                # Flat index without filtering
                scores, indices = index.search(query_vector, min(top_k, index.ntotal))  # type: ignore
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            if allowed_indices is not None and idx not in allowed_indices:
                continue
            
            chunk = self.doc_mapping[level].get(idx)
            if chunk:
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'meeting_id': chunk.meeting_id,
                    'level': level,
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'vector_score': float(score),  # Cosine similarity score
                    'index': int(idx)
                })
        
        return results
    
    def _bm25_search(self,
                     query_text: str,
                     level: str,
                     top_k: int,
                     allowed_indices: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query_text: Query text
            level: Level to search
            top_k: Number of results to return
            allowed_indices: Optional set of allowed indices to search in
            
        Returns:
            List of results with chunk info and BM25 scores
        """
        if level not in self.bm25_indices:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query_text)
        
        if not query_tokens:
            return []
        
        # Determine whether to search all docs or filtered docs
        search_all = allowed_indices is None or len(allowed_indices) == 0
        
        if search_all:
            # Search all documents
            bm25_scores = self.bm25_indices[level].get_scores(query_tokens)
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            
            # Format results using original indices
            results = []
            for idx in top_indices:
                if bm25_scores[idx] <= 0:
                    continue
                
                chunk = self.doc_mapping[level].get(idx)
                if chunk:
                    results.append({
                        'chunk_id': chunk.chunk_id,
                        'meeting_id': chunk.meeting_id,
                        'level': level,
                        'text': chunk.text,
                        'metadata': chunk.metadata,
                        'bm25_score': float(bm25_scores[idx]),
                        'index': int(idx)
                    })
        else:
            # Search only allowed_indices documents for efficiency
            # Note: allowed_indices is guaranteed to be non-None and non-empty here
            assert allowed_indices is not None and len(allowed_indices) > 0
            
            # Get tokenized documents for allowed indices
            sorted_allowed = sorted(allowed_indices)
            allowed_docs = [self.doc_text_tokenized[level][i] for i in sorted_allowed if i < len(self.doc_text_tokenized[level])]
            
            if not allowed_docs:
                return []
            
            # Create temporary BM25 index with only allowed documents
            temp_bm25 = BM25Okapi(allowed_docs)
            temp_scores = temp_bm25.get_scores(query_tokens)
            
            # Map temp indices back to original indices
            temp_indices = np.argsort(temp_scores)[::-1][:min(top_k, len(temp_scores))]
            
            results = []
            for temp_idx in temp_indices:
                if temp_scores[temp_idx] <= 0:
                    continue
                
                # Map back to original index
                original_idx = sorted_allowed[temp_idx]
                
                chunk = self.doc_mapping[level].get(original_idx)
                if chunk:
                    results.append({
                        'chunk_id': chunk.chunk_id,
                        'meeting_id': chunk.meeting_id,
                        'level': level,
                        'text': chunk.text,
                        'metadata': chunk.metadata,
                        'bm25_score': float(temp_scores[temp_idx]),
                        'index': int(original_idx)
                    })
        
        return results
    
    def _combine_and_rank_results(self,
                                  vector_results: List[Dict[str, Any]],
                                  bm25_results: List[Dict[str, Any]],
                                  top_k: int) -> List[Dict[str, Any]]:
        """
        Combine vector and BM25 results, normalize scores, and rank by hybrid score.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of results to return
            
        Returns:
            Combined and ranked results
        """
        
        if vector_results:
            vector_scores = [r['vector_score'] for r in vector_results]
            min_vec = min(vector_scores)
            max_vec = max(vector_scores)
            vec_range = max_vec - min_vec if max_vec != min_vec else 1.0
            
            for result in vector_results:
                # Normalize: (score - min) / range
                result['vector_score_norm'] = (result['vector_score'] - min_vec) / vec_range
        else:
            vector_scores = []
        
        # Normalize BM25 scores to [0, 1]
        if bm25_results:
            bm25_scores = [r['bm25_score'] for r in bm25_results]
            min_bm25 = min(bm25_scores)
            max_bm25 = max(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
            
            for result in bm25_results:
                # Normalize: (score - min) / range
                result['bm25_score_norm'] = (result['bm25_score'] - min_bm25) / bm25_range
        else:
            bm25_scores = []
            
            
            
            
        # Combine results and calculate hybrid scores
        # Create a dictionary to merge results by index
        combined_results: Dict[int, Dict[str, Any]] = {}
        
        # Add vector search results
        for result in vector_results:
            idx = result['index']
            combined_results[idx] = result.copy()
            combined_results[idx]['bm25_score_norm'] = 0.0  # Default if not found in BM25
        
        # Add/update with BM25 search results
        for result in bm25_results:
            idx = result['index']
            if idx in combined_results:
                combined_results[idx]['bm25_score_norm'] = result['bm25_score_norm']
            else:
                combined_results[idx] = result.copy()
                combined_results[idx]['vector_score_norm'] = 0.0  # Default if not found in vector
        
        # Calculate hybrid scores using alpha parameter
        # hybrid_score = alpha * vector_score_norm + (1 - alpha) * bm25_score_norm
        for result in combined_results.values():
            vector_norm = result.get('vector_score_norm', 0.0)
            bm25_norm = result.get('bm25_score_norm', 0.0)
            result['hybrid_score'] = self.alpha * vector_norm + (1 - self.alpha) * bm25_norm
            result['score'] = result['hybrid_score']  # For compatibility
        
        # Sort by hybrid score and return top_k
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:top_k]
        
        return final_results
        
    
    def search(self, query_text: str,
               level: str,
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + BM25) on a specific level.
        
        Args:
            query_text: Query text
            level: Level to search ('metadata', 'summary', 'meeting')
            top_k: Number of results to return
            
        Returns:
            List of search results with hybrid scores
        """
        if level not in ['metadata', 'summary', 'meeting']:
            raise ValueError(f"Invalid level: {level}")
        
        vector_results = self._vector_search(query_text, level, top_k)
        bm25_results = self._bm25_search(query_text, level, top_k)
        
        
        return self._combine_and_rank_results(vector_results, bm25_results, top_k)
    
    
    
    def search_with_entry_ids_filter(self,
                                     query_text: str,
                                     level: str,
                                     entry_ids: Set[str],
                                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with entry ID filtering (for meeting level).
        
        Args:
            query_text: Query text
            level: Level to search (must be 'meeting')
            entry_ids: Set of entry IDs to filter by
            top_k: Number of results to return
            
        Returns:
            List of search results with hybrid scores
        """
        
        if level != 'meeting':
            raise ValueError(f"Entry ID filtering is only supported for 'meeting' level, got '{level}'")
        
        if not entry_ids:
            return []
        
        relevant_indices = set()
        for entry_id in entry_ids:
            relevant_indices.update(
                self.entry_id_to_meeting_indices.get(entry_id, set())
            )
            
        if not relevant_indices:
            print(f"Warning: no relevant ID found for entry_ids: {entry_ids}")
            return []

        print(f"  Filtered to {len(relevant_indices)} relevant FAISS indices from {len(entry_ids)} entry_ids")    
        
        vector_results = self._vector_search(
            query_text, 
            level, 
            top_k, 
            allowed_indices=relevant_indices
        )
        bm25_results = self._bm25_search(
            query_text, 
            level, 
            top_k, 
            allowed_indices=relevant_indices
        )
                
        return self._combine_and_rank_results(vector_results, bm25_results, top_k)
    
    
    def search_with_meeting_ids_filter(self,
                                       query_text: str,
                                       level: str,
                                       meeting_ids: Optional[Set[str]] = None,
                                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with optional meeting ID filtering.
        
        Args:
            query_text: Query text
            level: Level to search (only 'summary' supported)
            meeting_ids: Optional set of meeting IDs to filter. If None or empty, searches all
            top_k: Number of results to return
            
        Returns:
            List of search results with hybrid scores
        """
        if level != 'summary':
            raise ValueError(f"Meeting ID filtering is only supported for 'summary' level, got '{level}'")
        
        # If meeting_ids is None or empty, search all documents
        if not meeting_ids:
            print("  No meeting IDs provided, searching all documents")
            vector_results = self._vector_search(query_text, level, top_k)
            bm25_results = self._bm25_search(query_text, level, top_k)
            return self._combine_and_rank_results(vector_results, bm25_results, top_k)
        
        relevant_indices = set()
        for meeting_id in meeting_ids:
            relevant_indices.update(
                self.meeting_id_to_summary_indices.get(meeting_id, set())
            )
        
        # If no relevant indices found, search all documents to ensure RAG continues smoothly
        if not relevant_indices:
            print(f"  Warning: no relevant indices found for meeting_ids: {meeting_ids}, searching all documents")
            vector_results = self._vector_search(query_text, level, top_k)
            bm25_results = self._bm25_search(query_text, level, top_k)
            return self._combine_and_rank_results(vector_results, bm25_results, top_k)
        
        print(f"  Filtered to {len(relevant_indices)} relevant indices from {len(meeting_ids)} meeting_ids")
        
        vector_results = self._vector_search(
            query_text, 
            level, 
            top_k, 
            allowed_indices=relevant_indices
        )
        bm25_results = self._bm25_search(
            query_text, 
            level, 
            top_k, 
            allowed_indices=relevant_indices
        )
        
        return self._combine_and_rank_results(vector_results, bm25_results, top_k)
            
