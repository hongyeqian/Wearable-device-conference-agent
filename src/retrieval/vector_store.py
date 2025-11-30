from typing import List, Dict, Any, Optional, Set
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import re
import pickle
from pathlib import Path

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.chunking.chunker import ChunkMetadata
from src.embeddings.generator import EmbeddingGenerator
from config.settings import EMBEDDING_MODEL


class HybridSearchVectorStore:
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator]= None, alpha: float =0.6):
        
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        self.alpha = alpha
        
        # faiss indice for each level
        self.faiss_indices: Dict[str, faiss.Index] ={}
        
        # BM25 indices for each level
        self.bm25_indices: Dict[str, BM25Okapi] ={}
        
        
        # mapping back to the chunkmetadata
        self.doc_mapping: Dict[str, Dict[int, ChunkMetadata]] = {
            'metadata':{},
            'summary':{},
            'meeting': {}
        }
        
        self.doc_text_tokenized: Dict[str, List[List[str]]] = {
            'metadata': [],
            'summary': [],
            'meeting': []
        }
        
        self.meeting_id_to_summary_indices: Dict[str, Set[int]] = {}
        self.entry_id_to_meeting_indices: Dict[str, Set[int]] = {}
        
    def _tokenize(self, text: str):
        # lowercase, split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
        
        
    def add_chunks(self, chunks: List[ChunkMetadata],
                    Level: str,
                    generate_embedding: bool):
        if Level not in ['metadata', 'summary', 'meeting']:
            raise ValueError(f"Invalid level: {Level}")
        
        if not chunks:
            print(f"No chunks in the {Level}")
            return
            
            
        print(f"Adding {len(chunks)} chunks to {Level} level")
        
        
        if generate_embedding:
            print(f"Generate embeddings...")
            embeddings = self.embedding_generator.generate_chunks_embeddings(chunks)
        else:
            raise ValueError("Embedding must be generated for FAISS index")
        
        embedding_array = np.array([emb.tolist() for emb in embeddings]).astype('float32')
        
        
        if Level not in self.faiss_indices:
            faiss.normalize_L2(embedding_array)
            
            # Create IVF index with IndexFlatIP as quantizer
            nlist = min(100, max(1, len(embedding_array) // 10))  # nlist: number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            
            # Train the index before adding vectors
            if len(embedding_array) >= nlist:
                index.train(embedding_array)  # type: ignore
            else:
                # If not enough vectors for training, use all vectors
                index.train(embedding_array)  # type: ignore
            
            index.add(embedding_array)  # type: ignore
            self.faiss_indices[Level] = index
            print(f"Created FAISS IVF index for {Level} level (nlist={nlist})")
            
            
        else:
            faiss.normalize_L2(embedding_array)
            self.faiss_indices[Level].add(embedding_array)  # type: ignore
            print(f"Updated FAISS index for {Level} level")
            
        # store FAISS mapping
        current_size = len(self.doc_mapping[Level])
        for idx, chunk in enumerate(chunks):
            global_idx = current_size + idx
            self.doc_mapping[Level][current_size+idx] = chunk
            
            
        # Building meeting_id -> summary_indice mapping
            if Level=='summary':
                meeting_id = chunk.meeting_id
                if meeting_id:
                    if meeting_id not in self.meeting_id_to_summary_indices:
                        self.meeting_id_to_summary_indices[meeting_id] = set()
                    self.meeting_id_to_summary_indices[meeting_id].add(global_idx)
                    
                    
            elif Level == 'meeting':
                entry_id = chunk.metadata.get('entry_id')
                if entry_id:
                    if entry_id not in self.entry_id_to_meeting_indices:
                        self.entry_id_to_meeting_indices[entry_id] = set()
                    self.entry_id_to_meeting_indices[entry_id].add(global_idx)
            
        # build BM25 mapping
        tokenized_docs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tokenized_docs.append(tokens)
            self.doc_text_tokenized[Level].append(tokens)
            
        if tokenized_docs:
            if Level not in self.bm25_indices:
                self.bm25_indices[Level] = BM25Okapi(tokenized_docs)
                print(f" Created BM25 index for {Level} level")
            else:
                # BM25 doesn't support incremental updates easily, rebuild
                # For now, we'll rebuild the entire index
                all_tokenized = self.doc_text_tokenized[Level]
                self.bm25_indices[Level] = BM25Okapi(all_tokenized)
                print(f" Rebuilt BM25 index for {Level} level")
        
        print(f"  Successfully indexed {len(chunks)} {Level} chunks")
        
        
        
    def _vector_search(self,
                      query_text: str,
                      level: str,
                      top_k: int,
                      allowed_indices: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search using FAISS (STEP 2A & 3A in diagram)
        
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
        
        # Use SearchParametersIVF with IDSelectorArray for filtering
        if allowed_indices is not None and len(allowed_indices) > 0:
            allowed_ids = np.array(sorted(allowed_indices), dtype=np.int64)
            id_selector = faiss.IDSelectorArray(len(allowed_ids), faiss.swig_ptr(allowed_ids))
            
            # Extract IVF index and use SearchParametersIVF
            ivf_index = faiss.extract_index_ivf(index)  # type: ignore
            search_params = faiss.SearchParametersIVF()  # type: ignore
            search_params.sel = id_selector  # type: ignore
            
            search_k = min(top_k, len(allowed_ids), index.ntotal)
            print(f"searck_k: {search_k} numbers of search k")
            scores, indices = ivf_index.search(query_vector, search_k, params=search_params)  # type: ignore
        else:
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
        Perform BM25 keyword search
        
        Args:
            query_text: Query text
            level: Level to search
            top_k: Number of results to return
            
        Returns:
            List of results with chunk info and BM25 scores
        """
        if level not in self.bm25_indices:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query_text)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        bm25_scores = self.bm25_indices[level].get_scores(query_tokens)
        
        # # Get top_k indices
        # top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Filter scores by allowed_indices if provided
        if allowed_indices is not None and len(allowed_indices) > 0:
            effective_top_k = min(top_k, len(allowed_indices))
            # Create a filtered score array
            filtered_scores = np.full(len(bm25_scores), -np.inf)
            for idx in allowed_indices:
                if 0 <= idx < len(bm25_scores):
                    filtered_scores[idx] = bm25_scores[idx]
            
            # Get top_k indices from filtered scores
            top_indices = np.argsort(filtered_scores)[::-1][:effective_top_k]
        else:
            # Get top_k indices from all scores
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if bm25_scores[idx] <= 0:  # Skip zero scores
                continue
            
            # Find corresponding chunk (idx in BM25 corresponds to position in doc_mappings)
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
        
        return results
    
    def _combine_and_rank_results(self,
                              vector_results: List[Dict[str, Any]],
                              bm25_results: List[Dict[str, Any]],
                              top_k: int) -> List[Dict[str, Any]]:
        
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
        
    
    def search(self, query_text:str,
               Level: str,
               top_k: int=5) -> List[Dict[str, Any]]: # type: ignore
        if Level not in ['metadata', 'summary', 'meeting']:
            raise ValueError(f"Invalid level: {Level}")
        
        vector_results = self._vector_search(query_text, Level, top_k)
        bm25_results = self._bm25_search(query_text, Level, top_k)
        
        
        return self._combine_and_rank_results(vector_results, bm25_results, top_k)
    
    
    
    def search_with_entry_ids_filter(self,
                                     query_text: str,
                                     level: str,
                                     entry_ids: Set[str],
                                     top_k: int=5) -> List[Dict[str, Any]]:
        
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
            print(f"Warning: no relevant id found for entry_ids: {entry_ids}")
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
                                   meeting_ids: Set[str],
                                   top_k: int = 5) -> List[Dict[str, Any]]:

        if level != 'summary':
            raise ValueError(f"Meeting ID filtering is only supported for 'summary' level, got '{level}'")
        
        if not meeting_ids:
            return []
        
        relevant_indices = set()
        for meeting_id in meeting_ids:
            relevant_indices.update(
                self.meeting_id_to_summary_indices.get(meeting_id, set())
            )
        
        if not relevant_indices:
            print(f"Warning: no relevant id found for meeting_ids: {meeting_ids}")
            return []
        
        print(f"  Filtered to {len(relevant_indices)} relevant FAISS indices from {len(meeting_ids)} meeting_ids")
        
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
            
        
            
    
    
    
    
        
        
        
    def search_multi_query(
        self,
        original_query: str,
        query_rewrite: Dict[str, Any],
        level: str,
        top_k_vector: int = 5,
        top_k_bm25: int = 20,
        num_paraphrases: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Multi-query search with query rewriting:
        1. Vector search: original_query + normalized_query + paraphrases (top_k_vector each)
        2. BM25 search: people + keywords (top_k_bm25)
        3. Merge by chunk_id, keep max vector score, combine with BM25 score
        
        Args:
            original_query: Original user query
            query_rewrite: Query rewrite result from QueryRewriter
            level: Level to search ('metadata', 'summary', 'meeting')
            top_k_vector: Top k for each vector query (default: 5)
            top_k_bm25: Top k for BM25 keyword query (default: 20)
            num_paraphrases: Number of paraphrases to use (default: 2)
            
        Returns:
            List of results with score_vector and score_bm25, sorted by hybrid score
        """
        import random
        
        if level not in ['metadata', 'summary', 'meeting']:
            raise ValueError(f"Invalid level: {level}")
        
        # ========== Step 1: Vector Multi-Query Search ==========
        vector_queries = []
        
        # Add original query
        vector_queries.append(original_query)
        
        # Add normalized query
        normalized_query = query_rewrite.get('normalized_query', '')
        if normalized_query:
            vector_queries.append(normalized_query)
        
        # Add paraphrases (randomly select num_paraphrases)
        paraphrases = query_rewrite.get('paraphrases', [])
        if paraphrases:
            selected_paraphrases = random.sample(
                paraphrases, 
                min(num_paraphrases, len(paraphrases))
            )
            vector_queries.extend(selected_paraphrases)
        
        print(f"  Vector multi-query: {len(vector_queries)} queries")
        for i, q in enumerate(vector_queries, 1):
            print(f"    {i}. {q[:80]}...")
        
        # Perform vector search for each query
        all_vector_results = []
        for query in vector_queries:
            results = self._vector_search(query, level, top_k_vector)
            all_vector_results.extend(results)
        
        # Merge vector results by chunk_id, keep max vector_score
        vector_results_by_chunk: Dict[str, Dict[str, Any]] = {}
        for result in all_vector_results:
            chunk_id = result['chunk_id']
            vector_score = result['vector_score']
            
            if chunk_id not in vector_results_by_chunk:
                vector_results_by_chunk[chunk_id] = result.copy()
            else:
                # Keep the result with highest vector_score
                if vector_score > vector_results_by_chunk[chunk_id]['vector_score']:
                    vector_results_by_chunk[chunk_id] = result.copy()
        
        vector_candidates = list(vector_results_by_chunk.values())
        print(f"  Vector search: {len(all_vector_results)} raw results -> {len(vector_candidates)} unique chunks")
        
        # ========== Step 2: BM25 Keyword Search ==========
        # Build keyword query from people + keywords
        entities = query_rewrite.get('entities', {})
        people = entities.get('people', [])
        keywords = entities.get('keywords', [])
        
        # Combine people and keywords
        keyword_terms = people + keywords
        keyword_query = ' '.join(keyword_terms)
        
        print(f"  BM25 keyword query: '{keyword_query}'")
        
        # Perform BM25 search
        bm25_results = self._bm25_search(keyword_query, level, top_k_bm25)
        print(f"  BM25 search: {len(bm25_results)} results")
        
        # ========== Step 3: Merge Results ==========
        # Create a dictionary to merge by chunk_id
        merged_results: Dict[str, Dict[str, Any]] = {}
        
        # Add vector results
        for result in vector_candidates:
            chunk_id = result['chunk_id']
            merged_results[chunk_id] = {
                'chunk_id': chunk_id,
                'meeting_id': result.get('meeting_id'),
                'level': level,
                'text': result.get('text'),
                'metadata': result.get('metadata'),
                'score_vector': result['vector_score'],  # Raw vector score
                'score_bm25': 0.0,  # Default, will be updated if found in BM25
                'index': result.get('index')
            }
        
        # Add/update with BM25 results
        for result in bm25_results:
            chunk_id = result['chunk_id']
            bm25_score = result['bm25_score']
            
            if chunk_id in merged_results:
                # Update existing result with BM25 score
                merged_results[chunk_id]['score_bm25'] = bm25_score
            else:
                # Add new result from BM25 only
                merged_results[chunk_id] = {
                    'chunk_id': chunk_id,
                    'meeting_id': result.get('meeting_id'),
                    'level': level,
                    'text': result.get('text'),
                    'metadata': result.get('metadata'),
                    'score_vector': 0.0,  # Default, not found in vector search
                    'score_bm25': bm25_score,
                    'index': result.get('index')
                }
        
        # Normalize scores and calculate hybrid score
        all_vector_scores = [r['score_vector'] for r in merged_results.values() if r['score_vector'] > 0]
        all_bm25_scores = [r['score_bm25'] for r in merged_results.values() if r['score_bm25'] > 0]
        
        # Normalize vector scores
        if all_vector_scores:
            min_vec = min(all_vector_scores)
            max_vec = max(all_vector_scores)
            vec_range = max_vec - min_vec if max_vec != min_vec else 1.0
        else:
            vec_range = 1.0
        
        # Normalize BM25 scores
        if all_bm25_scores:
            min_bm25 = min(all_bm25_scores)
            max_bm25 = max(all_bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
        else:
            bm25_range = 1.0
        
        # Calculate hybrid scores
        final_results = []
        for result in merged_results.values():
            # Normalize scores
            if result['score_vector'] > 0:
                result['score_vector_norm'] = (result['score_vector'] - min_vec) / vec_range
            else:
                result['score_vector_norm'] = 0.0
            
            if result['score_bm25'] > 0:
                result['score_bm25_norm'] = (result['score_bm25'] - min_bm25) / bm25_range
            else:
                result['score_bm25_norm'] = 0.0
            
            # Calculate hybrid score
            result['hybrid_score'] = (
                self.alpha * result['score_vector_norm'] + 
                (1 - self.alpha) * result['score_bm25_norm']
            )
            result['score'] = result['hybrid_score']  # For compatibility
        
        # Sort by hybrid score
        final_results = sorted(
            merged_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        print(f"  Merged results: {len(final_results)} total chunks")
        print(f"    - Vector only: {sum(1 for r in final_results if r['score_bm25'] == 0)}")
        print(f"    - BM25 only: {sum(1 for r in final_results if r['score_vector'] == 0)}")
        print(f"    - Both: {sum(1 for r in final_results if r['score_vector'] > 0 and r['score_bm25'] > 0)}")
        
        return final_results
            
        
            
    
    
    
    
        
        
        