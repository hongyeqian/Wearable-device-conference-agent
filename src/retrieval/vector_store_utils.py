"""
Utility functions for HybridSearchVectorStore
Provides save/load, statistics, and clear functionality
"""
from typing import Dict, Any, Optional, Set
from pathlib import Path
import pickle
import faiss

from src.retrieval.vector_store import HybridSearchVectorStore
from config.settings import EMBEDDING_MODEL


class VectorStoreUtilsMixin(HybridSearchVectorStore):
    """
    Extended HybridSearchVectorStore with utility methods
    Provides save/load, statistics, and clear functionality
    """
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about the indexed data for each level
        
        Returns:
            Dictionary with stats for each level:
            {
                'metadata': {
                    'doc_count': int,
                    'faiss_index_size': int,
                    'has_bm25': bool,
                    'tokenized_docs_count': int
                },
                ...
            }
        """
        stats = {}
        for level in ['metadata', 'summary', 'meeting']:
            doc_count = len(self.doc_mapping[level])
            faiss_size = self.faiss_indices[level].ntotal if level in self.faiss_indices else 0
            has_bm25 = level in self.bm25_indices
            
            stats[level] = {
                'doc_count': doc_count,
                'faiss_index_size': faiss_size,
                'has_bm25': has_bm25,
                'tokenized_docs_count': len(self.doc_text_tokenized[level])
            }
        
        return stats

    def get_processed_meeting_ids(self) -> Set[str]:
        """
        Get set of meeting IDs that have been processed and stored in the vector store.

        Returns:
            Set of processed meeting IDs
        """
        processed_meeting_ids = set()

        #extract meeting ids from summary level
        processed_meeting_ids.update(self.meeting_id_to_summary_indices.keys())

        # can also extract meeting ids from metadata and meeting level, but summary level should be enough
        return processed_meeting_ids

    def clear(self, level: Optional[str] = None):
        """
        Clear indices for a specific level or all levels
        
        Args:
            level: Level to clear ('metadata', 'summary', 'meeting'). 
                  If None, clears all levels
        """
        if level:
            if level not in ['metadata', 'summary', 'meeting']:
                raise ValueError(f"Invalid level: {level}")
            
            # Clear FAISS index
            if level in self.faiss_indices:
                del self.faiss_indices[level]
            
            # Clear BM25 index
            if level in self.bm25_indices:
                del self.bm25_indices[level]
            
            # Clear mappings
            self.doc_mapping[level] = {}
            self.doc_text_tokenized[level] = []
            
            print(f"✅ Cleared all indices for {level} level")
        else:
            # Clear all levels
            self.faiss_indices.clear()
            self.bm25_indices.clear()
            for level_key in ['metadata', 'summary', 'meeting']:
                self.doc_mapping[level_key] = {}
                self.doc_text_tokenized[level_key] = []
            print("✅ Cleared all indices for all levels")
    
    def save(self, save_dir: Path):
        """
        Save all indices and mappings to disk
        
        Args:
            save_dir: Directory path to save the indices
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving indices to {save_dir}...")
        
        # Save FAISS indices
        faiss_dir = save_dir / "faiss"
        faiss_dir.mkdir(exist_ok=True)
        
        for level, index in self.faiss_indices.items():
            faiss_file = faiss_dir / f"{level}.index"
            faiss.write_index(index, str(faiss_file))
            print(f"  ✅ Saved FAISS index for {level} level")
        
        # Save BM25 indices, doc_mapping, and doc_text_tokenized using pickle
        pickle_dir = save_dir / "pickle"
        pickle_dir.mkdir(exist_ok=True)
        
        # Save BM25 indices
        bm25_file = pickle_dir / "bm25_indices.pkl"
        with open(bm25_file, 'wb') as f:
            pickle.dump(self.bm25_indices, f)
        print(f"  ✅ Saved BM25 indices")
        
        # Save doc_mapping
        mapping_file = pickle_dir / "doc_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(self.doc_mapping, f)
        print(f"  ✅ Saved doc_mapping")
        
        # Save doc_text_tokenized
        tokenized_file = pickle_dir / "doc_text_tokenized.pkl"
        with open(tokenized_file, 'wb') as f:
            pickle.dump(self.doc_text_tokenized, f)
        print(f"  ✅ Saved doc_text_tokenized")

        # Save meeting_id_to_summary_indices
        meeting_indices_file = pickle_dir / "meeting_id_to_summary_indices.pkl"
        with open(meeting_indices_file, 'wb') as f:
            pickle.dump(self.meeting_id_to_summary_indices, f)
        print(f"  ✅ Saved meeting_id_to_summary_indices")

        # Save entry_id_to_meeting_indices
        entry_indices_file = pickle_dir / "entry_id_to_meeting_indices.pkl"
        with open(entry_indices_file, 'wb') as f:
            pickle.dump(self.entry_id_to_meeting_indices, f)
        print(f"  ✅ Saved entry_id_to_meeting_indices")
        
        # Save metadata (alpha, embedding_dim)
        metadata_file = pickle_dir / "metadata.pkl"
        metadata = {
            'alpha': self.alpha,
            'embedding_dim': self.embedding_dim,
            'embedding_model': EMBEDDING_MODEL,
            'faiss_nlist': self.faiss_nlist,
            'faiss_nprobe': self.faiss_nprobe
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  ✅ Saved metadata")
        
        print(f"✅ Successfully saved all indices to {save_dir}")
    
    @classmethod
    def load(cls, save_dir: Path, embedding_generator=None):
        """
        Load indices and mappings from disk
        
        Args:
            save_dir: Directory path where indices are saved
            embedding_generator: Optional EmbeddingGenerator instance.
                                If None, creates a new one
        
        Returns:
            VectorStoreUtilsMixin instance (extends HybridSearchVectorStore) with loaded indices
        """
        from src.embeddings.generator import EmbeddingGenerator
        
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            raise FileNotFoundError(f"Save directory not found: {save_dir}")
        
        print(f"Loading indices from {save_dir}...")
        
        # Load metadata first
        metadata_file = save_dir / "pickle" / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        if embedding_generator is None:
            embedding_generator = EmbeddingGenerator()
        
        instance = cls(embedding_generator=embedding_generator, alpha=metadata['alpha'])
        
        # Verify embedding dimension matches
        if instance.embedding_dim != metadata['embedding_dim']:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"loaded={metadata['embedding_dim']}, "
                f"current={instance.embedding_dim}"
            )
        
        # Load faiss_nlist and faiss_nprobe from metadata
        if 'faiss_nlist' in metadata:
            instance.faiss_nlist = metadata['faiss_nlist']
            print(f"  ✅ Loaded faiss_nlist: {instance.faiss_nlist}")
        
        if 'faiss_nprobe' in metadata:
            instance.faiss_nprobe = metadata['faiss_nprobe']
            print(f"  ✅ Loaded faiss_nprobe: {instance.faiss_nprobe}")
        
        # Load FAISS indices
        faiss_dir = save_dir / "faiss"
        if faiss_dir.exists():
            for level in ['metadata', 'summary', 'meeting']:
                faiss_file = faiss_dir / f"{level}.index"
                if faiss_file.exists():
                    index = faiss.read_index(str(faiss_file))
                    instance.faiss_indices[level] = index
                    print(f"  ✅ Loaded FAISS index for {level} level ({index.ntotal} vectors)")
        
        # Load BM25 indices, doc_mapping, and doc_text_tokenized
        pickle_dir = save_dir / "pickle"
        
        # Load BM25 indices
        bm25_file = pickle_dir / "bm25_indices.pkl"
        if bm25_file.exists():
            with open(bm25_file, 'rb') as f:
                instance.bm25_indices = pickle.load(f)
            print(f"  ✅ Loaded BM25 indices for {len(instance.bm25_indices)} levels")
        
        # Load doc_mapping
        mapping_file = pickle_dir / "doc_mapping.pkl"
        if mapping_file.exists():
            with open(mapping_file, 'rb') as f:
                instance.doc_mapping = pickle.load(f)
            total_docs = sum(len(mapping) for mapping in instance.doc_mapping.values())
            print(f"  ✅ Loaded doc_mapping ({total_docs} total documents)")
        
        # Load doc_text_tokenized
        tokenized_file = pickle_dir / "doc_text_tokenized.pkl"
        if tokenized_file.exists():
            with open(tokenized_file, 'rb') as f:
                instance.doc_text_tokenized = pickle.load(f)
            total_tokenized = sum(len(docs) for docs in instance.doc_text_tokenized.values())
            print(f"  ✅ Loaded doc_text_tokenized ({total_tokenized} documents)")

        # Load meeting_id_to_summary_indices
        meeting_indices_file = pickle_dir / "meeting_id_to_summary_indices.pkl"
        if meeting_indices_file.exists():
            with open(meeting_indices_file, 'rb') as f:
                instance.meeting_id_to_summary_indices = pickle.load(f)
            print(f"  ✅ Loaded meeting_id_to_summary_indices ({len(instance.meeting_id_to_summary_indices)} meeting IDs)")

        # Load entry_id_to_meeting_indices
        entry_indices_file = pickle_dir / "entry_id_to_meeting_indices.pkl"
        if entry_indices_file.exists():
            with open(entry_indices_file, 'rb') as f:
                instance.entry_id_to_meeting_indices = pickle.load(f)
            print(f"  ✅ Loaded entry_id_to_meeting_indices ({len(instance.entry_id_to_meeting_indices)} entry IDs)")
        
        print(f"✅ Successfully loaded all indices from {save_dir}")
        
        return instance
    
    
