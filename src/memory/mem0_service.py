import logging
import json
from pathlib import Path
from mem0 import Memory
# pyrefly: ignore [missing-import]
from config.settings import DATA_DIR, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class Mem0MemoryService:
    """
    Mem0 Long-Term Memory Service.
    Responsible for extracting high-density user preferences and facts from conversation summaries.
    Currently uses the default Mem0 extraction prompt to evaluate baseline extraction quality.
    """
    def __init__(self):
        # Store ChromaDB files locally in the datademo/.mem0_store directory
        chroma_path = str(Path(DATA_DIR) / ".mem0_store")
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "meeting_rag_memory",
                    "path": chroma_path
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": OPENAI_API_KEY
                }
            }
        }
        # Initialize Mem0; it will automatically connect to ChromaDB and prepare the LLM
        self.memory = Memory.from_config(config)

    def _sanitize_id(self, user_id: str) -> str:
        """Mem0 does not allow whitespace in user_id, so we replace them with underscores."""
        return user_id.replace(" ", "_")

    def add_session_memories(self, user_id: str, summary_text: str):
        sanitized_id = self._sanitize_id(user_id)
        logger.info(f"[Mem0] Background extraction started for user: {sanitized_id}")
        try:
            self.memory.add(summary_text, user_id=sanitized_id)
            logger.info(f"[Mem0] Extraction complete for user: {sanitized_id}")
            
            # --- 实时导出为实体文件，方便监控和调试 ---
            results = self.memory.get_all(filters={"user_id": sanitized_id})
            
            # 存放到你创建的 long_term_memory 文件夹下
            export_dir = Path("long_term_memory")
            export_dir.mkdir(parents=True, exist_ok=True)
            export_file = export_dir / f"{sanitized_id}_memory.json"
            
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"[Mem0] Successfully exported memory to {export_file}")
            
        except Exception as e:
            logger.error(f"[Mem0] Extraction or export failed: {e}")

    def search_memories(self, user_id: str, query: str) -> str:
        """
        Search the user's long-term memory in Mem0.
        (TODO: Currently not actively injected into agents; implemented for manual testing and future Phase 2 integration).
        """
        sanitized_id = self._sanitize_id(user_id)
        results = self.memory.search(query, user_id=sanitized_id)
        if not results:
            return ""
        
        # Format memory items as a bulleted list
        memory_lines = []
        for item in results:
            memory_text = item.get("memory", "")
            if memory_text:
                memory_lines.append(f"- {memory_text}")
        
        return "\n".join(memory_lines)

# Global singleton instance
mem0_service = Mem0MemoryService()
