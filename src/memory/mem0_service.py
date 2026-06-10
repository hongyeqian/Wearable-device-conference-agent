import logging
import json
from pathlib import Path
from mem0 import Memory
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal
# pyrefly: ignore [missing-import]
from config.settings import DATA_DIR, OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

class MemoryItem(BaseModel):
    text: str
    category: Literal["system", "interesting"]

class MemoryExtraction(BaseModel):
    memories: List[MemoryItem]

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

    def _extract_memories_with_custom_prompt(self, summary_text: str, existing_memories) -> list[str]:
        """
        Use OpenAI API to extract memories based on Omi's strict categorization rules.
        """
        # Safely extract the list if existing_memories is a dict like {"memories": [...]}
        existing_list = []
        if isinstance(existing_memories, dict):
            existing_list = existing_memories.get("results") or existing_memories.get("memories") or []
        elif isinstance(existing_memories, list):
            existing_list = existing_memories
            
        existing_memories_str = "\n".join([item.get("memory", item.get("text", "")) if isinstance(item, dict) else str(item) for item in existing_list]) if existing_list else "None"
        
        system_prompt = f"""
You are an expert memory extractor. Extract core, high-value, long-term memories about the user from the provided meeting summary.

Existing Memories (use for deduplication and updates, NEVER extract duplicate content):
{existing_memories_str}

Extraction Rules:
1. Extract a MAXIMUM of 2 memories. If no valuable information is found, output an empty array.
2. Each memory MUST be extremely concise, 15 words maximum.
3. Extract ONLY the following two categories:
   - "system": Factual information about the user (e.g., core identity, preferences, habits, relationships).
   - "interesting": Valuable insights or advice the user received from others (must include attribution, e.g., "John: suggested reading more code").
4. ABSOLUTELY DO NOT EXTRACT:
   - "User discussed/mentioned/talked about..."
   - Vague statements containing "possibly", "likely", or "seems".
   - Transient or short-term states using "is working on" or "is building".
   - News, general knowledge, or product announcements.
5. Deduplication: If the semantic meaning matches an existing memory, SKIP it. If it is an update or reversal of an existing memory, extract the updated fact.
"""
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.beta.chat.completions.parse(
                model=OPENAI_MODEL or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Meeting Summary:\n{summary_text}"}
                ],
                response_format=MemoryExtraction
            )
            parsed = response.choices[0].message.parsed
            if not parsed:
                return []
            return [m.text for m in parsed.memories if m.text]
        except Exception as e:
            logger.error(f"[Mem0] Custom extraction failed: {e}")
            return []

    def add_session_memories(self, user_id: str, summary_text: str):
        sanitized_id = self._sanitize_id(user_id)
        logger.info(f"[Mem0] Background extraction started for user: {sanitized_id}")
        try:
            # 1. Fetch existing memories to avoid duplication
            existing_results = self.memory.get_all(filters={"user_id": sanitized_id})
            
            # 2. Extract with custom strict prompt
            memories_to_add = self._extract_memories_with_custom_prompt(summary_text, existing_results)
            
            # 3. Add to mem0 if any
            if memories_to_add:
                for mem in memories_to_add:
                    self.memory.add(mem, user_id=sanitized_id)
                logger.info(f"[Mem0] Extraction complete for user: {sanitized_id}, added {len(memories_to_add)} memories.")
            else:
                logger.info(f"[Mem0] Extraction complete for user: {sanitized_id}, no high-value memories found.")
            
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

    def search_memories(self, user_id: str, query: str, limit: int = 20) -> str:
        """
        Search the user's long-term memory in Mem0.
        """
        sanitized_id = self._sanitize_id(user_id)
        # Mem0 search uses filters instead of top-level user_id now
        results = self.memory.search(query, filters={"user_id": sanitized_id}, limit=limit)
        if not results:
            return ""
        
        # Format memory items as a bulleted list
        memory_lines = []
        
        results_list = []
        if isinstance(results, dict):
            results_list = results.get("results") or results.get("memories") or []
        elif isinstance(results, list):
            results_list = results
            
        for item in results_list:
            if isinstance(item, dict):
                memory_text = item.get("memory", item.get("text", ""))
            else:
                memory_text = str(item)
            if memory_text:
                memory_lines.append(f"- {memory_text}")
        
        return "\n".join(memory_lines)

# Global singleton instance
mem0_service = Mem0MemoryService()
