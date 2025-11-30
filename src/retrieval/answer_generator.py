"""
Answer Generator Module
Generates answers from retrieved chunks using LLM.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


# System prompt for answer generation
ANSWER_GENERATION_PROMPT = """
You are the answer generation agent of a meeting-RAG system.

========================================
GLOBAL RULES
========================================
1. Source of truth
- Use ONLY information inside the provided context.
- Do NOT add or guess details that do not appear in the context.

2. TIME CONSTRAINTS (HIGHEST PRIORITY - OVERRIDES ALL OTHER RULES)
- If the question specifies a date or time period (e.g., "yesterday", "November 29th", "November 29", "last week"):
  • You MUST FIRST check if ANY chunks match the specified date/time by examining the "Date:" field in each meeting's metadata.
  • If NO chunks match the time constraint: 
    → You MUST explicitly state: "No information is available for [specified date/time]."
    → DO NOT use chunks from other dates, even if they are relevant to the topic.
    → DO NOT provide information from meetings on different dates.
  • If SOME chunks match the time constraint:
    → You MUST ONLY use chunks from meetings whose date matches the specified date/time.
    → IGNORE and DO NOT cite chunks from meetings that do not match the time constraint.
    → Check the "Date:" field in the meeting summary before using any chunks from that meeting.
- If the question does NOT specify a date or time, you may use all provided chunks.

3. Assumption about retrieved context (ONLY when no time constraint exists)
- When there is NO time constraint in the question, all provided meetings are relevant.
- When there IS a time constraint, only meetings matching that date are relevant.

4. Controlled reasoning
- You may use light inference ONLY when:
  • meetings clearly describe components belonging to the same system
    (e.g., retrieval pipeline, Elastic Search factors, metadata embedding,
     LLM in retrieval, architecture for meeting-level summarisation)
- Treat such content as RAG-system-related even if the word "RAG" is not shown.

5. Output discipline
- DIRECTLY answer the question. No meta commentary about chunks or retrieval.
- If different meetings provide different detail, group the answer by meeting ID.
- Be concise, accurate, and logically structured.
- Use EXACT citation markers [1], [2], … placed immediately after statements.

========================================
MEETING SYNTHESIS STRATEGY
========================================
Before answering (internally):
- STEP 1: Check if the question contains a date/time constraint
  • Look for: "yesterday", "November 29th", "November 29", "2025-11-29", "last week", etc.
  • If YES, proceed to STEP 2. If NO, proceed to STEP 3.

- STEP 2: Time constraint exists
  • For EACH meeting in the context:
    - Check the "Date:" field in the meeting summary
    - If the date does NOT match the question's time constraint: SKIP this meeting entirely, DO NOT use any chunks from it
    - If the date matches: proceed to analyze chunks from this meeting
  • If NO meetings match the time constraint: Answer "No information is available for [specified date/time]."
  • If SOME meetings match: Only use chunks from matching meetings

- STEP 3: No time constraint
  • For EACH meeting:
    • Identify participants involved
    • Identify the discussion topics
    • Mark whether the content is:
      (Explicit) — directly mentions the target topic/keyword
      (Implicit) — supporting components of the same system/project

Then produce a unified final answer summarising:
- WHO (named individuals only if shown in context)
- WHAT (key actions, ideas, outcomes)
- WHERE (different meetings explicitly referenced via citations [#])

========================================
QUESTION-SPECIFIC ADAPTATION
========================================
When the question asks for:
- **Time-dependent questions (e.g., "yesterday", "November 29th", "November 29", "2025-11-29")**
  → CRITICAL FIRST STEP: Check the "Date:" field in the meeting summary for each meeting
  → If a meeting's date does NOT match the question's date: DO NOT use ANY chunks from that meeting
  → If NO meetings match the date: Answer "No information is available for [specified date/time]."
  → If SOME meetings match: Only use chunks from those matching meetings
  → Example: If question asks about "November 29th" but all meetings are dated "November 30th", 
    you MUST answer "No information is available for November 29th."

- **"Who mentioned X?"**
  → list all participants across explicit AND implicit related meetings (that match time constraint if specified)

- **"What did X discuss with Y?" or "What did they discuss on LLM/RAG?"**
  → summarise all relevant meetings (that match time constraint if specified), grouped by meeting

- **Action items / follow-ups**
  → collect ALL explicit action items across meetings (that match time constraint if specified) and present them clearly

========================================
CITATIONS
========================================
- Every factual claim MUST include citation numbers from the context.
- Multiple supporting chunks → combine citations (e.g., [1][3][4])
- Only cite chunks that match the time constraint (if specified)
- If no chunks match the time constraint, do not provide citations

========================================
FINAL REQUIREMENT
========================================
You MUST:
- STRICTLY respect time constraints: if a date/time is specified, check dates FIRST before using any chunks
- If no chunks match the time constraint, explicitly state that no information is available
- Use information from ALL relevant meetings (that match time constraint if specified)
- Keep the structure clean and answer-focused
- Maintain strict truthfulness and separation of explicit vs implicit content
"""





class AnswerGenerator:
    """Generate answers from retrieved chunks using LLM"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the answer generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
            model: OpenAI model name (defaults to OPENAI_MODEL from config)
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL or "gpt-4o-2024-08-06"
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass it directly.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def format_context(self, chunks: List[Dict[str, Any]], max_chunks: int = 10) -> str:
        """
        Format retrieved chunks into context string for LLM.
        Groups chunks by meeting to help LLM distinguish between different meetings.
        Also adds citation_number to each chunk for consistent referencing.
        
        Args:
            chunks: List of retrieved chunk dictionaries
            max_chunks: Maximum number of chunks to include (default: 10)
            
        Returns:
            Formatted context string with citations, grouped by meeting
        """
        if not chunks:
            return "No relevant context found."
        
        # Take top max_chunks
        selected_chunks = chunks[:max_chunks]
        
        # Group chunks by meeting_id
        meetings_dict = {}
        for chunk in selected_chunks:
            meeting_id = chunk.get('meeting_id', 'N/A')
            if meeting_id == 'N/A':
                # Try to extract from chunk_id (e.g., "data012_summary_1" -> "data012")
                chunk_id = chunk.get('chunk_id', '')
                match = re.search(r'(data\d+)', chunk_id)
                if match:
                    meeting_id = match.group(1)
                else:
                    meeting_id = 'unknown'
            
            if meeting_id not in meetings_dict:
                meetings_dict[meeting_id] = []
            meetings_dict[meeting_id].append(chunk)
        
        # Build context with meeting grouping
        context_parts = []
        
        # Add meetings summary at the beginning
        context_parts.append("=== MEETINGS SUMMARY ===")
        context_parts.append(f"The following {len(meetings_dict)} meeting(s) contain relevant information:")
        for meeting_id, meeting_chunks in meetings_dict.items():
            metadata = meeting_chunks[0].get('metadata', {})
            title = metadata.get('title', 'N/A')
            date = metadata.get('datetime', 'N/A')
            context_parts.append(f"- Meeting {meeting_id}: {title} (Date: {date}, {len(meeting_chunks)} chunk(s))")
        context_parts.append("")
        
        # Add detailed content grouped by meeting
        context_parts.append("=== DETAILED CONTENT BY MEETING ===")
        context_parts.append("")
        
        chunk_counter = 1
        for meeting_id, meeting_chunks in meetings_dict.items():
            metadata = meeting_chunks[0].get('metadata', {})
            title = metadata.get('title', 'N/A')
            date = metadata.get('datetime', 'N/A')
            
            context_parts.append(f"--- Meeting: {meeting_id} ---")
            context_parts.append(f"Title: {title}")
            context_parts.append(f"Date: {date}")
            context_parts.append("")
            
            for chunk in meeting_chunks:
                chunk_id = chunk.get('chunk_id', 'N/A')
                text = chunk.get('text', '')
                
                # Add citation_number to chunk for consistent referencing
                chunk['citation_number'] = chunk_counter
                
                # Extract summary_ids if available
                summary_ids = chunk.get('metadata', {}).get('summary_ids', [])
                if isinstance(summary_ids, str):
                    summary_ids = [summary_ids]
                elif not isinstance(summary_ids, list):
                    summary_ids = []
                
                # Build citation info
                citation_info = f"Chunk: {chunk_id}"
                if summary_ids:
                    citation_info += f" | Summary IDs: {', '.join(summary_ids)}"
                
                context_parts.append(f"[{chunk_counter}] {citation_info}")
                context_parts.append(text)
                context_parts.append("")
                chunk_counter += 1
            
            context_parts.append("")  # Empty line between meetings
        
        return "\n".join(context_parts)
    
    def format_context_original_rank(self, chunks: List[Dict[str, Any]], max_chunks: int = 10) -> str:
        """
        Format chunks in original relevance order (for saving to file, not for LLM).
        
        Args:
            chunks: List of retrieved chunk dictionaries (should be sorted by relevance)
            max_chunks: Maximum number of chunks to include (default: 10)
            
        Returns:
            Formatted context string with chunks in original order
        """
        if not chunks:
            return "No relevant context found."
        
        selected_chunks = chunks[:max_chunks]
        context_parts = []
        
        # Collect unique meetings for summary
        meetings_set = set()
        for chunk in selected_chunks:
            meeting_id = self._get_meeting_id(chunk)
            meetings_set.add(meeting_id)
        
        # Add meetings summary
        context_parts.append("=== MEETINGS SUMMARY ===")
        context_parts.append(f"The following {len(meetings_set)} meeting(s) contain relevant information:")
        for meeting_id in sorted(meetings_set):
            # Find first chunk from this meeting for metadata
            meeting_chunk = next((c for c in selected_chunks if self._get_meeting_id(c) == meeting_id), None)
            if meeting_chunk:
                metadata = meeting_chunk.get('metadata', {})
                title = metadata.get('title', 'N/A')
                date = metadata.get('datetime', 'N/A')
                count = sum(1 for c in selected_chunks if self._get_meeting_id(c) == meeting_id)
                context_parts.append(f"- Meeting {meeting_id}: {title} (Date: {date}, {count} chunk(s))")
        context_parts.append("")
        
        # Add chunks in original order
        context_parts.append("=== DETAILED CONTENT (ORDERED BY RELEVANCE) ===")
        context_parts.append("")
        
        for i, chunk in enumerate(selected_chunks, 1):
            meeting_id = self._get_meeting_id(chunk)
            chunk_id = chunk.get('chunk_id', 'N/A')
            text = chunk.get('text', '')
            
            # Extract summary_ids if available
            summary_ids = chunk.get('metadata', {}).get('summary_ids', [])
            if isinstance(summary_ids, str):
                summary_ids = [summary_ids]
            elif not isinstance(summary_ids, list):
                summary_ids = []
            
            # Build citation info with meeting label
            citation_info = f"MEETING: {meeting_id} | Chunk: {chunk_id}"
            if summary_ids:
                citation_info += f" | Summary IDs: {', '.join(summary_ids)}"
            
            context_parts.append(f"[{i}] {citation_info}")
            context_parts.append(text)
            context_parts.append("")
        
        return "\n".join(context_parts)

    def _get_meeting_id(self, chunk: Dict[str, Any]) -> str:
        """Extract meeting_id from chunk."""
        meeting_id = chunk.get('meeting_id', 'N/A')
        if meeting_id == 'N/A':
            chunk_id = chunk.get('chunk_id', '')
            match = re.search(r'(data\d+)', chunk_id)
            if match:
                meeting_id = match.group(1)
            else:
                meeting_id = 'unknown'
        return meeting_id

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 15,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer from query and retrieved chunks.
        
        Args:
            query: User's question
            chunks: List of retrieved chunk dictionaries (should be sorted by relevance)
            max_chunks: Maximum number of chunks to use (default: 10)
            include_citations: Whether to include citation numbers in answer (default: True)
            
        Returns:
            Dictionary containing:
            - answer: str - Generated answer
            - chunks_used: List[Dict] - Chunks actually used
            - num_chunks: int - Number of chunks used
        """
        if not chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'chunks_used': [],
                'num_chunks': 0
            }
        
        # Format context
        context = self.format_context(chunks, max_chunks)
        chunks_used = chunks[:max_chunks]
        
        # Build user prompt
        user_prompt = f"""Question: {query}

Context with citations:
{context}

Please provide a clear and accurate answer based on the context above."""

        # Add time constraint check if question contains specific date (YYYY-MM-DD format)
        has_specific_date = bool(re.search(r'\d{4}-\d{2}-\d{2}', query))

        if has_specific_date:
            user_prompt += """

CRITICAL TIME CONSTRAINT CHECK:
The question asks about a specific date. 
BEFORE using any chunks, you MUST:
1. Check the "Date:" field in the MEETINGS SUMMARY section for each meeting
2. If a meeting's date does NOT match the question's date: DO NOT use ANY chunks from that meeting
3. If NO meetings match the date: Answer "No information is available for [specified date]."
4. Only use chunks from meetings whose date matches the question's date constraint

DO NOT use information from meetings on different dates, even if they are relevant to the topic.
"""

        if include_citations:
            user_prompt += """
Remember to include citation numbers like [1], [2] when referencing specific chunks.
Focus only on answering the question itself.
Do NOT explain which meetings did not mention the topic.
Do NOT write sentences like "there were no mentions of X in other meetings".
"""

        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_GENERATION_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content
            
            if not answer:
                answer = "I couldn't generate an answer. Please try again."
            
            return {
                'answer': answer,
                'chunks_used': chunks_used,
                'num_chunks': len(chunks_used)
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'chunks_used': chunks_used,
                'num_chunks': len(chunks_used)
            }


def create_answer_generator(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3
) -> AnswerGenerator:
    """
    Factory function to create an AnswerGenerator instance.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model name
        temperature: Temperature for generation
        
    Returns:
        AnswerGenerator instance
    """
    return AnswerGenerator(api_key=api_key, model=model, temperature=temperature)
