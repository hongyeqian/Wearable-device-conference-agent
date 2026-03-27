"""
Query Rewrite Agent - Single Agent Architecture
Uses three-layer Stage 3: Regex → spaCy Matcher → LLM fallback (single tool call)
"""
import logging
import sys
import json
import re
from pathlib import Path
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk.models import LiteLlm
from config.settings import OPENAI_API_KEY, CURRENT_USER, DATA_DIR
from config.meeting_patterns import MEETING_AMBIGUITY_PATTERNS, EMBEDDING_THRESHOLD

# Configure logging for query rewriter
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
import dateparser
import numpy as np
from sub_agents.pandas_utils import get_meetings_df

from sub_agents.date_resolver import DateResolver
from sub_agents.person_matcher import get_person_matcher

# Module-level spaCy model cache (loaded lazily or via warmup_all())
_spacy_nlp = None


# ============ Embedding Generator for Meeting Patterns ============

_embedding_generator = None

def get_embedding_generator():
    """Lazy initialize embedding generator for meeting patterns."""
    global _embedding_generator
    if _embedding_generator is not None:
        return _embedding_generator
    
    from src.embeddings.generator import EmbeddingGenerator
    _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


# ============ Stage 3 Layer 2: Embedding Threshold Detection ============

_pattern_embeddings = None

def _get_pattern_embeddings():
    """
    Pre-compute embeddings for all meeting ambiguity patterns.
    Uses lazy initialization.
    """
    global _pattern_embeddings
    if _pattern_embeddings is not None:
        return _pattern_embeddings
    
    generator = get_embedding_generator()
    embeddings = generator.generate_embeddings_batch(MEETING_AMBIGUITY_PATTERNS)
    _pattern_embeddings = np.array([emb.tolist() for emb in embeddings]).astype('float32')
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(_pattern_embeddings, axis=1, keepdims=True)
    _pattern_embeddings = _pattern_embeddings / norms
    
    return _pattern_embeddings


def check_embedding_threshold(query: str, threshold: float = EMBEDDING_THRESHOLD) -> tuple[bool, List[str]]:
    """
    Stage 3 - Layer 2: Embedding threshold detection
    
    - Uses embedding model to convert query to vector
    - Computes cosine similarity with preset meeting pattern library
    - If max similarity >= threshold: regex didn't fully resolve meeting ambiguity, returns matched patterns
    - If max similarity < threshold: no meeting ambiguity issue, returns empty list
    
    Args:
        query: Resolved query from Stage 2
        threshold: Similarity threshold (default from settings)
    
    Returns:
        (needs_further_processing, matched_patterns)
    """
    generator = get_embedding_generator()
    pattern_embeddings = _get_pattern_embeddings()
    
    # Generate embedding for query
    query_embedding = generator.generate_embedding(query)
    query_vector = np.array([query_embedding.tolist()]).astype('float32')
    
    # Normalize for cosine similarity
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # Compute cosine similarities
    similarities = np.dot(query_vector, pattern_embeddings.T)[0]
    
    # Find max similarity
    max_idx = np.argmax(similarities)
    max_similarity = similarities[max_idx]
    
    logger.info(f"[Stage3-Layer2-Embedding] Max similarity: {max_similarity:.3f} with pattern: '{MEETING_AMBIGUITY_PATTERNS[max_idx]}'")
    
    if max_similarity >= threshold:
        matched_pattern = MEETING_AMBIGUITY_PATTERNS[max_idx]
        logger.info(f"[Stage3-Layer2-Embedding] Threshold met, pattern matched: '{matched_pattern}'")
        return (True, [matched_pattern])
    
    logger.info(f"[Stage3-Layer2-Embedding] Threshold not met, no ambiguity detected")
    return (False, [])


# ============ Pydantic Models ============

class RewriteOutput(BaseModel):
    """Rewrite output structure - simplified format"""
    model_config = {"extra": "forbid"}
    
    rewritten_query: str = Field(description="Rewritten query with ambiguities resolved")
    relevant_meeting_ids: List[str] = Field(description="Filtered meeting ID list based on resolved ambiguities")


# ============ Tool Functions ============

# Global counter for pandas_query calls (enforced limit)
_pandas_query_call_count = 0
_PANDAS_QUERY_MAX_CALLS = 3

# Store last pandas_query result for extraction
_last_pandas_query_result: Optional[Dict[str, Any]] = None

def reset_pandas_query_counter():
    """Reset the pandas_query call counter - call this before each new query"""
    global _pandas_query_call_count, _last_pandas_query_result
    _pandas_query_call_count = 0
    _last_pandas_query_result = None

def get_last_pandas_query_result() -> Optional[Dict[str, Any]]:
    """Get the last pandas_query result (for extraction of meeting_ids)"""
    global _last_pandas_query_result
    return _last_pandas_query_result

def reset_counter() -> str:
    """
    Reset the pandas_query call counter.
    Call this at the start of each new query processing to ensure fresh counting.
    
    Returns:
        JSON formatted reset confirmation
    """
    reset_pandas_query_counter()
    return json.dumps({
        "status": "reset",
        "message": "pandas_query call counter has been reset to 0",
        "max_calls_per_query": _PANDAS_QUERY_MAX_CALLS
    })

def pandas_query(
    query_type: str,
    fuzzy_name: Optional[str] = None,
    n: Optional[int] = None,
    date_str: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    person_name: Optional[str] = None,
) -> str:
    """
    Pandas query tool for resolving ambiguities
    
    Args:
        query_type: Query type
            - "find_person": Fuzzy name matching
            - "last_n_meetings": Get last N meetings
            - "date_from_relative": Relative to absolute date
            - "filter_by_person": Filter by person name
            - "filter_by_date": Filter by date (year/month)
            - "filter_by_date_range": Filter by date range
            - "get_all_participants": Get all participants
        
        fuzzy_name: Fuzzy name (for find_person)
        n: Count (for last_n_meetings)
        date_str: Date string (for date_from_relative)
        year: Year (for filter_by_date)
        month: Month (for filter_by_date)
        start_date: Start date (for filter_by_date_range), format: YYYY-MM-DD
        end_date: End date (for filter_by_date_range), format: YYYY-MM-DD
        person_name: Person name (for last_n_meetings, optional - if provided, only returns meetings attended by this person)
    
    Returns:
        JSON formatted query result
    """
    global _pandas_query_call_count, _last_pandas_query_result
    
    # Check call count limit (code enforced)
    if _pandas_query_call_count >= _PANDAS_QUERY_MAX_CALLS:
        return json.dumps({
            "error": "MAX_CALLS_REACHED",
            "message": f"pandas_query has been called {_pandas_query_call_count} times. Maximum {_PANDAS_QUERY_MAX_CALLS} calls allowed per query. Please proceed with the information you have."
        })
    
    # Increment counter
    _pandas_query_call_count += 1
    
    mdf = get_meetings_df()
    
    # Include call count in response for transparency
    call_info = {
        "call_number": _pandas_query_call_count,
        "max_calls": _PANDAS_QUERY_MAX_CALLS,
        "remaining": _PANDAS_QUERY_MAX_CALLS - _pandas_query_call_count
    }
    
    if query_type == "find_person":
        if not fuzzy_name:
            return json.dumps({"error": "fuzzy_name is required"})
        
        matches = mdf.find_person(fuzzy_name)
        return json.dumps({
            "query_type": "find_person",
            "fuzzy_name": fuzzy_name,
            "matches": matches,
            "count": len(matches)
        })
    
    elif query_type == "last_n_meetings":
        if not n:
            n = 3
        
        # if apper someone's name, return his/her recent N meetings
        meetings = mdf.get_last_n_meetings(n=n, person_name=person_name)
        dates = [m["date"] for m in meetings]
        meeting_ids = [m["meeting_id"] for m in meetings]
        result = {
            "query_type": "last_n_meetings",
            "n": n,
            "person_name": person_name,
            "meetings": meetings,
            "dates": dates,
            "meeting_ids": meeting_ids,
            "count": len(meetings)
        }
        _last_pandas_query_result = result
        return json.dumps(result)
    
    elif query_type == "date_from_relative":
        if not date_str:
            return json.dumps({"error": "date_str is required"})
        
        resolver = DateResolver()
        resolved = resolver.resolve(date_str)
        return json.dumps({
            "query_type": "date_from_relative",
            "input": date_str,
            "resolved": resolved,
        })
    
    elif query_type == "filter_by_person":
        if not fuzzy_name:
            return json.dumps({"error": "fuzzy_name is required"})
        
        df = mdf.filter_by_person(fuzzy_name)
        meeting_ids = df["meeting_id"].tolist()
        
        return json.dumps({
            "query_type": "filter_by_person",
            "person": fuzzy_name,
            "meeting_ids": meeting_ids,
            "count": len(meeting_ids)
        })
    
    elif query_type == "filter_by_date":
        df_filtered = mdf.filter_by_year_month(
            year=year if year is not None else None,
            month=month if month is not None else None
        )
        meeting_ids = df_filtered["meeting_id"].tolist()
        
        return json.dumps({
            "query_type": "filter_by_date",
            "year": year,
            "month": month,
            "meeting_ids": meeting_ids,
            "count": len(meeting_ids)
        })
    
    elif query_type == "filter_by_date_range":
        if not start_date and not end_date:
            return json.dumps({"error": "At least one of start_date or end_date is required"})
        
        # pandas_utils.filter_by_date_range accepts Optional[str]
        df_filtered = mdf.filter_by_date_range(
            start_date=start_date, 
            end_date=end_date
        )
        meeting_ids = df_filtered["meeting_id"].tolist()
        dates = df_filtered["date"].tolist()
        
        return json.dumps({
            "query_type": "filter_by_date_range",
            "start_date": start_date,
            "end_date": end_date,
            "meeting_ids": meeting_ids,
            "dates": dates,
            "count": len(meeting_ids)
        })
    
    elif query_type == "get_all_participants":
        participants = mdf.get_all_participants()
        return json.dumps({
            "query_type": "get_all_participants",
            "participants": participants,
            "count": len(participants)
        })
    
    else:
        return json.dumps({"error": f"Unknown query_type: {query_type}"})



def check_ambiguity(query: str) -> str:
    """
    Check ambiguities in query using NLP libraries
    
    Uses spaCy for time entity detection and PersonMatcher (with pandas participants) for person entity detection.
    Person detection uses ListRecognizer with actual meeting participants for 100% accurate matching.
    
    Args:
        query: User query
    
    Returns:
        JSON formatted ambiguity check result
    """
    import spacy
    import dateparser
    
    global _spacy_nlp
    
    # Lazy load spaCy model if not already loaded by warmup_all()
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _spacy_nlp = spacy.load("en_core_web_sm")
    
    # Use spaCy to detect DATE/TIME entities
    # Return ALL detected time entities (Stage 2 will handle resolution)
    doc = _spacy_nlp(query)
    time_entities = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    
    # Return all spaCy-detected time entities
    # Stage 2 will use dateparser to resolve them
    # Note: "last meeting", "recent meetings" etc. won't be detected by spaCy
    # They will be handled by LLM in Stage 3
    found_time = time_entities
    
    # NOTE: We don't filter by dateparser here - we return all detected entities
    # Stage 2 is responsible for resolving absolute times using dateparser
    
    # Person detection using PersonMatcher with pandas participants
    # This uses ListRecognizer for 100% accurate matching against actual meeting participants
    matcher = get_person_matcher()
    person_detection_result = matcher.detect_and_resolve(query)
    
    # Get detected and resolved person names
    detected_persons = person_detection_result.get("detected", [])  # Original detected names
    resolved_persons = person_detection_result.get("resolved", [])  # Resolved canonical names
    
    # Build person_mapping directly from detected and resolved
    person_mapping = {}
    for detected, resolved in zip(detected_persons, resolved_persons):
        person_mapping[detected] = resolved
    
    # Get resolved person names
    found_person = resolved_persons
    
    # Also detect pronouns
    # Todo: Becasue we do not realize the memory logic for agent, so we do not consider he, she at the current stage
    person_ambiguous = {"i", "me", "my", "mine", "we", "us", "our", "ours", 
                        "he", "she", "they", "him", "her", "them"}
    query_lower = query.lower()
    found_person_pronouns = []
    for word in person_ambiguous:
        if f" {word} " in query_lower or f" {word}?" in query_lower or f" {word}," in query_lower:
            found_person_pronouns.append(word)
    
    # Combine: resolved persons + pronouns
    found_person = found_person + found_person_pronouns
    
    return json.dumps({
        "has_person_ambiguity": len(found_person) > 0,
        "person_terms": found_person,
        "person_mapping": person_mapping,  # Mapping from original name to resolved name
        "has_time_ambiguity": len(found_time) > 0,
        "time_terms": found_time,
    })


def resolve_entities(original_query: str, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: Resolution - Direct replacement (Option A)
    
    1. First-person pronouns → Replace with CURRENT_USER
    2. Absolute time expressions → Parse to specific dates using dateparser
    3. Person names → Replace using person_mapping
    4. "last meeting" type ambiguous times → Pass through to Stage 3 as-is
    
    Args:
        original_query: Original user query
        stage1_result: Detection result from Stage 1
    
    Returns:
        Dictionary containing resolved_query, replacement_log, resolved_persons, resolved_times
    """
    query = original_query
    person_terms = stage1_result.get('person_terms', [])
    person_mapping = stage1_result.get('person_mapping', {})  # {original_name: resolved_name}
    time_terms = stage1_result.get('time_terms', [])  # Only spaCy-detected absolute times
    
    replacement_log = []
    resolved_persons = []  # Track resolved person names for merge
    resolved_times = []    # Track resolved time strings for merge
    
    # 1. Handle pronouns -> Replace with actual user name (use word boundary regex)
    first_person = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
    for term in person_terms:
        term_lower = term.lower()
        if term_lower in first_person:
            replacement_log.append((term, CURRENT_USER))
            resolved_persons.append(CURRENT_USER)  # Already resolved to "Hongye Qian" from settings
            # Use word boundary to avoid replacing substrings
            query = re.sub(r'\b' + re.escape(term) + r'\b', CURRENT_USER, query, flags=re.IGNORECASE)
    
    # 2. Handle time terms -> dateparser解析
    for term in time_terms:
        parsed = dateparser.parse(term)
        if parsed:
            date_str = parsed.strftime("%Y-%m-%d")
            replacement_log.append((term, date_str))
            resolved_times.append(date_str)
            query = query.replace(term, date_str)
    
    # 3. Person names -> Replace original person names with resolved names using person_mapping
    # person_mapping: {original_name: resolved_name}, e.g., {"Hongye": "Hongye Qian"}
    for original_name, resolved_name in person_mapping.items():
        if original_name != resolved_name:  # Only replace if different
            replacement_log.append((original_name, resolved_name))
            resolved_persons.append(resolved_name)
            query = re.sub(r'\b' + re.escape(original_name) + r'\b', resolved_name, query, flags=re.IGNORECASE)
    
    # 4. "last meeting" is NOT handled here - passes through to Stage 3
    # (spaCy doesn't detect it, so it's not in time_terms)
    
    return {
        'resolved_query': query,
        'replacement_log': replacement_log,
        'resolved_persons': resolved_persons,
        'resolved_times': resolved_times
    }


# ============ Stage 3 Layer 1: Regex Matching ============

# Regex patterns for standard meeting-time expressions
_MEETING_REGEX_PATTERNS = [
    # "last 3 meetings", "last three meetings"
    (r'\b(?:the\s+)?last\s+(\d+)\s+meetings?\b', lambda m: int(m.group(1))),
    (r'\b(?:the\s+)?last\s+meeting\b', lambda m: 1),
    # "recent 5 meetings", "recent meetings"
    (r'\b(?:the\s+)?recent\s+(\d+)\s+meetings?\b', lambda m: int(m.group(1))),
    (r'\b(?:the\s+)?recent\s+meetings?\b', lambda m: 3),
    # "previous 2 meetings", "previous meeting"
    (r'\b(?:the\s+)?previous\s+(\d+)\s+meetings?\b', lambda m: int(m.group(1))),
    (r'\b(?:the\s+)?previous\s+meeting\b', lambda m: 1),
    # "past 3 meetings", "past meetings"
    (r'\b(?:the\s+)?past\s+(\d+)\s+meetings?\b', lambda m: int(m.group(1))),
    (r'\b(?:the\s+)?past\s+meetings?\b', lambda m: 3),
]


def resolve_by_regex(query: str, person_name: Optional[str] = None) -> Optional[RewriteOutput]:
    """
    Stage 3 - Layer 1: Regex-based resolution for standard meeting-time patterns.
    Directly calls pandas if a pattern matches, no LLM needed.
    
    Args:
        query: Resolved query from Stage 2
        person_name: Optional person name for filtering (from Stage 1)
    
    Returns:
        RewriteOutput if matched, None if no regex match
    """
    for pattern, n_extractor in _MEETING_REGEX_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            n = n_extractor(match)
            logger.info(f"[Stage3-Layer1-Regex] Matched pattern: '{match.group()}' → n={n}, person={person_name}")
            
            # Directly call pandas - no LLM needed
            mdf = get_meetings_df()
            meetings = mdf.get_last_n_meetings(n=n, person_name=person_name)
            meeting_ids = [m["meeting_id"] for m in meetings]
            dates = [m["date"] for m in meetings]
            
            # Replace the matched pattern with actual dates
            date_str = ", ".join(dates)
            rewritten = re.sub(pattern, f"meetings on {date_str}", query, flags=re.IGNORECASE)
            
            logger.info(f"[Stage3-Layer1-Regex] Rewritten: '{rewritten}', meeting_ids: {meeting_ids}")
            return RewriteOutput(
                rewritten_query=rewritten,
                relevant_meeting_ids=meeting_ids
            )
    
    return None  # No regex match


# ============ Stage 3 Layer 3: Embedding Resolution ============

# Word to number mapping for extraction
_WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a': 1, 'an': 1,
}


def _extract_number_from_query(query: str) -> Optional[int]:
    """
    Extract numeric value from query (digits or words).
    
    Args:
        query: The query string to extract number from
        
    Returns:
        Extracted number as int, or None if not found
    """
    # Try digit first
    digit_match = re.search(r'\b(\d+)\b', query)
    if digit_match:
        return int(digit_match.group(1))
    
    # Try word matching
    query_lower = query.lower()
    for word, num in _WORD_TO_NUM.items():
        if re.search(rf'\b{word}\b', query_lower):
            return num
    
    return None


def _find_best_pattern_match(query: str) -> tuple[Optional[str], float]:
    """
    Find the best matching meeting ambiguity pattern using embedding similarity.
    
    Args:
        query: The query to match against patterns
        
    Returns:
        (best_pattern, similarity_score) or (None, 0) if below threshold
    """
    generator = get_embedding_generator()
    pattern_embeddings = _get_pattern_embeddings()
    
    # Generate embedding for query
    query_embedding = generator.generate_embedding(query)
    query_vector = np.array([query_embedding.tolist()]).astype('float32')
    
    # Normalize
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # Compute similarities
    similarities = np.dot(query_vector, pattern_embeddings.T)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    best_pattern = MEETING_AMBIGUITY_PATTERNS[best_idx] if best_score >= EMBEDDING_THRESHOLD else None
    
    return best_pattern, float(best_score)


def resolve_by_embedding(query: str, person_name: Optional[str] = None) -> Optional[RewriteOutput]:
    """
    Stage 3 - Layer 3: Embedding-based resolution for ambiguous meeting patterns.
    
    This is called when regex didn't fully resolve the meeting ambiguity (threshold >= 0.8).
    Extracts number from query, matches with MEETING_AMBIGUITY_PATTERNS using embeddings,
    then calls pandas to resolve to specific meeting dates.
    
    Args:
        query: The rewritten query after Stage 1/2 (may still have meeting ambiguity)
        person_name: Optional person name for filtering
        
    Returns:
        RewriteOutput with resolved query and meeting IDs, or None if failed
    """
    # Step 1: Extract number from query
    extracted_num = _extract_number_from_query(query)
    n = extracted_num if extracted_num is not None else 3  # Default to 3 if no number found
    
    # Step 2: Find best matching pattern
    best_pattern, similarity = _find_best_pattern_match(query)
    
    if not best_pattern:
        logger.warning(f"[Stage3-Layer3-Embedding] No pattern matched above threshold, skipping")
        return None
    
    logger.info(f"[Stage3-Layer3-Embedding] Matched pattern: '{best_pattern}' (similarity={similarity:.3f}), n={n}")
    
    # Step 3: Call pandas to get meetings
    mdf = get_meetings_df()
    meetings = mdf.get_last_n_meetings(n=n, person_name=person_name)
    
    if not meetings:
        logger.warning(f"[Stage3-Layer3-Embedding] No meetings found for n={n}, person={person_name}")
        return None
    
    meeting_ids = [m["meeting_id"] for m in meetings]
    dates = [m["date"] for m in meetings]
    
    # Step 4: Replace the ambiguous pattern with actual dates
    date_str = ", ".join(dates)
    rewritten = re.sub(best_pattern, f"meetings on {date_str}", query, flags=re.IGNORECASE)
    
    # If no replacement happened, just append the date info
    if rewritten == query:
        rewritten = f"{query} (meetings on {date_str})"
    
    logger.info(f"[Stage3-Layer3-Embedding] Rewritten: '{rewritten}', meeting_ids: {meeting_ids}")
    
    return RewriteOutput(
        rewritten_query=rewritten,
        relevant_meeting_ids=meeting_ids
    )


# ============ Create Tools ============

pandas_query_tool = FunctionTool(func=pandas_query)

check_ambiguity_tool = FunctionTool(func=check_ambiguity)

reset_counter_tool = FunctionTool(func=reset_counter)


# ============ Main Entry Point: Three-Stage Pipeline ============

def merge_all_meeting_ids(
    stage2_result: Dict[str, Any],
    stage3_meeting_ids: List[str]
) -> List[str]:
    """
    Final unified processing: collect resolved persons/times from Stage 2,
    make pandas queries, and merge with Stage 3 meeting_ids
    
    Args:
        stage2_result: Stage 2 resolution result (resolved_persons, resolved_times)
        stage3_meeting_ids: Meeting IDs from Stage 3
    
    Returns:
        Merged list of meeting IDs
    """
    from sub_agents.pandas_utils import get_meetings_df
    
    all_meeting_ids = set(stage3_meeting_ids)
    mdf = get_meetings_df()
    
    # 1. Query by resolved person names
    resolved_persons = stage2_result.get('resolved_persons', [])
    for person in resolved_persons:
        df = mdf.filter_by_person(person)
        person_meeting_ids = df["meeting_id"].tolist()
        all_meeting_ids.update(person_meeting_ids)
        logger.info(f"[Merge] Added {len(person_meeting_ids)} meetings for person: {person}")
    
    # 2. Query by resolved time strings
    resolved_times = stage2_result.get('resolved_times', [])
    for date_str in resolved_times:
        df_filtered = mdf.filter_by_date_range(start_date=date_str, end_date=date_str)
        time_meeting_ids = df_filtered["meeting_id"].tolist()
        all_meeting_ids.update(time_meeting_ids)
        logger.info(f"[Merge] Added {len(time_meeting_ids)} meetings for time: {date_str}")
    
    logger.info(f"[Merge] Final meeting IDs count: {len(all_meeting_ids)}")
    return list(all_meeting_ids)


async def rewrite_query_async(query: str) -> RewriteOutput:
    """
    Three-stage Query Rewriter Pipeline (Async Entry)
    
    This is the async entry point for use in async environments like ADK.
    It avoids the "asyncio.run() cannot be called from a running event loop" error.
    
    Note: No LLM fallback - embedding threshold determines if query is clean
    
    Args:
        query: Original user query
        
    Returns:
        RewriteOutput with rewritten_query and relevant_meeting_ids
    """
    logger.info(f"[QueryRewriter] Input query: '{query}'")
    
    # ===== Stage 1: Detection (Synchronous) =====
    stage1_result_json = check_ambiguity(query)
    stage1_result = json.loads(stage1_result_json)
    logger.info(f"[Stage1-Detection] time={stage1_result.get('time_terms')}, person={stage1_result.get('person_terms')}")
    
    # ===== Stage 2: Resolution (Synchronous) =====
    stage2_result = resolve_entities(query, stage1_result)
    resolved_query = stage2_result['resolved_query']
    logger.info(f"[Stage2-Resolution] Resolved: '{resolved_query}', log={stage2_result.get('replacement_log')}")
    
    # Extract person name for Stage 3 (from resolved persons in Stage 2)
    resolved_persons = stage2_result.get('resolved_persons', [])
    person_name = resolved_persons[0] if resolved_persons else None
    
    # ===== Stage 3: Pipeline resolution (Regex → Embedding threshold check → Embedding if needed) =====
    # Layer 1: Regex - Try to resolve specific meeting patterns
    regex_result = resolve_by_regex(resolved_query, person_name)
    
    # Use regex result if available, otherwise use resolved_query
    base_query = regex_result.rewritten_query if regex_result else resolved_query
    
    # Layer 2: Check embedding threshold on the base query
    needs_embedding, matched_patterns = check_embedding_threshold(base_query)
    
    if needs_embedding:
        # Layer 3: Threshold met - need embedding resolution
        logger.info(f"[Stage3] Layer 2 threshold met, triggering Layer 3 embedding resolution")
        
        embedding_result = resolve_by_embedding(base_query, person_name)
        
        if embedding_result:
            stage3_meeting_ids = embedding_result.relevant_meeting_ids
            rewritten_query = embedding_result.rewritten_query
            logger.info(f"[Stage3] Resolved by Layer 3 (Embedding), meeting_ids: {stage3_meeting_ids}")
        else:
            # Embedding failed - fall back to regex result or base query
            stage3_meeting_ids = regex_result.relevant_meeting_ids if regex_result else []
            rewritten_query = base_query
            logger.info(f"[Stage3] Embedding failed, using fallback result")
    else:
        # Threshold not met - regex resolved it well (or no ambiguity)
        stage3_meeting_ids = regex_result.relevant_meeting_ids if regex_result else []
        rewritten_query = base_query
        logger.info(f"[Stage3] Layer 2 confirms clean, no embedding needed")
    
    # ===== Final: Merge Stage 2/3 meeting IDs =====
    final_meeting_ids = merge_all_meeting_ids(stage2_result, stage3_meeting_ids)
    
    return RewriteOutput(
        rewritten_query=rewritten_query,
        relevant_meeting_ids=final_meeting_ids
    )


# ============ Startup Warmup ============

def warmup_all():
    """Pre-load all lazy-initialized NLP components at startup.

    Call this once during application startup so the first user query
    does not incur the overhead of loading spaCy, Presidio, pandas
    DataFrames, and embedding models.
    """
    import spacy
    global _spacy_nlp

    logger.info("[Warmup] Pre-loading NLP components...")

    # 1. pandas MeetingsDataFrame
    logger.info("[Warmup]  Loading MeetingsDataFrame...")
    get_meetings_df()

    # 2. PersonMatcher singleton + Presidio AnalyzerEngine
    logger.info("[Warmup]  Loading PersonMatcher + Presidio AnalyzerEngine...")
    matcher = get_person_matcher()
    matcher._get_analyzer()

    # 3. spaCy model
    logger.info("[Warmup]  Loading spaCy model (en_core_web_sm)...")
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _spacy_nlp = spacy.load("en_core_web_sm")

    # 4. EmbeddingGenerator + pre-computed pattern embeddings
    logger.info("[Warmup]  Loading EmbeddingGenerator + pattern embeddings...")
    get_embedding_generator()
    _get_pattern_embeddings()

    logger.info("[Warmup] All NLP components loaded successfully!")
