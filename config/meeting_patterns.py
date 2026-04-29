"""
Meeting Ambiguity Patterns Configuration
Stores preset meeting patterns for embedding-based threshold detection in Stage 3
"""
from typing import List

# Meeting ambiguity patterns that users might use to reference meetings
# These patterns will be embedded and used for similarity matching
MEETING_AMBIGUITY_PATTERNS: List[str] = [
    # "last" patterns
    "last meeting",
    "last meetings",
    "the last meeting",
    "the last meetings",
    
    # "recent" patterns
    "recent meeting",
    "recent meetings",
    "the recent meeting",
    "the recent meetings",
    
    # "previous" patterns
    "previous meeting",
    "previous meetings",
    "the previous meeting",
    "the previous meetings",
    
    # "past" patterns
    "past meeting",
    "past meetings",
    "the past meeting",
    "the past meetings",
    
    # "latest" patterns
    "latest meeting",
    "latest meetings",
    "the latest meeting",
    "the latest meetings",
    
    # "earliest" patterns
    "earliest meeting",
    "earliest meetings",
    "the earliest meeting",
    "the earliest meetings",
    
    # "this" patterns
    "this meeting",
    "that meeting",
    
    # "my" patterns
    "my meeting",
    "my meetings",
    "our meeting",
    "our meetings",

    # future action_tasks
    "future action_tasks",
    "future action_tasks",
    "the future action_tasks",
    "the future action_tasks",
]

# Embedding threshold for matching (default: 0.8)
EMBEDDING_THRESHOLD: float = 0.8

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE: int = 100
