"""
Meeting Count Regex Patterns Configuration
Stage3 Layer1 — deterministic fast path for meeting-count ambiguity resolution

All patterns are externalized here.
Do NOT hardcode regex in query_rewriter_agent.py.
"""
import re
from typing import Dict, Any, Optional


# ============================================================
# Word-to-number mapping
# Covers one ~ ten (sufficient for practical meeting count queries)
# ============================================================
WORD_TO_NUM: Dict[str, int] = {
    "one":   1,
    "two":   2,
    "three": 3,
    "four":  4,
    "five":  5,
    "six":   6,
    "seven": 7,
    "eight": 8,
    "nine":  9,
    "ten":   10,
}


# ============================================================
# Typo normalization map
# Applied via word-boundary regex BEFORE pattern matching.
# Only add typos that are clearly unambiguous corrections.
# ============================================================
TYPO_NORMALIZATION_MAP: Dict[str, str] = {
    "previouse": "previous",
}


# ============================================================
# Default N for vague plural expressions
# e.g., "recent meetings", "past meetings" (no explicit number)
# ============================================================
DEFAULT_N_FOR_VAGUE: int = 3


# ============================================================
# Internal helpers (not part of the public API)
# ============================================================

# Build alternation string for word numbers: one|two|three|...|ten
_WORD_NUM_ALT: str = "|".join(WORD_TO_NUM.keys())

# Main compiled pattern.
#
# Matches expressions such as:
#   last meeting              group(1)="last",        group(2)=None,    group(3)="meeting"
#   the last meeting          group(1)="last",        group(2)=None,    group(3)="meeting"
#   last 3 meetings           group(1)="last",        group(2)="3",     group(3)="meetings"
#   last three meetings       group(1)="last",        group(2)="three", group(3)="meetings"
#   the last 3 meetings       group(1)="last",        group(2)="3",     group(3)="meetings"
#   most recent meeting       group(1)="most recent", group(2)=None,    group(3)="meeting"
#   latest meeting            group(1)="latest",      group(2)=None,    group(3)="meeting"
#   previous meeting          group(1)="previous",    group(2)=None,    group(3)="meeting"
#   previous 3 meetings       group(1)="previous",    group(2)="3",     group(3)="meetings"
#   recent meetings           group(1)="recent",      group(2)=None,    group(3)="meetings"
#   recent 5 meetings         group(1)="recent",      group(2)="5",     group(3)="meetings"
#   past meetings             group(1)="past",        group(2)=None,    group(3)="meetings"
#   past 2 meetings           group(1)="past",        group(2)="2",     group(3)="meetings"
#
# group(0) = full matched text  → used as original_phrase for string replacement
# group(1) = keyword            → determines n logic
# group(2) = number token       → digit string or word number, may be None
# group(3) = "meeting" or "meetings"  → singular forces n=1
_MEETING_COUNT_COMPILED = re.compile(
    r"\b(?:the\s+)?"                                        # optional article "the" (non-capturing)
    r"(most\s+recent|last|latest|previous|recent|past)"     # group 1: keyword
    r"(?:\s+(\d+|" + _WORD_NUM_ALT + r"))?"                 # group 2: optional number
    r"\s+(meetings?)\b",                                    # group 3: "meeting" or "meetings"
    re.IGNORECASE,
)


def _handle_meeting_count_match(match: re.Match) -> Dict[str, Any]:
    """
    Handler for _MEETING_COUNT_COMPILED matches.
    Returns a standardized parse result dict.

    n decision logic:
      keyword in (most recent, latest)       → n = 1 (always singular intent)
      explicit number present                → n = parsed number
      keyword + singular "meeting", no num   → n = 1
      keyword + plural "meetings",  no num   → n = DEFAULT_N_FOR_VAGUE

    confidence:
      "high"   — number is explicit, or singular noun forces n=1
      "medium" — falling back to DEFAULT_N_FOR_VAGUE
    """
    keyword_raw: str = match.group(1)
    num_str: Optional[str] = match.group(2)
    plurality: str = match.group(3).lower()  # "meeting" or "meetings"

    # Normalize keyword whitespace (handles "most  recent" edge case)
    keyword: str = " ".join(keyword_raw.lower().split())

    # ---- Determine n ----
    if keyword in ("most recent", "latest"):
        n = 1
        confidence = "high"
    elif num_str is not None:
        num_lower = num_str.lower()
        if num_lower in WORD_TO_NUM:
            n = WORD_TO_NUM[num_lower]
        else:
            try:
                n = int(num_str)
            except ValueError:
                n = DEFAULT_N_FOR_VAGUE
        n = max(1, n)  # guard against n=0
        confidence = "high"
    elif plurality == "meeting":
        # singular without explicit number → exactly 1
        n = 1
        confidence = "high"
    else:
        # plural without explicit number → vague
        n = DEFAULT_N_FOR_VAGUE
        confidence = "medium"

    # ---- Build canonical phrase ----
    canonical = "last 1 meeting" if n == 1 else f"last {n} meetings"

    return {
        "match_type": "last_n_meetings",
        "n": n,
        "canonical_phrase": canonical,
        "original_phrase": match.group(0),   # exact text to replace in query string
        "confidence": confidence,
    }


# ============================================================
# Public: MEETING_PATTERN_RULES
#
# A list of rule dicts. Each rule has:
#   "name"    : str         — identifier for logging / debugging
#   "pattern" : re.Pattern  — compiled regex
#   "handler" : callable    — (re.Match) -> Dict[str, Any]
#
# _try_regex_resolve_meeting_count iterates this list in order;
# first match wins.  To add a new pattern type, append a new
# rule here — no changes needed in query_rewriter_agent.py.
# ============================================================
MEETING_PATTERN_RULES = [
    {
        "name": "meeting_count_ambiguity",
        "pattern": _MEETING_COUNT_COMPILED,
        "handler": _handle_meeting_count_match,
    },
    # Future rules can be appended here, e.g.:
    # {
    #     "name": "next_meeting",
    #     "pattern": re.compile(r"\b(next|upcoming)\s+meeting\b", re.IGNORECASE),
    #     "handler": _handle_next_meeting_match,
    # },
]
