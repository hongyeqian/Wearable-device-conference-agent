"""
evaluate_asr_quality.py
=======================
Compute WER and CER between a reference transcript and a predicted transcript.

Both input files must use the raw transcript format produced by
generate_raw_transcript.py:

    [mm:ss] Speaker N: utterance text

The script strips timestamps and speaker labels before computing metrics,
leaving only the utterance text for evaluation.

Usage
-----
    python generate/evaluate_asr_quality.py reference.txt predicted.txt

Output
------
    WER = 0.14
    CER = 0.08
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Transcript cleaning
# ---------------------------------------------------------------------------

# Matches the leading "[mm:ss] Speaker N:" prefix on each paragraph line.
# mm can exceed 99 for long recordings, so \d+ (not \d{2}) is used.
# The speaker label is any non-empty run of characters up to the first colon.
_LINE_PREFIX = re.compile(r"^\[\d+:\d{2}\]\s+[^:]+:\s*")


def clean_transcript_text(file_path: str) -> str:
    """
    Read a transcript file and return the utterance text only.

    Removes every ``[mm:ss] Speaker N:`` prefix so that the returned string
    contains nothing but the spoken content, ready for WER / CER computation.

    Empty lines (paragraph separators in the raw transcript) are skipped;
    all remaining lines are joined with a single space.

    Parameters
    ----------
    file_path : str
        Path to a transcript file in ``[mm:ss] Speaker N: text`` format.

    Returns
    -------
    str
        Cleaned utterance text as one continuous string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains no recognisable utterance lines after cleaning.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip blank separator lines
        # Strip the [mm:ss] Speaker N: prefix if present
        text = _LINE_PREFIX.sub("", line).strip()
        if text:
            cleaned.append(text)

    if not cleaned:
        raise ValueError(
            f"No utterance text found after cleaning '{file_path}'.\n"
            "Expected lines in the format: [mm:ss] Speaker N: text"
        )

    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# CER fallback (used when jiwer.cer is unavailable)
# ---------------------------------------------------------------------------

def _levenshtein(ref: str, hyp: str) -> int:
    """Standard Levenshtein edit distance between two strings."""
    m, n = len(ref), len(hyp)
    # Use a single-row DP to keep memory O(n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev_row = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev_row[j - 1]
            else:
                dp[j] = 1 + min(prev_row[j], dp[j - 1], prev_row[j - 1])
    return dp[n]


def _manual_cer(reference: str, hypothesis: str) -> float:
    """
    Compute CER as character-level edit distance / reference character count.

    Spaces are removed before comparison so that word-boundary differences
    do not inflate the error rate — this is the standard approach for
    Chinese ASR evaluation.
    """
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return _levenshtein(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_asr_quality(reference_path: str, hypothesis_path: str) -> dict:
    """
    Compute WER and CER between a reference and a predicted transcript.

    Parameters
    ----------
    reference_path : str
        Path to the ground-truth transcript (``[mm:ss] Speaker N: text`` format).
    hypothesis_path : str
        Path to the predicted transcript (same format).

    Returns
    -------
    dict
        ``{"wer": float, "cer": float}``
        Both values are in the range [0, 1] (e.g. 0.14 means 14 %).

    Side effects
    ------------
    Prints ``WER = ...`` and ``CER = ...`` to stdout.
    Exits with code 1 if jiwer is not installed.
    """
    try:
        import jiwer
    except ImportError:
        print("[ERROR] jiwer is not installed.")
        print("  Install: pip install jiwer")
        sys.exit(1)

    reference  = clean_transcript_text(reference_path)
    hypothesis = clean_transcript_text(hypothesis_path)

    # --- WER ---
    wer_score = jiwer.wer(reference, hypothesis)

    # --- CER ---
    # jiwer.cer() was added in jiwer 2.3; fall back to manual implementation
    # for older versions so the script works regardless of installed version.
    if hasattr(jiwer, "cer"):
        cer_score = jiwer.cer(reference, hypothesis)
    else:
        cer_score = _manual_cer(reference, hypothesis)

    print(f"WER = {wer_score:.2f}")
    print(f"CER = {cer_score:.2f}")

    return {"wer": wer_score, "cer": cer_score}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate/evaluate_asr_quality.py reference.txt predicted.txt")
        sys.exit(1)

    ref_path  = sys.argv[1]
    hyp_path  = sys.argv[2]

    evaluate_asr_quality(ref_path, hyp_path)
