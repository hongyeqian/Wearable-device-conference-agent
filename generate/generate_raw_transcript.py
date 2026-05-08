"""
generate_raw_transcript.py — Phase 1 (OpenAI API edition)
==========================================================
Convert a video or audio file into a raw transcript .txt
using the OpenAI Whisper API for transcription and optional
GPT-4o-mini-based speaker diarization.

Pipeline:
    input (mp4/mp3/wav/...) → ffmpeg → 16kHz mono mp3
                             → OpenAI Whisper API → raw segments
                             → (optional) GPT diarization → speaker labels
                             → merge_segments → paragraphs
                             → format_output → raw transcript .txt

Output format (compatible with generate_transcript.py):
    [mm:ss] Speaker 1: utterance text

    [mm:ss] Speaker 1: utterance text

Usage:
    python generate/generate_raw_transcript.py input.mp4
    python generate/generate_raw_transcript.py input.mp4 --output out.txt
    python generate/generate_raw_transcript.py input.mp4 --diarization
    python generate/generate_raw_transcript.py input.mp4 --gap 2.0 --max-duration 90
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
import argparse
from pathlib import Path

# Remove stale SSL_CERT_FILE that breaks OpenAI HTTPS on some Windows setups
if "SSL_CERT_FILE" in os.environ:
    _cert = os.environ["SSL_CERT_FILE"]
    if _cert and not os.path.exists(_cert):
        del os.environ["SSL_CERT_FILE"]


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config.settings import OPENAI_API_KEY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Punctuation that signals the previous fragment ended a complete sentence.
SENTENCE_END_MARKS: frozenset = frozenset(".?!。？！")

# Punctuation that signals the previous fragment is mid-sentence.
CONTINUATION_MARKS: frozenset = frozenset(",，;；:：")

# OpenAI Whisper API file-size hard limit; warn if extracted audio exceeds this.
OPENAI_AUDIO_MAX_BYTES: int = 25 * 1024 * 1024  # 25 MB


# ---------------------------------------------------------------------------
# 1. check_dependencies
# ---------------------------------------------------------------------------

def check_dependencies(diarize: bool = False) -> None:
    """
    Verify that ffmpeg is in PATH and the openai package is importable.
    Prints a clear install hint and exits with code 1 on any failure.
    Called once at startup before any processing begins.
    """
    if shutil.which("ffmpeg") is None:
        print("[ERROR] ffmpeg not found in PATH.")
        print("  Windows install:  winget install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        print("  After installing, make sure 'ffmpeg' is on your system PATH.")
        sys.exit(1)

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        print("[ERROR] openai package not installed.")
        print("  Install: pip install openai")
        sys.exit(1)

    mode = "GPT diarization enabled" if diarize else "single-speaker mode"
    print(f"[OK] Dependencies: ffmpeg and openai are available ({mode}).")


# ---------------------------------------------------------------------------
# 2. extract_audio
# ---------------------------------------------------------------------------

def extract_audio(input_path: str, output_audio: str) -> str:
    """
    Extract audio from any video or audio file and save as 16 kHz mono mp3.

    mp3 at 32 kbps ≈ 14 MB/hour, well under the OpenAI 25 MB upload limit
    for typical meeting recordings (WAV at 16 kHz would be ~110 MB/hour).
    ffmpeg handles virtually all common formats (mp4, mkv, mp3, m4a, etc.).

    Parameters
    ----------
    input_path : str
        Path to the input video or audio file.
    output_audio : str
        Destination path for the temporary mp3 file (must end in .mp3).

    Returns
    -------
    str
        The output_audio path (same value passed in).

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    RuntimeError
        If ffmpeg exits with a non-zero return code or times out.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",           # Overwrite output without prompting
        "-i", input_path,
        "-ar", "16000", # Sample rate: 16 kHz
        "-ac", "1",     # Channels: mono
        "-vn",          # Drop video stream
        "-b:a", "32k",  # Audio bitrate: 32 kbps (keeps file small for API upload)
        output_audio,
    ]

    print(f"[INFO] Extracting audio from: {input_path}")
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=3600,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed (exit code {exc.returncode}).\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out after 3600 seconds.") from exc

    size_bytes = Path(output_audio).stat().st_size
    size_mb = size_bytes / 1024 / 1024
    print(f"[INFO] Audio extracted to: {output_audio} ({size_mb:.1f} MB)")
    if size_bytes > OPENAI_AUDIO_MAX_BYTES:
        print(
            f"[WARN] Extracted audio is {size_mb:.1f} MB, exceeding the OpenAI "
            "25 MB limit.  The API upload may fail for very long recordings."
        )
    return output_audio


# ---------------------------------------------------------------------------
# 3. transcribe_with_openai  (replaces whisperx transcribe_audio)
# ---------------------------------------------------------------------------

def transcribe_with_openai(audio_path: str) -> list:
    """
    Transcribe an audio file using the OpenAI Whisper API (model: whisper-1).

    Uses response_format="verbose_json" with segment-level timestamps so the
    returned segments are drop-in compatible with the existing merge_segments()
    pipeline (same {"start", "end", "text"} dict format as whisperx output).

    Parameters
    ----------
    audio_path : str
        Path to a local audio file accepted by the OpenAI API
        (mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg, oga).

    Returns
    -------
    list of dict
        Each element: {"start": float, "end": float, "text": str}
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"[INFO] Uploading to OpenAI Whisper API: {audio_path}")
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    raw_segs = getattr(result, "segments", None) or []
    for seg in raw_segs:
        start = getattr(seg, "start", None)
        end   = getattr(seg, "end",   None)
        text  = getattr(seg, "text",  "").strip()
        if start is None or end is None or not text:
            continue
        segments.append({"start": float(start), "end": float(end), "text": text})

    print(f"[INFO] OpenAI Whisper returned {len(segments)} segments.")
    return segments


# ---------------------------------------------------------------------------
# 4. _seconds_to_mmss  (moved before diarize_with_gpt — needed in prompt)
# ---------------------------------------------------------------------------

def _seconds_to_mmss(seconds: float) -> str:
    """
    Convert a float number of seconds to a zero-padded mm:ss string.

    Examples
    --------
    0.0   → "00:00"
    5.9   → "00:05"
    63.5  → "01:03"
    3661  → "61:01"  (no hour component; mm rolls over 60)
    """
    total = int(seconds)
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


# ---------------------------------------------------------------------------
# 5. diarize_with_gpt  (replaces pyannote diarize_audio + assign_speakers)
# ---------------------------------------------------------------------------

def diarize_with_gpt(segments: list) -> list:
    """
    Use GPT-4o-mini to assign speaker labels to transcript segments.

    Sends a numbered, timestamped segment list to GPT and asks it to return
    a JSON map from segment index → speaker_0 / speaker_1 / … based on
    natural conversation flow and speech patterns.  Speaker labels are then
    mapped deterministically to "Speaker 1" / "Speaker 2" / … in
    first-appearance order so the output matches the format expected by
    generate_transcript.py.

    Falls back silently to all "Speaker 1" on any API or JSON-parse failure,
    preserving Phase 1 behaviour rather than aborting the run.

    Parameters
    ----------
    segments : list of dict
        Raw segments from transcribe_with_openai(): {"start", "end", "text"}.

    Returns
    -------
    list of dict
        Same segments with a "speaker" field added to each item.
        {"start": float, "end": float, "text": str, "speaker": str}
    """
    from openai import OpenAI

    if not segments:
        return segments

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build a compact numbered transcript for the prompt
    numbered = "\n".join(
        f"{i}: [{_seconds_to_mmss(s['start'])}] {s['text']}"
        for i, s in enumerate(segments)
    )

    system_prompt = (
        "You are a speaker diarization assistant. "
        "You receive a numbered list of transcript segments with timestamps. "
        "Assign a speaker label (speaker_0, speaker_1, speaker_2, …) to each "
        "segment index based on natural conversation flow and speech patterns. "
        "Reply ONLY with a compact JSON object mapping each segment index "
        "(as a string key) to its speaker label. "
        'Example: {"0":"speaker_0","1":"speaker_1","2":"speaker_0"}'
    )

    print("[INFO] Running GPT-based speaker diarization ...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": numbered},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # Strip optional markdown code fence that the model sometimes adds
        if raw.startswith("```"):
            lines = raw.splitlines()
            end_idx = -1 if lines[-1].strip().startswith("```") else len(lines)
            raw = "\n".join(lines[1:end_idx])
        assignments: dict = json.loads(raw)
    except Exception as exc:
        print(f"[WARN] GPT diarization failed: {exc}")
        print("[WARN] Falling back to single-speaker mode (Speaker 1).")
        return [dict(s, speaker="Speaker 1") for s in segments]

    # Map speaker_0, speaker_1, … → Speaker 1, Speaker 2, … (first-appearance order)
    label_map: dict = {}
    counter = 1
    labeled = []
    for i, seg in enumerate(segments):
        raw_label = assignments.get(str(i), "speaker_0")
        if raw_label not in label_map:
            label_map[raw_label] = f"Speaker {counter}"
            counter += 1
        labeled.append(dict(seg, speaker=label_map[raw_label]))

    print(f"[INFO] GPT diarization assigned {len(label_map)} speaker(s), "
          f"{len(labeled)} segments.")
    return labeled


# ---------------------------------------------------------------------------
# 6. _effective_threshold  (internal helper — unchanged)
# ---------------------------------------------------------------------------

def _effective_threshold(last_text: str, gap_threshold: float) -> float:
    """
    Adjust the gap threshold based on the punctuation at the end of
    the previous segment's text.

    Rules
    -----
    Sentence-end punctuation (. ? ! 。 ？ ！)
        Lower the threshold: effective = max(gap_threshold * 0.7, 0.8)

    Continuation punctuation (, ， ; ； : ：) or alphanumeric ending
        Raise the threshold: effective = gap_threshold * 1.4

    Anything else
        Use the default threshold unchanged.
    """
    text = last_text.rstrip()
    if not text:
        return gap_threshold

    last_char = text[-1]

    if last_char in SENTENCE_END_MARKS:
        return max(gap_threshold * 0.7, 0.8)

    if last_char in CONTINUATION_MARKS or last_char.isalnum():
        return gap_threshold * 1.4

    return gap_threshold


# ---------------------------------------------------------------------------
# 7. merge_segments  (unchanged)
# ---------------------------------------------------------------------------

def merge_segments(
    segments: list,
    gap_threshold: float = 1.5,
    max_duration: float = 60.0,
    max_chars: int = 300,
    speaker: str = "Speaker 1",
) -> list:
    """
    Merge raw Whisper segments into longer, readable paragraphs.

    Compatible with both diarized segments (have a "speaker" field) and
    plain segments (no "speaker" field — uses the global `speaker` fallback).

    Cut conditions (evaluated in priority order):
    ─────────────────────────────────────────────
    Hard 0. (diarized) Speaker label changed → must split.
    Hard 1. Adding this segment would push merged duration > max_duration.
    Hard 2. Adding this segment would push total chars > max_chars.
    Soft 3. The silence gap exceeds the punctuation-adjusted threshold.

    Parameters
    ----------
    segments : list of dict
        Segments with at least {"start", "end", "text"}.
        Diarized segments also carry a "speaker" field.
    gap_threshold : float
        Base silence gap (seconds) that triggers a paragraph split.
    max_duration : float
        Maximum duration (seconds) of any merged paragraph.
    max_chars : int
        Maximum character count of any merged paragraph.
    speaker : str
        Fallback speaker label when a segment has no "speaker" field.

    Returns
    -------
    list of dict
        Merged paragraphs: {"start": float, "speaker": str, "text": str}
    """
    segments = [s for s in segments if s["text"].strip()]

    if not segments:
        print("[WARN] No non-empty segments to merge.")
        return []

    def _speaker_of(seg: dict) -> str:
        return seg.get("speaker", speaker)

    merged = []
    group = [segments[0]]

    for seg in segments[1:]:
        last = group[-1]
        gap = seg["start"] - last["end"]

        group_start   = group[0]["start"]
        group_text    = " ".join(s["text"] for s in group)
        group_speaker = _speaker_of(group[0])

        # Hard 0: speaker change
        if _speaker_of(seg) != group_speaker:
            merged.append({
                "start":   group_start,
                "speaker": group_speaker,
                "text":    group_text.strip(),
            })
            group = [seg]
            continue

        # Hard 1: duration limit
        if seg["end"] - group_start > max_duration:
            merged.append({
                "start":   group_start,
                "speaker": group_speaker,
                "text":    group_text.strip(),
            })
            group = [seg]
            continue

        # Hard 2: character limit
        if len(group_text) + len(seg["text"]) > max_chars:
            merged.append({
                "start":   group_start,
                "speaker": group_speaker,
                "text":    group_text.strip(),
            })
            group = [seg]
            continue

        # Soft: gap with punctuation adjustment
        effective = _effective_threshold(last["text"], gap_threshold)
        if gap > effective:
            merged.append({
                "start":   group_start,
                "speaker": group_speaker,
                "text":    group_text.strip(),
            })
            group = [seg]
        else:
            group.append(seg)

    if group:
        merged.append({
            "start":   group[0]["start"],
            "speaker": _speaker_of(group[0]),
            "text":    " ".join(s["text"] for s in group).strip(),
        })

    print(f"[INFO] Merged {len(segments)} raw segments → {len(merged)} paragraphs.")
    return merged


# ---------------------------------------------------------------------------
# 8. format_output  (unchanged)
# ---------------------------------------------------------------------------

def format_output(merged_segments: list) -> str:
    """
    Format merged paragraphs into a raw transcript string.

    Output format per paragraph:
        [mm:ss] Speaker N: utterance text

    Paragraphs are separated by a single blank line.
    This format is compatible with generate_transcript.py's parser.

    Parameters
    ----------
    merged_segments : list of dict
        Output from merge_segments(): {"start", "speaker", "text"}.

    Returns
    -------
    str
        The complete raw transcript text, ready to write to a .txt file.
    """
    lines = []
    for seg in merged_segments:
        ts      = _seconds_to_mmss(seg["start"])
        speaker = seg["speaker"]
        text    = seg["text"]
        lines.append(f"[{ts}] {speaker}: {text}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 9. generate_raw_transcript  (main pipeline — updated for OpenAI API)
# ---------------------------------------------------------------------------

def generate_raw_transcript(
    input_path: str,
    output_path: str = None,
    model_size: str = "base",       # Kept for backward compat with run_pipeline.py; not used
    gap_threshold: float = 1.5,
    max_duration: float = 60.0,
    max_chars: int = 300,
    diarize: bool = False,
    hf_token=None,                  # Kept for backward compat with run_pipeline.py; not used
) -> str:
    """
    Full pipeline: video/audio → raw transcript .txt via OpenAI Whisper API.

    Phase 1 (diarize=False):
        extract_audio → transcribe_with_openai → merge_segments → format_output

    Phase 2 (diarize=True):
        extract_audio → transcribe_with_openai → diarize_with_gpt
        → merge_segments → format_output

    Parameters model_size and hf_token are accepted for backward compatibility
    with run_pipeline.py but are no longer used (WhisperX / pyannote replaced).

    Parameters
    ----------
    input_path : str
        Path to the source video or audio file.
    output_path : str, optional
        Where to write the .txt file.
        Defaults to the same directory and stem as input_path, with .txt suffix.
    model_size : str
        Deprecated (was WhisperX model size). Accepted but ignored.
    gap_threshold : float
        Base silence gap in seconds before splitting a paragraph.
    max_duration : float
        Maximum paragraph duration in seconds.
    max_chars : int
        Maximum paragraph character count.
    diarize : bool
        If True, run GPT-based speaker diarization. Default False.
    hf_token : str, optional
        Deprecated (was HuggingFace token for pyannote). Accepted but ignored.

    Returns
    -------
    str
        Absolute path of the written .txt file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_suffix(".txt")
    output_path = Path(output_path)

    # Temp file as mp3 — much smaller than WAV for API upload
    tmp_fd, tmp_audio = tempfile.mkstemp(suffix=".mp3", prefix="astar_openai_")
    os.close(tmp_fd)

    try:
        # Step 1: Extract audio → compressed mp3
        extract_audio(str(input_path), tmp_audio)

        # Step 2: Transcribe with OpenAI Whisper API
        segments = transcribe_with_openai(tmp_audio)

        # Step 3 (optional): GPT-based speaker diarization
        if diarize:
            segments = diarize_with_gpt(segments)

        # Step 4: Merge fragments into readable paragraphs
        merged = merge_segments(
            segments,
            gap_threshold=gap_threshold,
            max_duration=max_duration,
            max_chars=max_chars,
            speaker="Speaker 1",   # Fallback for non-diarized path
        )

        # Step 5: Format as raw transcript text
        content = format_output(merged)

        # Step 6: Write output file (UTF-8 for Chinese/English mixed content)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"[OK] Saved raw transcript: {output_path}")

    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)
            print(f"[INFO] Temporary audio removed: {tmp_audio}")

    return str(output_path)


# ---------------------------------------------------------------------------
# 10. __main__  entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_raw_transcript.py",
        description=(
            "Convert a video or audio file into a raw transcript .txt\n"
            "using the OpenAI Whisper API (no local GPU required).\n\n"
            "Phase 1 (default): single speaker, OpenAI Whisper only.\n"
            "Phase 2 (--diarization): multi-speaker via GPT-4o-mini.\n\n"
            "Examples:\n"
            "  python generate/generate_raw_transcript.py meeting.mp4\n"
            "  python generate/generate_raw_transcript.py meeting.mp4 "
            "--output ./datademo/con99/raw.txt\n"
            "  python generate/generate_raw_transcript.py meeting.mp4 "
            "--diarization"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Path to the input video or audio file (mp4, mp3, wav, m4a, etc.).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Path for the output .txt file. "
            "Default: same directory and filename as input, with .txt extension."
        ),
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help=(
            "Base silence gap (seconds) that triggers a paragraph split. "
            "Default: 1.5."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Maximum duration (seconds) per merged paragraph. Default: 60.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=300,
        metavar="N",
        help="Maximum character count per merged paragraph. Default: 300.",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help=(
            "Enable speaker diarization via GPT-4o-mini. "
            "Identifies multiple speakers from conversation context. "
            "Without this flag the script runs in single-speaker mode."
        ),
    )
    # Kept for backward compat with older scripts; silently ignored
    parser.add_argument(
        "--model",
        default="base",
        metavar="SIZE",
        help=argparse.SUPPRESS,   # Hidden — no longer applies (OpenAI API used)
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help=argparse.SUPPRESS,   # Hidden — no longer needed (pyannote removed)
    )

    args = parser.parse_args()

    check_dependencies(diarize=args.diarization)

    generate_raw_transcript(
        input_path    = args.input,
        output_path   = args.output,
        gap_threshold = args.gap,
        max_duration  = args.max_duration,
        max_chars     = args.max_chars,
        diarize       = args.diarization,
    )
