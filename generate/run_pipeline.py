"""
run_pipeline.py
===============
One-command pipeline: video / audio → all meeting artifacts.

Directory layout
----------------
  datademo/{con_id}/   ← Steps 1-5 + summary_metadata.json
  transfer/{con_id}/   ← Steps 6-7  (structured summary + report)

Execution order
---------------
  Step 1  generate_raw_transcript()          →  datademo/conX/raw_transcriptXXX.txt
  Step 2  generate_data_style_transcript()   →  datademo/conX/dataXXX.md
  Step 3  generate_meet_level()              →  datademo/conX/meetLevelXXX.md
  Step 4  generate_summary_from_meetlevel()  →  datademo/conX/summaryXXX.md
  Step 5  generate_metadata_from_transcript()→  datademo/conX/metaDataXXX.json
  Step 6  StructuredSummaryGenerator         →  transfer/conX/dataXXX.md
                                                transfer/conX/dataXXX.json
                                                datademo/conX/summary_metadata.json  (always)
  Step 7  DetailedReportGenerator            →  transfer/conX/reportXXX.md           (optional)

Steps 1-5 are mandatory.
Step 6 can be skipped; summary_metadata.json is still written (from metaData fallback).
Step 7 can be skipped independently.

Usage
-----
  # Basic run
  python generate/run_pipeline.py "C:/path/to/meeting.mp4" --con-id con6

  # With speaker diarization
  python generate/run_pipeline.py "C:/path/to/meeting.mp4" --con-id con6 \\
      --diarization --model small

  # Skip optional steps
  python generate/run_pipeline.py "C:/path/to/meeting.mp4" --con-id con6 \\
      --skip-structured-summary --skip-detailed-report
"""

import re
import sys
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — generate/ has no __init__.py, so add both dirs to sys.path
# ---------------------------------------------------------------------------
_GENERATE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _GENERATE_DIR.parent

for _p in (str(_GENERATE_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from generate_raw_transcript     import generate_raw_transcript as _gen_raw  # noqa: E402
from generate_transcript         import generate_data_style_transcript        # noqa: E402
from generate_meetLevel          import generate_meet_level                   # noqa: E402
from generate_summary            import generate_summary_from_meetlevel       # noqa: E402
from generate_metaData           import generate_metadata_from_transcript     # noqa: E402
from generate_structured_summary import StructuredSummaryGenerator            # noqa: E402
from generate_detailed_report    import DetailedReportGenerator               # noqa: E402

try:
    from config.settings import HF_TOKEN
except Exception:
    HF_TOKEN = None


# ---------------------------------------------------------------------------
# con-id parsing
# ---------------------------------------------------------------------------

def parse_con_id(con_id: str):
    """
    Parse a con-id string into (con_id, meeting_no).

    Examples
    --------
    "con6"   → ("con6",  "006")
    "con11"  → ("con11", "011")
    "con123" → ("con123","123")

    Raises SystemExit with a clear message if the format is invalid.
    """
    m = re.fullmatch(r"con(\d+)", con_id.strip())
    if not m:
        print(
            f"[ERROR] --con-id must look like con6 / con7 / con11 "
            f"(got: '{con_id}')"
        )
        sys.exit(1)
    meeting_no = m.group(1).zfill(3)
    return con_id.strip(), meeting_no


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)

def _step(msg: str) -> None: _log("STEP",  msg)
def _ok(msg: str)   -> None: _log("OK",    msg)
def _warn(msg: str) -> None: _log("WARN",  msg)
def _err(msg: str)  -> None: _log("ERROR", msg)


# ---------------------------------------------------------------------------
# summary_metadata.json helpers
# ---------------------------------------------------------------------------

def _write_summary_metadata(datademo_dir: Path, new_data: dict) -> Path:
    """
    Write or update datademo_dir/summary_metadata.json.

    Uses dict.update() semantics so manually added top-level keys are
    preserved across repeated runs.
    """
    path = datademo_dir / "summary_metadata.json"
    existing: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
            else:
                _warn("summary_metadata.json had unexpected root type — resetting")
        except json.JSONDecodeError:
            _warn("summary_metadata.json contained invalid JSON — resetting")

    existing.update(new_data)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_summary_metadata(meta_path: Path, ml_path: Path) -> dict:
    """
    Build the lightweight summary_metadata dict aligned to the old pipeline schema:

      {
        "meeting_id":   <from metaDataXXX.json>,
        "datetime":     <from metaDataXXX.json>,
        "participants": [<name>, ...],          # participant objects → name list
        "topics":       [<Topic Title>, ...],   # extracted from meetLevelXXX.md
        "actions":      []
      }
    """
    meeting_id: str | None = None
    datetime_val = None
    participants: list = []
    topics: list = []
    actions: list = []

    # ── metaDataXXX.json ────────────────────────────────────────────────────
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meeting_id   = meta.get("meeting_id")
        datetime_val = meta.get("datetime")
        for p in meta.get("participants", []):
            if isinstance(p, dict):
                name = p.get("name", "").strip()
                if name:
                    participants.append(name)
            elif isinstance(p, str) and p.strip():
                participants.append(p.strip())
        # carry over actions only if they are already lightweight strings/dicts
        raw_actions = meta.get("actions", [])
        if isinstance(raw_actions, list):
            actions = raw_actions
    except Exception:
        pass

    # ── meetLevelXXX.md  →  Topic Title lines ───────────────────────────────
    if ml_path and ml_path.exists():
        try:
            for line in ml_path.read_text(encoding="utf-8").splitlines():
                m = re.match(r".*\*\*Topic Title:\*\*\s*(.+)", line)
                if m:
                    title = m.group(1).strip()
                    if title:
                        topics.append(title)
        except Exception:
            pass

    return {
        "meeting_id":   meeting_id,
        "datetime":     datetime_val,
        "participants": participants,
        "topics":       topics,
        "actions":      actions,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    con_id: str,
    meeting_no: str,
    model_size: str = "base",
    diarize: bool = False,
    hf_token=None,
    gap: float = 1.5,
    max_duration: float = 60.0,
    max_chars: int = 300,
    skip_structured_summary: bool = False,
    skip_detailed_report: bool = False,
) -> dict:
    """
    Run the full meeting transcript pipeline.

    Parameters
    ----------
    input_path : str
        Path to the source video or audio file.
    con_id : str
        Conversation folder ID, e.g. "con6".
    meeting_no : str
        Zero-padded 3-digit meeting number derived from con_id, e.g. "006".
    model_size : str
        WhisperX model size (tiny / base / small / medium / large).
    diarize : bool
        Enable speaker diarization via pyannote.audio.
    hf_token : str | None
        HuggingFace token (required when diarize=True).
    gap : float
        Silence gap in seconds for paragraph splitting.
    max_duration : float
        Maximum paragraph duration in seconds.
    max_chars : int
        Maximum paragraph character count.
    skip_structured_summary : bool
        Skip Step 6.  summary_metadata.json is still written from metaData.
    skip_detailed_report : bool
        Skip Step 7.

    Returns
    -------
    dict
        All output paths, or None for skipped/failed steps.
    """
    results: dict = {
        "raw_transcript":    None,
        "data_markdown":     None,
        "meetlevel":         None,
        "summary":           None,
        "metadata":          None,
        "transfer_data_md":  None,   # transfer/conX/dataXXX.md
        "transfer_data_json":None,   # transfer/conX/dataXXX.json
        "summary_metadata":  None,   # datademo/conX/summary_metadata.json
        "detailed_report":   None,
    }

    no = meeting_no  # e.g. "006"

    # ── Directory setup ─────────────────────────────────────────────────────
    datademo_dir = _PROJECT_ROOT / "datademo" / con_id
    transfer_dir = _PROJECT_ROOT / "transfer"  / con_id

    datademo_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  con_id      : {con_id}")
    print(f"  meeting_no  : {no}")
    print(f"  datademo_dir: {datademo_dir}")
    print(f"  transfer_dir: {transfer_dir}")
    print()

    # ── Step 1: raw transcript ───────────────────────────────────────────────
    raw_path = datademo_dir / f"raw_transcript{no}.txt"
    _step(f"Step 1 — raw transcript  →  {raw_path.name}")
    try:
        _gen_raw(
            input_path    = str(input_path),
            output_path   = str(raw_path),
            model_size    = model_size,
            gap_threshold = gap,
            max_duration  = max_duration,
            max_chars     = max_chars,
            diarize       = diarize,
            hf_token      = hf_token,
        )
        results["raw_transcript"] = str(raw_path)
        _ok(f"raw transcript  →  {raw_path}")
    except Exception as exc:
        _err(f"Step 1 failed: {exc}")
        raise

    # ── Step 2: dataXXX.md ──────────────────────────────────────────────────
    # generate_data_style_transcript saves to raw_path's parent directory,
    # which is datademo_dir — correct by construction.
    _step(f"Step 2 — transcript formatting  →  data{no}.md")
    try:
        data_path_str = generate_data_style_transcript(
            raw_paths  = str(raw_path),
            meeting_no = no,
        )
        data_path = Path(data_path_str)
        results["data_markdown"] = str(data_path)
        _ok(f"data markdown  →  {data_path}")
    except Exception as exc:
        _err(f"Step 2 failed: {exc}")
        raise

    # ── Step 3: meetLevelXXX.md ─────────────────────────────────────────────
    # generate_meet_level saves to data_path's parent = datademo_dir.
    _step(f"Step 3 — meet level  →  meetLevel{no}.md")
    try:
        ml_path_str = generate_meet_level(transcript_path=str(data_path))
        ml_path = Path(ml_path_str)
        results["meetlevel"] = str(ml_path)
        _ok(f"meet level  →  {ml_path}")
    except Exception as exc:
        _err(f"Step 3 failed: {exc}")
        raise

    # ── Step 4: summaryXXX.md ───────────────────────────────────────────────
    # generate_summary_from_meetlevel saves to ml_path's parent = datademo_dir.
    _step(f"Step 4 — summary  →  summary{no}.md")
    try:
        summary_path_str = generate_summary_from_meetlevel(meetlevel_path=str(ml_path))
        results["summary"] = summary_path_str
        _ok(f"summary  →  {summary_path_str}")
    except Exception as exc:
        _err(f"Step 4 failed: {exc}")
        raise

    # ── Step 5: metaDataXXX.json ────────────────────────────────────────────
    # generate_metadata_from_transcript saves to data_path's parent = datademo_dir.
    _step(f"Step 5 — metadata  →  metaData{no}.json")
    try:
        meta_path_str = generate_metadata_from_transcript(transcript_path=str(data_path))
        meta_path = Path(meta_path_str)
        results["metadata"] = str(meta_path)
        _ok(f"metadata  →  {meta_path}")
    except Exception as exc:
        _err(f"Step 5 failed: {exc}")
        raise

    # ── Step 6: structured summary → transfer_dir ───────────────────────────
    # transfer/conX/dataXXX.md + dataXXX.json  (old-style naming)
    # datademo/conX/summary_metadata.json       (always written, old location)
    if not skip_structured_summary:
        _step(
            f"Step 6 — structured summary  →  "
            f"data{no}.md / data{no}.json  +  summary_metadata.json"
        )
        try:
            transcript_text = data_path.read_text(encoding="utf-8")
            gen    = StructuredSummaryGenerator()
            output = gen.generate_summary(transcript_text)

            td_md_path   = transfer_dir / f"data{no}.md"
            td_json_path = transfer_dir / f"data{no}.json"

            td_md_path.write_text(output["formatted_text"], encoding="utf-8")
            td_json_path.write_text(
                json.dumps(output["json_data"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results["transfer_data_md"]   = str(td_md_path)
            results["transfer_data_json"] = str(td_json_path)
            _ok(f"structured summary  →  {td_md_path.name}  {td_json_path.name}")

            sm_data = _build_summary_metadata(meta_path, ml_path)
            sm_path = _write_summary_metadata(datademo_dir, sm_data)
            results["summary_metadata"] = str(sm_path)
            _ok(f"summary_metadata.json  →  {sm_path}")

        except Exception as exc:
            _warn(f"Step 6 structured summary failed: {exc}")
            _warn("Writing fallback summary_metadata.json from metaData + meetLevel")
            try:
                sm_data = _build_summary_metadata(meta_path, ml_path)
                sm_path = _write_summary_metadata(datademo_dir, sm_data)
                results["summary_metadata"] = str(sm_path)
                _ok(f"summary_metadata.json (fallback)  →  {sm_path}")
            except Exception as exc2:
                _err(f"summary_metadata.json fallback failed: {exc2}")
    else:
        _warn("Step 6 skipped (--skip-structured-summary)")
        _step("Step 6 — writing summary_metadata.json from metaData + meetLevel")
        try:
            sm_data = _build_summary_metadata(meta_path, ml_path)
            sm_path = _write_summary_metadata(datademo_dir, sm_data)
            results["summary_metadata"] = str(sm_path)
            _ok(f"summary_metadata.json  →  {sm_path}")
        except Exception as exc:
            _err(f"summary_metadata.json write failed: {exc}")

    # ── Step 7: detailed report → transfer_dir ──────────────────────────────
    if not skip_detailed_report:
        _step(f"Step 7 — detailed report  →  report{no}.md")
        if results["transfer_data_json"] is None:
            _warn(
                "Step 7 skipped — transfer data JSON not available "
                "(Step 6 failed or was skipped)"
            )
        else:
            try:
                json_data = json.loads(
                    Path(results["transfer_data_json"]).read_text(encoding="utf-8")
                )
                transcript_text = data_path.read_text(encoding="utf-8")

                report_gen  = DetailedReportGenerator()
                report_text = report_gen.generate_report(
                    json_data      = json_data,
                    transcript     = transcript_text,
                    meeting_number = no,
                )
                report_path = transfer_dir / f"report{no}.md"
                report_path.write_text(report_text, encoding="utf-8")
                results["detailed_report"] = str(report_path)
                _ok(f"detailed report  →  {report_path}")
            except Exception as exc:
                _warn(f"Step 7 failed (non-fatal): {exc}")
    else:
        _warn("Step 7 skipped (--skip-detailed-report)")

    # ── Final summary ────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("Pipeline finished successfully.")
    print("=" * 64)
    label_map = {
        "raw_transcript":    "raw_transcript",
        "data_markdown":     "data_markdown",
        "meetlevel":         "meetlevel",
        "summary":           "summary",
        "metadata":          "metadata",
        "transfer_data_md":  "transfer_data_md",
        "transfer_data_json":"transfer_data_json",
        "summary_metadata":  "summary_metadata",
        "detailed_report":   "detailed_report",
    }
    for key, path in results.items():
        status = path if path is not None else "(skipped / failed)"
        print(f"  {label_map[key]:<24}  {status}")
    print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "One-command pipeline: video/audio → all meeting artifacts.\n"
            "Outputs split into datademo/{con_id}/ (Steps 1-5) "
            "and transfer/{con_id}/ (Steps 6-7)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate/run_pipeline.py meeting.mp4 --con-id con6\n\n"
            "  python generate/run_pipeline.py meeting.mp4 --con-id con6 \\\n"
            "      --diarization --model small\n\n"
            "  python generate/run_pipeline.py meeting.mp4 --con-id con6 \\\n"
            "      --skip-structured-summary --skip-detailed-report"
        ),
    )

    # Positional
    parser.add_argument(
        "input",
        help="Path to the input video or audio file (mp4, mp3, wav, m4a, ...).",
    )

    # Required
    parser.add_argument(
        "--con-id", required=True, metavar="conN",
        help="Conversation folder ID, e.g. con6, con7, con11.",
    )

    # WhisperX / transcription
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="WhisperX model size. Default: base.",
    )
    parser.add_argument(
        "--gap", type=float, default=1.5, metavar="SEC",
        help="Silence gap threshold (s) for paragraph splitting. Default: 1.5.",
    )
    parser.add_argument(
        "--max-duration", type=float, default=60.0, metavar="SEC",
        help="Max paragraph duration in seconds. Default: 60.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=300, metavar="N",
        help="Max paragraph character count. Default: 300.",
    )

    # Diarization
    parser.add_argument(
        "--diarization", action="store_true", default=False,
        help="Enable speaker diarization via pyannote.audio.",
    )
    parser.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help=(
            "HuggingFace token for pyannote model download. "
            "Required with --diarization. "
            "Alternative: set HF_TOKEN in config/.env."
        ),
    )

    # Optional step control
    parser.add_argument(
        "--skip-structured-summary", action="store_true", default=False,
        help=(
            "Skip Step 6 (structured summary). "
            "summary_metadata.json is still written from metaData."
        ),
    )
    parser.add_argument(
        "--skip-detailed-report", action="store_true", default=False,
        help="Skip Step 7 (detailed report).",
    )

    args = parser.parse_args()

    # Parse con-id → con_id + meeting_no
    con_id, meeting_no = parse_con_id(args.con_id)

    # Resolve HF token
    hf_token = args.hf_token or HF_TOKEN

    if args.diarization and not hf_token:
        print("[ERROR] --diarization requires a HuggingFace token.")
        print("  Option 1: pass --hf-token YOUR_TOKEN")
        print("  Option 2: set HF_TOKEN in config/.env")
        sys.exit(1)

    run_pipeline(
        input_path              = args.input,
        con_id                  = con_id,
        meeting_no              = meeting_no,
        model_size              = args.model,
        diarize                 = args.diarization,
        hf_token                = hf_token,
        gap                     = args.gap,
        max_duration            = args.max_duration,
        max_chars               = args.max_chars,
        skip_structured_summary = args.skip_structured_summary,
        skip_detailed_report    = args.skip_detailed_report,
    )
