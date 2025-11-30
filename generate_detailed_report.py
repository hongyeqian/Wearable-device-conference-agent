"""
Generate Detailed Meeting Reports
Processes JSON outline files and original transcripts to generate comprehensive meeting reports
with high-level summaries, topic breakdowns, and action items.
"""
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from config.settings import OPENAI_API_KEY

# System prompt for detailed report generation
DETAILED_REPORT_PROMPT = """You will generate a detailed meeting report using two inputs:

1. A JSON outline with:
   - high_level_topic
   - second_level_topics with bullet points

2. The original transcript

Output format MUST follow the exact structure below:

Discussion on <high_level_topic>

# High-Level Summary

(2–4 sentences summarizing what the meeting is about, referencing major topics.)

---

For each second-level topic (use number: 1, 2, 3️, etc.):

## 1️. <topic_name>

Brief Summary:

- 2–3 sentences describing what was discussed under this topic

Key Takeaways:

- key takeaway 1
- key takeaway 2
- key takeaway 3 (only if exists)

---

## 2. <topic_name>

Brief Summary:

- 2–3 sentences summarizing discussion

Key Takeaways:

- key takeaway 1
- key takeaway 2

---

## Action Items

(Summarize meeting outcomes + next steps)

For every action item mentioned in the transcript, produce:

- Responsible Person: <Name>

  - Topic id: <MXXX-A##> (Follow meeting number from outline and ascending numbering per item)

  - Task: <the actionable request>

  - Context: <why this action was raised, based on topic discussion>

  - Duration: [start time–end time] if available, else omit this line

  - Deadline: <describe> or "TBD"

(Repeat for all action items)

---

## Conclusion
2 to 4 sentences about the total summary of this meeting

STRICT RULES:

- DO NOT modify or create new high-level or second-level topic names. Use exactly what is in the JSON.
- DO NOT invent actions; only include what is explicitly stated in the transcript.
- Preserve chronological and structural alignment with the transcript.
- No transcript quotes required, but timestamps and paragraph IDs ARE required when available.
- Use professional, concise wording.
- If any required metadata is missing, fill with "TBD".
- Extract paragraph IDs (e.g., P001, P002) from the transcript format [#P###].
- Extract timestamps from the transcript format [[HH:MM](...)] if available.
- If no action items are mentioned in the transcript, write "No action items were mentioned in this meeting."
- All output must be in English.

Your job is to produce only the formatted report text. Do NOT output JSON.

Now process the following:

JSON Outline:

<<JSON_HERE>>

Transcript:

<<TRANSCRIPT_HERE>>"""


class DetailedReportGenerator:
    """Generate detailed meeting reports from JSON outline and original transcript"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        """
        Initialize the detailed report generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
            model: OpenAI model name (defaults to "gpt-4o-mini")
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass it directly.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_report(
        self,
        json_data: Dict[str, Any],
        transcript: str,
        meeting_number: str
    ) -> str:
        """
        Generate detailed report from JSON outline and transcript.
        
        Args:
            json_data: The JSON outline with high_level_topic and second_level_topics
            transcript: The original transcript text
            meeting_number: Meeting number string (e.g., "001") for topic IDs
            
        Returns:
            Formatted detailed report text
        """
        # Format JSON for prompt
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        
        # Update prompt with meeting number context
        prompt = DETAILED_REPORT_PROMPT.replace(
            "<MXXX-A##>",
            f"M{meeting_number}-A##"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"JSON Outline:\n\n<<JSON_HERE>>\n{json_str}\n\nTranscript:\n\n<<TRANSCRIPT_HERE>>\n{transcript}"
                    }
                ],
                temperature=self.temperature
            )
            
            report = response.choices[0].message.content
            
            if not report:
                raise ValueError("Empty response from OpenAI")
            
            return report.strip()
            
        except Exception as e:
            raise Exception(f"Error generating detailed report: {e}")


def find_json_and_transcript_files(
    transfer_dir: Path,
    datademo_dir: Path,
    start_num: Optional[int] = None,
    end_num: Optional[int] = None
) -> List[Tuple[Path, Path, str, str]]:
    """
    Find matching JSON files in transfer directory and corresponding transcript files in datademo.
    
    Args:
        transfer_dir: Path to transfer directory containing JSON files
        datademo_dir: Path to datademo directory containing original transcripts
        start_num: Starting number (e.g., 1 for data001)
        end_num: Ending number (e.g., 13 for data013)
        
    Returns:
        List of tuples: (json_file_path, transcript_file_path, file_number, con_folder)
    """
    file_pairs = []
    
    # Find all JSON files in transfer directory (search recursively in con* subdirectories)
    for json_file in transfer_dir.rglob("data*.json"):
        match = re.search(r'data(\d+)\.json', json_file.name)
        if match:
            file_num = int(match.group(1))
            
            # Apply filters if specified
            if start_num is not None and file_num < start_num:
                continue
            if end_num is not None and file_num > end_num:
                continue
            
            # Extract con folder from JSON file path
            con_match = re.search(r'con\d+', str(json_file))
            if con_match:
                con_folder = con_match.group(0)
            else:
                # Try to find con folder in parent directories
                con_folder = None
                for parent in json_file.parents:
                    if parent.name.startswith('con'):
                        con_folder = parent.name
                        break
                if not con_folder:
                    con_folder = f"con{file_num:03d}"  # Default fallback
            
            # Find corresponding transcript file in datademo
            transcript_file = None
            for transcript in datademo_dir.rglob(f"data{file_num:03d}.md"):
                transcript_file = transcript
                break
            
            if transcript_file and transcript_file.exists():
                file_pairs.append((json_file, transcript_file, f"{file_num:03d}", con_folder))
            else:
                print(f"Warning: Could not find transcript file for {json_file.name}")
    
    # Sort by file number
    file_pairs.sort(key=lambda x: int(x[2]))
    
    return file_pairs


def process_reports(
    start: str = "data001",
    end: str = "data013",
    transfer_dir: Optional[Path] = None,
    datademo_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    """
    Process JSON files and transcripts to generate detailed reports.
    
    Args:
        start: Starting file name (e.g., "data001")
        end: Ending file name (e.g., "data013")
        transfer_dir: Path to transfer directory with JSON files (defaults to project_root/transfer)
        datademo_dir: Path to datademo directory with transcripts (defaults to project_root/datademo)
        output_dir: Path to output directory (defaults to project_root/transfer)
    """
    # Set up paths
    if transfer_dir is None:
        transfer_dir = project_root / "transfer"
    if datademo_dir is None:
        datademo_dir = project_root / "datademo"
    if output_dir is None:
        output_dir = project_root / "transfer"
    
    # Create base output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract numbers from start/end strings
    start_match = re.search(r'data(\d+)', start)
    end_match = re.search(r'data(\d+)', end)
    
    if not start_match or not end_match:
        raise ValueError("start and end must be in format 'dataXXX' (e.g., 'data001', 'data013')")
    
    start_num = int(start_match.group(1))
    end_num = int(end_match.group(1))
    
    print(f"Processing reports from {start} to {end} (numbers {start_num} to {end_num})...")
    
    # Find matching file pairs
    file_pairs = find_json_and_transcript_files(
        transfer_dir,
        datademo_dir,
        start_num=start_num,
        end_num=end_num
    )
    
    if not file_pairs:
        print(f"No matching JSON and transcript files found in range {start} to {end}")
        return
    
    print(f"Found {len(file_pairs)} file pair(s) to process")
    
    # Initialize generator
    generator = DetailedReportGenerator()
    
    # Process each file pair
    for json_file, transcript_file, file_num, con_folder in file_pairs:
        print(f"\nProcessing data{file_num}...")
        print(f"  JSON: {json_file}")
        print(f"  Transcript: {transcript_file}")
        print(f"  Con folder: {con_folder}")
        
        try:
            # Create con-specific output directory
            con_output_dir = output_dir / con_folder
            con_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read JSON outline
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Read transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            # Generate detailed report
            report = generator.generate_report(json_data, transcript, file_num)
            
            # Save report
            report_output_file = con_output_dir / f"report{file_num}.md"
            with open(report_output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  Saved detailed report to {report_output_file}")
            
        except Exception as e:
            print(f"  Error processing data{file_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete! Reports saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate detailed meeting reports from JSON outlines and transcripts")
    parser.add_argument(
        "--start",
        type=str,
        default="data001",
        help="Starting file name (e.g., 'data001')"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="data013",
        help="Ending file name (e.g., 'data013')"
    )
    parser.add_argument(
        "--transfer-dir",
        type=str,
        help="Path to transfer directory with JSON files (defaults to project_root/transfer)"
    )
    parser.add_argument(
        "--datademo-dir",
        type=str,
        help="Path to datademo directory with transcripts (defaults to project_root/datademo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory (defaults to project_root/transfer)"
    )
    
    args = parser.parse_args()
    
    transfer_path = Path(args.transfer_dir) if args.transfer_dir else None
    datademo_path = Path(args.datademo_dir) if args.datademo_dir else None
    output_path = Path(args.output_dir) if args.output_dir else None
    
    process_reports(
        start=args.start,
        end=args.end,
        transfer_dir=transfer_path,
        datademo_dir=datademo_path,
        output_dir=output_path
    )
