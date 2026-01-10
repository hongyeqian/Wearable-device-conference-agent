"""
Generate Structured Meeting Summaries
Processes data00*.md files from datademo directory and generates structured summaries
using OpenAI GPT-4 mini with JSON output format.
"""
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from config.settings import OPENAI_API_KEY


import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]


# Pydantic model for structured JSON output
class BulletPoint(BaseModel):
    """Single bullet point under a second-level topic"""
    point: str = Field(description="The bullet point text")

class SecondLevelTopic(BaseModel):
    """Second-level topic with its bullet points"""
    topic_name: str = Field(description="Name of the second-level topic")
    bullet_points: List[BulletPoint] = Field(description="List of bullet points under this topic")

class MeetingSummary(BaseModel):
    """Complete meeting summary structure"""
    high_level_topic: str = Field(description="The high-level topic summarizing the overall meeting")
    second_level_topics: List[SecondLevelTopic] = Field(description="List of 2-5 second-level topics with their bullet points")

# System prompt for meeting summarization
MEETING_SUMMARY_PROMPT = """You are a meeting summarization assistant.

Your input: a raw transcript.

Your task: extract a structured summary strictly in the following format:

# Discussion on <High-Level Topic>

- **<Second-Level Topic 1>** (the level 2 topic should be bolded)
  - Bullet point 1
  - Bullet point 2
  - Bullet point 3 (only if exists)

- **<Second-Level Topic 2>** (the level 2 topic should be bolded)
  - Bullet point 1
  - Bullet point 2

- **<Second-Level Topic 3>** (the level 2 topic should be bolded)
  - Bullet point 1
  - Bullet point 2

Rules:
- Identify only 1 high-level topic name summarizing the overall meeting.
- Extract 2–10 second-level topics discussed.
- Under each second-level topic, write 3–6 bullet points capturing key decisions / ideas.
- Do NOT add extra commentary or reorder content logically if not present.
- Only use information explicitly from the transcript.
- DO NOT make up any content that is not in the transcript.
- If something is not mentioned in the transcript, do not include it.

Now take the following transcript as input and produce the output in that exact structure:"""


class StructuredSummaryGenerator:
    """Generate structured meeting summaries using OpenAI GPT-4 mini"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        """
        Initialize the summary generator.
        
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
    
    def generate_summary(self, transcript: str) -> Dict[str, Any]:
        """
        Generate structured summary from transcript.
        
        Args:
            transcript: The raw transcript text
            
        Returns:
            Dictionary containing:
            - formatted_text: str (the formatted markdown-style summary)
            - json_data: dict (the structured JSON with high_level_topic and second_level_topics)
        """
        try:
            # Use OpenAI structured outputs with Pydantic model
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": MEETING_SUMMARY_PROMPT},
                    {"role": "user", "content": f"<<TRANSCRIPT>>\n\n{transcript}"}
                ],
                response_format=MeetingSummary,
                temperature=self.temperature
            )
            
            # Get parsed JSON structure
            parsed_summary = response.choices[0].message.parsed
            
            # Convert to dict
            json_data = parsed_summary.model_dump()
            
            # Generate formatted text output
            formatted_text = self._format_summary_text(json_data)
            
            return {
                "formatted_text": formatted_text,
                "json_data": json_data
            }
            
        except Exception as e:
            # Fallback: try with JSON mode if structured outputs not available
            print(f"Warning: Structured outputs failed ({e}), trying JSON mode...")
            return self._generate_with_json_mode(transcript)
    
    def _generate_with_json_mode(self, transcript: str) -> Dict[str, Any]:
        """
        Fallback method using JSON mode for older models.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MEETING_SUMMARY_PROMPT + "\n\nYou must return a JSON object with the following structure:\n{\n  \"high_level_topic\": \"<topic>\",\n  \"second_level_topics\": [\n    {\n      \"topic_name\": \"<topic name>\",\n      \"bullet_points\": [\n        {\"point\": \"<point 1>\"},\n        {\"point\": \"<point 2>\"}\n      ]\n    }\n  ]\n}"},
                {"role": "user", "content": f"<<TRANSCRIPT>>\n\n{transcript}"}
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature
        )
        
        content = response.choices[0].message.content
        json_data = json.loads(content)
        
        # Generate formatted text output
        formatted_text = self._format_summary_text(json_data)
        
        return {
            "formatted_text": formatted_text,
            "json_data": json_data
        }
    
    def _format_summary_text(self, json_data: Dict[str, Any]) -> str:
        """
        Format JSON data into the required text structure.
        
        Args:
            json_data: Dictionary with high_level_topic and second_level_topics
            
        Returns:
            Formatted text string
        """
        lines = []
        lines.append(f"Discussion on {json_data['high_level_topic']}\n")
        
        for topic in json_data['second_level_topics']:
            lines.append(f"- {topic['topic_name']}")
            for bullet in topic['bullet_points']:
                lines.append(f"  - {bullet['point']}")
            lines.append("")  # Empty line between topics
        
        return "\n".join(lines).strip()


def find_data_files(datademo_dir: Path, start_num: Optional[int] = None, end_num: Optional[int] = None) -> List[Path]:
    """
    Find all data00*.md files in datademo directory, optionally filtered by number range.
    
    Args:
        datademo_dir: Path to datademo directory
        start_num: Starting number (e.g., 1 for data001)
        end_num: Ending number (e.g., 13 for data013)
        
    Returns:
        List of paths to data files, sorted by number
    """
    data_files = []
    
    # Find all data00*.md files recursively
    for data_file in datademo_dir.rglob("data*.md"):
        # Extract number from filename (e.g., "data001.md" -> 1)
        match = re.search(r'data(\d+)\.md', data_file.name)
        if match:
            file_num = int(match.group(1))
            
            # Apply filters if specified
            if start_num is not None and file_num < start_num:
                continue
            if end_num is not None and file_num > end_num:
                continue
            
            data_files.append((file_num, data_file))
    
    # Sort by number
    data_files.sort(key=lambda x: x[0])
    
    return [path for _, path in data_files]


def process_data_files(
    start: str = "data001",
    end: str = "data013",
    datademo_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    """
    Process data files and generate structured summaries.
    
    Args:
        start: Starting file name (e.g., "data001")
        end: Ending file name (e.g., "data013")
        datademo_dir: Path to datademo directory (defaults to project_root/datademo)
        output_dir: Path to output directory (defaults to project_root/transfer)
    """
    # Set up paths
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
    
    print(f"Processing files from {start} to {end} (numbers {start_num} to {end_num})...")
    
    # Find all matching data files
    data_files = find_data_files(datademo_dir, start_num=start_num, end_num=end_num)
    
    if not data_files:
        print(f"No data files found in range {start} to {end}")
        return
    
    print(f"Found {len(data_files)} file(s) to process")
    
    # Initialize generator
    generator = StructuredSummaryGenerator()
    
    # Process each file
    for data_file in data_files:
        file_num = re.search(r'data(\d+)', data_file.name).group(1)
        print(f"\nProcessing {data_file.name}...")
        
        try:
            # Extract con folder name from file path (e.g., "con1" from "datademo/con1/data001.md")
            con_match = re.search(r'con\d+', str(data_file))
            if con_match:
                con_folder = con_match.group(0)
            else:
                # Fallback: try to find con folder in parent directories
                con_folder = None
                for parent in data_file.parents:
                    if parent.name.startswith('con'):
                        con_folder = parent.name
                        break
                if not con_folder:
                    con_folder = f"con{file_num.zfill(3)}"  # Default fallback
            
            # Create con-specific output directory
            con_output_dir = output_dir / con_folder
            con_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read transcript
            with open(data_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            # Generate summary
            result = generator.generate_summary(transcript)
            
            # Save formatted text output
            text_output_file = con_output_dir / f"data{file_num}.md"
            with open(text_output_file, 'w', encoding='utf-8') as f:
                f.write(result['formatted_text'])
            print(f"  Saved formatted text to {text_output_file}")
            
            # Save JSON output
            json_output_file = con_output_dir / f"data{file_num}.json"
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(result['json_data'], f, indent=2, ensure_ascii=False)
            print(f"  Saved JSON data to {json_output_file}")
            
        except Exception as e:
            print(f"  Error processing {data_file.name}: {e}")
            continue
    
    print(f"\nProcessing complete! Output saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate structured meeting summaries from data files")
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
        "--datademo-dir",
        type=str,
        help="Path to datademo directory (defaults to project_root/datademo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory (defaults to project_root/transfer)"
    )
    
    args = parser.parse_args()
    
    datademo_path = Path(args.datademo_dir) if args.datademo_dir else None
    output_path = Path(args.output_dir) if args.output_dir else None
    
    process_data_files(
        start=args.start,
        end=args.end,
        datademo_dir=datademo_path,
        output_dir=output_path
    )
