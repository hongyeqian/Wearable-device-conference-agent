"""
This module is data loader for the project
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Meeting:
    # from metaData.json
    meeting_id: str
    title: str
    datetime: datetime
    participants: List[Dict[str, Any]]
    organizations: List[str]
    topics: List[str]
    keywords: List[str]
    summary_brief: str

    # from data*.md
    conversation_text: str
    # from meetLevel*.md
    meeting_level_text: str
    # from summary*.md
    summary_text: str

    # from metaData.json
    metadata: Dict[str, Any]



class DataLoader:
    """Load data from datademo"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        
    def load_all_meetings(self) -> List[Meeting]:
        """Load all meetings data"""
        meetings = []
        
        # iterate all con* folders
        for con_dir in sorted(self.data_dir.glob("con*")):
            if not con_dir.is_dir():
                continue
                
            try:
                meeting = self._load_meeting(con_dir)
                meetings.append(meeting)
                print(f"Success Loaded: {meeting.meeting_id} - {meeting.title}")
            except Exception as e:
                print(f"Fail Error loading {con_dir.name}: {e}")
                
        print(f"\n Total meetings loaded: {len(meetings)}")
        return meetings
    
    def _load_meeting(self, con_dir: Path) -> Meeting:
        """Load single meeting data"""
        # 1. read metadata JSON
        metadata_file = self._find_file(con_dir, "metaData*.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 2. read text files
        conversation_file = self._find_file(con_dir, metadata['related_files']['transcript'])
        meeting_level_file = self._find_file(con_dir, metadata['related_files']['meeting_summary'])
        summary_file = self._find_file(con_dir, metadata['related_files']['summary_embedding'])
        
        # Add paragraph index to conversation file
        self._add_paragraph_index(conversation_file)
        
        conversation_text = self._read_text_file(conversation_file)
        meeting_level_text = self._read_text_file(meeting_level_file)
        summary_text = self._read_text_file(summary_file)
        
        # 3. construct Meeting object
        meeting = Meeting(
            meeting_id=metadata['meeting_id'],
            title=metadata['title'],
            datetime=metadata['datetime'],
            participants=metadata['participants'],
            organizations=metadata['organizations'],
            topics=metadata['topics'],
            keywords=metadata['keywords'],
            summary_brief=metadata['summary_brief'],
            conversation_text=conversation_text,
            meeting_level_text=meeting_level_text,
            summary_text=summary_text,
            metadata=metadata
        )

        # 4. Generate and save summary_metadata.json
        self._generate_summary_metadata(meeting, con_dir)

        return meeting

    def _generate_summary_metadata(self, meeting: Meeting, con_dir: Path) -> None:
        """
        Generate summary_metadata.json by parsing summary_text and metadata.
        This file contains structured topic and action information for filtering.
        """
        # Extract topics from summary_text
        # Format: [S013-T01] Topic: Topic Name
        topics = []
        topic_pattern = re.compile(r'\[S\d+-T\d+\]\s*Topic:\s*(.+?)(?:\n|$)')
        for match in topic_pattern.finditer(meeting.summary_text):
            topic_name = match.group(1).strip()
            # Remove markdown formatting like ** at the end
            topic_name = topic_name.rstrip('*').strip()
            if topic_name:
                topics.append(topic_name)

        # Extract actions from summary_text
        # Format: [S013-A01] Action: Description
        actions = []
        action_pattern = re.compile(r'\[S\d+-A\d+\]\s*Action:\s*(.+?)(?:\n|$)')
        for match in action_pattern.finditer(meeting.summary_text):
            action_desc = match.group(1).strip()
            # Remove markdown formatting like ** at the end
            action_desc = action_desc.rstrip('*').strip()
            if action_desc:
                # Try to extract assignee from the action description
                # Format often: "Action: Name to do something"
                assignee = None
                if " to " in action_desc:
                    # Pattern: "Hongye Qian to Begin with..."
                    parts = action_desc.split(" to ", 1)
                    potential_assignee = parts[0].strip()
                    # Check if this matches a participant name
                    for p in meeting.participants:
                        if isinstance(p, dict):
                            p_name = p.get("name", "")
                        else:
                            p_name = str(p)
                        if p_name.lower() == potential_assignee.lower():
                            assignee = p_name
                            break
                        # Also check if participant name is in the text
                        if potential_assignee.lower() in p_name.lower():
                            assignee = p_name
                            break

                # Also check metadata for action info
                metadata_actions = meeting.metadata.get("actions", [])
                action_id = None
                for md_action in metadata_actions:
                    if md_action.get("description_plain", "").lower() in action_desc.lower():
                        action_id = md_action.get("action_id")
                        if not assignee and isinstance(md_action.get("assignee"), dict):
                            assignee = md_action["assignee"].get("name")
                        break

                action_entry = {
                    "action_id": action_id or f"{meeting.meeting_id.replace('data', 'M')}-A{len(actions)+1:02d}",
                    "task": action_desc,
                }
                if assignee:
                    action_entry["assignee"] = assignee
                actions.append(action_entry)

        # Extract participant names
        participants = []
        for p in meeting.participants:
            if isinstance(p, dict):
                name = p.get("name", "").strip()
            else:
                name = str(p).strip()
            if name:
                participants.append(name)

        # Handle datetime - can be string or None
        datetime_str = None
        if meeting.datetime:
            if isinstance(meeting.datetime, str):
                datetime_str = meeting.datetime
            elif hasattr(meeting.datetime, 'isoformat'):
                datetime_str = meeting.datetime.isoformat()

        # Build the summary metadata
        summary_metadata = {
            "meeting_id": meeting.meeting_id,
            "datetime": datetime_str,
            "participants": participants,
            "topics": topics if topics else meeting.topics,  # Fallback to metadata topics if none parsed
            "actions": actions
        }

        # Save to file
        output_file = con_dir / "summary_metadata.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_metadata, f, ensure_ascii=False, indent=2)
    
    def _find_file(self, directory: Path, filename: str) -> Path:
        """find file (support wildcard and case-insensitive)"""
        # try direct match first
        direct_path = directory / filename
        if direct_path.exists():
            return direct_path
        
        # try wildcard match
        matches = list(directory.glob(filename))
        if matches:
            return matches[0]
        
        # try case-insensitive match
        for file in directory.iterdir():
            if file.name.lower() == filename.lower():
                return file
        
        raise FileNotFoundError(f"File not found: {filename} in {directory}")
    
    def _read_text_file(self, filepath: Path) -> str:
        """read text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _add_paragraph_index(self, filepath: Path) -> None:
        """
        Add paragraph index to conversation text file.
        Add index at the start of each paragraph (non-empty line after empty line or file start).
        Skip title lines (starting with #) and lines that already have index.
        Index format: [#P001], [#P002], etc.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        indexed_lines = []
        paragraph_counter = 1
        prev_line_empty = True  # Treat file start as "previous line empty"
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                indexed_lines.append(line)
                prev_line_empty = True
                continue
            
            # Skip title lines (starting with #)
            if stripped.startswith('#'):
                indexed_lines.append(line)
                prev_line_empty = False
                continue
            
            # Skip lines that already have index
            if '[#P' in stripped:
                indexed_lines.append(line)
                prev_line_empty = False
                continue
            
            # If previous line was empty (or file start), this is a new paragraph
            if prev_line_empty:
                # Add paragraph index at the beginning of the line
                index_tag = f"[#P{paragraph_counter:03d}] "
                leading_spaces = len(line) - len(line.lstrip())
                indexed_lines.append(' ' * leading_spaces + index_tag + line.lstrip())
                paragraph_counter += 1
            else:
                indexed_lines.append(line)
            
            prev_line_empty = False
        
        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(indexed_lines)


# testing code
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from config.settings import DATA_DIR
    
    loader = DataLoader(DATA_DIR)
    meetings = loader.load_all_meetings()
    
    # print first meeting information
    if meetings:
        m = meetings[3]
        print(f"1.  meeting_id: {m.meeting_id}")
        print(f"2.  title: {m.title}")
        print(f"3.  datetime: {m.datetime}")
        print(f"4.  participants: {m.participants}")
        print(f"5.  organizations: {m.organizations}")
        print(f"6.  topics: {m.topics}")
        print(f"7.  keywords: {m.keywords}")
        print(f"8.  summary_brief: {m.summary_brief}")
        print(f"9.  conversation_text: {m.conversation_text[:100]}...")
        print(f"10. meeting_level_text: {m.meeting_level_text[:100]}...")
        print(f"11. summary_text: {m.summary_text[:100]}...")
        print(f"12. metadata: {m.metadata}")
