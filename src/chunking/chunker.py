"""
This module is chunker for the project
Supports both standard hierarchical chunking and topic-aware graph chunking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import re

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.loader import Meeting
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP



@dataclass
class ChunkMetadata:
    meeting_id: str
    level: str
    chunk_id: str
    text: str
    #metadata: Dict[str, Any]
    metadata: Dict[str, Any]





class HierarchicalChunker:
    """
    Hierarchical chunker for meeting transcripts
    
    Architecture:
        Metadata Level       ← Meeting metadata (with topic filter)
          ↓ summarizes
        Summary Level        ← Meeting summaries (split by individual summaries)
          ↓ details_in
        Meeting Level        ← Structured meeting content (split by topics with smart splitting)
    
    Summary level splits by individual summary sections. Meeting level splits by topics,
    ensuring each topic stays intact when possible, with overlap when splitting is needed.
    """


    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the hierarchical chunker

        Args:
            chunk_size: int = CHUNK_SIZE
            chunk_overlap: int = CHUNK_OVERLAP
        
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "；", "，", " ", ""]
            )



    def chunk_all_levels(self, 
                         meetings: List[Meeting], 
                         include_chunk_level: bool = False) -> Dict[str, List[ChunkMetadata]]:
        """
        Chunk all levels for all meetings
        
        Args:
            meetings: list of meetings
            include_chunk_level: whether to include fine-grained chunk level (default: False)
            
        Returns:
            dictionary containing all level chunking results
        """
        # Initialize result dictionary
        result = {
            'metadata': [],
            'summary': [],
            'meeting': [],
        }
        
        if include_chunk_level:
            result['chunk'] = []
        
        # Process each meeting
        for meeting in meetings:
            result['metadata'].extend(self.chunk_metadata_level(meeting))
            result['summary'].extend(self.chunk_summary_level(meeting))
            result['meeting'].extend(self.chunk_meeting_level(meeting))
            
            if include_chunk_level:
                result['chunk'].extend(self.chunk_fine_grained_level(meeting))
        
        # Print statistics
        self._print_statistics(result)
        
        return result
    
    
    def _print_statistics(self, result: Dict[str, List[ChunkMetadata]]):
        """Print chunking statistics"""
        print(f"\n{'='*80}")
        print("HIERARCHICAL CHUNKING STATISTICS")
        print(f"{'='*80}")
        
        print(f"  Metadata level     : {len(result['metadata'])} chunks")
        print(f"  Summary level      : {len(result['summary'])} chunks")
        print(f"  Meeting level      : {len(result['meeting'])} chunks")
        
        if 'chunk' in result:
            if result['chunk']:
                print(f"  Chunk level        : {len(result['chunk'])} chunks [Legacy - Not Recommended]")
            else:
                print(f"  Chunk level        : [DISABLED]")
        else:
            print(f"  Chunk level        : [DISABLED] (use include_chunk_level=True to enable)")
        
        print(f"{'='*80}\n")


    def chunk_metadata_level(self, meeting: Meeting) -> List[ChunkMetadata]:
        """
        Metadata level: convert structured metadata (JSON) to text format for embedding
        """
        
        
        embedding_text_parts = [];
    
        
        # add brief summary (for semantic understanding)
        if meeting.summary_brief:
            embedding_text_parts.append(f"Brief Summary: {meeting.summary_brief}")
        
        
        actions = meeting.metadata.get('actions', [])
        if actions:
            for action in actions:
                description_plain =action.get('description_plain', '')
                if description_plain:
                    embedding_text_parts.append(description_plain)
                    
                    
        embedding_text = "\n".join(embedding_text_parts) if embedding_text_parts else ""
        
        metadata_dict ={
            'meeting_id': meeting.meeting_id,
            'title': meeting.title,
            'datetime': str(meeting.datetime),
            'duration_sec': meeting.duration_sec,
            'language': meeting.language,
            'meeting_type': meeting.meeting_type,
            'participants': meeting.participants,
            'organizations': meeting.organizations,
            'topics': meeting.topics,
            'keywords': meeting.keywords,
            'summary_brief': meeting.summary_brief
        }
        
        if actions:
            metadata_dict['actions'] = actions
        
        return [ChunkMetadata(
            meeting_id=meeting.meeting_id,
            level='metadata',
            chunk_id=f"{meeting.meeting_id}_metadata",
            text=embedding_text,  # this text will be embedded, for semantic search
            metadata= metadata_dict
        )]



    def chunk_summary_level(self, meeting: Meeting) -> List[ChunkMetadata]:
        """
        Summary level: treat the entire summary as a single text document.
        Split using bullet-level overlap when text exceeds 1200 characters.
        Extract summary_id(s) and reference(s) from each chunk to establish
        retrieval hierarchy mapping to meeting-level topics/actions.
        """
        chunks = []
        chunk_counter = 0

        # Split entire summary text with bullet-level overlap
        sub_chunks = self._split_with_bullet_overlap(meeting.summary_text, chunk_size=1200)

        for sub_chunk in sub_chunks:
            if not sub_chunk.strip():
                continue

            # Extract all summary_ids and references from this chunk
            summary_ids = self._extract_all_summary_ids(sub_chunk)
            references = self._extract_all_references(sub_chunk)

            chunk_id = f"{meeting.meeting_id}_summary_{chunk_counter}"

            full_text = f"{sub_chunk}\n\n" \
                      f"Meeting: {meeting.title}\n" \
                      f"Date: {meeting.datetime}"

            chunks.append(ChunkMetadata(
                meeting_id=meeting.meeting_id,
                level='summary',
                chunk_id=chunk_id,
                text=full_text,
                metadata={
                    'title': meeting.title,
                    'datetime': str(meeting.datetime),
                    'summary_ids': summary_ids,     # List of S001-T01, S001-A01, etc.
                    'references': references,       # List of M001-T01, M001-A01, etc.
                }
            ))
            chunk_counter += 1

        return chunks
    
    def _extract_all_summary_ids(self, text: str) -> List[str]:
        """Extract all summary IDs (e.g., S001-T01, S001-A01) from chunk text"""
        id_matches = re.findall(r'\[S\d+-[TA]\d+\]', text)
        return [m.strip('[]') for m in id_matches]

    def _extract_all_references(self, text: str) -> List[str]:
        """
        Extract all meeting-level references from chunk text.
        Returns list of references like ['M001-T01', 'M001-T02', 'M001-A01'].
        """
        references = []

        # Find all lines with "Reference:" pattern
        for line in text.split('\n'):
            if 'Reference:' in line:
                # Extract the reference part after "Reference:"
                ref_match = re.search(r'Reference:\s*(.+)', line)
                if ref_match:
                    ref_text = ref_match.group(1).strip()
                    # Extract all M###-T## or M###-A## patterns from the reference text
                    ref_ids = re.findall(r'M\d+-[TA]\d+', ref_text)
                    references.extend(ref_ids)

        return references

    def _split_with_bullet_overlap(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text at bullet point boundaries with bullet-level overlap.
        Each chunk (except the first) starts with the last bullet point from the previous chunk.
        """
        # Split text into segments (bullet points and non-bullet text)
        segments = []
        current_segment = []
        is_bullet = False
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('- '):
                # Save previous segment if exists
                if current_segment:
                    segments.append(('\n'.join(current_segment), is_bullet))
                # Start new bullet segment
                current_segment = [line]
                is_bullet = True
            else:
                # Continue current segment
                if current_segment:
                    current_segment.append(line)
                else:
                    # Start new non-bullet segment
                    current_segment = [line]
                    is_bullet = False
        
        # Save last segment
        if current_segment:
            segments.append(('\n'.join(current_segment), is_bullet))
        
        # If no segments found, return original text
        if not segments:
            return [text] if text.strip() else []
        
        # If no bullet points found, return original text as single chunk
        if not any(is_bullet for _, is_bullet in segments):
            return [text] if text.strip() else []
        
        # Group segments into chunks with bullet-level overlap
        chunks = []
        current_chunk = []
        current_size = 0
        last_bullet_segment = None
        
        for segment, is_bullet_seg in segments:
            segment_size = len(segment)
            
            # If adding this segment would exceed chunk_size, start a new chunk
            if current_size + segment_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with last bullet from previous chunk (overlap)
                if last_bullet_segment:
                    current_chunk = [last_bullet_segment, segment]
                    current_size = len(last_bullet_segment) + segment_size
                else:
                    current_chunk = [segment]
                    current_size = segment_size
            else:
                # Add to current chunk
                current_chunk.append(segment)
                current_size += segment_size
            
            # Update last bullet segment for overlap
            if is_bullet_seg:
                last_bullet_segment = segment
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def chunk_meeting_level(self, meeting: Meeting) -> List[ChunkMetadata]:
        """
        Meeting level: split by markdown headers (## Key Topics, ## Action Items),
        then split by topic/item entries (- **Topic Title:** or - **Responsible Person:**).
        Only use bullet-level overlap when a single entry exceeds 1200 characters.
        """
        chunks = []

        # Split by ## headers (Key Topics, Action Items, etc.)
        section_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Section")])
        sections = section_splitter.split_text(meeting.meeting_level_text)

        chunk_counter = 0

        for section in sections:
            section_name = section.metadata.get("Section", "Unknown")
            section_content = section.page_content.strip()
            if not section_content:
                continue

            # Split by entry headers: - **Topic Title:** or - **Responsible Person:**
            entries = self._split_into_entries(section_content)

            for entry in entries:
                if not entry.strip():
                    continue

                # Extract topic_id/action_id and reference
                entry_id = self._extract_id(entry)
                reference = self._extract_reference(entry)
                paragraph_indices = self._parse_reference_to_paragraph_indices(reference) if reference else []

                # Only split with bullet-level overlap if entry exceeds 1200 characters
                if len(entry) > 1200:
                    sub_chunks = self._split_with_bullet_overlap(entry, chunk_size=1200)
                else:
                    sub_chunks = [entry]

                for sub_chunk in sub_chunks:
                    chunk_id = f"{meeting.meeting_id}_meeting_{chunk_counter}"

                    full_text = f"{sub_chunk}\n\n" \
                              f"Meeting: {meeting.title}\n" \
                              f"Date: {meeting.datetime}"

                    chunks.append(ChunkMetadata(
                        meeting_id=meeting.meeting_id,
                        level='meeting',
                        chunk_id=chunk_id,
                        text=full_text,
                        metadata={
                            'title': meeting.title,
                            'datetime': str(meeting.datetime),
                            'section': section_name,
                            'entry_id': entry_id,
                            'reference': reference,
                            'paragraph_indices': paragraph_indices
                        }
                    ))
                    chunk_counter += 1

        return chunks


    def _split_into_entries(self, text: str) -> List[str]:
        """
        Split section content by entry headers.
        Entry headers: - **Topic Title:** or - **Responsible Person:**
        """
        # Pattern to match entry headers
        pattern = r'^-\s*\*\*(?:Topic Title|Responsible Person):\*\*'

        lines = text.split('\n')
        entries = []
        current_entry = []

        for line in lines:
            # Check if line starts a new entry
            if re.match(pattern, line.strip()):
                # Save previous entry if exists
                if current_entry:
                    entries.append('\n'.join(current_entry))
                # Start new entry
                current_entry = [line]
            else:
                # Add to current entry
                current_entry.append(line)

        # Save last entry
        if current_entry:
            entries.append('\n'.join(current_entry))

        return entries

    def _extract_id(self, text: str) -> Optional[str]:
        """Extract Topic id or Action id from chunk text"""
        id_match = re.search(r'-\s*(?:Topic id|Action id):\s*(.+)', text)
        if id_match:
            return id_match.group(1).strip()
        return None

    def _extract_reference(self, text: str) -> Optional[str]:
        """
        Extract Reference from chunk text.
        Supports both formats:
        - Meeting level: "- Reference: M001-T01"
        - Summary level: "   Reference: M001-T01" (indented)
        """
        # Try meeting level format first (with leading -)
        ref_match = re.search(r'-\s*Reference:\s*(.+)', text)
        if ref_match:
            return ref_match.group(1).strip()

        # Try summary level format (indented, no leading -)
        ref_match = re.search(r'\s+Reference:\s*(.+)', text)
        if ref_match:
            return ref_match.group(1).strip()

        return None
    
    
    def _parse_reference_to_paragraph_indices(self, reference: str) -> List[str]:
        """
        Convert a Reference string (e.g. "P001-P003,P005") into formatted paragraph indices.
        """
        if not reference:
            return []
        
        normalized = reference.replace('–', '-')
        paragraph_indices: List[str] = []
        
        for part in normalized.split(','):
            segment = part.strip()
            if not segment:
                continue
            
            if '-' in segment:
                start_raw, end_raw = segment.split('-', 1)
                start_num = start_raw.strip().lstrip('P')
                end_num = end_raw.strip().lstrip('P')
                if start_num.isdigit() and end_num.isdigit():
                    start = int(start_num)
                    end = int(end_num)
                    if start > end:
                        start, end = end, start
                    for value in range(start, end + 1):
                        formatted = f"#P{value:03d}"
                        if formatted not in paragraph_indices:
                            paragraph_indices.append(formatted)
            else:
                idx = segment.lstrip('P')
                if idx.isdigit():
                    formatted = f"#P{int(idx):03d}"
                    if formatted not in paragraph_indices:
                        paragraph_indices.append(formatted)
        
        return paragraph_indices


    def chunk_fine_grained_level(self, meeting: Meeting) -> List[ChunkMetadata]:
        """
        [LEGACY - NOT RECOMMENDED]
        Chunk level: the finest granularity, use RecursiveCharacterTextSplitter
        to split raw conversation text without semantic boundaries.
        
        This is kept for compatibility but not recommended. Use meeting level 
        with smart splitting instead, which preserves semantic coherence.
        """
        chunks = []
        
        # use langchain's text splitter on raw conversation
        text_chunks = self.splitter.split_text(meeting.conversation_text)
        
        for idx, text in enumerate(text_chunks):
            chunks.append(ChunkMetadata(
                meeting_id=meeting.meeting_id,
                level='chunk',
                chunk_id=f"{meeting.meeting_id}_chunk_{idx}",
                text=text,
                # metadata={
                #     'title': meeting.title,
                #     'datetime': str(meeting.datetime),
                #     'chunk_index': idx,
                #     'total_chunks': len(text_chunks)
                # }
                metadata={}
            ))
        
        return chunks
