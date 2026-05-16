"""
Person fuzzy matching utility
Uses Presidio + TheFuzz for robust person name matching against meeting participants
"""
from typing import List, Optional, Dict, Any
# pyrefly: ignore [missing-import]
from thefuzz import fuzz, process


class PersonMatcher:
    """Person fuzzy matcher using Presidio + TheFuzz"""
    
    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: Similarity threshold, below this value considered no match
        """
        self.threshold = threshold
        self._analyzer = None
    
    def _get_analyzer(self):
        """Lazy load Presidio analyzer"""
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
        return self._analyzer
    
    def detect_person_entities(self, text: str, participants: Optional[List[str]] = None) -> List[str]:
        """
        Detect person entities in text using Presidio
        
        Args:
            text: Input text
            participants: Not used directly - kept for API compatibility
            
        Returns:
            List of detected person names from Presidio
        """
        analyzer = self._get_analyzer()
        results = analyzer.analyze(text=text, language='en')
        persons = [text[ent.start:ent.end] for ent in results if ent.entity_type == "PERSON"]
        return persons
    
    def find_match(self, fuzzy_name: str, candidates: List[str]) -> Optional[str]:
        """
        Find best match using TheFuzz against candidate list
        
        Args:
            fuzzy_name: Fuzzy input
            candidates: Candidate list from pandas (meeting participants)
            
        Returns:
            Matched name or None
        """
        if not candidates:
            return None
        
        # Use TheFuzz for fuzzy matching
        matches = process.extract(
            fuzzy_name,
            candidates,
            scorer=fuzz.token_sort_ratio,
            limit=3
        )
        
        # Check threshold
        for match_name, score in matches:
            if score / 100 >= self.threshold:
                return match_name
        
        return None
    
    def find_all_matches(self, fuzzy_name: str, candidates: List[str]) -> List[str]:
        """
        Find all matches (multiple possibilities)
        
        Args:
            fuzzy_name: Fuzzy input
            candidates: Candidate list
            
        Returns:
            All matched names list
        """
        if not candidates:
            return []
        
        matches = process.extract(
            fuzzy_name,
            candidates,
            scorer=fuzz.token_sort_ratio,
            limit=5
        )
        
        result = []
        for match_name, score in matches:
            if score / 100 >= self.threshold:
                result.append(match_name)
        
        return result
    
    
    # Todo: I keep it, future may still needs it.
    def resolve_pronoun(self, pronoun: str, context_participants: List[str]) -> Optional[str]:
        """
        Resolve pronoun
        
        Args:
            pronoun: Pronoun (he, she, they etc)
            context_participants: Participants in context
            
        Returns:
            Resolved person name or None
        """
        pronoun_lower = pronoun.lower().strip()
        
        # If only one participant, return directly
        if len(context_participants) == 1:
            return context_participants[0]
        
        # Multiple participants - pronoun needs context to resolve
        return None
    
    def detect_and_resolve(self, text: str, participants: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect person entities and resolve to canonical names from meeting participants
        
        Flow:
        1. Use Presidio to detect full person names in text
        2. Also use TheFuzz to fuzzy match text against pandas participants (handles partial names)
        3. Combine results and resolve to canonical names
        
        Args:
            text: Input text
            participants: Participant list from pandas (meeting participants)
            
        Returns:
            Dict with detected entities and resolved names
        """
        # Get participants from pandas if not provided
        if participants is None:
            from sub_agents.metadata_manager import get_meetings_metadata
            current_user = getattr(self, 'current_user', "")
            mdf = get_meetings_metadata(current_user)
            participants = mdf.get_all_participants()
        
        # Step 1: Detect persons using Presidio (good for full names)
        detected_by_presidio = self.detect_person_entities(text, participants)
        
        # Step 2: Use TheFuzz to find fuzzy matches in text against participants
        # This handles partial names like "Hongye" -> "Hongye Qian"
        # Only match if the text length is reasonable (avoid false positives like "weather")
        detected_by_fuzz = []  # Store original matched text (e.g., "Hongye")
        text_lower = text.lower()
        for participant in participants:
            # Skip generic/placeholder names
            if participant.lower() in ["host", "speaker", "unknown speaker", "speaker 1", "speaker 2", 
                                        "speaker 3", "speaker 4", "speaker 5"]:
                continue
            
            # Check if any part of the participant name appears in the text
            parts = participant.split()
            for part in parts:
                # Skip short parts (less than 4 chars) to avoid false positives
                if len(part) < 4:
                    continue
                # Use token_set_ratio for better matching
                match_score = fuzz.token_set_ratio(part.lower(), text_lower)
                if match_score >= 80:  # Higher threshold for detection
                    detected_by_fuzz.append(part)  # Store original matched part (e.g., "Hongye")
                    break
        
        # Combine results (union)
        all_detected = list(set(detected_by_presidio + detected_by_fuzz))
        
        # Step 3: Resolve to canonical names using TheFuzz
        resolved = []
        for person in all_detected:
            match = self.find_match(person, participants)  # e.g., "Hongye" -> "Hongye Qian"
            if match:
                resolved.append(match)
        
        # Remove duplicates while preserving order
        resolved = list(dict.fromkeys(resolved))
        
        return {
            "detected": all_detected,
            "resolved": resolved,
            "participants": participants
        }


# ============ Module-level Singleton ============

_person_matcher_instance = None

def get_person_matcher(current_user: str = "", threshold: float = 0.6) -> "PersonMatcher":
    """Get or create the module-level PersonMatcher singleton.
    
    Avoids re-initialising AnalyzerEngine on every query.
    The instance (and its internal Presidio analyzer) is created once
    and reused for the lifetime of the process.
    """
    global _person_matcher_instance
    if _person_matcher_instance is None:
        _person_matcher_instance = PersonMatcher(threshold=threshold)
    _person_matcher_instance.current_user = current_user
    return _person_matcher_instance


# ============ Test Code ============
if __name__ == "__main__":
    # Test with mock participants
    test_participants = [
        "Hongye Qian", 
        "Ankit Kumar", 
        "Jensen Huang", 
        "Satya Nadella",
        "Elon Musk"
    ]
    
    matcher = PersonMatcher(threshold=0.6)
    
    # Test detect_and_resolve
    print("=== Test detect_and_resolve ===")
    test_texts = [
        "What did Hongye discuss?",
        "What did Ankit and Jensen talk about?",
        "What did I discuss in the meeting?",
    ]
    
    for text in test_texts:
        result = matcher.detect_and_resolve(text, test_participants)
        print(f"Text: {text}")
        print(f"  Detected (Presidio): {result['detected']}")
        print(f"  Resolved (TheFuzz): {result['resolved']}\n")
    
    # Test fuzzy matching
    print("=== Test fuzzy matching ===")
    test_names = ["Hongye", "Qian", "Ankit", "Jensen", "Unknown"]
    
    for name in test_names:
        match = matcher.find_match(name, test_participants)
        all_matches = matcher.find_all_matches(name, test_participants)
        print(f"'{name}' -> best match: {match}, all matches: {all_matches}")
