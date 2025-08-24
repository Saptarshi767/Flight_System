"""
Intent recognition and entity extraction using spaCy.

This module provides advanced intent recognition capabilities for flight-related queries
using spaCy's NLP pipeline and custom patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import re
from datetime import datetime, timedelta
from dataclasses import dataclass

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EntityMatch:
    """Represents an extracted entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float


@dataclass
class IntentResult:
    """Result of intent recognition with confidence scores."""
    primary_intent: str
    confidence: float
    secondary_intents: List[Tuple[str, float]]
    entities: List[EntityMatch]
    patterns_matched: List[str]


class FlightIntentRecognizer:
    """
    Advanced intent recognition for flight-related queries using spaCy.
    
    Recognizes intents like:
    - Best time to fly
    - Delay analysis
    - Congestion patterns
    - Schedule optimization
    - Cascading impact analysis
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the intent recognizer.
        
        Args:
            model_name: spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Please install with: python -m spacy download {model_name}")
            raise
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Define intent patterns
        self._setup_intent_patterns()
        
        # Define entity patterns
        self._setup_entity_patterns()
        
        # Airport and airline mappings
        self._setup_domain_knowledge()
        
        logger.info("FlightIntentRecognizer initialized successfully")
    
    def _setup_intent_patterns(self):
        """Set up pattern matching for different intents."""
        
        # Best time intent patterns
        best_time_patterns = [
            [{"LOWER": {"IN": ["best", "optimal", "good", "ideal"}}],
            [{"LOWER": "when"}, {"LOWER": "to"}, {"LOWER": {"IN": ["fly", "travel", "depart", "leave"]}}],
            [{"LOWER": {"IN": ["avoid", "minimize"}}}, {"LOWER": {"IN": ["delays", "delay"]}}],
            [{"LOWER": "least"}, {"LOWER": {"IN": ["delayed", "busy", "crowded"]}}],
            [{"LOWER": {"IN": ["recommend", "suggest"]}}, {"LOWER": {"IN": ["time", "slot"]}}]
        ]
        
        for i, pattern in enumerate(best_time_patterns):
            self.matcher.add(f"BEST_TIME_{i}", [pattern])
        
        # Delay analysis patterns
        delay_patterns = [
            [{"LOWER": {"IN": ["delay", "delays", "delayed"]}}],
            [{"LOWER": {"IN": ["late", "punctual", "on-time", "ontime"]}}],
            [{"LOWER": "delay"}, {"LOWER": {"IN": ["analysis", "pattern", "statistics", "stats"]}}],
            [{"LOWER": {"IN": ["causes", "reasons"]}}, {"LOWER": "of"}, {"LOWER": "delay"}],
            [{"LOWER": {"IN": ["flight", "flights"]}}, {"LOWER": {"IN": ["delayed", "late"]}}]
        ]
        
        for i, pattern in enumerate(delay_patterns):
            self.matcher.add(f"DELAY_ANALYSIS_{i}", [pattern])
        
        # Congestion patterns
        congestion_patterns = [
            [{"LOWER": {"IN": ["busy", "busiest", "crowded", "congested"]}}],
            [{"LOWER": {"IN": ["peak", "rush"]}}, {"LOWER": {"IN": ["hour", "hours", "time"]}}],
            [{"LOWER": {"IN": ["traffic", "congestion", "volume"]}}],
            [{"LOWER": {"IN": ["avoid", "skip"]}}, {"LOWER": {"IN": ["busy", "crowded", "peak"]}}],
            [{"LOWER": {"IN": ["quietest", "least", "slowest"]}}, {"LOWER": {"IN": ["busy", "crowded"]}}]
        ]
        
        for i, pattern in enumerate(congestion_patterns):
            self.matcher.add(f"CONGESTION_{i}", [pattern])
        
        # Schedule impact patterns
        schedule_patterns = [
            [{"LOWER": "schedule"}, {"LOWER": {"IN": ["change", "modification", "adjustment"]}}],
            [{"LOWER": {"IN": ["what", "how"]}}, {"LOWER": "if"}],
            [{"LOWER": {"IN": ["reschedule", "rescheduling"]}}],
            [{"LOWER": "schedule"}, {"LOWER": {"IN": ["optimization", "tuning", "improvement"]}}],
            [{"LOWER": {"IN": ["impact", "effect"]}}, {"LOWER": "of"}, {"LOWER": {"IN": ["changing", "modifying"]}}]
        ]
        
        for i, pattern in enumerate(schedule_patterns):
            self.matcher.add(f"SCHEDULE_IMPACT_{i}", [pattern])
        
        # Cascading impact patterns
        cascading_patterns = [
            [{"LOWER": {"IN": ["cascading", "cascade", "domino"]}}],
            [{"LOWER": {"IN": ["critical", "important"]}}, {"LOWER": {"IN": ["flight", "flights"]}}],
            [{"LOWER": {"IN": ["network", "connection"]}}, {"LOWER": {"IN": ["impact", "effect"]}}],
            [{"LOWER": {"IN": ["downstream", "upstream"]}}, {"LOWER": {"IN": ["effect", "impact"]}}],
            [{"LOWER": {"IN": ["connected", "connecting"]}}, {"LOWER": {"IN": ["flight", "flights"]}}]
        ]
        
        for i, pattern in enumerate(cascading_patterns):
            self.matcher.add(f"CASCADING_IMPACT_{i}", [pattern])
    
    def _setup_entity_patterns(self):
        """Set up patterns for extracting flight-related entities."""
        
        # Flight number patterns
        flight_patterns = [
            [{"TEXT": {"REGEX": r"^[A-Z]{2,3}\d{1,4}$"}}],  # AA123, UAL1234
            [{"LOWER": {"IN": ["flight", "flt"]}}, {"TEXT": {"REGEX": r"^[A-Z]{2,3}\d{1,4}$"}}]
        ]
        
        for i, pattern in enumerate(flight_patterns):
            self.matcher.add(f"FLIGHT_NUMBER_{i}", [pattern])
        
        # Time patterns
        time_patterns = [
            [{"TEXT": {"REGEX": r"^\d{1,2}:\d{2}$"}}],  # 14:30
            [{"TEXT": {"REGEX": r"^\d{1,2}(am|pm)$"}}],  # 2pm
            [{"LOWER": {"IN": ["morning", "afternoon", "evening", "night"]}}],
            [{"LOWER": {"IN": ["early", "late"]}}, {"LOWER": {"IN": ["morning", "afternoon", "evening"]}}]
        ]
        
        for i, pattern in enumerate(time_patterns):
            self.matcher.add(f"TIME_REFERENCE_{i}", [pattern])
    
    def _setup_domain_knowledge(self):
        """Set up domain-specific knowledge for airports and airlines."""
        
        # Airport mappings
        self.airport_mappings = {
            # Indian airports
            'mumbai': 'BOM',
            'bombay': 'BOM',
            'bom': 'BOM',
            'delhi': 'DEL',
            'new delhi': 'DEL',
            'del': 'DEL',
            'bangalore': 'BLR',
            'bengaluru': 'BLR',
            'blr': 'BLR',
            'chennai': 'MAA',
            'maa': 'MAA',
            'kolkata': 'CCU',
            'calcutta': 'CCU',
            'ccu': 'CCU',
            'hyderabad': 'HYD',
            'hyd': 'HYD',
            'pune': 'PNQ',
            'pnq': 'PNQ',
            'ahmedabad': 'AMD',
            'amd': 'AMD',
            'kochi': 'COK',
            'cochin': 'COK',
            'cok': 'COK',
            'goa': 'GOI',
            'goi': 'GOI',
            
            # International airports (common ones)
            'london': 'LHR',
            'heathrow': 'LHR',
            'lhr': 'LHR',
            'dubai': 'DXB',
            'dxb': 'DXB',
            'singapore': 'SIN',
            'sin': 'SIN',
            'hong kong': 'HKG',
            'hkg': 'HKG',
            'new york': 'JFK',
            'jfk': 'JFK',
            'los angeles': 'LAX',
            'lax': 'LAX'
        }
        
        # Common airline mappings
        self.airline_mappings = {
            'airline a': 'AA',
            'airline b': 'AB',
            'airline c': 'AC',
            'airline d': 'AD',
            'airline e': 'AE',
            'airline f': 'AF',
            'emirates': 'EK',
            'qatar airways': 'QR',
            'singapore airlines': 'SQ',
            'lufthansa': 'LH',
            'british airways': 'BA',
            'american airlines': 'AA',
            'united airlines': 'UA',
            'delta': 'DL'
        }
        
        # Create phrase matchers for airports and airlines
        airport_patterns = [self.nlp(airport) for airport in self.airport_mappings.keys()]
        self.phrase_matcher.add("AIRPORT", airport_patterns)
        
        airline_patterns = [self.nlp(airline) for airline in self.airline_mappings.keys()]
        self.phrase_matcher.add("AIRLINE", airline_patterns)
    
    def recognize_intent(self, text: str) -> IntentResult:
        """
        Recognize intent from text using pattern matching and NLP.
        
        Args:
            text: Input text to analyze
            
        Returns:
            IntentResult with recognized intent and entities
        """
        # Process text with spaCy
        doc = self.nlp(text.lower())
        
        # Find pattern matches
        matches = self.matcher(doc)
        phrase_matches = self.phrase_matcher(doc)
        
        # Count intent matches
        intent_scores = {
            'best_time': 0,
            'delay_analysis': 0,
            'congestion': 0,
            'schedule_impact': 0,
            'cascading_impact': 0,
            'general_info': 0
        }
        
        patterns_matched = []
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            patterns_matched.append(label)
            
            if 'BEST_TIME' in label:
                intent_scores['best_time'] += 1
            elif 'DELAY_ANALYSIS' in label:
                intent_scores['delay_analysis'] += 1
            elif 'CONGESTION' in label:
                intent_scores['congestion'] += 1
            elif 'SCHEDULE_IMPACT' in label:
                intent_scores['schedule_impact'] += 1
            elif 'CASCADING_IMPACT' in label:
                intent_scores['cascading_impact'] += 1
        
        # Extract entities
        entities = self._extract_entities(doc, phrase_matches)
        
        # Determine primary intent
        if max(intent_scores.values()) == 0:
            primary_intent = 'general_info'
            confidence = 0.5
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[primary_intent] / 3.0, 1.0)  # Normalize confidence
        
        # Get secondary intents
        secondary_intents = [
            (intent, score / 3.0) for intent, score in intent_scores.items() 
            if intent != primary_intent and score > 0
        ]
        secondary_intents.sort(key=lambda x: x[1], reverse=True)
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents[:2],  # Top 2 secondary intents
            entities=entities,
            patterns_matched=patterns_matched
        )
    
    def _extract_entities(self, doc: Doc, phrase_matches: List[Tuple]) -> List[EntityMatch]:
        """
        Extract entities from the processed document.
        
        Args:
            doc: spaCy processed document
            phrase_matches: Phrase matcher results
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract named entities from spaCy
        for ent in doc.ents:
            entities.append(EntityMatch(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.8  # Default confidence for spaCy entities
            ))
        
        # Extract phrase matches (airports, airlines)
        for match_id, start, end in phrase_matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            entities.append(EntityMatch(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char,
                confidence=0.9  # High confidence for exact phrase matches
            ))
        
        # Extract flight numbers using regex
        flight_number_pattern = re.compile(r'\b[A-Z]{2,3}\d{1,4}\b')
        for match in flight_number_pattern.finditer(doc.text.upper()):
            entities.append(EntityMatch(
                text=match.group(),
                label="FLIGHT_NUMBER",
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
        
        # Extract time references
        time_pattern = re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}(am|pm)\b', re.IGNORECASE)
        for match in time_pattern.finditer(doc.text):
            entities.append(EntityMatch(
                text=match.group(),
                label="TIME",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
        
        return entities
    
    def extract_airports(self, text: str) -> List[str]:
        """
        Extract airport codes from text.
        
        Args:
            text: Input text
            
        Returns:
            List of airport codes
        """
        airports = []
        text_lower = text.lower()
        
        # Check for direct airport code mentions
        airport_code_pattern = re.compile(r'\b(BOM|DEL|BLR|MAA|CCU|HYD|PNQ|AMD|COK|GOI|LHR|DXB|SIN|HKG|JFK|LAX)\b', re.IGNORECASE)
        for match in airport_code_pattern.finditer(text):
            airports.append(match.group().upper())
        
        # Check for airport name mentions
        for airport_name, code in self.airport_mappings.items():
            if airport_name in text_lower:
                airports.append(code)
        
        return list(set(airports))  # Remove duplicates
    
    def extract_airlines(self, text: str) -> List[str]:
        """
        Extract airline codes from text.
        
        Args:
            text: Input text
            
        Returns:
            List of airline codes
        """
        airlines = []
        text_lower = text.lower()
        
        # Check for airline name mentions
        for airline_name, code in self.airline_mappings.items():
            if airline_name in text_lower:
                airlines.append(code)
        
        return list(set(airlines))  # Remove duplicates
    
    def extract_time_references(self, text: str) -> Dict[str, Any]:
        """
        Extract time references from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with time reference information
        """
        time_refs = {
            'specific_times': [],
            'relative_times': [],
            'time_ranges': []
        }
        
        # Specific times (HH:MM, 2pm, etc.)
        time_pattern = re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}(am|pm)\b', re.IGNORECASE)
        for match in time_pattern.finditer(text):
            time_refs['specific_times'].append(match.group())
        
        # Relative time references
        relative_patterns = [
            'morning', 'afternoon', 'evening', 'night',
            'early morning', 'late evening', 'midnight',
            'today', 'tomorrow', 'yesterday',
            'this week', 'next week', 'last week',
            'this month', 'next month', 'last month'
        ]
        
        text_lower = text.lower()
        for pattern in relative_patterns:
            if pattern in text_lower:
                time_refs['relative_times'].append(pattern)
        
        return time_refs
    
    def get_intent_explanation(self, intent_result: IntentResult) -> str:
        """
        Get human-readable explanation of the recognized intent.
        
        Args:
            intent_result: Result from intent recognition
            
        Returns:
            Human-readable explanation
        """
        explanations = {
            'best_time': "Looking for optimal flight scheduling times",
            'delay_analysis': "Analyzing flight delays and punctuality",
            'congestion': "Examining airport congestion and busy periods",
            'schedule_impact': "Evaluating impact of schedule changes",
            'cascading_impact': "Analyzing cascading effects of flight disruptions",
            'general_info': "General flight information query"
        }
        
        explanation = explanations.get(intent_result.primary_intent, "Unknown intent")
        
        if intent_result.secondary_intents:
            secondary = ", ".join([explanations.get(intent, intent) for intent, _ in intent_result.secondary_intents])
            explanation += f" (also related to: {secondary})"
        
        return explanation