"""
Context management for natural language conversations.

This module handles conversation context, follow-up questions, and session management
for the flight scheduling analysis system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import uuid

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryContext:
    """Represents the context of a single query."""
    query_id: str
    query_text: str
    intent: str
    entities: Dict[str, Any]
    timestamp: datetime
    response: Optional[str] = None
    data_used: Optional[Dict[str, Any]] = None
    follow_up_questions: List[str] = field(default_factory=list)
    user_feedback: Optional[str] = None


@dataclass
class ConversationSession:
    """Represents a conversation session with multiple queries."""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    queries: List[QueryContext] = field(default_factory=list)
    persistent_context: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


class ConversationContextManager:
    """
    Manages conversation context and follow-up questions for flight queries.
    
    Features:
    - Session management
    - Context persistence across queries
    - Follow-up question generation
    - User preference learning
    - Query history tracking
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize the context manager.
        
        Args:
            session_timeout_minutes: Minutes after which a session expires
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_context_queries = 10  # Maximum queries to keep in context
        
        logger.info("ConversationContextManager initialized")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession if found and not expired, None otherwise
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if datetime.now() - session.last_activity > self.session_timeout:
            logger.info(f"Session {session_id} expired, removing")
            del self.sessions[session_id]
            return None
        
        return session
    
    def add_query_to_session(
        self, 
        session_id: str, 
        query_text: str, 
        intent: str, 
        entities: Dict[str, Any],
        response: Optional[str] = None,
        data_used: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a query to a session.
        
        Args:
            session_id: Session identifier
            query_text: The user's query
            intent: Recognized intent
            entities: Extracted entities
            response: Generated response
            data_used: Data used to generate response
            
        Returns:
            Query ID
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
        
        query_id = str(uuid.uuid4())
        now = datetime.now()
        
        query_context = QueryContext(
            query_id=query_id,
            query_text=query_text,
            intent=intent,
            entities=entities,
            timestamp=now,
            response=response,
            data_used=data_used
        )
        
        session.queries.append(query_context)
        session.last_activity = now
        
        # Limit context size
        if len(session.queries) > self.max_context_queries:
            session.queries = session.queries[-self.max_context_queries:]
        
        # Update persistent context
        self._update_persistent_context(session, query_context)
        
        logger.info(f"Added query {query_id} to session {session_id}")
        return query_id
    
    def get_context_for_query(self, session_id: str) -> Dict[str, Any]:
        """
        Get relevant context for processing a new query.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context dictionary with relevant information
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        context = {
            'session_id': session_id,
            'conversation_length': len(session.queries),
            'persistent_context': session.persistent_context.copy(),
            'user_preferences': session.preferences.copy(),
            'recent_queries': []
        }
        
        # Add recent queries (last 3)
        recent_queries = session.queries[-3:] if session.queries else []
        for query in recent_queries:
            context['recent_queries'].append({
                'query': query.query_text,
                'intent': query.intent,
                'entities': query.entities,
                'timestamp': query.timestamp.isoformat()
            })
        
        # Add frequently mentioned entities
        context['frequent_entities'] = self._get_frequent_entities(session)
        
        # Add conversation patterns
        context['conversation_patterns'] = self._analyze_conversation_patterns(session)
        
        return context
    
    def _update_persistent_context(self, session: ConversationSession, query: QueryContext):
        """
        Update persistent context based on the new query.
        
        Args:
            session: Conversation session
            query: New query context
        """
        # Track frequently mentioned airports
        if 'airports' in query.entities:
            airports = query.entities['airports']
            if 'frequent_airports' not in session.persistent_context:
                session.persistent_context['frequent_airports'] = {}
            
            for airport in airports:
                session.persistent_context['frequent_airports'][airport] = \
                    session.persistent_context['frequent_airports'].get(airport, 0) + 1
        
        # Track frequently mentioned airlines
        if 'airlines' in query.entities:
            airlines = query.entities['airlines']
            if 'frequent_airlines' not in session.persistent_context:
                session.persistent_context['frequent_airlines'] = {}
            
            for airline in airlines:
                session.persistent_context['frequent_airlines'][airline] = \
                    session.persistent_context['frequent_airlines'].get(airline, 0) + 1
        
        # Track query types
        if 'query_types' not in session.persistent_context:
            session.persistent_context['query_types'] = {}
        
        session.persistent_context['query_types'][query.intent] = \
            session.persistent_context['query_types'].get(query.intent, 0) + 1
        
        # Track time preferences
        if 'time_range' in query.entities and query.entities['time_range']:
            if 'time_preferences' not in session.persistent_context:
                session.persistent_context['time_preferences'] = []
            
            session.persistent_context['time_preferences'].append(query.entities['time_range'])
    
    def _get_frequent_entities(self, session: ConversationSession) -> Dict[str, Any]:
        """
        Get frequently mentioned entities in the conversation.
        
        Args:
            session: Conversation session
            
        Returns:
            Dictionary of frequent entities
        """
        frequent = {}
        
        if 'frequent_airports' in session.persistent_context:
            # Get top 3 airports
            airports = session.persistent_context['frequent_airports']
            top_airports = sorted(airports.items(), key=lambda x: x[1], reverse=True)[:3]
            frequent['airports'] = [airport for airport, count in top_airports]
        
        if 'frequent_airlines' in session.persistent_context:
            # Get top 3 airlines
            airlines = session.persistent_context['frequent_airlines']
            top_airlines = sorted(airlines.items(), key=lambda x: x[1], reverse=True)[:3]
            frequent['airlines'] = [airline for airline, count in top_airlines]
        
        return frequent
    
    def _analyze_conversation_patterns(self, session: ConversationSession) -> Dict[str, Any]:
        """
        Analyze patterns in the conversation.
        
        Args:
            session: Conversation session
            
        Returns:
            Dictionary of conversation patterns
        """
        patterns = {}
        
        if not session.queries:
            return patterns
        
        # Analyze query progression
        query_intents = [q.intent for q in session.queries]
        patterns['query_progression'] = query_intents
        
        # Check for follow-up patterns
        if len(query_intents) >= 2:
            patterns['is_follow_up'] = self._is_follow_up_query(session.queries[-2], session.queries[-1])
        
        # Analyze time patterns
        query_times = [q.timestamp.hour for q in session.queries]
        if query_times:
            patterns['query_time_pattern'] = {
                'avg_hour': sum(query_times) / len(query_times),
                'time_range': (min(query_times), max(query_times))
            }
        
        # Check for refinement patterns
        patterns['refinement_pattern'] = self._detect_refinement_pattern(session.queries)
        
        return patterns
    
    def _is_follow_up_query(self, prev_query: QueryContext, current_query: QueryContext) -> bool:
        """
        Determine if current query is a follow-up to the previous one.
        
        Args:
            prev_query: Previous query context
            current_query: Current query context
            
        Returns:
            True if current query is a follow-up
        """
        # Time-based check (within 5 minutes)
        time_diff = current_query.timestamp - prev_query.timestamp
        if time_diff > timedelta(minutes=5):
            return False
        
        # Intent similarity check
        related_intents = {
            'delay_analysis': ['best_time', 'congestion'],
            'best_time': ['delay_analysis', 'schedule_impact'],
            'congestion': ['best_time', 'delay_analysis'],
            'schedule_impact': ['delay_analysis', 'cascading_impact'],
            'cascading_impact': ['schedule_impact', 'delay_analysis']
        }
        
        if current_query.intent in related_intents.get(prev_query.intent, []):
            return True
        
        # Entity overlap check
        prev_entities = set()
        curr_entities = set()
        
        for entity_list in prev_query.entities.values():
            if isinstance(entity_list, list):
                prev_entities.update(entity_list)
            elif isinstance(entity_list, str):
                prev_entities.add(entity_list)
        
        for entity_list in current_query.entities.values():
            if isinstance(entity_list, list):
                curr_entities.update(entity_list)
            elif isinstance(entity_list, str):
                curr_entities.add(entity_list)
        
        # If there's significant entity overlap, it's likely a follow-up
        if prev_entities and curr_entities:
            overlap = len(prev_entities.intersection(curr_entities))
            return overlap / len(prev_entities.union(curr_entities)) > 0.3
        
        return False
    
    def _detect_refinement_pattern(self, queries: List[QueryContext]) -> Dict[str, Any]:
        """
        Detect if user is refining their queries.
        
        Args:
            queries: List of query contexts
            
        Returns:
            Refinement pattern information
        """
        if len(queries) < 2:
            return {'is_refining': False}
        
        # Check if queries are getting more specific
        recent_queries = queries[-3:] if len(queries) >= 3 else queries
        
        entity_counts = []
        for query in recent_queries:
            count = 0
            for entities in query.entities.values():
                if isinstance(entities, list):
                    count += len(entities)
                elif entities:
                    count += 1
            entity_counts.append(count)
        
        # If entity count is increasing, user is refining
        is_refining = len(entity_counts) >= 2 and entity_counts[-1] > entity_counts[0]
        
        return {
            'is_refining': is_refining,
            'entity_progression': entity_counts,
            'refinement_direction': 'more_specific' if is_refining else 'same_level'
        }
    
    def generate_follow_up_questions(
        self, 
        session_id: str, 
        current_intent: str, 
        current_entities: Dict[str, Any]
    ) -> List[str]:
        """
        Generate contextual follow-up questions.
        
        Args:
            session_id: Session identifier
            current_intent: Intent of current query
            current_entities: Entities from current query
            
        Returns:
            List of follow-up questions
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        context = self.get_context_for_query(session_id)
        follow_ups = []
        
        # Intent-specific follow-ups
        if current_intent == 'delay_analysis':
            follow_ups.extend([
                "Would you like to see the main causes of these delays?",
                "Are you interested in comparing delays across different time periods?",
                "Would you like to see delay patterns for specific airlines?"
            ])
            
            # Context-aware follow-ups
            if context.get('frequent_entities', {}).get('airports'):
                airports = context['frequent_entities']['airports']
                follow_ups.append(f"Would you like to compare delays between {' and '.join(airports)}?")
        
        elif current_intent == 'best_time':
            follow_ups.extend([
                "Would you like to see how weather affects these recommendations?",
                "Are you interested in comparing different days of the week?",
                "Would you like to see alternative time slots if your preferred time is busy?"
            ])
            
            if 'airports' in current_entities and len(current_entities['airports']) == 1:
                airport = current_entities['airports'][0]
                follow_ups.append(f"Would you like to see the best times for other routes from {airport}?")
        
        elif current_intent == 'congestion':
            follow_ups.extend([
                "Would you like to see congestion patterns for different days?",
                "Are you interested in runway-specific congestion data?",
                "Would you like recommendations for less busy time slots?"
            ])
            
            if context.get('conversation_patterns', {}).get('is_follow_up'):
                follow_ups.append("Would you like to see how this congestion affects delay patterns?")
        
        elif current_intent == 'schedule_impact':
            follow_ups.extend([
                "Would you like to see the cascading effects of this schedule change?",
                "Are you interested in alternative schedule modifications?",
                "Would you like to see the impact on specific airlines or routes?"
            ])
        
        elif current_intent == 'cascading_impact':
            follow_ups.extend([
                "Would you like to see mitigation strategies for these critical flights?",
                "Are you interested in the network impact of specific delays?",
                "Would you like to see how schedule changes could reduce cascading effects?"
            ])
        
        # Limit to 3 most relevant follow-ups
        return follow_ups[:3]
    
    def update_user_preferences(
        self, 
        session_id: str, 
        preferences: Dict[str, Any]
    ):
        """
        Update user preferences for the session.
        
        Args:
            session_id: Session identifier
            preferences: User preferences to update
        """
        session = self.get_session(session_id)
        if session:
            session.preferences.update(preferences)
            logger.info(f"Updated preferences for session {session_id}")
    
    def add_user_feedback(
        self, 
        session_id: str, 
        query_id: str, 
        feedback: str
    ):
        """
        Add user feedback for a specific query.
        
        Args:
            session_id: Session identifier
            query_id: Query identifier
            feedback: User feedback
        """
        session = self.get_session(session_id)
        if not session:
            return
        
        for query in session.queries:
            if query.query_id == query_id:
                query.user_feedback = feedback
                logger.info(f"Added feedback for query {query_id}")
                break
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'start_time': session.start_time.isoformat(),
            'duration_minutes': (session.last_activity - session.start_time).total_seconds() / 60,
            'total_queries': len(session.queries),
            'query_types': list(session.persistent_context.get('query_types', {}).keys()),
            'frequent_airports': list(session.persistent_context.get('frequent_airports', {}).keys()),
            'frequent_airlines': list(session.persistent_context.get('frequent_airlines', {}).keys()),
            'user_preferences': session.preferences
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to free memory."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if now - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")