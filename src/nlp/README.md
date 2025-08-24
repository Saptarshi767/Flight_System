# Natural Language Processing Interface

This module implements a comprehensive natural language processing interface for the Flight Scheduling Analysis System. It provides advanced query processing capabilities using OpenAI GPT-4, spaCy for entity extraction, and LangChain for complex query orchestration.

## Overview

The NLP interface enables users to interact with flight data using natural language queries such as:
- "What's the best time to fly from Mumbai to Delhi?"
- "Which flights cause the most delays?"
- "Show me congestion patterns for Delhi airport"
- "What's the impact of changing the schedule by 30 minutes?"

## Components

### 1. Query Processor (`query_processor.py`)

**FlightQueryProcessor** - Main class for processing natural language queries about flight data.

**Features:**
- OpenAI GPT-4 integration for query understanding and response generation
- spaCy-based entity extraction and intent recognition
- Context management for follow-up questions
- Query validation and error handling
- Support for multiple query types (delay analysis, congestion, schedule optimization, etc.)

**Key Methods:**
- `process_query(query, context)` - Process a natural language query
- `validate_query(query)` - Validate query before processing
- `_extract_intent_and_entities(query)` - Extract intent and entities using spaCy
- `_generate_response(query, intent, data, context)` - Generate response using OpenAI

### 2. Intent Recognizer (`intent_recognizer.py`)

**FlightIntentRecognizer** - Advanced intent recognition using spaCy NLP pipeline.

**Features:**
- Pattern-based intent classification
- Entity extraction for airports, airlines, flight numbers, and time references
- Confidence scoring for intent recognition
- Support for multiple intent types:
  - Best time recommendations
  - Delay analysis
  - Congestion analysis
  - Schedule impact modeling
  - Cascading impact analysis

**Key Methods:**
- `recognize_intent(text)` - Recognize intent from text
- `extract_airports(text)` - Extract airport codes
- `extract_airlines(text)` - Extract airline codes
- `extract_time_references(text)` - Extract time-related entities

### 3. Context Manager (`context_manager.py`)

**ConversationContextManager** - Manages conversation context and follow-up questions.

**Features:**
- Session management with automatic expiration
- Context persistence across queries
- Follow-up question generation
- User preference learning
- Query history tracking
- Conversation pattern analysis

**Key Methods:**
- `create_session(user_id)` - Create new conversation session
- `add_query_to_session(session_id, query, intent, entities)` - Add query to session
- `get_context_for_query(session_id)` - Get relevant context for processing
- `generate_follow_up_questions(session_id, intent, entities)` - Generate contextual follow-ups

### 4. LangChain Orchestrator (`langchain_orchestrator.py`)

**FlightQueryOrchestrator** - LangChain-based orchestration for complex queries.

**Features:**
- Multi-step query processing using LangChain agents
- Custom tools for flight data analysis
- Memory management for conversation context
- Specialized chains for different analysis types
- Agent-based reasoning for complex scenarios

**Custom Tools:**
- **FlightDataTool** - Retrieve flight data based on filters
- **DelayAnalysisTool** - Perform delay analysis and provide insights
- **CongestionAnalysisTool** - Analyze airport congestion patterns
- **ScheduleImpactTool** - Analyze impact of schedule changes

**Key Methods:**
- `process_complex_query(query, context)` - Process complex multi-step queries
- `_classify_query_complexity(query)` - Determine processing approach
- `get_conversation_summary()` - Get conversation summary
- `add_custom_tool(tool)` - Add custom analysis tools

### 5. Response Generator (`response_generator.py`)

**FlightResponseGenerator** - Advanced response generation with templates and visualizations.

**Features:**
- Template-based response generation for different query types
- Data visualization recommendations
- Follow-up question suggestions
- Response caching for performance
- Context-aware formatting
- Confidence scoring

**Response Types:**
- Delay Analysis
- Congestion Analysis
- Best Time Recommendations
- Schedule Impact Analysis
- Cascading Impact Analysis
- General Information
- Error Handling

**Visualization Types:**
- Bar Charts
- Line Charts
- Heatmaps
- Histograms
- Network Graphs
- Time Series
- Scatter Plots

**Key Methods:**
- `generate_response(response_type, data, context)` - Generate formatted response
- `get_response_templates()` - Get available response templates
- `get_visualization_types()` - Get visualization type information
- `add_custom_template(response_type, template)` - Add custom response templates

## Usage Examples

### Basic Query Processing

```python
from src.nlp import FlightQueryProcessor

# Initialize processor
processor = FlightQueryProcessor(openai_api_key="your-api-key")

# Process a query
response = processor.process_query("What's the best time to fly from Mumbai?")
print(response.answer)
```

### Complex Query Orchestration

```python
from src.nlp import FlightQueryOrchestrator

# Initialize orchestrator
orchestrator = FlightQueryOrchestrator(openai_api_key="your-api-key")

# Process complex query
result = orchestrator.process_complex_query(
    "Show me delay patterns and then recommend optimal times for Mumbai to Delhi flights"
)
print(result['response'])
```

### Context-Aware Conversations

```python
from src.nlp import ConversationContextManager, FlightQueryProcessor

# Initialize components
context_manager = ConversationContextManager()
processor = FlightQueryProcessor(openai_api_key="your-api-key")

# Create session
session_id = context_manager.create_session("user123")

# Process queries with context
context_manager.add_query_to_session(session_id, "Show delays for Mumbai", "delay_analysis", {"airports": ["BOM"]})
context = context_manager.get_context_for_query(session_id)

response = processor.process_query("What about congestion patterns?", context)
```

### Response Generation with Templates

```python
from src.nlp import FlightResponseGenerator, ResponseType

# Initialize generator
generator = FlightResponseGenerator()

# Generate formatted response
data = {
    'total_flights': 150,
    'average_delay': 22.5,
    'on_time_percentage': 78.3,
    'most_delayed_route': 'BOM-DEL'
}

response = generator.generate_response(ResponseType.DELAY_ANALYSIS, data)
print(response['response_text'])
print("Visualizations:", response['visualizations'])
print("Follow-ups:", response['follow_up_questions'])
```

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Database Configuration (for data retrieval)
DATABASE_URL=postgresql://user:password@localhost:5432/flightdb
REDIS_URL=redis://localhost:6379/0

# Cache Configuration
CACHE_TTL=3600  # Response cache TTL in seconds
```

### Dependencies

The NLP interface requires the following dependencies:

```
openai>=1.3.7
langchain>=0.0.350
langchain-openai>=0.0.2
spacy>=3.7.2
pandas>=2.1.4
pydantic>=2.5.0
```

### spaCy Model Installation

```bash
python -m spacy download en_core_web_sm
```

## Architecture

The NLP interface follows a modular architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Processor │───▶│ Intent Recognizer│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│Context Manager  │◀───│  LangChain       │───▶│Database Operations│
└─────────────────┘    │  Orchestrator    │    └─────────────────┘
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │Response Generator│
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Formatted Response│
                       │ + Visualizations │
                       │ + Follow-ups     │
                       └──────────────────┘
```

## Error Handling

The NLP interface includes comprehensive error handling:

- **Query Validation**: Checks for empty, too long, or potentially harmful queries
- **API Error Handling**: Graceful handling of OpenAI API errors with fallback responses
- **Data Validation**: Ensures required data is available before processing
- **Timeout Handling**: Manages session timeouts and cleanup
- **Logging**: Comprehensive logging for debugging and monitoring

## Performance Considerations

- **Response Caching**: Frequently requested responses are cached to improve performance
- **Session Management**: Automatic cleanup of expired sessions to manage memory
- **Query Optimization**: Efficient database queries for data retrieval
- **Batch Processing**: Support for batch query processing when needed

## Testing

The module includes comprehensive tests:

- Unit tests for individual components
- Integration tests for end-to-end workflows
- Mock-based tests to avoid external dependencies during development
- Performance tests for response time validation

Run tests with:
```bash
python -m pytest tests/test_nlp_*.py -v
```

## Future Enhancements

Planned improvements include:

1. **Multi-language Support**: Support for queries in multiple languages
2. **Voice Interface**: Integration with speech-to-text for voice queries
3. **Advanced Analytics**: More sophisticated analysis capabilities
4. **Custom Model Training**: Fine-tuned models for aviation domain
5. **Real-time Processing**: Streaming responses for long-running queries
6. **API Rate Limiting**: Built-in rate limiting for OpenAI API calls

## Contributing

When contributing to the NLP interface:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for any new features
4. Ensure compatibility with existing query types
5. Consider performance implications of changes

## License

This module is part of the Flight Scheduling Analysis System and follows the same licensing terms.