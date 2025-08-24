"""
LangChain orchestration for complex flight data queries.

This module implements LangChain chains for multi-step query processing,
memory management, and custom tools for flight data analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import asyncio

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain, SequentialChain
from langchain.chains.router import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_route_prompt import MULTI_ROUTE_PROMPT
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import BaseTool, Tool
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import pandas as pd

from ..utils.logger import get_logger
from ..database.operations import DatabaseOperations
from ..analysis.delay_analyzer import DelayAnalyzer
from ..analysis.congestion_analyzer import CongestionAnalyzer
from ..analysis.schedule_impact_analyzer import ScheduleImpactAnalyzer
from ..analysis.cascading_impact_analyzer import CascadingImpactAnalyzer

logger = get_logger(__name__)


class FlightDataTool(BaseTool):
    """Custom tool for retrieving flight data."""
    
    name = "flight_data_retrieval"
    description = "Retrieve flight data based on filters like airports, airlines, and time ranges"
    
    def __init__(self, db_operations: DatabaseOperations):
        super().__init__()
        self.db_ops = db_operations
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool to retrieve flight data."""
        try:
            # Parse query to extract filters
            filters = self._parse_query_filters(query)
            
            # Get flight data
            df = self.db_ops.get_general_flight_data(filters)
            
            if df is None or df.empty:
                return "No flight data found matching the specified criteria."
            
            # Return summary of data
            summary = {
                'total_flights': len(df),
                'airports': list(df['origin_airport'].unique()) + list(df['destination_airport'].unique()),
                'airlines': list(df['airline'].unique()) if 'airline' in df.columns else [],
                'date_range': {
                    'start': df['scheduled_departure'].min().isoformat() if 'scheduled_departure' in df.columns else None,
                    'end': df['scheduled_departure'].max().isoformat() if 'scheduled_departure' in df.columns else None
                }
            }
            
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            logger.error(f"Error in flight data tool: {str(e)}")
            return f"Error retrieving flight data: {str(e)}"
    
    def _parse_query_filters(self, query: str) -> Dict[str, Any]:
        """Parse query to extract filters."""
        filters = {}
        query_lower = query.lower()
        
        # Extract airports
        airport_codes = ['bom', 'del', 'blr', 'maa', 'ccu', 'hyd']
        airports = []
        for code in airport_codes:
            if code in query_lower:
                airports.append(code.upper())
        if airports:
            filters['airports'] = airports
        
        # Extract airlines
        airline_keywords = {
            'air india': 'AI',
            'indigo': '6E',
            'spicejet': 'SG',
            'vistara': 'UK'
        }
        airlines = []
        for keyword, code in airline_keywords.items():
            if keyword in query_lower:
                airlines.append(code)
        if airlines:
            filters['airlines'] = airlines
        
        return filters


class DelayAnalysisTool(BaseTool):
    """Custom tool for delay analysis."""
    
    name = "delay_analysis"
    description = "Analyze flight delays and provide insights on delay patterns"
    
    def __init__(self, delay_analyzer: DelayAnalyzer, db_operations: DatabaseOperations):
        super().__init__()
        self.delay_analyzer = delay_analyzer
        self.db_ops = db_operations
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute delay analysis."""
        try:
            # Parse query for filters
            filters = self._parse_analysis_filters(query)
            
            # Get delay data
            df = self.db_ops.get_delay_analysis_data(filters)
            
            if df is None or df.empty:
                return "No delay data available for analysis."
            
            # Perform delay analysis
            analysis_result = self.delay_analyzer.analyze_delays(df)
            
            # Format results
            result = {
                'total_flights_analyzed': len(df),
                'average_delay_minutes': df['delay_minutes'].mean() if 'delay_minutes' in df.columns else 0,
                'on_time_percentage': (df['delay_minutes'] <= 15).mean() * 100 if 'delay_minutes' in df.columns else 0,
                'most_delayed_route': self._get_most_delayed_route(df),
                'peak_delay_hours': self._get_peak_delay_hours(df)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in delay analysis tool: {str(e)}")
            return f"Error performing delay analysis: {str(e)}"
    
    def _parse_analysis_filters(self, query: str) -> Dict[str, Any]:
        """Parse query for analysis filters."""
        # Similar to FlightDataTool but can be extended for analysis-specific filters
        return {}
    
    def _get_most_delayed_route(self, df: pd.DataFrame) -> str:
        """Get the most delayed route."""
        if 'origin_airport' not in df.columns or 'destination_airport' not in df.columns:
            return "Unknown"
        
        df['route'] = df['origin_airport'] + '-' + df['destination_airport']
        route_delays = df.groupby('route')['delay_minutes'].mean()
        return route_delays.idxmax() if not route_delays.empty else "Unknown"
    
    def _get_peak_delay_hours(self, df: pd.DataFrame) -> List[int]:
        """Get hours with highest delays."""
        if 'scheduled_departure' not in df.columns:
            return []
        
        df['hour'] = pd.to_datetime(df['scheduled_departure']).dt.hour
        hourly_delays = df.groupby('hour')['delay_minutes'].mean()
        return hourly_delays.nlargest(3).index.tolist()


class CongestionAnalysisTool(BaseTool):
    """Custom tool for congestion analysis."""
    
    name = "congestion_analysis"
    description = "Analyze airport congestion patterns and identify busy time slots"
    
    def __init__(self, congestion_analyzer: CongestionAnalyzer, db_operations: DatabaseOperations):
        super().__init__()
        self.congestion_analyzer = congestion_analyzer
        self.db_ops = db_operations
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute congestion analysis."""
        try:
            filters = self._parse_congestion_filters(query)
            df = self.db_ops.get_congestion_data(filters)
            
            if df is None or df.empty:
                return "No congestion data available for analysis."
            
            # Perform congestion analysis
            result = {
                'total_flights': len(df),
                'peak_hours': self._get_peak_hours(df),
                'quietest_hours': self._get_quietest_hours(df),
                'busiest_airports': self._get_busiest_airports(df),
                'congestion_score': self._calculate_congestion_score(df)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in congestion analysis tool: {str(e)}")
            return f"Error performing congestion analysis: {str(e)}"
    
    def _parse_congestion_filters(self, query: str) -> Dict[str, Any]:
        """Parse query for congestion filters."""
        return {}
    
    def _get_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """Get peak congestion hours."""
        if 'hour' not in df.columns:
            return []
        
        hourly_counts = df.groupby('hour').size()
        return hourly_counts.nlargest(3).index.tolist()
    
    def _get_quietest_hours(self, df: pd.DataFrame) -> List[int]:
        """Get quietest hours."""
        if 'hour' not in df.columns:
            return []
        
        hourly_counts = df.groupby('hour').size()
        return hourly_counts.nsmallest(3).index.tolist()
    
    def _get_busiest_airports(self, df: pd.DataFrame) -> List[str]:
        """Get busiest airports."""
        if 'origin_airport' not in df.columns:
            return []
        
        airport_counts = df['origin_airport'].value_counts()
        return airport_counts.head(3).index.tolist()
    
    def _calculate_congestion_score(self, df: pd.DataFrame) -> float:
        """Calculate overall congestion score."""
        if df.empty:
            return 0.0
        
        # Simple congestion score based on flight density
        hourly_counts = df.groupby('hour').size() if 'hour' in df.columns else pd.Series([len(df)])
        return float(hourly_counts.std() / hourly_counts.mean()) if hourly_counts.mean() > 0 else 0.0


class ScheduleImpactTool(BaseTool):
    """Custom tool for schedule impact analysis."""
    
    name = "schedule_impact_analysis"
    description = "Analyze the impact of schedule changes on flight operations"
    
    def __init__(self, schedule_analyzer: ScheduleImpactAnalyzer, db_operations: DatabaseOperations):
        super().__init__()
        self.schedule_analyzer = schedule_analyzer
        self.db_ops = db_operations
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute schedule impact analysis."""
        try:
            filters = self._parse_schedule_filters(query)
            df = self.db_ops.get_schedule_impact_data(filters)
            
            if df is None or df.empty:
                return "No schedule data available for impact analysis."
            
            result = {
                'flights_analyzed': len(df),
                'average_turnaround_time': df['turnaround_time_minutes'].mean() if 'turnaround_time_minutes' in df.columns else 0,
                'schedule_efficiency_score': self._calculate_efficiency_score(df),
                'bottleneck_airports': self._identify_bottlenecks(df),
                'optimization_opportunities': self._identify_optimization_opportunities(df)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in schedule impact tool: {str(e)}")
            return f"Error performing schedule impact analysis: {str(e)}"
    
    def _parse_schedule_filters(self, query: str) -> Dict[str, Any]:
        """Parse query for schedule filters."""
        return {}
    
    def _calculate_efficiency_score(self, df: pd.DataFrame) -> float:
        """Calculate schedule efficiency score."""
        if df.empty:
            return 0.0
        
        # Simple efficiency based on delay ratios
        if 'departure_delay_minutes' in df.columns:
            on_time_ratio = (df['departure_delay_minutes'] <= 15).mean()
            return float(on_time_ratio * 100)
        
        return 50.0  # Default neutral score
    
    def _identify_bottlenecks(self, df: pd.DataFrame) -> List[str]:
        """Identify bottleneck airports."""
        if 'origin_airport' not in df.columns or 'departure_delay_minutes' not in df.columns:
            return []
        
        airport_delays = df.groupby('origin_airport')['departure_delay_minutes'].mean()
        return airport_delays.nlargest(3).index.tolist()
    
    def _identify_optimization_opportunities(self, df: pd.DataFrame) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        if 'turnaround_time_minutes' in df.columns:
            avg_turnaround = df['turnaround_time_minutes'].mean()
            if avg_turnaround > 60:  # More than 1 hour
                opportunities.append("Reduce aircraft turnaround times")
        
        if 'departure_delay_minutes' in df.columns:
            high_delay_ratio = (df['departure_delay_minutes'] > 30).mean()
            if high_delay_ratio > 0.2:  # More than 20% significantly delayed
                opportunities.append("Improve on-time performance")
        
        return opportunities


class FlightQueryOrchestrator:
    """
    LangChain orchestrator for complex flight data queries.
    
    This class manages multi-step query processing, conversation memory,
    and coordinates between different analysis tools.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the orchestrator.
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Initialize database operations and analyzers
        self.db_ops = DatabaseOperations()
        
        # Initialize analysis components (these would need to be implemented)
        try:
            from ..analysis.delay_analyzer import DelayAnalyzer
            from ..analysis.congestion_analyzer import CongestionAnalyzer
            from ..analysis.schedule_impact_analyzer import ScheduleImpactAnalyzer
            from ..analysis.cascading_impact_analyzer import CascadingImpactAnalyzer
            
            self.delay_analyzer = DelayAnalyzer()
            self.congestion_analyzer = CongestionAnalyzer()
            self.schedule_analyzer = ScheduleImpactAnalyzer()
            self.cascading_analyzer = CascadingImpactAnalyzer()
        except ImportError:
            logger.warning("Analysis components not available, using mock analyzers")
            self.delay_analyzer = None
            self.congestion_analyzer = None
            self.schedule_analyzer = None
            self.cascading_analyzer = None
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize chains
        self.chains = self._create_chains()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        logger.info("FlightQueryOrchestrator initialized successfully")
    
    def _create_tools(self) -> List[BaseTool]:
        """Create custom tools for flight data analysis."""
        tools = [
            FlightDataTool(self.db_ops)
        ]
        
        # Add analysis tools if analyzers are available
        if self.delay_analyzer:
            tools.append(DelayAnalysisTool(self.delay_analyzer, self.db_ops))
        
        if self.congestion_analyzer:
            tools.append(CongestionAnalysisTool(self.congestion_analyzer, self.db_ops))
        
        if self.schedule_analyzer:
            tools.append(ScheduleImpactTool(self.schedule_analyzer, self.db_ops))
        
        return tools
    
    def _create_chains(self) -> Dict[str, LLMChain]:
        """Create specialized chains for different query types."""
        chains = {}
        
        # Delay analysis chain
        delay_prompt = PromptTemplate(
            input_variables=["query", "data"],
            template="""
            You are a flight delay analysis expert. Based on the following query and data, provide insights about flight delays.
            
            Query: {query}
            Data: {data}
            
            Please provide:
            1. Key delay insights
            2. Patterns identified
            3. Recommendations for improvement
            4. Specific metrics and statistics
            
            Response:
            """
        )
        chains["delay_analysis"] = LLMChain(llm=self.llm, prompt=delay_prompt)
        
        # Congestion analysis chain
        congestion_prompt = PromptTemplate(
            input_variables=["query", "data"],
            template="""
            You are an airport congestion analysis expert. Based on the following query and data, provide insights about airport congestion.
            
            Query: {query}
            Data: {data}
            
            Please provide:
            1. Peak congestion periods
            2. Congestion patterns
            3. Recommendations for avoiding busy times
            4. Alternative time slots
            
            Response:
            """
        )
        chains["congestion_analysis"] = LLMChain(llm=self.llm, prompt=congestion_prompt)
        
        # Schedule optimization chain
        schedule_prompt = PromptTemplate(
            input_variables=["query", "data"],
            template="""
            You are a flight schedule optimization expert. Based on the following query and data, provide insights about schedule optimization.
            
            Query: {query}
            Data: {data}
            
            Please provide:
            1. Schedule optimization opportunities
            2. Impact analysis of proposed changes
            3. Risk assessment
            4. Implementation recommendations
            
            Response:
            """
        )
        chains["schedule_optimization"] = LLMChain(llm=self.llm, prompt=schedule_prompt)
        
        return chains
    
    def _create_agent(self) -> AgentExecutor:
        """Create an agent with access to all tools."""
        if not self.tools:
            logger.warning("No tools available for agent")
            return None
        
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent
    
    def process_complex_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a complex query using LangChain orchestration.
        
        Args:
            query: User's complex query
            context: Optional context from previous interactions
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Processing complex query: {query}")
            
            # Determine query complexity and routing
            query_type = self._classify_query_complexity(query)
            
            if query_type == "simple":
                # Use direct chain processing
                response = self._process_simple_query(query, context)
            elif query_type == "multi_step":
                # Use agent for multi-step processing
                response = self._process_multi_step_query(query, context)
            else:
                # Use specialized chain
                response = self._process_specialized_query(query, query_type, context)
            
            # Add metadata
            result = {
                "response": response,
                "query_type": query_type,
                "processing_method": self._get_processing_method(query_type),
                "tools_used": self._get_tools_used(),
                "confidence": self._calculate_confidence(response),
                "follow_up_suggestions": self._generate_follow_ups(query, response)
            }
            
            logger.info("Complex query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing complex query: {str(e)}")
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "query_type": "error",
                "processing_method": "error_handling",
                "tools_used": [],
                "confidence": 0.0,
                "follow_up_suggestions": []
            }
    
    def _classify_query_complexity(self, query: str) -> str:
        """Classify query complexity to determine processing approach."""
        query_lower = query.lower()
        
        # Multi-step indicators
        multi_step_keywords = [
            "and then", "after that", "also show", "compare", "both", 
            "what if", "impact of", "how does", "relationship between"
        ]
        
        if any(keyword in query_lower for keyword in multi_step_keywords):
            return "multi_step"
        
        # Specialized analysis indicators
        if any(keyword in query_lower for keyword in ["delay", "late", "punctual"]):
            return "delay_analysis"
        elif any(keyword in query_lower for keyword in ["busy", "congestion", "peak", "crowded"]):
            return "congestion_analysis"
        elif any(keyword in query_lower for keyword in ["schedule", "optimize", "change", "impact"]):
            return "schedule_optimization"
        
        return "simple"
    
    def _process_simple_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Process simple queries using basic conversation chain."""
        try:
            # Create a simple conversation chain
            conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=True
            )
            
            # Add context if available
            if context:
                context_str = f"Context from previous conversation: {json.dumps(context, default=str)}\n\n"
                query = context_str + query
            
            response = conversation.predict(input=query)
            return response
            
        except Exception as e:
            logger.error(f"Error in simple query processing: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def _process_multi_step_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Process multi-step queries using agent."""
        try:
            if not self.agent:
                return "Multi-step query processing not available (no tools configured)"
            
            # Add context to query if available
            if context:
                context_str = f"Previous context: {json.dumps(context, default=str)}\n\n"
                query = context_str + query
            
            response = self.agent.run(query)
            return response
            
        except Exception as e:
            logger.error(f"Error in multi-step query processing: {str(e)}")
            return f"Error processing multi-step query: {str(e)}"
    
    def _process_specialized_query(self, query: str, query_type: str, context: Optional[Dict[str, Any]]) -> str:
        """Process specialized queries using specific chains."""
        try:
            if query_type not in self.chains:
                return f"Specialized processing for {query_type} not available"
            
            chain = self.chains[query_type]
            
            # Get relevant data using tools
            data = self._get_relevant_data(query, query_type)
            
            # Add context if available
            if context:
                query = f"Context: {json.dumps(context, default=str)}\n\nQuery: {query}"
            
            response = chain.run(query=query, data=data)
            return response
            
        except Exception as e:
            logger.error(f"Error in specialized query processing: {str(e)}")
            return f"Error processing specialized query: {str(e)}"
    
    def _get_relevant_data(self, query: str, query_type: str) -> str:
        """Get relevant data for the query using appropriate tools."""
        try:
            # Use the first available tool to get data
            if self.tools:
                tool = self.tools[0]  # FlightDataTool
                data = tool._run(query)
                return data
            else:
                return "No data available"
                
        except Exception as e:
            logger.error(f"Error getting relevant data: {str(e)}")
            return f"Error retrieving data: {str(e)}"
    
    def _get_processing_method(self, query_type: str) -> str:
        """Get processing method description."""
        methods = {
            "simple": "Direct conversation chain",
            "multi_step": "Agent-based multi-step processing",
            "delay_analysis": "Specialized delay analysis chain",
            "congestion_analysis": "Specialized congestion analysis chain",
            "schedule_optimization": "Specialized schedule optimization chain"
        }
        return methods.get(query_type, "Unknown method")
    
    def _get_tools_used(self) -> List[str]:
        """Get list of tools that were used."""
        # This would be enhanced to track actual tool usage
        return [tool.name for tool in self.tools]
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for the response."""
        # Simple confidence calculation based on response characteristics
        if "error" in response.lower():
            return 0.1
        elif "not available" in response.lower():
            return 0.3
        elif len(response) < 50:
            return 0.5
        else:
            return 0.8
    
    def _generate_follow_ups(self, query: str, response: str) -> List[str]:
        """Generate follow-up question suggestions."""
        follow_ups = []
        
        query_lower = query.lower()
        
        if "delay" in query_lower:
            follow_ups.extend([
                "What are the main causes of these delays?",
                "How do delays vary by time of day?",
                "Which airlines have the best on-time performance?"
            ])
        elif "congestion" in query_lower:
            follow_ups.extend([
                "What are the alternative time slots with less congestion?",
                "How does weather affect congestion patterns?",
                "Which runways are most congested?"
            ])
        elif "schedule" in query_lower:
            follow_ups.extend([
                "What would be the impact of changing departure times by 30 minutes?",
                "How would this affect connecting flights?",
                "What are the cost implications of this schedule change?"
            ])
        
        return follow_ups[:3]  # Limit to 3 suggestions
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        try:
            if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
                messages = self.memory.chat_memory.messages
                summary = f"Conversation with {len(messages)} messages. "
                
                # Get recent topics
                recent_messages = messages[-4:] if len(messages) > 4 else messages
                topics = []
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        content = msg.content.lower()
                        if 'delay' in content:
                            topics.append('delays')
                        elif 'congestion' in content:
                            topics.append('congestion')
                        elif 'schedule' in content:
                            topics.append('scheduling')
                
                if topics:
                    summary += f"Recent topics: {', '.join(set(topics))}"
                
                return summary
            else:
                return "No conversation history available"
                
        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}")
            return "Error retrieving conversation summary"
    
    def clear_memory(self):
        """Clear conversation memory."""
        try:
            self.memory.clear()
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def add_custom_tool(self, tool: BaseTool):
        """Add a custom tool to the orchestrator."""
        try:
            self.tools.append(tool)
            
            # Recreate agent with new tools
            if len(self.tools) > 0:
                self.agent = self._create_agent()
            
            logger.info(f"Added custom tool: {tool.name}")
            
        except Exception as e:
            logger.error(f"Error adding custom tool: {str(e)}")
    
    def get_available_capabilities(self) -> Dict[str, Any]:
        """Get information about available capabilities."""
        return {
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "chains": list(self.chains.keys()),
            "memory_type": type(self.memory).__name__,
            "llm_model": "gpt-4",
            "max_iterations": 3,
            "supported_query_types": [
                "simple", "multi_step", "delay_analysis", 
                "congestion_analysis", "schedule_optimization"
            ]
        }