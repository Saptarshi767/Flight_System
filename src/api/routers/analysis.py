"""
Analysis router for flight scheduling analysis operations.

This module provides endpoints for delay analysis, congestion analysis,
schedule impact modeling, and cascading impact analysis.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from src.api.models import (
    DelayAnalysisRequest, DelayAnalysisResult,
    CongestionAnalysisRequest, CongestionAnalysisResult,
    ScheduleImpactRequest, ScheduleImpactResult,
    CascadingImpactRequest, CascadingImpactResult,
    BaseResponse
)
from src.database.connection import db_session_scope
from src.database.operations import get_flights_dataframe, flight_repo
from src.analysis.delay_analyzer import DelayAnalyzer
from src.analysis.congestion_analyzer import CongestionAnalyzer
from src.analysis.schedule_impact_analyzer import ScheduleImpactAnalyzer
from src.analysis.cascading_impact_analyzer import CascadingImpactAnalyzer
from src.utils.logging import logger

router = APIRouter()


def get_db():
    """Dependency to get database session"""
    with db_session_scope() as session:
        yield session


@router.get("/")
async def analysis_root():
    """Analysis root endpoint with available operations."""
    return {
        "success": True,
        "message": "Flight Analysis API",
        "data": {
            "available_endpoints": [
                "POST /delay - Analyze flight delays and optimal times",
                "POST /congestion - Analyze airport congestion patterns",
                "POST /schedule-impact - Model schedule change impacts",
                "POST /cascading-impact - Analyze cascading delay impacts",
                "GET /delay/{airport_code} - Get delay analysis for airport",
                "GET /congestion/{airport_code} - Get congestion analysis for airport"
            ]
        }
    }


@router.post("/delay", response_model=DelayAnalysisResult)
async def analyze_delays(
    request: DelayAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze flight delays and identify optimal takeoff/landing times.
    
    This endpoint implements EXPECTATION 2: scheduled vs actual time comparison
    to find best takeoff/landing times using open source AI tools.
    """
    try:
        # Get flight data for analysis
        df = get_flights_dataframe(
            airport_code=request.airport_code,
            start_date=request.date_from,
            end_date=request.date_to
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No flight data found for airport {request.airport_code} in the specified date range"
            )
        
        # Initialize delay analyzer
        analyzer = DelayAnalyzer()
        
        # Convert DataFrame to FlightData objects (simplified for API)
        flight_data = []
        for _, row in df.iterrows():
            flight_data.append({
                'flight_id': row['flight_id'],
                'flight_number': row['flight_number'],
                'airline_code': row['airline_code'],
                'origin_airport': row['origin_airport'],
                'destination_airport': row['destination_airport'],
                'scheduled_departure': row['scheduled_departure'],
                'actual_departure': row['actual_departure'],
                'scheduled_arrival': row['scheduled_arrival'],
                'actual_arrival': row['actual_arrival'],
                'delay_minutes': row.get('delay_minutes', 0)
            })
        
        # Perform delay analysis
        analysis_result = analyzer.analyze_delays_from_data(
            flight_data, 
            request.airport_code,
            include_weather=request.include_weather,
            granularity=request.granularity
        )
        
        # Store analysis result in background
        background_tasks.add_task(
            store_analysis_result,
            "delay",
            request.airport_code,
            analysis_result
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delay analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delay analysis failed: {str(e)}")


@router.post("/congestion", response_model=CongestionAnalysisResult)
async def analyze_congestion(
    request: CongestionAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze airport congestion patterns and identify busiest time slots.
    
    This endpoint implements EXPECTATION 3: flight density calculation
    to find busiest time slots using open source AI tools.
    """
    try:
        # Get flight data for analysis
        df = get_flights_dataframe(
            airport_code=request.airport_code,
            start_date=request.date_from,
            end_date=request.date_to
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No flight data found for airport {request.airport_code} in the specified date range"
            )
        
        # Initialize congestion analyzer
        runway_capacity = request.runway_capacity or 60  # Default capacity
        analyzer = CongestionAnalyzer(runway_capacity=runway_capacity)
        
        # Convert DataFrame to FlightData objects
        flight_data = []
        for _, row in df.iterrows():
            flight_data.append({
                'flight_id': row['flight_id'],
                'scheduled_departure': row['scheduled_departure'],
                'origin_airport': row['origin_airport'],
                'destination_airport': row['destination_airport'],
                'airline_code': row['airline_code']
            })
        
        # Perform congestion analysis
        analysis_result = analyzer.analyze_congestion_from_data(
            flight_data,
            request.airport_code,
            runway_capacity
        )
        
        # Store analysis result in background
        background_tasks.add_task(
            store_analysis_result,
            "congestion",
            request.airport_code,
            analysis_result
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in congestion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Congestion analysis failed: {str(e)}")


@router.post("/schedule-impact", response_model=ScheduleImpactResult)
async def analyze_schedule_impact(
    request: ScheduleImpactRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Model the impact of schedule changes on flight delays.
    
    This endpoint implements EXPECTATION 4: schedule change simulation
    to tune flight schedules and analyze delay impact.
    """
    try:
        # Get the specific flight
        flight = flight_repo.get_flight_by_id(db, str(request.flight_id))
        if not flight:
            raise HTTPException(status_code=404, detail="Flight not found")
        
        # Get related flights for impact analysis
        impact_start = request.proposed_departure - timedelta(hours=request.impact_radius)
        impact_end = request.proposed_departure + timedelta(hours=request.impact_radius)
        
        df = get_flights_dataframe(
            airport_code=flight.origin_airport,
            start_date=impact_start,
            end_date=impact_end
        )
        
        # Initialize schedule impact analyzer
        analyzer = ScheduleImpactAnalyzer()
        
        # Create schedule change object
        schedule_change = {
            'flight_id': str(request.flight_id),
            'original_departure': flight.scheduled_departure,
            'new_departure': request.proposed_departure,
            'new_arrival': request.proposed_arrival,
            'impact_radius': request.impact_radius
        }
        
        # Perform impact analysis
        analysis_result = analyzer.analyze_schedule_change(
            schedule_change,
            df.to_dict('records') if not df.empty else []
        )
        
        # Store analysis result in background
        background_tasks.add_task(
            store_analysis_result,
            "schedule_impact",
            flight.origin_airport,
            analysis_result
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in schedule impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schedule impact analysis failed: {str(e)}")


@router.post("/cascading-impact", response_model=CascadingImpactResult)
async def analyze_cascading_impact(
    request: CascadingImpactRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze cascading delay impacts and identify critical flights.
    
    This endpoint implements EXPECTATION 5: flight network graph modeling
    to isolate flights with biggest cascading impact using NetworkX.
    """
    try:
        # Get flight data for network analysis
        df = get_flights_dataframe(
            airport_code=request.airport_code,
            start_date=request.date_from,
            end_date=request.date_to
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No flight data found for airport {request.airport_code} in the specified date range"
            )
        
        # Initialize cascading impact analyzer
        analyzer = CascadingImpactAnalyzer()
        
        # Convert DataFrame to flight data
        flight_data = df.to_dict('records')
        
        # Perform cascading impact analysis
        analysis_result = analyzer.analyze_network_impact(
            flight_data,
            request.airport_code,
            network_depth=request.network_depth
        )
        
        # Store analysis result in background
        background_tasks.add_task(
            store_analysis_result,
            "cascading_impact",
            request.airport_code,
            analysis_result
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cascading impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cascading impact analysis failed: {str(e)}")


@router.get("/delay/{airport_code}")
async def get_delay_analysis(
    airport_code: str,
    days_back: int = Query(7, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """Get recent delay analysis results for an airport."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Get flight data
        df = get_flights_dataframe(
            airport_code=airport_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return {
                "success": True,
                "message": f"No recent flight data found for airport {airport_code}",
                "data": {
                    "airport_code": airport_code,
                    "analysis_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                    "total_flights": 0,
                    "average_delay": 0,
                    "delay_statistics": {}
                }
            }
        
        # Calculate basic delay statistics
        delay_stats = {
            "total_flights": len(df),
            "flights_with_delays": len(df[df['delay_minutes'] > 0]),
            "average_delay_minutes": df['delay_minutes'].mean(),
            "median_delay_minutes": df['delay_minutes'].median(),
            "max_delay_minutes": df['delay_minutes'].max(),
            "delay_by_hour": df.groupby(df['scheduled_departure'].dt.hour)['delay_minutes'].mean().to_dict(),
            "delay_by_airline": df.groupby('airline_code')['delay_minutes'].mean().to_dict()
        }
        
        return {
            "success": True,
            "message": f"Delay analysis for airport {airport_code}",
            "data": {
                "airport_code": airport_code,
                "analysis_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                **delay_stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting delay analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get delay analysis: {str(e)}")


@router.get("/congestion/{airport_code}")
async def get_congestion_analysis(
    airport_code: str,
    days_back: int = Query(7, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """Get recent congestion analysis results for an airport."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Get flight data
        df = get_flights_dataframe(
            airport_code=airport_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return {
                "success": True,
                "message": f"No recent flight data found for airport {airport_code}",
                "data": {
                    "airport_code": airport_code,
                    "analysis_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                    "total_flights": 0,
                    "congestion_statistics": {}
                }
            }
        
        # Calculate congestion statistics
        hourly_flights = df.groupby(df['scheduled_departure'].dt.hour).size()
        daily_flights = df.groupby(df['scheduled_departure'].dt.date).size()
        
        congestion_stats = {
            "total_flights": len(df),
            "peak_hour": hourly_flights.idxmax(),
            "peak_hour_flights": hourly_flights.max(),
            "average_flights_per_hour": hourly_flights.mean(),
            "flights_by_hour": hourly_flights.to_dict(),
            "flights_by_day": {str(k): v for k, v in daily_flights.to_dict().items()},
            "busiest_time_slots": hourly_flights.nlargest(5).to_dict()
        }
        
        return {
            "success": True,
            "message": f"Congestion analysis for airport {airport_code}",
            "data": {
                "airport_code": airport_code,
                "analysis_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                **congestion_stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting congestion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get congestion analysis: {str(e)}")


async def store_analysis_result(analysis_type: str, airport_code: str, result: Dict[str, Any]):
    """Background task to store analysis results in database."""
    try:
        with db_session_scope() as db:
            from src.database.operations import analysis_result_repo
            
            result_data = {
                'analysis_type': analysis_type,
                'airport_code': airport_code,
                'analysis_date': datetime.utcnow(),
                'metrics': result,
                'confidence_score': result.get('confidence_score', 0.8)
            }
            
            analysis_result_repo.create_analysis_result(db, result_data)
            logger.info(f"Stored {analysis_type} analysis result for airport {airport_code}")
            
    except Exception as e:
        logger.error(f"Failed to store analysis result: {str(e)}")


@router.options("/", include_in_schema=False)
async def analysis_options():
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}