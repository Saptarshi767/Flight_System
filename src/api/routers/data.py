"""
Data management router for flight data operations.

This module provides endpoints for flight data upload, retrieval, filtering, and export.
"""

import io
import csv
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import pandas as pd

from src.api.models import (
    FlightData, FlightDataCreate, FlightDataUpdate, FlightDataResponse, 
    FlightDataListResponse, FlightDataFilter, PaginationParams,
    FileUploadResponse, ExportRequest, ExportResponse, ResponseFormat,
    BaseResponse, ErrorResponse
)
from src.database.connection import db_session_scope
from src.database.operations import (
    flight_repo, airport_repo, airline_repo, aircraft_repo,
    get_flights_dataframe
)
from src.data.processors import ExcelDataProcessor
from src.utils.logging import logger

router = APIRouter()


def get_db():
    """Dependency to get database session"""
    with db_session_scope() as session:
        yield session


@router.get("/")
async def data_root():
    """Data management root endpoint with available operations."""
    return {
        "success": True,
        "message": "Flight Data Management API",
        "data": {
            "available_endpoints": [
                "GET /flights - List flights with filtering and pagination",
                "POST /flights - Create new flight record",
                "GET /flights/{flight_id} - Get specific flight",
                "PUT /flights/{flight_id} - Update flight record",
                "DELETE /flights/{flight_id} - Delete flight record",
                "POST /upload - Upload flight data files",
                "GET /export - Export flight data",
                "GET /statistics - Get flight statistics"
            ]
        }
    }


@router.get("/flights", response_model=FlightDataListResponse)
async def get_flights(
    pagination: PaginationParams = Depends(),
    airline: Optional[str] = Query(None, description="Filter by airline code"),
    origin_airport: Optional[str] = Query(None, description="Filter by origin airport"),
    destination_airport: Optional[str] = Query(None, description="Filter by destination airport"),
    date_from: Optional[datetime] = Query(None, description="Filter flights from this date"),
    date_to: Optional[datetime] = Query(None, description="Filter flights to this date"),
    delay_category: Optional[str] = Query(None, description="Filter by delay category"),
    min_delay: Optional[int] = Query(None, description="Minimum delay in minutes"),
    max_delay: Optional[int] = Query(None, description="Maximum delay in minutes"),
    db: Session = Depends(get_db)
):
    """
    Retrieve flights with filtering and pagination.
    
    Supports filtering by:
    - Airline code
    - Origin/destination airports
    - Date range
    - Delay category and thresholds
    """
    try:
        from sqlalchemy import and_, or_, func
        from src.database.models import Flight
        
        # Build query with filters
        query = db.query(Flight)
        
        # Apply filters
        if airline:
            query = query.filter(Flight.airline_code == airline)
        
        if origin_airport:
            query = query.filter(Flight.origin_airport == origin_airport)
        
        if destination_airport:
            query = query.filter(Flight.destination_airport == destination_airport)
        
        if date_from:
            query = query.filter(Flight.scheduled_departure >= date_from)
        
        if date_to:
            query = query.filter(Flight.scheduled_departure <= date_to)
        
        if delay_category:
            query = query.filter(Flight.delay_category == delay_category)
        
        if min_delay is not None:
            query = query.filter(
                or_(
                    Flight.departure_delay_minutes >= min_delay,
                    Flight.arrival_delay_minutes >= min_delay
                )
            )
        
        if max_delay is not None:
            query = query.filter(
                and_(
                    or_(Flight.departure_delay_minutes <= max_delay, Flight.departure_delay_minutes.is_(None)),
                    or_(Flight.arrival_delay_minutes <= max_delay, Flight.arrival_delay_minutes.is_(None))
                )
            )
        
        # Get total count for pagination
        total = query.count()
        
        # Apply pagination
        flights = query.offset(pagination.offset).limit(pagination.size).all()
        
        # Calculate pagination metadata
        pages = (total + pagination.size - 1) // pagination.size
        has_next = pagination.page < pages
        has_prev = pagination.page > 1
        
        return FlightDataListResponse(
            data=flights,
            page=pagination.page,
            size=pagination.size,
            total=total,
            pages=pages,
            has_next=has_next,
            has_prev=has_prev,
            message=f"Retrieved {len(flights)} flights"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving flights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve flights: {str(e)}")


@router.post("/flights", response_model=FlightDataResponse)
async def create_flight(
    flight_data: FlightDataCreate,
    db: Session = Depends(get_db)
):
    """Create a new flight record."""
    try:
        # Convert Pydantic model to dict
        flight_dict = flight_data.model_dump()
        flight_dict['flight_id'] = f"{flight_data.flight_number}_{flight_data.scheduled_departure.strftime('%Y%m%d_%H%M')}"
        flight_dict['airline_code'] = flight_dict.pop('airline')  # Map airline to airline_code
        flight_dict['data_source'] = 'api'
        flight_dict['created_at'] = datetime.utcnow()
        flight_dict['updated_at'] = datetime.utcnow()
        
        # Calculate delay if actual times are provided
        if flight_data.actual_departure and flight_data.scheduled_departure:
            delay_minutes = (flight_data.actual_departure - flight_data.scheduled_departure).total_seconds() / 60
            flight_dict['departure_delay_minutes'] = int(delay_minutes)
        
        # Create flight record
        flight = flight_repo.create_flight(db, flight_dict)
        
        return FlightDataResponse(
            data=flight,
            message="Flight created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating flight: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to create flight: {str(e)}")


@router.get("/flights/{flight_id}", response_model=FlightDataResponse)
async def get_flight(
    flight_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific flight by ID."""
    try:
        flight = flight_repo.get_flight_by_id(db, flight_id)
        if not flight:
            raise HTTPException(status_code=404, detail="Flight not found")
        
        return FlightDataResponse(
            data=flight,
            message="Flight retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving flight {flight_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve flight: {str(e)}")


@router.put("/flights/{flight_id}", response_model=FlightDataResponse)
async def update_flight(
    flight_id: str,
    flight_update: FlightDataUpdate,
    db: Session = Depends(get_db)
):
    """Update a flight record."""
    try:
        # Convert to dict and remove None values
        update_dict = {k: v for k, v in flight_update.model_dump().items() if v is not None}
        
        if not update_dict:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        # Update flight
        flight = flight_repo.update_flight(db, flight_id, update_dict)
        if not flight:
            raise HTTPException(status_code=404, detail="Flight not found")
        
        return FlightDataResponse(
            data=flight,
            message="Flight updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating flight {flight_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update flight: {str(e)}")


@router.delete("/flights/{flight_id}", response_model=BaseResponse)
async def delete_flight(
    flight_id: str,
    db: Session = Depends(get_db)
):
    """Delete a flight record."""
    try:
        success = flight_repo.delete_flight(db, flight_id)
        if not success:
            raise HTTPException(status_code=404, detail="Flight not found")
        
        return BaseResponse(message="Flight deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flight {flight_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete flight: {str(e)}")


@router.post("/upload", response_model=FileUploadResponse)
async def upload_flight_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Flight data file (Excel or CSV)"),
    db: Session = Depends(get_db)
):
    """
    Upload flight data from Excel or CSV files.
    
    Supports:
    - Excel files (.xlsx, .xls)
    - CSV files (.csv)
    """
    try:
        # Validate file type
        allowed_extensions = ['.xlsx', '.xls', '.csv']
        file_extension = None
        for ext in allowed_extensions:
            if file.filename.lower().endswith(ext):
                file_extension = ext
                break
        
        if not file_extension:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Process file based on type
        if file_extension in ['.xlsx', '.xls']:
            # Process Excel file
            df = pd.read_excel(io.BytesIO(content))
        else:
            # Process CSV file
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File contains no data")
        
        # Process data in background
        background_tasks.add_task(
            process_uploaded_data,
            df,
            file.filename,
            file_extension
        )
        
        return FileUploadResponse(
            filename=file.filename,
            size=len(content),
            records_processed=len(df),
            records_created=0,  # Will be updated by background task
            records_updated=0,  # Will be updated by background task
            message="File uploaded successfully. Processing in background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


async def process_uploaded_data(df: pd.DataFrame, filename: str, file_extension: str):
    """Background task to process uploaded flight data."""
    try:
        with db_session_scope() as db:
            # Standardize column names
            column_mapping = {
                'Flight': 'flight_number',
                'From': 'origin_airport', 
                'To': 'destination_airport',
                'Departure': 'scheduled_departure',
                'Arrival': 'scheduled_arrival',
                'Aircraft': 'aircraft_type',
                'Airline': 'airline_code'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Process each row
            records_created = 0
            records_updated = 0
            errors = []
            
            for index, row in df.iterrows():
                try:
                    # Create flight data dict
                    flight_data = {
                        'flight_number': str(row.get('flight_number', '')),
                        'airline_code': str(row.get('airline_code', '')),
                        'origin_airport': str(row.get('origin_airport', '')),
                        'destination_airport': str(row.get('destination_airport', '')),
                        'aircraft_type': str(row.get('aircraft_type', '')),
                        'data_source': 'file_upload',
                        'created_at': datetime.utcnow(),
                        'updated_at': datetime.utcnow()
                    }
                    
                    # Handle datetime columns
                    for col in ['scheduled_departure', 'scheduled_arrival', 'actual_departure', 'actual_arrival']:
                        if col in row and pd.notna(row[col]):
                            flight_data[col] = pd.to_datetime(row[col])
                    
                    # Generate flight_id
                    if flight_data.get('scheduled_departure'):
                        flight_data['flight_id'] = f"{flight_data['flight_number']}_{flight_data['scheduled_departure'].strftime('%Y%m%d_%H%M')}"
                    else:
                        flight_data['flight_id'] = f"{flight_data['flight_number']}_{uuid4().hex[:8]}"
                    
                    # Check if flight exists
                    existing_flight = flight_repo.get_flight_by_id(db, flight_data['flight_id'])
                    
                    if existing_flight:
                        # Update existing flight
                        flight_repo.update_flight(db, flight_data['flight_id'], flight_data)
                        records_updated += 1
                    else:
                        # Create new flight
                        flight_repo.create_flight(db, flight_data)
                        records_created += 1
                        
                except Exception as e:
                    errors.append(f"Row {index + 1}: {str(e)}")
                    continue
            
            logger.info(f"Processed {filename}: {records_created} created, {records_updated} updated, {len(errors)} errors")
            
    except Exception as e:
        logger.error(f"Error processing uploaded data: {str(e)}")


@router.get("/export")
async def export_flight_data(
    format: ResponseFormat = Query(ResponseFormat.CSV, description="Export format"),
    airline: Optional[str] = Query(None, description="Filter by airline code"),
    origin_airport: Optional[str] = Query(None, description="Filter by origin airport"),
    destination_airport: Optional[str] = Query(None, description="Filter by destination airport"),
    date_from: Optional[datetime] = Query(None, description="Filter flights from this date"),
    date_to: Optional[datetime] = Query(None, description="Filter flights to this date"),
    include_analysis: bool = Query(False, description="Include analysis results")
):
    """
    Export flight data in various formats.
    
    Supports:
    - CSV format
    - JSON format
    - Excel format (XLSX)
    """
    try:
        # Get flight data with filters
        filters = {}
        if airline:
            filters['airlines'] = [airline]
        if origin_airport or destination_airport:
            airports = []
            if origin_airport:
                airports.append(origin_airport)
            if destination_airport:
                airports.append(destination_airport)
            filters['airports'] = airports
        if date_from or date_to:
            filters['time_range'] = {}
            if date_from:
                filters['time_range']['start'] = date_from
            if date_to:
                filters['time_range']['end'] = date_to
        
        # Get data as DataFrame
        df = get_flights_dataframe(
            airport_code=origin_airport or destination_airport,
            start_date=date_from,
            end_date=date_to
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No flight data found matching criteria")
        
        # Filter by airline if specified
        if airline:
            df = df[df['airline_code'] == airline]
        
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"flight_data_{timestamp}"
        
        # Export based on format
        if format == ResponseFormat.CSV:
            output = io.StringIO()
            df.to_csv(output, index=False)
            content = output.getvalue()
            
            return StreamingResponse(
                io.StringIO(content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
            )
            
        elif format == ResponseFormat.JSON:
            content = df.to_json(orient='records', date_format='iso', indent=2)
            
            return StreamingResponse(
                io.StringIO(content),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}.json"}
            )
            
        elif format == ResponseFormat.EXCEL:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Flight Data', index=False)
            
            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.getvalue()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}.xlsx"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/statistics", response_model=BaseResponse)
async def get_flight_statistics(
    airport_code: Optional[str] = Query(None, description="Filter by airport code"),
    start_date: Optional[datetime] = Query(None, description="Statistics from this date"),
    end_date: Optional[datetime] = Query(None, description="Statistics to this date"),
    db: Session = Depends(get_db)
):
    """
    Get flight statistics and metrics.
    
    Returns:
    - Total flights
    - Delay statistics
    - Airport performance metrics
    - Airline performance metrics
    """
    try:
        # Get flight statistics
        stats = flight_repo.get_flight_statistics(
            db, 
            airport_code=airport_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add additional metrics
        with db_session_scope() as session:
            from src.database.models import Flight, Airport, Airline
            from sqlalchemy import func, distinct
            
            query = session.query(Flight)
            
            # Apply filters
            if airport_code:
                from sqlalchemy import or_
                query = query.filter(
                    or_(
                        Flight.origin_airport == airport_code,
                        Flight.destination_airport == airport_code
                    )
                )
            
            if start_date:
                query = query.filter(Flight.scheduled_departure >= start_date)
            if end_date:
                query = query.filter(Flight.scheduled_departure <= end_date)
            
            # Additional statistics
            unique_routes = query.with_entities(
                distinct(func.concat(Flight.origin_airport, '-', Flight.destination_airport))
            ).count()
            
            unique_airlines = query.with_entities(distinct(Flight.airline_code)).count()
            unique_aircraft = query.with_entities(distinct(Flight.aircraft_type)).count()
            
            stats.update({
                'unique_routes': unique_routes,
                'unique_airlines': unique_airlines,
                'unique_aircraft_types': unique_aircraft,
                'analysis_period': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'airport_code': airport_code
                }
            })
        
        return {
            "success": True,
            "message": "Flight statistics retrieved successfully",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting flight statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.options("/", include_in_schema=False)
async def data_options():
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}