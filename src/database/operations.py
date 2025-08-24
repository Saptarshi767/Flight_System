"""
Database CRUD operations for flight data
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd

from src.database.models import Flight, Airport, Airline, Aircraft, AnalysisResult, DataIngestionLog
from src.database.connection import db_session_scope
from src.utils.logging import logger


class FlightRepository:
    """Repository for flight data operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_flight(self, session: Session, flight_data: Dict[str, Any]) -> Flight:
        """Create a new flight record"""
        try:
            flight = Flight(**flight_data)
            session.add(flight)
            session.flush()  # Get the ID without committing
            self.logger.debug(f"Created flight: {flight.flight_id}")
            return flight
        except Exception as e:
            self.logger.error(f"Failed to create flight: {e}")
            raise
    
    def get_flight_by_id(self, session: Session, flight_id: str) -> Optional[Flight]:
        """Get flight by flight_id"""
        try:
            return session.query(Flight).filter(Flight.flight_id == flight_id).first()
        except Exception as e:
            self.logger.error(f"Failed to get flight by ID {flight_id}: {e}")
            return None
    
    def get_flights_by_route(self, session: Session, origin: str, destination: str, 
                           start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> List[Flight]:
        """Get flights by route and optional date range"""
        try:
            query = session.query(Flight).filter(
                Flight.origin_airport == origin,
                Flight.destination_airport == destination
            )
            
            if start_date:
                query = query.filter(Flight.scheduled_departure >= start_date)
            if end_date:
                query = query.filter(Flight.scheduled_departure <= end_date)
            
            return query.order_by(Flight.scheduled_departure).all()
        except Exception as e:
            self.logger.error(f"Failed to get flights by route {origin}-{destination}: {e}")
            return []
    
    def get_flights_by_airline(self, session: Session, airline_code: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> List[Flight]:
        """Get flights by airline and optional date range"""
        try:
            query = session.query(Flight).filter(Flight.airline_code == airline_code)
            
            if start_date:
                query = query.filter(Flight.scheduled_departure >= start_date)
            if end_date:
                query = query.filter(Flight.scheduled_departure <= end_date)
            
            return query.order_by(Flight.scheduled_departure).all()
        except Exception as e:
            self.logger.error(f"Failed to get flights by airline {airline_code}: {e}")
            return []
    
    def get_delayed_flights(self, session: Session, min_delay_minutes: int = 15,
                          airport_code: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Flight]:
        """Get flights with delays above threshold"""
        try:
            query = session.query(Flight).filter(
                or_(
                    Flight.departure_delay_minutes >= min_delay_minutes,
                    Flight.arrival_delay_minutes >= min_delay_minutes
                )
            )
            
            if airport_code:
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
            
            return query.order_by(desc(Flight.departure_delay_minutes)).all()
        except Exception as e:
            self.logger.error(f"Failed to get delayed flights: {e}")
            return []
    
    def update_flight(self, session: Session, flight_id: str, update_data: Dict[str, Any]) -> Optional[Flight]:
        """Update flight record"""
        try:
            flight = session.query(Flight).filter(Flight.flight_id == flight_id).first()
            if flight:
                for key, value in update_data.items():
                    if hasattr(flight, key):
                        setattr(flight, key, value)
                flight.updated_at = datetime.utcnow()
                session.flush()
                self.logger.debug(f"Updated flight: {flight_id}")
                return flight
            return None
        except Exception as e:
            self.logger.error(f"Failed to update flight {flight_id}: {e}")
            raise
    
    def delete_flight(self, session: Session, flight_id: str) -> bool:
        """Delete flight record"""
        try:
            flight = session.query(Flight).filter(Flight.flight_id == flight_id).first()
            if flight:
                session.delete(flight)
                session.flush()
                self.logger.debug(f"Deleted flight: {flight_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete flight {flight_id}: {e}")
            raise
    
    def bulk_insert_flights(self, session: Session, flights_data: List[Dict[str, Any]]) -> int:
        """Bulk insert flight records"""
        try:
            flights = [Flight(**data) for data in flights_data]
            session.add_all(flights)
            session.flush()
            count = len(flights)
            self.logger.info(f"Bulk inserted {count} flights")
            return count
        except Exception as e:
            self.logger.error(f"Failed to bulk insert flights: {e}")
            raise
    
    def get_flight_statistics(self, session: Session, airport_code: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get flight statistics for analysis"""
        try:
            query = session.query(Flight)
            
            if airport_code:
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
            
            # Calculate statistics
            total_flights = query.count()
            avg_departure_delay = query.with_entities(func.avg(Flight.departure_delay_minutes)).scalar() or 0
            avg_arrival_delay = query.with_entities(func.avg(Flight.arrival_delay_minutes)).scalar() or 0
            max_departure_delay = query.with_entities(func.max(Flight.departure_delay_minutes)).scalar() or 0
            max_arrival_delay = query.with_entities(func.max(Flight.arrival_delay_minutes)).scalar() or 0
            
            # Count delayed flights (>15 minutes)
            delayed_flights = query.filter(
                or_(
                    Flight.departure_delay_minutes > 15,
                    Flight.arrival_delay_minutes > 15
                )
            ).count()
            
            return {
                'total_flights': total_flights,
                'delayed_flights': delayed_flights,
                'delay_percentage': (delayed_flights / total_flights * 100) if total_flights > 0 else 0,
                'avg_departure_delay_minutes': round(avg_departure_delay, 2),
                'avg_arrival_delay_minutes': round(avg_arrival_delay, 2),
                'max_departure_delay_minutes': max_departure_delay,
                'max_arrival_delay_minutes': max_arrival_delay
            }
        except Exception as e:
            self.logger.error(f"Failed to get flight statistics: {e}")
            return {}


class AirportRepository:
    """Repository for airport data operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_airport(self, session: Session, airport_data: Dict[str, Any]) -> Airport:
        """Create a new airport record"""
        try:
            airport = Airport(**airport_data)
            session.add(airport)
            session.flush()
            self.logger.debug(f"Created airport: {airport.code}")
            return airport
        except Exception as e:
            self.logger.error(f"Failed to create airport: {e}")
            raise
    
    def get_airport_by_code(self, session: Session, airport_code: str) -> Optional[Airport]:
        """Get airport by code"""
        try:
            return session.query(Airport).filter(Airport.code == airport_code).first()
        except Exception as e:
            self.logger.error(f"Failed to get airport {airport_code}: {e}")
            return None
    
    def get_all_airports(self, session: Session) -> List[Airport]:
        """Get all airports"""
        try:
            return session.query(Airport).order_by(Airport.code).all()
        except Exception as e:
            self.logger.error(f"Failed to get all airports: {e}")
            return []
    
    def update_airport(self, session: Session, airport_code: str, update_data: Dict[str, Any]) -> Optional[Airport]:
        """Update airport record"""
        try:
            airport = session.query(Airport).filter(Airport.code == airport_code).first()
            if airport:
                for key, value in update_data.items():
                    if hasattr(airport, key):
                        setattr(airport, key, value)
                airport.updated_at = datetime.utcnow()
                session.flush()
                self.logger.debug(f"Updated airport: {airport_code}")
                return airport
            return None
        except Exception as e:
            self.logger.error(f"Failed to update airport {airport_code}: {e}")
            raise


class AirlineRepository:
    """Repository for airline data operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_airline(self, session: Session, airline_data: Dict[str, Any]) -> Airline:
        """Create a new airline record"""
        try:
            airline = Airline(**airline_data)
            session.add(airline)
            session.flush()
            self.logger.debug(f"Created airline: {airline.code}")
            return airline
        except Exception as e:
            self.logger.error(f"Failed to create airline: {e}")
            raise
    
    def get_airline_by_code(self, session: Session, airline_code: str) -> Optional[Airline]:
        """Get airline by code"""
        try:
            return session.query(Airline).filter(Airline.code == airline_code).first()
        except Exception as e:
            self.logger.error(f"Failed to get airline {airline_code}: {e}")
            return None
    
    def get_all_airlines(self, session: Session) -> List[Airline]:
        """Get all airlines"""
        try:
            return session.query(Airline).order_by(Airline.name).all()
        except Exception as e:
            self.logger.error(f"Failed to get all airlines: {e}")
            return []


class AircraftRepository:
    """Repository for aircraft data operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_aircraft(self, session: Session, aircraft_data: Dict[str, Any]) -> Aircraft:
        """Create a new aircraft record"""
        try:
            aircraft = Aircraft(**aircraft_data)
            session.add(aircraft)
            session.flush()
            self.logger.debug(f"Created aircraft: {aircraft.type_code}")
            return aircraft
        except Exception as e:
            self.logger.error(f"Failed to create aircraft: {e}")
            raise
    
    def get_aircraft_by_type(self, session: Session, type_code: str) -> Optional[Aircraft]:
        """Get aircraft by type code"""
        try:
            return session.query(Aircraft).filter(Aircraft.type_code == type_code).first()
        except Exception as e:
            self.logger.error(f"Failed to get aircraft {type_code}: {e}")
            return None


class AnalysisResultRepository:
    """Repository for analysis results operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_analysis_result(self, session: Session, result_data: Dict[str, Any]) -> AnalysisResult:
        """Create a new analysis result record"""
        try:
            result = AnalysisResult(**result_data)
            session.add(result)
            session.flush()
            self.logger.debug(f"Created analysis result: {result.analysis_type}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to create analysis result: {e}")
            raise
    
    def get_analysis_results(self, session: Session, analysis_type: Optional[str] = None,
                           airport_code: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[AnalysisResult]:
        """Get analysis results with filters"""
        try:
            query = session.query(AnalysisResult)
            
            if analysis_type:
                query = query.filter(AnalysisResult.analysis_type == analysis_type)
            if airport_code:
                query = query.filter(AnalysisResult.airport_code == airport_code)
            if start_date:
                query = query.filter(AnalysisResult.analysis_date >= start_date)
            if end_date:
                query = query.filter(AnalysisResult.analysis_date <= end_date)
            
            return query.order_by(desc(AnalysisResult.analysis_date)).all()
        except Exception as e:
            self.logger.error(f"Failed to get analysis results: {e}")
            return []


class DataIngestionLogRepository:
    """Repository for data ingestion log operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_ingestion_log(self, session: Session, log_data: Dict[str, Any]) -> DataIngestionLog:
        """Create a new ingestion log record"""
        try:
            log = DataIngestionLog(**log_data)
            session.add(log)
            session.flush()
            self.logger.debug(f"Created ingestion log: {log.source_type}")
            return log
        except Exception as e:
            self.logger.error(f"Failed to create ingestion log: {e}")
            raise
    
    def update_ingestion_log(self, session: Session, log_id: str, update_data: Dict[str, Any]) -> Optional[DataIngestionLog]:
        """Update ingestion log record"""
        try:
            log = session.query(DataIngestionLog).filter(DataIngestionLog.id == log_id).first()
            if log:
                for key, value in update_data.items():
                    if hasattr(log, key):
                        setattr(log, key, value)
                session.flush()
                self.logger.debug(f"Updated ingestion log: {log_id}")
                return log
            return None
        except Exception as e:
            self.logger.error(f"Failed to update ingestion log {log_id}: {e}")
            raise


# Repository instances
flight_repo = FlightRepository()
airport_repo = AirportRepository()
airline_repo = AirlineRepository()
aircraft_repo = AircraftRepository()
analysis_result_repo = AnalysisResultRepository()
ingestion_log_repo = DataIngestionLogRepository()


# Convenience functions for common operations
def get_flights_dataframe(airport_code: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Get flights as pandas DataFrame for analysis"""
    with db_session_scope() as session:
        query = session.query(Flight)
        
        if airport_code:
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
        
        # Convert to DataFrame
        flights = query.all()
        if not flights:
            return pd.DataFrame()
        
        data = []
        for flight in flights:
            data.append({
                'flight_id': flight.flight_id,
                'flight_number': flight.flight_number,
                'airline_code': flight.airline_code,
                'origin_airport': flight.origin_airport,
                'destination_airport': flight.destination_airport,
                'aircraft_type': flight.aircraft_type,
                'scheduled_departure': flight.scheduled_departure,
                'scheduled_arrival': flight.scheduled_arrival,
                'actual_departure': flight.actual_departure,
                'actual_arrival': flight.actual_arrival,
                'departure_delay_minutes': flight.departure_delay_minutes,
                'arrival_delay_minutes': flight.arrival_delay_minutes,
                'delay_category': flight.delay_category,
                'passenger_count': flight.passenger_count,
                'status': flight.status
            })
        
        return pd.DataFrame(data)


def insert_sample_data():
    """Insert sample data for testing"""
    with db_session_scope() as session:
        # Sample airports
        airports = [
            {
                'code': 'BOM',
                'name': 'Chhatrapati Shivaji Maharaj International Airport',
                'city': 'Mumbai',
                'country': 'India',
                'timezone': 'Asia/Kolkata',
                'runway_count': 2,
                'runway_capacity': 45,
                'latitude': 19.0896,
                'longitude': 72.8656
            },
            {
                'code': 'DEL',
                'name': 'Indira Gandhi International Airport',
                'city': 'Delhi',
                'country': 'India',
                'timezone': 'Asia/Kolkata',
                'runway_count': 4,
                'runway_capacity': 65,
                'latitude': 28.5562,
                'longitude': 77.1000
            }
        ]
        
        for airport_data in airports:
            existing = airport_repo.get_airport_by_code(session, airport_data['code'])
            if not existing:
                airport_repo.create_airport(session, airport_data)
        
        # Sample airlines
        airlines = [
            {'code': 'AI', 'name': 'Air India', 'country': 'India'},
            {'code': '6E', 'name': 'IndiGo', 'country': 'India'},
            {'code': 'SG', 'name': 'SpiceJet', 'country': 'India'},
            {'code': 'UK', 'name': 'Vistara', 'country': 'India'}
        ]
        
        for airline_data in airlines:
            existing = airline_repo.get_airline_by_code(session, airline_data['code'])
            if not existing:
                airline_repo.create_airline(session, airline_data)
        
        # Sample aircraft
        aircraft_types = [
            {
                'type_code': 'A320',
                'manufacturer': 'Airbus',
                'model': 'A320',
                'typical_seating': 150,
                'max_seating': 180,
                'max_range': 3300,
                'cruise_speed': 450
            },
            {
                'type_code': 'B737',
                'manufacturer': 'Boeing',
                'model': '737-800',
                'typical_seating': 162,
                'max_seating': 189,
                'max_range': 3200,
                'cruise_speed': 453
            }
        ]
        
        for aircraft_data in aircraft_types:
            existing = aircraft_repo.get_aircraft_by_type(session, aircraft_data['type_code'])
            if not existing:
                aircraft_repo.create_aircraft(session, aircraft_data)
        
        logger.info("Sample data inserted successfully")


class DatabaseOperations:
    """
    High-level database operations for NLP query processing.
    
    This class provides methods specifically designed for the NLP query processor
    to retrieve relevant flight data based on different query types.
    """
    
    def __init__(self):
        self.flight_repo = flight_repo
        self.airport_repo = airport_repo
        self.airline_repo = airline_repo
        self.analysis_repo = analysis_result_repo
        self.logger = logger
    
    def get_delay_analysis_data(self, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Get flight data for delay analysis queries.
        
        Args:
            filters: Dictionary containing query filters
            
        Returns:
            DataFrame with flight delay data
        """
        try:
            with db_session_scope() as session:
                query = session.query(Flight)
                
                # Apply filters
                if 'airports' in filters:
                    airport_conditions = []
                    for airport in filters['airports']:
                        airport_conditions.extend([
                            Flight.origin_airport == airport,
                            Flight.destination_airport == airport
                        ])
                    query = query.filter(or_(*airport_conditions))
                
                if 'airlines' in filters:
                    query = query.filter(Flight.airline_code.in_(filters['airlines']))
                
                if 'time_range' in filters and filters['time_range']:
                    time_range = filters['time_range']
                    if 'start' in time_range:
                        query = query.filter(Flight.scheduled_departure >= time_range['start'])
                    if 'end' in time_range:
                        query = query.filter(Flight.scheduled_departure <= time_range['end'])
                
                # Get flights with delay information
                flights = query.all()
                
                if not flights:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for flight in flights:
                    # Calculate delay minutes
                    delay_minutes = 0
                    if flight.actual_departure and flight.scheduled_departure:
                        delay_minutes = (flight.actual_departure - flight.scheduled_departure).total_seconds() / 60
                    
                    data.append({
                        'flight_id': flight.flight_id,
                        'flight_number': flight.flight_number,
                        'airline': flight.airline_code,
                        'origin_airport': flight.origin_airport,
                        'destination_airport': flight.destination_airport,
                        'scheduled_departure': flight.scheduled_departure,
                        'actual_departure': flight.actual_departure,
                        'scheduled_arrival': flight.scheduled_arrival,
                        'actual_arrival': flight.actual_arrival,
                        'delay_minutes': max(0, delay_minutes),  # Only positive delays
                        'delay_category': flight.delay_category or 'unknown',
                        'aircraft_type': flight.aircraft_type,
                        'passenger_count': flight.passenger_count
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error getting delay analysis data: {str(e)}")
            return None
    
    def get_congestion_data(self, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Get flight data for congestion analysis queries.
        
        Args:
            filters: Dictionary containing query filters
            
        Returns:
            DataFrame with flight congestion data
        """
        try:
            with db_session_scope() as session:
                query = session.query(Flight)
                
                # Apply filters
                if 'airports' in filters:
                    airport_conditions = []
                    for airport in filters['airports']:
                        airport_conditions.extend([
                            Flight.origin_airport == airport,
                            Flight.destination_airport == airport
                        ])
                    query = query.filter(or_(*airport_conditions))
                
                if 'time_range' in filters and filters['time_range']:
                    time_range = filters['time_range']
                    if 'start' in time_range:
                        query = query.filter(Flight.scheduled_departure >= time_range['start'])
                    if 'end' in time_range:
                        query = query.filter(Flight.scheduled_departure <= time_range['end'])
                
                flights = query.all()
                
                if not flights:
                    return pd.DataFrame()
                
                # Convert to DataFrame with hourly aggregation
                data = []
                for flight in flights:
                    data.append({
                        'flight_id': flight.flight_id,
                        'scheduled_departure': flight.scheduled_departure,
                        'origin_airport': flight.origin_airport,
                        'destination_airport': flight.destination_airport,
                        'airline': flight.airline_code,
                        'aircraft_type': flight.aircraft_type,
                        'hour': flight.scheduled_departure.hour if flight.scheduled_departure else 0,
                        'day_of_week': flight.scheduled_departure.weekday() if flight.scheduled_departure else 0
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error getting congestion data: {str(e)}")
            return None
    
    def get_schedule_impact_data(self, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Get flight data for schedule impact analysis queries.
        
        Args:
            filters: Dictionary containing query filters
            
        Returns:
            DataFrame with flight schedule data
        """
        try:
            with db_session_scope() as session:
                query = session.query(Flight)
                
                # Apply filters similar to delay analysis
                if 'airports' in filters:
                    airport_conditions = []
                    for airport in filters['airports']:
                        airport_conditions.extend([
                            Flight.origin_airport == airport,
                            Flight.destination_airport == airport
                        ])
                    query = query.filter(or_(*airport_conditions))
                
                if 'airlines' in filters:
                    query = query.filter(Flight.airline_code.in_(filters['airlines']))
                
                if 'time_range' in filters and filters['time_range']:
                    time_range = filters['time_range']
                    if 'start' in time_range:
                        query = query.filter(Flight.scheduled_departure >= time_range['start'])
                    if 'end' in time_range:
                        query = query.filter(Flight.scheduled_departure <= time_range['end'])
                
                flights = query.all()
                
                if not flights:
                    return pd.DataFrame()
                
                # Convert to DataFrame with schedule impact metrics
                data = []
                for flight in flights:
                    # Calculate turnaround time if it's a connecting flight
                    turnaround_time = None
                    if flight.actual_arrival and flight.scheduled_departure:
                        # This is simplified - in reality, we'd need to track aircraft routing
                        turnaround_time = (flight.scheduled_departure - flight.actual_arrival).total_seconds() / 60
                    
                    data.append({
                        'flight_id': flight.flight_id,
                        'flight_number': flight.flight_number,
                        'airline': flight.airline_code,
                        'origin_airport': flight.origin_airport,
                        'destination_airport': flight.destination_airport,
                        'aircraft_type': flight.aircraft_type,
                        'scheduled_departure': flight.scheduled_departure,
                        'scheduled_arrival': flight.scheduled_arrival,
                        'actual_departure': flight.actual_departure,
                        'actual_arrival': flight.actual_arrival,
                        'departure_delay_minutes': flight.departure_delay_minutes or 0,
                        'arrival_delay_minutes': flight.arrival_delay_minutes or 0,
                        'turnaround_time_minutes': turnaround_time,
                        'passenger_count': flight.passenger_count
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error getting schedule impact data: {str(e)}")
            return None
    
    def get_cascading_impact_data(self, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Get flight data for cascading impact analysis queries.
        
        Args:
            filters: Dictionary containing query filters
            
        Returns:
            DataFrame with flight network data
        """
        try:
            with db_session_scope() as session:
                query = session.query(Flight)
                
                # Apply filters
                if 'airports' in filters:
                    airport_conditions = []
                    for airport in filters['airports']:
                        airport_conditions.extend([
                            Flight.origin_airport == airport,
                            Flight.destination_airport == airport
                        ])
                    query = query.filter(or_(*airport_conditions))
                
                if 'time_range' in filters and filters['time_range']:
                    time_range = filters['time_range']
                    if 'start' in time_range:
                        query = query.filter(Flight.scheduled_departure >= time_range['start'])
                    if 'end' in time_range:
                        query = query.filter(Flight.scheduled_departure <= time_range['end'])
                
                flights = query.all()
                
                if not flights:
                    return pd.DataFrame()
                
                # Convert to DataFrame with network analysis data
                data = []
                for flight in flights:
                    # Calculate network impact score (simplified)
                    network_impact_score = 1.0  # Base score
                    
                    # Increase score for hub airports
                    if flight.origin_airport in ['BOM', 'DEL', 'BLR']:
                        network_impact_score += 0.5
                    if flight.destination_airport in ['BOM', 'DEL', 'BLR']:
                        network_impact_score += 0.5
                    
                    # Increase score for larger aircraft (more passengers affected)
                    if flight.passenger_count and flight.passenger_count > 150:
                        network_impact_score += 0.3
                    
                    # Increase score for delays
                    if flight.departure_delay_minutes and flight.departure_delay_minutes > 30:
                        network_impact_score += 0.4
                    
                    data.append({
                        'flight_id': flight.flight_id,
                        'flight_number': flight.flight_number,
                        'airline': flight.airline_code,
                        'origin_airport': flight.origin_airport,
                        'destination_airport': flight.destination_airport,
                        'aircraft_type': flight.aircraft_type,
                        'scheduled_departure': flight.scheduled_departure,
                        'departure_delay_minutes': flight.departure_delay_minutes or 0,
                        'arrival_delay_minutes': flight.arrival_delay_minutes or 0,
                        'passenger_count': flight.passenger_count or 0,
                        'network_impact_score': network_impact_score,
                        'is_hub_flight': flight.origin_airport in ['BOM', 'DEL', 'BLR'] or 
                                       flight.destination_airport in ['BOM', 'DEL', 'BLR']
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error getting cascading impact data: {str(e)}")
            return None
    
    def get_general_flight_data(self, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Get general flight data for information queries.
        
        Args:
            filters: Dictionary containing query filters
            
        Returns:
            DataFrame with general flight information
        """
        try:
            with db_session_scope() as session:
                query = session.query(Flight)
                
                # Apply filters
                if 'airports' in filters:
                    airport_conditions = []
                    for airport in filters['airports']:
                        airport_conditions.extend([
                            Flight.origin_airport == airport,
                            Flight.destination_airport == airport
                        ])
                    query = query.filter(or_(*airport_conditions))
                
                if 'airlines' in filters:
                    query = query.filter(Flight.airline_code.in_(filters['airlines']))
                
                if 'time_range' in filters and filters['time_range']:
                    time_range = filters['time_range']
                    if 'start' in time_range:
                        query = query.filter(Flight.scheduled_departure >= time_range['start'])
                    if 'end' in time_range:
                        query = query.filter(Flight.scheduled_departure <= time_range['end'])
                
                # Limit results for general queries
                flights = query.limit(100).all()
                
                if not flights:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for flight in flights:
                    data.append({
                        'flight_id': flight.flight_id,
                        'flight_number': flight.flight_number,
                        'airline': flight.airline_code,
                        'origin_airport': flight.origin_airport,
                        'destination_airport': flight.destination_airport,
                        'aircraft_type': flight.aircraft_type,
                        'scheduled_departure': flight.scheduled_departure,
                        'scheduled_arrival': flight.scheduled_arrival,
                        'actual_departure': flight.actual_departure,
                        'actual_arrival': flight.actual_arrival,
                        'status': flight.status,
                        'passenger_count': flight.passenger_count,
                        'delay_category': flight.delay_category
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error getting general flight data: {str(e)}")
            return None