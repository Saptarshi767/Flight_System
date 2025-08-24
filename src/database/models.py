"""
SQLAlchemy database models for flight scheduling analysis
"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import String as SQLString
from datetime import datetime
import uuid

Base = declarative_base()


class Airport(Base):
    """Airport information table"""
    __tablename__ = 'airports'
    
    # Primary key
    code = Column(String(3), primary_key=True, comment="IATA airport code")
    
    # Basic information
    name = Column(String(100), nullable=False, comment="Airport name")
    city = Column(String(50), nullable=False, comment="City name")
    country = Column(String(50), nullable=False, comment="Country name")
    timezone = Column(String(50), nullable=False, default='UTC', comment="Airport timezone")
    
    # Operational information
    runway_count = Column(Integer, nullable=False, default=1, comment="Number of runways")
    runway_capacity = Column(Integer, nullable=False, default=30, comment="Flights per hour capacity")
    
    # Geographic information
    latitude = Column(Float, comment="Airport latitude")
    longitude = Column(Float, comment="Airport longitude")
    elevation = Column(Integer, comment="Airport elevation in feet")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    origin_flights = relationship("Flight", foreign_keys="Flight.origin_airport", back_populates="origin")
    destination_flights = relationship("Flight", foreign_keys="Flight.destination_airport", back_populates="destination")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('runway_count > 0', name='check_runway_count_positive'),
        CheckConstraint('runway_capacity > 0', name='check_runway_capacity_positive'),
        CheckConstraint('latitude >= -90 AND latitude <= 90', name='check_latitude_range'),
        CheckConstraint('longitude >= -180 AND longitude <= 180', name='check_longitude_range'),
        Index('idx_airports_city', 'city'),
        Index('idx_airports_country', 'country'),
    )
    
    def __repr__(self):
        return f"<Airport(code='{self.code}', name='{self.name}', city='{self.city}')>"


class Airline(Base):
    """Airline information table"""
    __tablename__ = 'airlines'
    
    # Primary key
    code = Column(String(3), primary_key=True, comment="IATA airline code")
    
    # Basic information
    name = Column(String(100), nullable=False, comment="Airline name")
    country = Column(String(50), comment="Country of origin")
    
    # Operational information
    is_active = Column(Boolean, default=True, nullable=False, comment="Is airline currently active")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    flights = relationship("Flight", back_populates="airline_info")
    
    # Indexes
    __table_args__ = (
        Index('idx_airlines_name', 'name'),
        Index('idx_airlines_country', 'country'),
        Index('idx_airlines_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Airline(code='{self.code}', name='{self.name}')>"


class Aircraft(Base):
    """Aircraft type information table"""
    __tablename__ = 'aircraft'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Aircraft information
    type_code = Column(String(10), nullable=False, unique=True, comment="Aircraft type code (e.g., A320, B737)")
    manufacturer = Column(String(50), comment="Aircraft manufacturer")
    model = Column(String(50), comment="Aircraft model")
    
    # Capacity information
    typical_seating = Column(Integer, comment="Typical passenger seating capacity")
    max_seating = Column(Integer, comment="Maximum passenger seating capacity")
    cargo_capacity = Column(Float, comment="Cargo capacity in tons")
    
    # Performance information
    max_range = Column(Integer, comment="Maximum range in nautical miles")
    cruise_speed = Column(Integer, comment="Typical cruise speed in knots")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    flights = relationship("Flight", back_populates="aircraft_info")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('typical_seating > 0', name='check_typical_seating_positive'),
        CheckConstraint('max_seating >= typical_seating', name='check_max_seating_valid'),
        Index('idx_aircraft_type', 'type_code'),
        Index('idx_aircraft_manufacturer', 'manufacturer'),
    )
    
    def __repr__(self):
        return f"<Aircraft(type_code='{self.type_code}', manufacturer='{self.manufacturer}')>"


class Flight(Base):
    """Flight information table"""
    __tablename__ = 'flights'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="Unique flight record ID")
    
    # Flight identification
    flight_id = Column(String(50), nullable=False, unique=True, comment="Business flight ID")
    flight_number = Column(String(10), nullable=False, comment="Flight number")
    airline_code = Column(String(3), ForeignKey('airlines.code'), nullable=False, comment="Airline code")
    
    # Route information
    origin_airport = Column(String(3), ForeignKey('airports.code'), nullable=False, comment="Origin airport code")
    destination_airport = Column(String(3), ForeignKey('airports.code'), nullable=False, comment="Destination airport code")
    
    # Aircraft information
    aircraft_type = Column(String(10), ForeignKey('aircraft.type_code'), comment="Aircraft type code")
    
    # Scheduled times
    scheduled_departure = Column(DateTime, nullable=False, comment="Scheduled departure time")
    scheduled_arrival = Column(DateTime, nullable=False, comment="Scheduled arrival time")
    
    # Actual times
    actual_departure = Column(DateTime, comment="Actual departure time")
    actual_arrival = Column(DateTime, comment="Actual arrival time")
    
    # Delay information
    departure_delay_minutes = Column(Integer, default=0, comment="Departure delay in minutes")
    arrival_delay_minutes = Column(Integer, default=0, comment="Arrival delay in minutes")
    delay_category = Column(String(20), comment="Category of delay (weather, operational, traffic, etc.)")
    
    # Operational information
    runway_used = Column(String(10), comment="Runway used for takeoff/landing")
    gate = Column(String(10), comment="Gate number")
    terminal = Column(String(10), comment="Terminal")
    
    # Passenger and cargo information
    passenger_count = Column(Integer, comment="Number of passengers")
    cargo_weight = Column(Float, comment="Cargo weight in kg")
    
    # Flight status
    status = Column(String(20), default='scheduled', comment="Flight status")
    
    # Weather and conditions
    weather_conditions = Column(JSON, comment="Weather conditions at departure/arrival")
    
    # Data source and quality
    data_source = Column(String(20), nullable=False, comment="Source of flight data")
    data_quality_score = Column(Float, comment="Data quality score (0-1)")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, comment="When the flight data was processed")
    
    # Relationships
    airline_info = relationship("Airline", back_populates="flights")
    origin = relationship("Airport", foreign_keys=[origin_airport], back_populates="origin_flights")
    destination = relationship("Airport", foreign_keys=[destination_airport], back_populates="destination_flights")
    aircraft_info = relationship("Aircraft", back_populates="flights")
    analysis_results = relationship("AnalysisResult", back_populates="flight")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('scheduled_arrival > scheduled_departure', name='check_arrival_after_departure'),
        CheckConstraint('departure_delay_minutes >= -60', name='check_departure_delay_reasonable'),
        CheckConstraint('arrival_delay_minutes >= -60', name='check_arrival_delay_reasonable'),
        CheckConstraint('passenger_count >= 0', name='check_passenger_count_non_negative'),
        CheckConstraint('cargo_weight >= 0', name='check_cargo_weight_non_negative'),
        CheckConstraint('data_quality_score >= 0 AND data_quality_score <= 1', name='check_quality_score_range'),
        
        # Indexes for performance
        Index('idx_flights_flight_number', 'flight_number'),
        Index('idx_flights_airline', 'airline_code'),
        Index('idx_flights_route', 'origin_airport', 'destination_airport'),
        Index('idx_flights_scheduled_departure', 'scheduled_departure'),
        Index('idx_flights_scheduled_arrival', 'scheduled_arrival'),
        Index('idx_flights_actual_departure', 'actual_departure'),
        Index('idx_flights_status', 'status'),
        Index('idx_flights_data_source', 'data_source'),
        Index('idx_flights_delay_category', 'delay_category'),
        Index('idx_flights_created_at', 'created_at'),
        
        # Composite indexes for common queries
        Index('idx_flights_route_date', 'origin_airport', 'destination_airport', 'scheduled_departure'),
        Index('idx_flights_airline_date', 'airline_code', 'scheduled_departure'),
        Index('idx_flights_delay_analysis', 'departure_delay_minutes', 'delay_category'),
    )
    
    def __repr__(self):
        return f"<Flight(flight_id='{self.flight_id}', flight_number='{self.flight_number}', route='{self.origin_airport}-{self.destination_airport}')>"


class AnalysisResult(Base):
    """Analysis results table"""
    __tablename__ = 'analysis_results'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Analysis information
    analysis_type = Column(String(50), nullable=False, comment="Type of analysis performed")
    analysis_date = Column(DateTime, nullable=False, default=datetime.utcnow, comment="When analysis was performed")
    
    # Scope of analysis
    airport_code = Column(String(3), ForeignKey('airports.code'), comment="Airport analyzed")
    flight_id = Column(String(36), ForeignKey('flights.id'), comment="Specific flight analyzed")
    
    # Results
    metrics = Column(JSON, nullable=False, comment="Analysis metrics and results")
    recommendations = Column(JSON, comment="Analysis recommendations")
    confidence_score = Column(Float, comment="Confidence score of analysis (0-1)")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    airport = relationship("Airport")
    flight = relationship("Flight", back_populates="analysis_results")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_score_range'),
        Index('idx_analysis_type', 'analysis_type'),
        Index('idx_analysis_date', 'analysis_date'),
        Index('idx_analysis_airport', 'airport_code'),
        Index('idx_analysis_flight', 'flight_id'),
        Index('idx_analysis_type_date', 'analysis_type', 'analysis_date'),
    )
    
    def __repr__(self):
        return f"<AnalysisResult(analysis_type='{self.analysis_type}', airport='{self.airport_code}')>"


class DataIngestionLog(Base):
    """Data ingestion logging table"""
    __tablename__ = 'data_ingestion_logs'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Ingestion information
    source_type = Column(String(20), nullable=False, comment="Type of data source")
    source_name = Column(String(100), comment="Name/identifier of data source")
    
    # Processing information
    start_time = Column(DateTime, nullable=False, comment="When ingestion started")
    end_time = Column(DateTime, comment="When ingestion completed")
    status = Column(String(20), nullable=False, comment="Ingestion status")
    
    # Statistics
    records_processed = Column(Integer, default=0, comment="Number of records processed")
    records_inserted = Column(Integer, default=0, comment="Number of records inserted")
    records_updated = Column(Integer, default=0, comment="Number of records updated")
    records_failed = Column(Integer, default=0, comment="Number of records that failed")
    
    # Error information
    error_message = Column(Text, comment="Error message if ingestion failed")
    error_details = Column(JSON, comment="Detailed error information")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('records_processed >= 0', name='check_records_processed_non_negative'),
        CheckConstraint('records_inserted >= 0', name='check_records_inserted_non_negative'),
        CheckConstraint('records_updated >= 0', name='check_records_updated_non_negative'),
        CheckConstraint('records_failed >= 0', name='check_records_failed_non_negative'),
        Index('idx_ingestion_source_type', 'source_type'),
        Index('idx_ingestion_status', 'status'),
        Index('idx_ingestion_start_time', 'start_time'),
        Index('idx_ingestion_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<DataIngestionLog(source_type='{self.source_type}', status='{self.status}', records={self.records_processed})>"