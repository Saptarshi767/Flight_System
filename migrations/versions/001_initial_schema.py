"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create airports table
    op.create_table('airports',
        sa.Column('code', sa.String(length=3), nullable=False, comment='IATA airport code'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='Airport name'),
        sa.Column('city', sa.String(length=50), nullable=False, comment='City name'),
        sa.Column('country', sa.String(length=50), nullable=False, comment='Country name'),
        sa.Column('timezone', sa.String(length=50), nullable=False, comment='Airport timezone'),
        sa.Column('runway_count', sa.Integer(), nullable=False, comment='Number of runways'),
        sa.Column('runway_capacity', sa.Integer(), nullable=False, comment='Flights per hour capacity'),
        sa.Column('latitude', sa.Float(), nullable=True, comment='Airport latitude'),
        sa.Column('longitude', sa.Float(), nullable=True, comment='Airport longitude'),
        sa.Column('elevation', sa.Integer(), nullable=True, comment='Airport elevation in feet'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('runway_count > 0', name='check_runway_count_positive'),
        sa.CheckConstraint('runway_capacity > 0', name='check_runway_capacity_positive'),
        sa.CheckConstraint('latitude >= -90 AND latitude <= 90', name='check_latitude_range'),
        sa.CheckConstraint('longitude >= -180 AND longitude <= 180', name='check_longitude_range'),
        sa.PrimaryKeyConstraint('code')
    )
    op.create_index('idx_airports_city', 'airports', ['city'])
    op.create_index('idx_airports_country', 'airports', ['country'])

    # Create airlines table
    op.create_table('airlines',
        sa.Column('code', sa.String(length=3), nullable=False, comment='IATA airline code'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='Airline name'),
        sa.Column('country', sa.String(length=50), nullable=True, comment='Country of origin'),
        sa.Column('is_active', sa.Boolean(), nullable=False, comment='Is airline currently active'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('code')
    )
    op.create_index('idx_airlines_name', 'airlines', ['name'])
    op.create_index('idx_airlines_country', 'airlines', ['country'])
    op.create_index('idx_airlines_active', 'airlines', ['is_active'])

    # Create aircraft table
    op.create_table('aircraft',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('type_code', sa.String(length=10), nullable=False, comment='Aircraft type code (e.g., A320, B737)'),
        sa.Column('manufacturer', sa.String(length=50), nullable=True, comment='Aircraft manufacturer'),
        sa.Column('model', sa.String(length=50), nullable=True, comment='Aircraft model'),
        sa.Column('typical_seating', sa.Integer(), nullable=True, comment='Typical passenger seating capacity'),
        sa.Column('max_seating', sa.Integer(), nullable=True, comment='Maximum passenger seating capacity'),
        sa.Column('cargo_capacity', sa.Float(), nullable=True, comment='Cargo capacity in tons'),
        sa.Column('max_range', sa.Integer(), nullable=True, comment='Maximum range in nautical miles'),
        sa.Column('cruise_speed', sa.Integer(), nullable=True, comment='Typical cruise speed in knots'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('typical_seating > 0', name='check_typical_seating_positive'),
        sa.CheckConstraint('max_seating >= typical_seating', name='check_max_seating_valid'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('type_code')
    )
    op.create_index('idx_aircraft_type', 'aircraft', ['type_code'])
    op.create_index('idx_aircraft_manufacturer', 'aircraft', ['manufacturer'])

    # Create flights table
    op.create_table('flights',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, comment='Unique flight record ID'),
        sa.Column('flight_id', sa.String(length=50), nullable=False, comment='Business flight ID'),
        sa.Column('flight_number', sa.String(length=10), nullable=False, comment='Flight number'),
        sa.Column('airline_code', sa.String(length=3), nullable=False, comment='Airline code'),
        sa.Column('origin_airport', sa.String(length=3), nullable=False, comment='Origin airport code'),
        sa.Column('destination_airport', sa.String(length=3), nullable=False, comment='Destination airport code'),
        sa.Column('aircraft_type', sa.String(length=10), nullable=True, comment='Aircraft type code'),
        sa.Column('scheduled_departure', sa.DateTime(), nullable=False, comment='Scheduled departure time'),
        sa.Column('scheduled_arrival', sa.DateTime(), nullable=False, comment='Scheduled arrival time'),
        sa.Column('actual_departure', sa.DateTime(), nullable=True, comment='Actual departure time'),
        sa.Column('actual_arrival', sa.DateTime(), nullable=True, comment='Actual arrival time'),
        sa.Column('departure_delay_minutes', sa.Integer(), nullable=True, comment='Departure delay in minutes'),
        sa.Column('arrival_delay_minutes', sa.Integer(), nullable=True, comment='Arrival delay in minutes'),
        sa.Column('delay_category', sa.String(length=20), nullable=True, comment='Category of delay (weather, operational, traffic, etc.)'),
        sa.Column('runway_used', sa.String(length=10), nullable=True, comment='Runway used for takeoff/landing'),
        sa.Column('gate', sa.String(length=10), nullable=True, comment='Gate number'),
        sa.Column('terminal', sa.String(length=10), nullable=True, comment='Terminal'),
        sa.Column('passenger_count', sa.Integer(), nullable=True, comment='Number of passengers'),
        sa.Column('cargo_weight', sa.Float(), nullable=True, comment='Cargo weight in kg'),
        sa.Column('status', sa.String(length=20), nullable=True, comment='Flight status'),
        sa.Column('weather_conditions', sa.JSON(), nullable=True, comment='Weather conditions at departure/arrival'),
        sa.Column('data_source', sa.String(length=20), nullable=False, comment='Source of flight data'),
        sa.Column('data_quality_score', sa.Float(), nullable=True, comment='Data quality score (0-1)'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True, comment='When the flight data was processed'),
        sa.CheckConstraint('scheduled_arrival > scheduled_departure', name='check_arrival_after_departure'),
        sa.CheckConstraint('departure_delay_minutes >= -60', name='check_departure_delay_reasonable'),
        sa.CheckConstraint('arrival_delay_minutes >= -60', name='check_arrival_delay_reasonable'),
        sa.CheckConstraint('passenger_count >= 0', name='check_passenger_count_non_negative'),
        sa.CheckConstraint('cargo_weight >= 0', name='check_cargo_weight_non_negative'),
        sa.CheckConstraint('data_quality_score >= 0 AND data_quality_score <= 1', name='check_quality_score_range'),
        sa.ForeignKeyConstraint(['aircraft_type'], ['aircraft.type_code'], ),
        sa.ForeignKeyConstraint(['airline_code'], ['airlines.code'], ),
        sa.ForeignKeyConstraint(['destination_airport'], ['airports.code'], ),
        sa.ForeignKeyConstraint(['origin_airport'], ['airports.code'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('flight_id')
    )
    
    # Create indexes for flights table
    op.create_index('idx_flights_flight_number', 'flights', ['flight_number'])
    op.create_index('idx_flights_airline', 'flights', ['airline_code'])
    op.create_index('idx_flights_route', 'flights', ['origin_airport', 'destination_airport'])
    op.create_index('idx_flights_scheduled_departure', 'flights', ['scheduled_departure'])
    op.create_index('idx_flights_scheduled_arrival', 'flights', ['scheduled_arrival'])
    op.create_index('idx_flights_actual_departure', 'flights', ['actual_departure'])
    op.create_index('idx_flights_status', 'flights', ['status'])
    op.create_index('idx_flights_data_source', 'flights', ['data_source'])
    op.create_index('idx_flights_delay_category', 'flights', ['delay_category'])
    op.create_index('idx_flights_created_at', 'flights', ['created_at'])
    
    # Create composite indexes for common queries
    op.create_index('idx_flights_route_date', 'flights', ['origin_airport', 'destination_airport', 'scheduled_departure'])
    op.create_index('idx_flights_airline_date', 'flights', ['airline_code', 'scheduled_departure'])
    op.create_index('idx_flights_delay_analysis', 'flights', ['departure_delay_minutes', 'delay_category'])

    # Create analysis_results table
    op.create_table('analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('analysis_type', sa.String(length=50), nullable=False, comment='Type of analysis performed'),
        sa.Column('analysis_date', sa.DateTime(), nullable=False, comment='When analysis was performed'),
        sa.Column('airport_code', sa.String(length=3), nullable=True, comment='Airport analyzed'),
        sa.Column('flight_id', postgresql.UUID(as_uuid=True), nullable=True, comment='Specific flight analyzed'),
        sa.Column('metrics', sa.JSON(), nullable=False, comment='Analysis metrics and results'),
        sa.Column('recommendations', sa.JSON(), nullable=True, comment='Analysis recommendations'),
        sa.Column('confidence_score', sa.Float(), nullable=True, comment='Confidence score of analysis (0-1)'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_score_range'),
        sa.ForeignKeyConstraint(['airport_code'], ['airports.code'], ),
        sa.ForeignKeyConstraint(['flight_id'], ['flights.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_analysis_type', 'analysis_results', ['analysis_type'])
    op.create_index('idx_analysis_date', 'analysis_results', ['analysis_date'])
    op.create_index('idx_analysis_airport', 'analysis_results', ['airport_code'])
    op.create_index('idx_analysis_flight', 'analysis_results', ['flight_id'])
    op.create_index('idx_analysis_type_date', 'analysis_results', ['analysis_type', 'analysis_date'])

    # Create data_ingestion_logs table
    op.create_table('data_ingestion_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_type', sa.String(length=20), nullable=False, comment='Type of data source'),
        sa.Column('source_name', sa.String(length=100), nullable=True, comment='Name/identifier of data source'),
        sa.Column('start_time', sa.DateTime(), nullable=False, comment='When ingestion started'),
        sa.Column('end_time', sa.DateTime(), nullable=True, comment='When ingestion completed'),
        sa.Column('status', sa.String(length=20), nullable=False, comment='Ingestion status'),
        sa.Column('records_processed', sa.Integer(), nullable=True, comment='Number of records processed'),
        sa.Column('records_inserted', sa.Integer(), nullable=True, comment='Number of records inserted'),
        sa.Column('records_updated', sa.Integer(), nullable=True, comment='Number of records updated'),
        sa.Column('records_failed', sa.Integer(), nullable=True, comment='Number of records that failed'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if ingestion failed'),
        sa.Column('error_details', sa.JSON(), nullable=True, comment='Detailed error information'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('records_processed >= 0', name='check_records_processed_non_negative'),
        sa.CheckConstraint('records_inserted >= 0', name='check_records_inserted_non_negative'),
        sa.CheckConstraint('records_updated >= 0', name='check_records_updated_non_negative'),
        sa.CheckConstraint('records_failed >= 0', name='check_records_failed_non_negative'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ingestion_source_type', 'data_ingestion_logs', ['source_type'])
    op.create_index('idx_ingestion_status', 'data_ingestion_logs', ['status'])
    op.create_index('idx_ingestion_start_time', 'data_ingestion_logs', ['start_time'])
    op.create_index('idx_ingestion_created_at', 'data_ingestion_logs', ['created_at'])


def downgrade() -> None:
    # Drop tables in reverse order to handle foreign key constraints
    op.drop_table('data_ingestion_logs')
    op.drop_table('analysis_results')
    op.drop_table('flights')
    op.drop_table('aircraft')
    op.drop_table('airlines')
    op.drop_table('airports')