-- Initialize the flight scheduling database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create airports table
CREATE TABLE IF NOT EXISTS airports (
    code VARCHAR(3) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    city VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL,
    runway_count INTEGER NOT NULL DEFAULT 1,
    runway_capacity INTEGER NOT NULL DEFAULT 60,
    timezone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    coordinates POINT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create flights table
CREATE TABLE IF NOT EXISTS flights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    flight_number VARCHAR(10) NOT NULL,
    airline_code VARCHAR(3) NOT NULL,
    aircraft_type VARCHAR(10),
    origin_airport VARCHAR(3) NOT NULL REFERENCES airports(code),
    destination_airport VARCHAR(3) NOT NULL REFERENCES airports(code),
    scheduled_departure TIMESTAMP NOT NULL,
    actual_departure TIMESTAMP,
    scheduled_arrival TIMESTAMP NOT NULL,
    actual_arrival TIMESTAMP,
    delay_minutes INTEGER DEFAULT 0,
    delay_category VARCHAR(20),
    runway_used VARCHAR(10),
    passenger_count INTEGER,
    weather_conditions JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_type VARCHAR(50) NOT NULL,
    airport_code VARCHAR(3) NOT NULL REFERENCES airports(code),
    analysis_date DATE NOT NULL,
    metrics JSONB NOT NULL,
    recommendations JSONB,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_flights_departure ON flights(scheduled_departure);
CREATE INDEX IF NOT EXISTS idx_flights_arrival ON flights(scheduled_arrival);
CREATE INDEX IF NOT EXISTS idx_flights_origin ON flights(origin_airport);
CREATE INDEX IF NOT EXISTS idx_flights_destination ON flights(destination_airport);
CREATE INDEX IF NOT EXISTS idx_flights_airline ON flights(airline_code);
CREATE INDEX IF NOT EXISTS idx_flights_delay ON flights(delay_minutes);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_airport ON analysis_results(airport_code);
CREATE INDEX IF NOT EXISTS idx_analysis_results_date ON analysis_results(analysis_date);

-- Insert sample airport data
INSERT INTO airports (code, name, city, country, runway_count, runway_capacity, timezone) VALUES
    ('BOM', 'Chhatrapati Shivaji Maharaj International Airport', 'Mumbai', 'India', 2, 120, 'Asia/Kolkata'),
    ('DEL', 'Indira Gandhi International Airport', 'Delhi', 'India', 4, 180, 'Asia/Kolkata')
ON CONFLICT (code) DO NOTHING;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_flights_updated_at BEFORE UPDATE ON flights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_airports_updated_at BEFORE UPDATE ON airports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();