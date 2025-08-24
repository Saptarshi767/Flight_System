"""
Demo Data Generator for Flight Analysis Dashboard
Creates realistic flight data for presentation screenshots
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

def generate_airport_data():
    """Generate realistic airport performance data"""
    airports = {
        'BOM': {
            'name': 'Mumbai',
            'city': 'Mumbai',
            'daily_flights': 950,
            'avg_delay': 18.3,
            'ontime_rate': 72.1,
            'status': 'Moderate',
            'status_color': '🟡',
            'runway_capacity': 45,
            'peak_hours': [7, 8, 9, 18, 19, 20],
            'weather_factor': 0.8
        },
        'DEL': {
            'name': 'Delhi',
            'city': 'New Delhi',
            'daily_flights': 1200,
            'avg_delay': 22.7,
            'ontime_rate': 68.5,
            'status': 'High Congestion',
            'status_color': '🔴',
            'runway_capacity': 50,
            'peak_hours': [6, 7, 8, 17, 18, 19],
            'weather_factor': 0.7
        },
        'BLR': {
            'name': 'Bangalore',
            'city': 'Bengaluru',
            'daily_flights': 680,
            'avg_delay': 8.9,
            'ontime_rate': 85.3,
            'status': 'Normal',
            'status_color': '🟢',
            'runway_capacity': 35,
            'peak_hours': [7, 8, 18, 19],
            'weather_factor': 0.9
        },
        'MAA': {
            'name': 'Chennai',
            'city': 'Chennai',
            'daily_flights': 520,
            'avg_delay': 15.2,
            'ontime_rate': 76.8,
            'status': 'Moderate',
            'status_color': '🟡',
            'runway_capacity': 30,
            'peak_hours': [7, 8, 18, 19],
            'weather_factor': 0.75
        },
        'CCU': {
            'name': 'Kolkata',
            'city': 'Kolkata',
            'daily_flights': 380,
            'avg_delay': 11.4,
            'ontime_rate': 81.2,
            'status': 'Normal',
            'status_color': '🟢',
            'runway_capacity': 25,
            'peak_hours': [7, 8, 18, 19],
            'weather_factor': 0.85
        },
        'HYD': {
            'name': 'Hyderabad',
            'city': 'Hyderabad',
            'daily_flights': 420,
            'avg_delay': 9.6,
            'ontime_rate': 83.7,
            'status': 'Normal',
            'status_color': '🟢',
            'runway_capacity': 28,
            'peak_hours': [7, 8, 18, 19],
            'weather_factor': 0.88
        }
    }
    return airports

def generate_hourly_delay_patterns(airport_code, airports_data):
    """Generate realistic hourly delay patterns for an airport"""
    airport = airports_data[airport_code]
    base_delay = airport['avg_delay']
    peak_hours = airport['peak_hours']
    
    hourly_delays = []
    for hour in range(24):
        if hour in peak_hours:
            # Higher delays during peak hours
            delay = base_delay * random.uniform(1.5, 2.5)
        elif hour in [0, 1, 2, 3, 4, 5]:
            # Lower delays during night hours
            delay = base_delay * random.uniform(0.2, 0.6)
        else:
            # Normal delays during off-peak hours
            delay = base_delay * random.uniform(0.7, 1.3)
        
        # Add some randomness
        delay += random.gauss(0, 3)
        delay = max(0, delay)  # Ensure non-negative delays
        
        hourly_delays.append(round(delay, 1))
    
    return hourly_delays

def generate_congestion_heatmap():
    """Generate congestion heatmap data"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    congestion_data = []
    for day_idx, day in enumerate(days):
        for hour in hours:
            # Higher congestion during business days and peak hours
            base_congestion = 30
            
            # Business day factor
            if day_idx < 5:  # Monday to Friday
                base_congestion += 20
            
            # Peak hour factor
            if hour in [7, 8, 9, 18, 19, 20]:
                base_congestion += 30
            elif hour in [6, 10, 17, 21]:
                base_congestion += 15
            
            # Weekend evening factor
            if day_idx >= 5 and hour in [18, 19, 20, 21]:
                base_congestion += 25
            
            # Night time reduction
            if hour in [0, 1, 2, 3, 4, 5]:
                base_congestion *= 0.3
            
            # Add randomness
            congestion = base_congestion + random.gauss(0, 10)
            congestion = max(0, min(100, congestion))  # Clamp between 0-100
            
            congestion_data.append({
                'day': day,
                'hour': hour,
                'congestion': round(congestion, 1)
            })
    
    return congestion_data

def generate_flight_network_data():
    """Generate flight network data for graph analysis"""
    airports = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU', 'HYD']
    routes = []
    
    # Generate routes between airports
    for i, origin in enumerate(airports):
        for j, destination in enumerate(airports):
            if i != j:
                # Calculate route importance and frequency
                origin_size = generate_airport_data()[origin]['daily_flights']
                dest_size = generate_airport_data()[destination]['daily_flights']
                
                # More flights between larger airports
                daily_flights = int((origin_size + dest_size) / 100 * random.uniform(0.5, 1.5))
                
                routes.append({
                    'origin': origin,
                    'destination': destination,
                    'daily_flights': daily_flights,
                    'avg_delay': random.uniform(5, 25),
                    'importance_score': daily_flights * random.uniform(0.8, 1.2)
                })
    
    return routes

def generate_ml_performance_data():
    """Generate ML model performance metrics"""
    # Simulate actual vs predicted delays
    n_samples = 1000
    actual_delays = np.random.exponential(15, n_samples)  # Exponential distribution for delays
    
    # Add some correlation with noise for predictions
    predicted_delays = actual_delays * 0.87 + np.random.normal(0, 3, n_samples)
    predicted_delays = np.maximum(0, predicted_delays)  # Ensure non-negative
    
    # Calculate performance metrics
    mae = np.mean(np.abs(actual_delays - predicted_delays))
    rmse = np.sqrt(np.mean((actual_delays - predicted_delays) ** 2))
    
    # R-squared
    ss_res = np.sum((actual_delays - predicted_delays) ** 2)
    ss_tot = np.sum((actual_delays - np.mean(actual_delays)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'actual_delays': actual_delays.tolist(),
        'predicted_delays': predicted_delays.tolist(),
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2_score': round(r2, 3),
        'accuracy': round(r2 * 100, 1)
    }

def generate_time_series_forecast():
    """Generate time series forecasting data"""
    # Generate 30 days of historical data
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    # Generate realistic flight volume with trends and seasonality
    base_volume = 800
    trend = np.linspace(0, 50, 30)  # Slight upward trend
    seasonal = 100 * np.sin(np.linspace(0, 4*np.pi, 30))  # Seasonal pattern
    noise = np.random.normal(0, 30, 30)
    
    historical_volume = base_volume + trend + seasonal + noise
    historical_volume = np.maximum(200, historical_volume)  # Minimum volume
    
    # Generate 7 days of forecast
    forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    forecast_trend = np.linspace(50, 60, 7)
    forecast_seasonal = 100 * np.sin(np.linspace(4*np.pi, 5*np.pi, 7))
    forecast_volume = base_volume + forecast_trend + forecast_seasonal
    
    # Add confidence intervals
    confidence_upper = forecast_volume + 50
    confidence_lower = forecast_volume - 50
    
    return {
        'historical_dates': [d.strftime('%Y-%m-%d') for d in dates],
        'historical_volume': historical_volume.tolist(),
        'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'forecast_volume': forecast_volume.tolist(),
        'confidence_upper': confidence_upper.tolist(),
        'confidence_lower': confidence_lower.tolist()
    }

def save_demo_data():
    """Save all demo data to JSON files"""
    
    # Generate all data
    airports_data = generate_airport_data()
    
    demo_data = {
        'airports': airports_data,
        'hourly_delays': {},
        'congestion_heatmap': generate_congestion_heatmap(),
        'flight_network': generate_flight_network_data(),
        'ml_performance': generate_ml_performance_data(),
        'time_series_forecast': generate_time_series_forecast()
    }
    
    # Generate hourly delays for each airport
    for airport_code in airports_data.keys():
        demo_data['hourly_delays'][airport_code] = generate_hourly_delay_patterns(airport_code, airports_data)
    
    # Save to JSON file
    with open('demo_data.json', 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print("✅ Demo data generated and saved to demo_data.json")
    print(f"📊 Generated data for {len(airports_data)} airports")
    print(f"🕐 Hourly patterns for 24-hour periods")
    print(f"🔥 Congestion heatmap with {len(demo_data['congestion_heatmap'])} data points")
    print(f"🛫 Flight network with {len(demo_data['flight_network'])} routes")
    print(f"🤖 ML performance data with {len(demo_data['ml_performance']['actual_delays'])} samples")
    print(f"📈 Time series forecast for 7 days")

def create_sample_flight_data():
    """Create sample flight data CSV for analysis"""
    
    airports_data = generate_airport_data()
    airlines = ['AI', '6E', 'SG', 'UK', 'G8', '9W', 'I5']
    aircraft_types = ['A320', 'B737', 'A321', 'B738', 'ATR72', 'A319']
    
    flights = []
    flight_id = 1000
    
    for airport_code, airport_info in airports_data.items():
        daily_flights = airport_info['daily_flights']
        
        # Generate flights for this airport
        for _ in range(int(daily_flights * 0.7)):  # 70% of daily flights as sample
            
            # Random flight details
            airline = random.choice(airlines)
            aircraft = random.choice(aircraft_types)
            
            # Random destination (different from origin)
            destinations = [code for code in airports_data.keys() if code != airport_code]
            destination = random.choice(destinations)
            
            # Random scheduled time
            hour = random.randint(5, 23)
            minute = random.choice([0, 15, 30, 45])
            scheduled_dep = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Calculate delay based on hour and airport characteristics
            base_delay = airport_info['avg_delay']
            if hour in airport_info['peak_hours']:
                delay = random.gauss(base_delay * 1.8, 8)
            else:
                delay = random.gauss(base_delay * 0.8, 5)
            
            delay = max(0, delay)  # No negative delays
            actual_dep = scheduled_dep + timedelta(minutes=delay)
            
            # Flight duration (random between 1-3 hours)
            duration = random.randint(60, 180)
            scheduled_arr = scheduled_dep + timedelta(minutes=duration)
            actual_arr = actual_dep + timedelta(minutes=duration)
            
            flights.append({
                'flight_id': f"{airline}{flight_id}",
                'airline': airline,
                'aircraft_type': aircraft,
                'origin_airport': airport_code,
                'destination_airport': destination,
                'scheduled_departure': scheduled_dep.strftime('%Y-%m-%d %H:%M:%S'),
                'actual_departure': actual_dep.strftime('%Y-%m-%d %H:%M:%S'),
                'scheduled_arrival': scheduled_arr.strftime('%Y-%m-%d %H:%M:%S'),
                'actual_arrival': actual_arr.strftime('%Y-%m-%d %H:%M:%S'),
                'delay_minutes': round(delay, 1),
                'passenger_count': random.randint(80, 180)
            })
            
            flight_id += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(flights)
    df.to_csv('sample_flight_data.csv', index=False)
    
    print(f"✅ Sample flight data created: {len(flights)} flights")
    print("📁 Saved to sample_flight_data.csv")
    
    return df

if __name__ == "__main__":
    print("🚀 Generating demo data for Flight Analysis Dashboard...")
    
    # Generate all demo data
    save_demo_data()
    
    # Create sample flight data CSV
    create_sample_flight_data()
    
    print("\n🎯 Demo data generation complete!")
    print("📋 Files created:")
    print("   - demo_data.json (Dashboard data)")
    print("   - sample_flight_data.csv (Flight records)")
    print("\n📸 Ready for screenshot capture!")