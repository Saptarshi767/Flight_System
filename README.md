# Flight Scheduling Analysis System

An AI-powered flight scheduling analysis system designed to optimize flight scheduling at busy airports, particularly Mumbai (BOM) and Delhi (DEL). The system uses open-source AI tools and OpenAI to analyze flight data and provide insights for reducing delays and improving scheduling efficiency.

## 🎯 Project Expectations

This system addresses five key expectations:

1. **Open Source AI Tools + NLP Interface**: Uses scikit-learn, XGBoost, Prophet, NetworkX, spaCy for analysis with OpenAI for natural language queries
2. **Best Takeoff/Landing Times**: Analyzes scheduled vs actual times to identify optimal scheduling windows
3. **Busiest Time Slots**: Identifies peak congestion periods to avoid
4. **Schedule Tuning Model**: Provides "what-if" analysis for schedule changes and delay impact
5. **Cascading Impact Analysis**: Identifies critical flights with biggest network impact

## 🏗️ Architecture

- **Backend**: FastAPI with Python 3.11+
- **Databases**: PostgreSQL (structured data) + InfluxDB (time series) + Redis (caching)
- **AI/ML**: Scikit-learn, XGBoost, TensorFlow, Prophet, NetworkX, spaCy
- **NLP**: OpenAI GPT-4 + LangChain
- **Frontend**: Streamlit dashboard with Plotly visualizations
- **Deployment**: Docker + Docker Compose

## 📊 Data Sources

- **Primary**: Flight_Data.xlsx (provided sample data)
- **Web Scraping**: 
  - Mumbai Airport: https://www.flightradar24.com/data/airports/bom
  - Delhi Airport: https://www.flightradar24.com/data/airports/del
  - FlightRadar24: https://www.flightradar24.com/
  - FlightAware: https://www.flightaware.com/

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flight-scheduling-analysis
   ```

2. **Install dependencies (Quick Fix)**
   ```bash
   # Quick fix for missing packages
   python fix_imports.py
   
   # Or install from requirements
   pip install -r requirements.txt
   
   # Or use the comprehensive installer
   python install_dependencies.py
   ```

3. **Set up environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your OpenAI API key
   # OPENAI_API_KEY=sk-proj-your-key-here
   ```

3. **Install dependencies**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Start services with Docker**
   ```bash
   docker-compose up -d
   ```

5. **Access the application**
   - API Documentation: http://localhost:8000/docs
   - Streamlit Dashboard: http://localhost:8501
   - Database: PostgreSQL on localhost:5432
   - Cache: Redis on localhost:6379
   - Time Series DB: InfluxDB on localhost:8086

## 📁 Project Structure

```
flight-scheduling-analysis/
├── src/
│   ├── api/           # FastAPI REST endpoints
│   ├── analysis/      # Analysis engines and ML models
│   ├── data/          # Data processing and ingestion
│   ├── database/      # Database models and connections
│   ├── nlp/           # Natural language processing
│   ├── utils/         # Utility functions
│   └── config.py      # Configuration management
├── tests/             # Test suite
├── data/              # Data files (created automatically)
├── logs/              # Log files (created automatically)
├── .kiro/specs/       # Project specifications
├── docker-compose.yml # Docker services configuration
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# OpenAI API Key (Required)
OPENAI_API_KEY=sk-proj-your-key-here

# Database URLs
DATABASE_URL=postgresql://user:password@localhost:5432/flightdb
REDIS_URL=redis://localhost:6379/0
INFLUXDB_URL=http://localhost:8086

# Data Source URLs
MUMBAI_AIRPORT_URL=https://www.flightradar24.com/data/airports/bom
DELHI_AIRPORT_URL=https://www.flightradar24.com/data/airports/del

# File Paths
FLIGHT_DATA_EXCEL=Flight_Data.xlsx
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data_processing.py
```

## 📈 Usage Examples

### Natural Language Queries

Once the system is running, you can ask questions like:

- "What's the best time to schedule a flight from Mumbai to Delhi?"
- "Which flights cause the most delays in the network?"
- "Show me congestion patterns for Delhi airport last week"
- "What would happen if I move flight AI101 to 2 hours earlier?"

### API Endpoints

- `POST /api/v1/data/upload` - Upload flight data files
- `GET /api/v1/analysis/delays` - Get delay analysis results
- `GET /api/v1/analysis/congestion` - Get congestion analysis
- `POST /api/v1/nlp/query` - Process natural language queries

## 🛠️ Development

### Adding New Features

1. Check the implementation plan in `.kiro/specs/flight-scheduling-analysis/tasks.md`
2. Update task status when starting work
3. Follow the modular architecture
4. Add tests for new functionality
5. Update documentation

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 📋 Implementation Status

- [x] Project structure and environment setup
- [ ] Data ingestion and processing
- [ ] Database infrastructure
- [ ] Core analysis engines
- [ ] Machine learning models
- [ ] Natural language processing
- [ ] REST API development
- [ ] Web dashboard
- [ ] System integration and deployment

## 🤝 Contributing

1. Follow the task-based development approach outlined in the specs
2. Write tests for new functionality
3. Follow Python best practices and PEP 8
4. Update documentation as needed

## 📄 License

This project is developed for flight scheduling optimization research and analysis.

## 🆘 Support

For issues and questions:
1. Check the project specifications in `.kiro/specs/`
2. Review the implementation tasks
3. Check existing issues and documentation