"""
Load testing scripts for the Flight Scheduling Analysis System
"""
import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import random
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str = "http://localhost:80"
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: int = 60  # seconds
    test_duration: int = 300  # seconds
    request_timeout: int = 30
    think_time: float = 1.0  # seconds between requests
    endpoints: List[str] = field(default_factory=list)


@dataclass
class RequestResult:
    """Individual request result"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0


class LoadTestRunner:
    """Load test runner for API endpoints"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.results_lock = threading.Lock()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Default endpoints if none provided
        if not self.config.endpoints:
            self.config.endpoints = [
                "/health",
                "/api/v1/data/flights",
                "/api/v1/analysis/delays",
                "/api/v1/analysis/congestion",
                "/api/v1/nlp/query",
                "/api/v1/reports/dashboard"
            ]
    
    async def setup_session(self):
        """Setup HTTP session"""
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "LoadTest/1.0"}
        )
    
    async def cleanup_session(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    async def make_request(self, endpoint: str, method: str = "GET", 
                          data: Dict[str, Any] = None) -> RequestResult:
        """Make a single HTTP request"""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    content = await response.read()
                    response_time = time.time() - start_time
                    
                    return RequestResult(
                        endpoint=endpoint,
                        method=method,
                        status_code=response.status,
                        response_time=response_time,
                        timestamp=datetime.now(),
                        success=200 <= response.status < 400,
                        response_size=len(content)
                    )
            
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    content = await response.read()
                    response_time = time.time() - start_time
                    
                    return RequestResult(
                        endpoint=endpoint,
                        method=method,
                        status_code=response.status,
                        response_time=response_time,
                        timestamp=datetime.now(),
                        success=200 <= response.status < 400,
                        response_size=len(content)
                    )
        
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    async def user_simulation(self, user_id: int):
        """Simulate a single user's behavior"""
        logger.info(f"Starting user {user_id}")
        
        # Ramp up delay
        ramp_delay = (user_id / self.config.concurrent_users) * self.config.ramp_up_time
        await asyncio.sleep(ramp_delay)
        
        requests_made = 0
        start_time = time.time()
        
        while (requests_made < self.config.requests_per_user and 
               time.time() - start_time < self.config.test_duration):
            
            # Select random endpoint
            endpoint = random.choice(self.config.endpoints)
            
            # Prepare request data based on endpoint
            method = "GET"
            data = None
            
            if "/nlp/query" in endpoint:
                method = "POST"
                data = {
                    "query": random.choice([
                        "What are the busiest hours at Mumbai airport?",
                        "Show me delay patterns for Air India flights",
                        "Which flights have the most cascading impact?",
                        "What's the best time to fly from Delhi to Mumbai?"
                    ]),
                    "context": {},
                    "session_id": f"load_test_user_{user_id}"
                }
            
            elif "/data/flights" in endpoint:
                # Add query parameters for flight data
                params = random.choice([
                    "?airline=AI&limit=50",
                    "?origin_airport=BOM&limit=100",
                    "?destination_airport=DEL&limit=100",
                    "?delay_category=weather&limit=50"
                ])
                endpoint += params
            
            # Make request
            result = await self.make_request(endpoint, method, data)
            
            # Store result
            with self.results_lock:
                self.results.append(result)
            
            requests_made += 1
            
            # Think time between requests
            if self.config.think_time > 0:
                await asyncio.sleep(self.config.think_time)
        
        logger.info(f"User {user_id} completed {requests_made} requests")
    
    async def run_load_test(self) -> Dict[str, Any]:
        """Run the complete load test"""
        logger.info(f"Starting load test with {self.config.concurrent_users} users")
        logger.info(f"Target: {self.config.requests_per_user} requests per user")
        logger.info(f"Duration: {self.config.test_duration} seconds")
        
        await self.setup_session()
        
        try:
            # Create user simulation tasks
            tasks = []
            for user_id in range(self.config.concurrent_users):
                task = asyncio.create_task(self.user_simulation(user_id))
                tasks.append(task)
            
            # Wait for all users to complete
            start_time = time.time()
            await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            return self.analyze_results(total_time)
        
        finally:
            await self.cleanup_session()
    
    def analyze_results(self, total_time: float) -> Dict[str, Any]:
        """Analyze load test results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.success])
        failed_requests = total_requests - successful_requests
        
        # Response time statistics
        response_times = [r.response_time for r in self.results]
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        p99_response_time = sorted(response_times)[int(0.99 * len(response_times))]
        
        # Throughput
        requests_per_second = total_requests / total_time
        
        # Status code distribution
        status_codes = {}
        for result in self.results:
            status_codes[result.status_code] = status_codes.get(result.status_code, 0) + 1
        
        # Endpoint performance
        endpoint_stats = {}
        for result in self.results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {
                    'count': 0,
                    'success_count': 0,
                    'response_times': []
                }
            
            endpoint_stats[result.endpoint]['count'] += 1
            if result.success:
                endpoint_stats[result.endpoint]['success_count'] += 1
            endpoint_stats[result.endpoint]['response_times'].append(result.response_time)
        
        # Calculate endpoint averages
        for endpoint, stats in endpoint_stats.items():
            stats['avg_response_time'] = statistics.mean(stats['response_times'])
            stats['success_rate'] = stats['success_count'] / stats['count'] * 100
        
        # Error analysis
        errors = [r for r in self.results if not r.success]
        error_types = {}
        for error in errors:
            error_key = error.error_message or f"HTTP_{error.status_code}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        return {
            'test_config': {
                'concurrent_users': self.config.concurrent_users,
                'requests_per_user': self.config.requests_per_user,
                'test_duration': self.config.test_duration,
                'total_time': total_time
            },
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests * 100,
                'requests_per_second': requests_per_second
            },
            'response_times': {
                'average': avg_response_time,
                'median': median_response_time,
                'p95': p95_response_time,
                'p99': p99_response_time,
                'min': min(response_times),
                'max': max(response_times)
            },
            'status_codes': status_codes,
            'endpoint_performance': endpoint_stats,
            'errors': error_types
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("LOAD TEST RESULTS")
        print("="*80)
        
        # Test configuration
        config = results['test_config']
        print(f"\nTest Configuration:")
        print(f"  Concurrent Users: {config['concurrent_users']}")
        print(f"  Requests per User: {config['requests_per_user']}")
        print(f"  Test Duration: {config['test_duration']}s")
        print(f"  Actual Duration: {config['total_time']:.2f}s")
        
        # Summary
        summary = results['summary']
        print(f"\nSummary:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Successful: {summary['successful_requests']}")
        print(f"  Failed: {summary['failed_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.2f}%")
        print(f"  Throughput: {summary['requests_per_second']:.2f} req/s")
        
        # Response times
        times = results['response_times']
        print(f"\nResponse Times:")
        print(f"  Average: {times['average']:.3f}s")
        print(f"  Median: {times['median']:.3f}s")
        print(f"  95th Percentile: {times['p95']:.3f}s")
        print(f"  99th Percentile: {times['p99']:.3f}s")
        print(f"  Min: {times['min']:.3f}s")
        print(f"  Max: {times['max']:.3f}s")
        
        # Status codes
        print(f"\nStatus Codes:")
        for code, count in sorted(results['status_codes'].items()):
            print(f"  {code}: {count}")
        
        # Endpoint performance
        print(f"\nEndpoint Performance:")
        for endpoint, stats in results['endpoint_performance'].items():
            print(f"  {endpoint}:")
            print(f"    Requests: {stats['count']}")
            print(f"    Success Rate: {stats['success_rate']:.2f}%")
            print(f"    Avg Response Time: {stats['avg_response_time']:.3f}s")
        
        # Errors
        if results['errors']:
            print(f"\nErrors:")
            for error, count in results['errors'].items():
                print(f"  {error}: {count}")


class StressTestRunner(LoadTestRunner):
    """Stress test runner that gradually increases load"""
    
    async def run_stress_test(self, max_users: int = 100, step_size: int = 10, 
                            step_duration: int = 60) -> List[Dict[str, Any]]:
        """Run stress test with gradually increasing load"""
        results = []
        
        for users in range(step_size, max_users + 1, step_size):
            logger.info(f"Running stress test with {users} users")
            
            # Update configuration
            self.config.concurrent_users = users
            self.config.test_duration = step_duration
            self.results = []  # Reset results
            
            # Run test
            result = await self.run_load_test()
            result['concurrent_users'] = users
            results.append(result)
            
            # Check if system is failing
            if result['summary']['success_rate'] < 50:
                logger.warning(f"System failing at {users} users, stopping stress test")
                break
            
            # Brief pause between test steps
            await asyncio.sleep(5)
        
        return results


async def main():
    """Main function for running load tests"""
    parser = argparse.ArgumentParser(description="Flight Analysis System Load Test")
    parser.add_argument("--base-url", default="http://localhost:80", help="Base URL for API")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=100, help="Requests per user")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=int, default=60, help="Ramp up time in seconds")
    parser.add_argument("--think-time", type=float, default=1.0, help="Think time between requests")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test")
    parser.add_argument("--max-users", type=int, default=100, help="Maximum users for stress test")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = LoadTestConfig(
        base_url=args.base_url,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        test_duration=args.duration,
        ramp_up_time=args.ramp_up,
        think_time=args.think_time
    )
    
    if args.stress_test:
        # Run stress test
        runner = StressTestRunner(config)
        results = await runner.run_stress_test(args.max_users)
        
        print("\n" + "="*80)
        print("STRESS TEST RESULTS")
        print("="*80)
        
        for result in results:
            users = result['concurrent_users']
            success_rate = result['summary']['success_rate']
            throughput = result['summary']['requests_per_second']
            avg_response = result['response_times']['average']
            
            print(f"Users: {users:3d} | Success: {success_rate:6.2f}% | "
                  f"Throughput: {throughput:6.2f} req/s | "
                  f"Avg Response: {avg_response:.3f}s")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    else:
        # Run regular load test
        runner = LoadTestRunner(config)
        results = await runner.run_load_test()
        runner.print_results(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())