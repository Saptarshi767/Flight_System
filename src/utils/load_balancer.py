"""
Load balancing and scalability utilities for the Flight Scheduling Analysis System
"""
import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty

from src.utils.logging import logger


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    RANDOM = "random"


@dataclass
class ServerNode:
    """Represents a server node in the load balancer"""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    is_active: bool = True
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor"""
        return self.current_connections / self.max_connections
    
    def update_response_time(self, response_time: float):
        """Update average response time using exponential moving average"""
        alpha = 0.1  # Smoothing factor
        self.avg_response_time = (alpha * response_time) + ((1 - alpha) * self.avg_response_time)
    
    def increment_connections(self):
        """Increment connection count"""
        self.current_connections += 1
        self.total_requests += 1
    
    def decrement_connections(self):
        """Decrement connection count"""
        self.current_connections = max(0, self.current_connections - 1)
    
    def record_failure(self):
        """Record a failed request"""
        self.failed_requests += 1
    
    def __str__(self):
        return f"ServerNode({self.id}, {self.host}:{self.port}, load={self.load_factor:.2f})"


class LoadBalancer:
    """Load balancer for distributing requests across multiple server nodes"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.nodes: List[ServerNode] = []
        self.current_index = 0
        self.lock = threading.Lock()
        self.logger = logger
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self._health_check_task = None
        self._running = False
    
    def add_node(self, node: ServerNode):
        """Add a server node to the load balancer"""
        with self.lock:
            self.nodes.append(node)
            self.logger.info(f"Added server node: {node}")
    
    def remove_node(self, node_id: str):
        """Remove a server node from the load balancer"""
        with self.lock:
            self.nodes = [node for node in self.nodes if node.id != node_id]
            self.logger.info(f"Removed server node: {node_id}")
    
    def get_healthy_nodes(self) -> List[ServerNode]:
        """Get list of healthy and active nodes"""
        return [node for node in self.nodes if node.is_healthy and node.is_active]
    
    def select_node(self, request_key: str = None) -> Optional[ServerNode]:
        """Select a server node based on the configured strategy"""
        healthy_nodes = self.get_healthy_nodes()
        
        if not healthy_nodes:
            self.logger.warning("No healthy nodes available")
            return None
        
        with self.lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.HASH_BASED:
                return self._hash_based_selection(healthy_nodes, request_key)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(healthy_nodes)
            else:
                return self._round_robin_selection(healthy_nodes)
    
    def _round_robin_selection(self, nodes: List[ServerNode]) -> ServerNode:
        """Round-robin selection"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index = (self.current_index + 1) % len(nodes)
        return node
    
    def _weighted_round_robin_selection(self, nodes: List[ServerNode]) -> ServerNode:
        """Weighted round-robin selection"""
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return self._round_robin_selection(nodes)
        
        # Create weighted list
        weighted_nodes = []
        for node in nodes:
            weighted_nodes.extend([node] * node.weight)
        
        node = weighted_nodes[self.current_index % len(weighted_nodes)]
        self.current_index = (self.current_index + 1) % len(weighted_nodes)
        return node
    
    def _least_connections_selection(self, nodes: List[ServerNode]) -> ServerNode:
        """Select node with least connections"""
        return min(nodes, key=lambda n: n.current_connections)
    
    def _least_response_time_selection(self, nodes: List[ServerNode]) -> ServerNode:
        """Select node with least average response time"""
        return min(nodes, key=lambda n: n.avg_response_time)
    
    def _hash_based_selection(self, nodes: List[ServerNode], request_key: str) -> ServerNode:
        """Hash-based selection for session affinity"""
        if not request_key:
            return self._round_robin_selection(nodes)
        
        hash_value = int(hashlib.md5(request_key.encode()).hexdigest(), 16)
        return nodes[hash_value % len(nodes)]
    
    def _random_selection(self, nodes: List[ServerNode]) -> ServerNode:
        """Random selection"""
        return random.choice(nodes)
    
    async def execute_request(self, request_func: Callable, request_key: str = None, 
                            timeout: float = 30.0) -> Tuple[Any, ServerNode]:
        """Execute a request using load balancing"""
        node = self.select_node(request_key)
        if not node:
            raise Exception("No healthy nodes available")
        
        node.increment_connections()
        start_time = time.time()
        
        try:
            # Execute the request
            if asyncio.iscoroutinefunction(request_func):
                result = await asyncio.wait_for(request_func(node), timeout=timeout)
            else:
                result = request_func(node)
            
            # Record successful request
            response_time = time.time() - start_time
            node.update_response_time(response_time)
            
            return result, node
            
        except Exception as e:
            # Record failed request
            node.record_failure()
            self.logger.error(f"Request failed on node {node.id}: {e}")
            raise
        finally:
            node.decrement_connections()
    
    def start_health_checks(self):
        """Start periodic health checks"""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("Started health check monitoring")
    
    def stop_health_checks(self):
        """Stop periodic health checks"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
        self.logger.info("Stopped health check monitoring")
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _perform_health_checks(self):
        """Perform health checks on all nodes"""
        tasks = []
        for node in self.nodes:
            task = asyncio.create_task(self._check_node_health(node))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_node_health(self, node: ServerNode):
        """Check health of a single node"""
        try:
            # Simple TCP connection test
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node.host, node.port),
                timeout=self.health_check_timeout
            )
            writer.close()
            await writer.wait_closed()
            
            # Mark as healthy
            if not node.is_healthy:
                self.logger.info(f"Node {node.id} is back online")
            node.is_healthy = True
            node.last_health_check = datetime.now()
            
        except Exception as e:
            # Mark as unhealthy
            if node.is_healthy:
                self.logger.warning(f"Node {node.id} health check failed: {e}")
            node.is_healthy = False
            node.last_health_check = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_nodes = self.get_healthy_nodes()
        total_requests = sum(node.total_requests for node in self.nodes)
        total_failures = sum(node.failed_requests for node in self.nodes)
        
        return {
            'strategy': self.strategy.value,
            'total_nodes': len(self.nodes),
            'healthy_nodes': len(healthy_nodes),
            'total_requests': total_requests,
            'total_failures': total_failures,
            'success_rate': (total_requests - total_failures) / total_requests if total_requests > 0 else 1.0,
            'nodes': [
                {
                    'id': node.id,
                    'host': f"{node.host}:{node.port}",
                    'is_healthy': node.is_healthy,
                    'current_connections': node.current_connections,
                    'total_requests': node.total_requests,
                    'success_rate': node.success_rate,
                    'avg_response_time': node.avg_response_time,
                    'load_factor': node.load_factor
                }
                for node in self.nodes
            ]
        }


class AutoScaler:
    """Auto-scaling manager for dynamic resource allocation"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.logger = logger
        self.scale_up_threshold = 0.8  # Scale up when load > 80%
        self.scale_down_threshold = 0.3  # Scale down when load < 30%
        self.min_nodes = 2
        self.max_nodes = 10
        self.scale_cooldown = 300  # 5 minutes cooldown
        self.last_scale_action = datetime.now() - timedelta(seconds=self.scale_cooldown)
        self._monitoring = False
        self._monitor_task = None
    
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started auto-scaling monitoring")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        self.logger.info("Stopped auto-scaling monitoring")
    
    async def _monitoring_loop(self):
        """Auto-scaling monitoring loop"""
        while self._monitoring:
            try:
                await self._check_scaling_conditions()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-scaling monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_scaling_conditions(self):
        """Check if scaling action is needed"""
        if not self._can_scale():
            return
        
        healthy_nodes = self.load_balancer.get_healthy_nodes()
        if not healthy_nodes:
            return
        
        # Calculate average load
        avg_load = sum(node.load_factor for node in healthy_nodes) / len(healthy_nodes)
        
        # Check for scale up
        if avg_load > self.scale_up_threshold and len(healthy_nodes) < self.max_nodes:
            await self._scale_up()
        
        # Check for scale down
        elif avg_load < self.scale_down_threshold and len(healthy_nodes) > self.min_nodes:
            await self._scale_down()
    
    def _can_scale(self) -> bool:
        """Check if scaling action is allowed (cooldown period)"""
        return datetime.now() - self.last_scale_action > timedelta(seconds=self.scale_cooldown)
    
    async def _scale_up(self):
        """Scale up by adding a new node"""
        try:
            # In a real implementation, this would provision a new server instance
            new_node_id = f"auto-node-{int(time.time())}"
            new_node = ServerNode(
                id=new_node_id,
                host="localhost",  # Would be actual server IP
                port=8000 + len(self.load_balancer.nodes),  # Would be actual port
                weight=1,
                max_connections=100
            )
            
            self.load_balancer.add_node(new_node)
            self.last_scale_action = datetime.now()
            
            self.logger.info(f"Scaled up: Added node {new_node_id}")
            
        except Exception as e:
            self.logger.error(f"Scale up failed: {e}")
    
    async def _scale_down(self):
        """Scale down by removing a node"""
        try:
            healthy_nodes = self.load_balancer.get_healthy_nodes()
            if len(healthy_nodes) <= self.min_nodes:
                return
            
            # Remove node with least connections
            node_to_remove = min(healthy_nodes, key=lambda n: n.current_connections)
            
            # Wait for connections to drain
            max_wait = 30  # seconds
            wait_time = 0
            while node_to_remove.current_connections > 0 and wait_time < max_wait:
                await asyncio.sleep(1)
                wait_time += 1
            
            self.load_balancer.remove_node(node_to_remove.id)
            self.last_scale_action = datetime.now()
            
            self.logger.info(f"Scaled down: Removed node {node_to_remove.id}")
            
        except Exception as e:
            self.logger.error(f"Scale down failed: {e}")
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get auto-scaling metrics"""
        healthy_nodes = self.load_balancer.get_healthy_nodes()
        avg_load = sum(node.load_factor for node in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0
        
        return {
            'current_nodes': len(healthy_nodes),
            'min_nodes': self.min_nodes,
            'max_nodes': self.max_nodes,
            'average_load': avg_load,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'last_scale_action': self.last_scale_action.isoformat(),
            'can_scale': self._can_scale(),
            'monitoring_active': self._monitoring
        }


class RequestQueue:
    """Request queue for handling high traffic loads"""
    
    def __init__(self, max_size: int = 1000, max_workers: int = 10):
        self.queue = Queue(maxsize=max_size)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger
        self._processing = False
        self._workers = []
    
    def start_processing(self):
        """Start request processing workers"""
        if self._processing:
            return
        
        self._processing = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} request processing workers")
    
    def stop_processing(self):
        """Stop request processing workers"""
        self._processing = False
        
        # Add sentinel values to wake up workers
        for _ in range(self.max_workers):
            try:
                self.queue.put(None, timeout=1)
            except:
                pass
        
        self.logger.info("Stopped request processing workers")
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing requests"""
        while self._processing:
            try:
                item = self.queue.get(timeout=1)
                if item is None:  # Sentinel value to stop worker
                    break
                
                request_func, args, kwargs, future = item
                
                try:
                    result = request_func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def submit_request(self, request_func: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit a request to the queue"""
        future = asyncio.Future()
        
        try:
            self.queue.put((request_func, args, kwargs, future), timeout=1)
            return future
        except:
            future.set_exception(Exception("Request queue is full"))
            return future
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'queue_size': self.queue.qsize(),
            'max_size': self.queue.maxsize,
            'active_workers': len([w for w in self._workers if w.is_alive()]),
            'max_workers': self.max_workers,
            'processing': self._processing
        }


# Global instances for the application
default_load_balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
auto_scaler = AutoScaler(default_load_balancer)
request_queue = RequestQueue(max_size=1000, max_workers=10)