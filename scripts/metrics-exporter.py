#!/usr/bin/env python3
"""
PRSM Custom Metrics Exporter
Collects and exports PRSM-specific metrics for Prometheus monitoring
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional
import yaml
import json
import httpx
import asyncpg
import redis.asyncio as redis
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from prometheus_client.core import CollectorRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PRSMMetricsExporter:
    """Custom metrics exporter for PRSM system metrics"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config = self._load_config(config_path)
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        # Connection pools
        self.db_pool = None
        self.redis_client = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'collection_interval': 15,
            'metrics_port': 9091,
            'prsm_api_url': os.getenv('PRSM_API_URL', 'http://localhost:8000'),
            'database_url': os.getenv('DATABASE_URL'),
            'redis_url': os.getenv('REDIS_URL'),
            'metrics': {
                'system': True,
                'tokenomics': True,
                'agents': True,
                'federation': True,
                'safety': True
            }
        }
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        
        # System metrics
        self.query_duration = Histogram(
            'prsm_query_processing_duration_seconds',
            'Time taken to process user queries through NWTN orchestrator',
            ['query_type', 'user_tier', 'complexity'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.concurrent_sessions = Gauge(
            'prsm_concurrent_sessions_total',
            'Number of active user sessions',
            ['session_type', 'region'],
            registry=self.registry
        )
        
        self.agent_pipeline_duration = Histogram(
            'prsm_agent_pipeline_execution_seconds',
            'End-to-end agent pipeline execution time',
            ['pipeline_stage', 'agent_type'],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # FTNS token metrics
        self.ftns_transactions_rate = Gauge(
            'ftns_transactions_per_second',
            'Rate of FTNS token transactions',
            ['transaction_type', 'user_tier'],
            registry=self.registry
        )
        
        self.ftns_cost_calculation_duration = Histogram(
            'ftns_cost_calculation_duration_seconds',
            'Time to calculate FTNS costs with microsecond precision',
            ['calculation_type'],
            buckets=[0.0001, 0.001, 0.01, 0.1, 1.0],
            registry=self.registry
        )
        
        self.ftns_balance_distribution = Histogram(
            'ftns_balance_distribution',
            'Distribution of FTNS balances across users',
            buckets=[10, 100, 1000, 10000, 100000],
            registry=self.registry
        )
        
        # Agent framework metrics
        self.agent_queue_depth = Gauge(
            'agent_task_queue_depth',
            'Number of tasks waiting in agent queues',
            ['agent_type', 'priority'],
            registry=self.registry
        )
        
        self.model_selection_duration = Histogram(
            'agent_model_selection_duration_seconds',
            'Time for router agents to select optimal models',
            ['router_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )
        
        self.compilation_success_rate = Gauge(
            'agent_compilation_success_rate',
            'Success rate of compiler agents aggregating results',
            ['compilation_type'],
            registry=self.registry
        )
        
        # P2P federation metrics
        self.p2p_connections = Gauge(
            'p2p_active_connections',
            'Number of active P2P connections',
            ['connection_type', 'region'],
            registry=self.registry
        )
        
        self.consensus_latency = Histogram(
            'p2p_consensus_latency_seconds',
            'Time to reach consensus in P2P network',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.model_registry_sync_duration = Histogram(
            'model_registry_sync_duration_seconds',
            'Time to synchronize model registry across network',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Safety metrics
        self.circuit_breaker_activations = Counter(
            'circuit_breaker_activations_total',
            'Number of circuit breaker activations',
            ['threat_level', 'component'],
            registry=self.registry
        )
        
        self.safety_validation_duration = Histogram(
            'safety_validation_duration_seconds',
            'Time to validate safety constraints',
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        self.security_threat_score = Gauge(
            'security_threat_score',
            'Current security threat assessment score',
            ['threat_type'],
            registry=self.registry
        )
        
        # System info
        self.prsm_info = Info(
            'prsm_system_info',
            'Information about PRSM system',
            registry=self.registry
        )
    
    async def start(self):
        """Start the metrics exporter"""
        await self._setup_connections()
        
        # Start Prometheus HTTP server
        start_http_server(
            self.config['metrics_port'], 
            registry=self.registry
        )
        
        logger.info(f"PRSM metrics exporter started on port {self.config['metrics_port']}")
        
        # Start metrics collection loop
        await self._collection_loop()
    
    async def _setup_connections(self):
        """Setup database and Redis connections"""
        try:
            if self.config.get('database_url'):
                self.db_pool = await asyncpg.create_pool(
                    self.config['database_url'],
                    min_size=1,
                    max_size=5
                )
                logger.info("Database connection pool created")
            
            if self.config.get('redis_url'):
                self.redis_client = redis.from_url(self.config['redis_url'])
                await self.redis_client.ping()
                logger.info("Redis connection established")
                
        except Exception as e:
            logger.error(f"Failed to setup connections: {e}")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        interval = self.config['collection_interval']
        
        while True:
            try:
                start_time = time.time()
                
                # Collect metrics from different sources
                if self.config['metrics']['system']:
                    await self._collect_system_metrics()
                
                if self.config['metrics']['tokenomics']:
                    await self._collect_tokenomics_metrics()
                
                if self.config['metrics']['agents']:
                    await self._collect_agent_metrics()
                
                if self.config['metrics']['federation']:
                    await self._collect_federation_metrics()
                
                if self.config['metrics']['safety']:
                    await self._collect_safety_metrics()
                
                collection_time = time.time() - start_time
                logger.debug(f"Metrics collection completed in {collection_time:.2f}s")
                
                # Sleep until next collection
                await asyncio.sleep(max(0, interval - collection_time))
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Get active sessions from API
            response = await self.http_client.get(
                f"{self.config['prsm_api_url']}/api/v1/metrics/sessions"
            )
            if response.status_code == 200:
                data = response.json()
                for session_type, count in data.get('active_sessions', {}).items():
                    self.concurrent_sessions.labels(
                        session_type=session_type,
                        region='unknown'
                    ).set(count)
            
            # Get query processing metrics from database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    query = """
                    SELECT query_type, user_tier, complexity, 
                           AVG(processing_duration) as avg_duration,
                           COUNT(*) as count
                    FROM query_metrics 
                    WHERE timestamp > NOW() - INTERVAL '5 minutes'
                    GROUP BY query_type, user_tier, complexity
                    """
                    rows = await conn.fetch(query)
                    
                    for row in rows:
                        self.query_duration.labels(
                            query_type=row['query_type'],
                            user_tier=row['user_tier'],
                            complexity=row['complexity']
                        ).observe(row['avg_duration'])
                        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_tokenomics_metrics(self):
        """Collect FTNS token metrics"""
        try:
            # Get FTNS transaction rates
            response = await self.http_client.get(
                f"{self.config['prsm_api_url']}/api/v1/metrics/ftns"
            )
            if response.status_code == 200:
                data = response.json()
                
                # Transaction rates
                for tx_type, rate in data.get('transaction_rates', {}).items():
                    self.ftns_transactions_rate.labels(
                        transaction_type=tx_type,
                        user_tier='all'
                    ).set(rate)
                
                # Cost calculation performance
                if 'cost_calculation_times' in data:
                    for calc_type, times in data['cost_calculation_times'].items():
                        for duration in times:
                            self.ftns_cost_calculation_duration.labels(
                                calculation_type=calc_type
                            ).observe(duration)
            
            # Get balance distribution from database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    query = "SELECT balance FROM user_balances WHERE balance > 0"
                    rows = await conn.fetch(query)
                    
                    for row in rows:
                        self.ftns_balance_distribution.observe(float(row['balance']))
                        
        except Exception as e:
            logger.error(f"Error collecting tokenomics metrics: {e}")
    
    async def _collect_agent_metrics(self):
        """Collect agent framework metrics"""
        try:
            # Get agent queue depths from Redis
            if self.redis_client:
                # Check task queues
                queue_patterns = [
                    'agent:architect:*',
                    'agent:prompter:*',
                    'agent:router:*',
                    'agent:executor:*',
                    'agent:compiler:*'
                ]
                
                for pattern in queue_patterns:
                    keys = await self.redis_client.keys(pattern)
                    for key in keys:
                        queue_length = await self.redis_client.llen(key)
                        agent_type = key.decode().split(':')[1]
                        
                        self.agent_queue_depth.labels(
                            agent_type=agent_type,
                            priority='normal'
                        ).set(queue_length)
            
            # Get performance metrics from API
            response = await self.http_client.get(
                f"{self.config['prsm_api_url']}/api/v1/metrics/agents"
            )
            if response.status_code == 200:
                data = response.json()
                
                # Model selection times
                for router_type, times in data.get('model_selection_times', {}).items():
                    for duration in times:
                        self.model_selection_duration.labels(
                            router_type=router_type
                        ).observe(duration)
                
                # Compilation success rates
                for comp_type, rate in data.get('compilation_success_rates', {}).items():
                    self.compilation_success_rate.labels(
                        compilation_type=comp_type
                    ).set(rate)
                        
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
    
    async def _collect_federation_metrics(self):
        """Collect P2P federation metrics"""
        try:
            response = await self.http_client.get(
                f"{self.config['prsm_api_url']}/api/v1/metrics/federation"
            )
            if response.status_code == 200:
                data = response.json()
                
                # P2P connections
                for conn_type, count in data.get('active_connections', {}).items():
                    self.p2p_connections.labels(
                        connection_type=conn_type,
                        region='unknown'
                    ).set(count)
                
                # Consensus latencies
                for latency in data.get('consensus_latencies', []):
                    self.consensus_latency.observe(latency)
                
                # Model registry sync times
                for sync_time in data.get('registry_sync_times', []):
                    self.model_registry_sync_duration.observe(sync_time)
                        
        except Exception as e:
            logger.error(f"Error collecting federation metrics: {e}")
    
    async def _collect_safety_metrics(self):
        """Collect safety and security metrics"""
        try:
            response = await self.http_client.get(
                f"{self.config['prsm_api_url']}/api/v1/metrics/safety"
            )
            if response.status_code == 200:
                data = response.json()
                
                # Circuit breaker activations
                for threat_level, components in data.get('circuit_breaker_activations', {}).items():
                    for component, count in components.items():
                        self.circuit_breaker_activations.labels(
                            threat_level=threat_level,
                            component=component
                        ).inc(count)
                
                # Safety validation times
                for validation_time in data.get('validation_times', []):
                    self.safety_validation_duration.observe(validation_time)
                
                # Security threat scores
                for threat_type, score in data.get('threat_scores', {}).items():
                    self.security_threat_score.labels(
                        threat_type=threat_type
                    ).set(score)
                        
        except Exception as e:
            logger.error(f"Error collecting safety metrics: {e}")
    
    async def close(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        await self.http_client.aclose()

async def main():
    """Main entry point"""
    config_path = os.getenv('CONFIG_PATH', 'config.yml')
    exporter = PRSMMetricsExporter(config_path)
    
    try:
        await exporter.start()
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await exporter.close()

if __name__ == "__main__":
    asyncio.run(main())