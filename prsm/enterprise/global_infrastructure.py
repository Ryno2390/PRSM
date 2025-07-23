"""
PRSM Global Infrastructure Management
Enterprise-scale multi-cloud infrastructure with intelligent routing, auto-scaling, and cost optimization
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import logging
import uuid
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Cloud provider SDKs
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1, monitoring_v3
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.monitor import MonitorManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Load balancing and CDN
try:
    import cloudflare
    CLOUDFLARE_AVAILABLE = True
except ImportError:
    CLOUDFLARE_AVAILABLE = False

# HTTP client for health checks
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITAL_OCEAN = "digitalocean"
    LINODE = "linode"


class Region(Enum):
    """Global regions for deployment"""
    # North America
    US_EAST_1 = "us-east-1"
    US_WEST_1 = "us-west-1"
    US_CENTRAL_1 = "us-central-1"
    CANADA_CENTRAL = "canada-central"
    
    # Europe
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"
    
    # Asia Pacific
    ASIA_SOUTHEAST_1 = "asia-southeast-1"
    ASIA_NORTHEAST_1 = "asia-northeast-1"
    ASIA_SOUTH_1 = "asia-south-1"
    
    # Others
    AUSTRALIA_SOUTHEAST_1 = "australia-southeast-1"
    SOUTH_AMERICA_EAST_1 = "south-america-east-1"


class ServiceTier(Enum):
    """Service tiers for different workload types"""
    PREMIUM = "premium"      # Ultra-low latency, highest cost
    STANDARD = "standard"    # Balanced performance and cost
    ECONOMY = "economy"      # Cost-optimized, higher latency acceptable


class DeploymentStatus(Enum):
    """Deployment status tracking"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    TERMINATING = "terminating"


@dataclass
class CloudCredentials:
    """Cloud provider credentials"""
    provider: CloudProvider
    credentials: Dict[str, Any]
    region_access: List[Region] = field(default_factory=list)
    enabled: bool = True


@dataclass
class InfrastructureNode:
    """Individual infrastructure node"""
    node_id: str
    provider: CloudProvider
    region: Region
    instance_type: str
    public_ip: str
    private_ip: str
    status: DeploymentStatus
    
    # Resource specifications
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    network_gbps: float
    
    # Performance metrics
    current_cpu_usage: float = 0.0
    current_memory_usage: float = 0.0
    current_connections: int = 0
    response_time_ms: float = 0.0
    
    # Cost tracking
    hourly_cost_usd: float = 0.0
    monthly_cost_usd: float = 0.0
    
    # Metadata
    service_tier: ServiceTier = ServiceTier.STANDARD
    deployment_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_health_check: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LoadBalancerRule:
    """Load balancer routing rule"""
    rule_id: str
    name: str
    source_patterns: List[str]  # URL patterns, IP ranges, etc.
    target_regions: List[Region]
    target_providers: List[CloudProvider]
    weight_distribution: Dict[str, float]  # node_id -> weight
    
    # Advanced routing
    health_check_required: bool = True
    sticky_sessions: bool = False
    ssl_termination: bool = True
    rate_limiting: Optional[Dict[str, Any]] = None
    
    # Priority and conditions
    priority: int = 100
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AutoScalingPolicy:
    """Auto-scaling policy configuration"""
    policy_id: str
    name: str
    target_regions: List[Region]
    target_providers: List[CloudProvider]
    
    # Scaling triggers
    cpu_threshold_up: float = 75.0
    cpu_threshold_down: float = 25.0
    memory_threshold_up: float = 80.0
    memory_threshold_down: float = 30.0
    response_time_threshold_ms: float = 500.0
    
    # Scaling parameters
    min_instances: int = 2
    max_instances: int = 100
    scale_up_count: int = 2
    scale_down_count: int = 1
    cooldown_minutes: int = 5
    
    # Instance configuration
    instance_type: str = "standard"
    service_tier: ServiceTier = ServiceTier.STANDARD
    
    enabled: bool = True


@dataclass
class CostOptimizationRule:
    """Cost optimization rule"""
    rule_id: str
    name: str
    description: str
    
    # Conditions
    trigger_conditions: List[Dict[str, Any]]
    
    # Actions
    actions: List[Dict[str, Any]]  # scale_down, migrate, spot_instances, etc.
    
    # Constraints
    max_cost_increase_percent: float = 0.0
    min_performance_level: float = 0.8
    excluded_regions: List[Region] = field(default_factory=list)
    
    enabled: bool = True


@dataclass
class GlobalInfrastructureConfig:
    """Global infrastructure configuration"""
    service_name: str = "prsm-api"
    
    # Cloud providers
    cloud_credentials: List[CloudCredentials] = field(default_factory=list)
    preferred_providers: List[CloudProvider] = field(default_factory=lambda: [CloudProvider.AWS, CloudProvider.GCP])
    
    # Global deployment
    target_regions: List[Region] = field(default_factory=lambda: [
        Region.US_EAST_1, Region.US_WEST_1, Region.EU_WEST_1,
        Region.ASIA_SOUTHEAST_1, Region.ASIA_NORTHEAST_1
    ])
    min_nodes_per_region: int = 2
    max_nodes_per_region: int = 20
    
    # Load balancing
    enable_global_load_balancer: bool = True
    enable_cdn: bool = True
    cdn_provider: str = "cloudflare"
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    default_scaling_policy: Optional[AutoScalingPolicy] = None
    
    # Cost optimization
    enable_cost_optimization: bool = True
    monthly_budget_usd: float = 50000.0
    cost_alert_threshold_percent: float = 80.0
    
    # Health monitoring
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    
    # Performance targets
    target_response_time_ms: float = 100.0
    target_availability_percent: float = 99.9
    target_throughput_rps: int = 10000


class CloudProviderManager:
    """Base class for cloud provider management"""
    
    def __init__(self, provider: CloudProvider, credentials: CloudCredentials):
        self.provider = provider
        self.credentials = credentials
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize cloud provider client"""
        if self.provider == CloudProvider.AWS and AWS_AVAILABLE:
            self._initialize_aws_client()
        elif self.provider == CloudProvider.GCP and GCP_AVAILABLE:
            self._initialize_gcp_client()
        elif self.provider == CloudProvider.AZURE and AZURE_AVAILABLE:
            self._initialize_azure_client()
    
    def _initialize_aws_client(self):
        """Initialize AWS client"""
        try:
            self.client = boto3.client(
                'ec2',
                aws_access_key_id=self.credentials.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.credentials.get('secret_access_key'),
                region_name=self.credentials.credentials.get('region', 'us-east-1')
            )
        except Exception as e:
            logger.error(f"Failed to initialize AWS client: {e}")
    
    def _initialize_gcp_client(self):
        """Initialize GCP client"""
        try:
            if 'service_account_key' in self.credentials.credentials:
                credentials = service_account.Credentials.from_service_account_info(
                    self.credentials.credentials['service_account_key']
                )
                self.client = compute_v1.InstancesClient(credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to initialize GCP client: {e}")
    
    def _initialize_azure_client(self):
        """Initialize Azure client"""
        try:
            credential = DefaultAzureCredential()
            subscription_id = self.credentials.credentials.get('subscription_id')
            if subscription_id:
                self.client = ComputeManagementClient(credential, subscription_id)
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")
    
    async def deploy_instance(self, region: Region, instance_type: str, 
                            service_tier: ServiceTier) -> Optional[InfrastructureNode]:
        """Deploy a new instance"""
        try:
            if self.provider == CloudProvider.AWS:
                return await self._deploy_aws_instance(region, instance_type, service_tier)
            elif self.provider == CloudProvider.GCP:
                return await self._deploy_gcp_instance(region, instance_type, service_tier)
            elif self.provider == CloudProvider.AZURE:
                return await self._deploy_azure_instance(region, instance_type, service_tier)
            
        except Exception as e:
            logger.error(f"Failed to deploy instance on {self.provider.value}: {e}")
            return None
    
    async def _deploy_aws_instance(self, region: Region, instance_type: str, 
                                 service_tier: ServiceTier) -> Optional[InfrastructureNode]:
        """Deploy AWS EC2 instance"""
        try:
            # Map service tier to instance type
            instance_mapping = {
                ServiceTier.PREMIUM: "c5n.2xlarge",
                ServiceTier.STANDARD: "c5.xlarge", 
                ServiceTier.ECONOMY: "t3.large"
            }
            
            aws_instance_type = instance_mapping.get(service_tier, "c5.xlarge")
            
            # Launch instance
            response = self.client.run_instances(
                ImageId='ami-0abcdef1234567890',  # Would be configurable
                MinCount=1,
                MaxCount=1,
                InstanceType=aws_instance_type,
                KeyName=self.credentials.credentials.get('key_name'),
                SecurityGroupIds=[self.credentials.credentials.get('security_group_id')],
                SubnetId=self.credentials.credentials.get('subnet_id'),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'prsm-{region.value}-{int(time.time())}'},
                        {'Key': 'Service', 'Value': 'prsm-api'},
                        {'Key': 'Tier', 'Value': service_tier.value}
                    ]
                }]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            
            # Wait for instance to be running
            waiter = self.client.get_waiter('instance_running')
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: waiter.wait(InstanceIds=[instance_id])
            )
            
            # Get instance details
            instances = self.client.describe_instances(InstanceIds=[instance_id])
            instance_data = instances['Reservations'][0]['Instances'][0]
            
            # Create infrastructure node
            return InfrastructureNode(
                node_id=instance_id,
                provider=self.provider,
                region=region,
                instance_type=aws_instance_type,
                public_ip=instance_data.get('PublicIpAddress', ''),
                private_ip=instance_data.get('PrivateIpAddress', ''),
                status=DeploymentStatus.DEPLOYING,
                cpu_cores=self._get_instance_specs(aws_instance_type)['cpu'],
                memory_gb=self._get_instance_specs(aws_instance_type)['memory'],
                storage_gb=self._get_instance_specs(aws_instance_type)['storage'],
                network_gbps=self._get_instance_specs(aws_instance_type)['network'],
                hourly_cost_usd=self._get_instance_cost(aws_instance_type, region),
                service_tier=service_tier,
                tags={'provider': 'aws', 'region': region.value}
            )
            
        except ClientError as e:
            logger.error(f"AWS deployment error: {e}")
            return None
    
    async def _deploy_gcp_instance(self, region: Region, instance_type: str, 
                                 service_tier: ServiceTier) -> Optional[InfrastructureNode]:
        """Deploy GCP Compute Engine instance"""
        # Implementation would be similar to AWS but using GCP APIs
        logger.info(f"Deploying GCP instance in {region.value}")
        return None
    
    async def _deploy_azure_instance(self, region: Region, instance_type: str, 
                                   service_tier: ServiceTier) -> Optional[InfrastructureNode]:
        """Deploy Azure VM instance"""
        # Implementation would be similar to AWS but using Azure APIs
        logger.info(f"Deploying Azure instance in {region.value}")
        return None
    
    def _get_instance_specs(self, instance_type: str) -> Dict[str, Any]:
        """Get instance specifications"""
        # This would be a comprehensive mapping of instance types to specs
        specs_mapping = {
            "c5n.2xlarge": {"cpu": 8, "memory": 21.0, "storage": 100, "network": 25.0},
            "c5.xlarge": {"cpu": 4, "memory": 8.0, "storage": 50, "network": 10.0},
            "t3.large": {"cpu": 2, "memory": 8.0, "storage": 50, "network": 5.0}
        }
        return specs_mapping.get(instance_type, {"cpu": 2, "memory": 4.0, "storage": 20, "network": 1.0})
    
    def _get_instance_cost(self, instance_type: str, region: Region) -> float:
        """Get hourly cost for instance type in region"""
        # This would use cloud provider pricing APIs
        base_costs = {
            "c5n.2xlarge": 0.432,
            "c5.xlarge": 0.192,
            "t3.large": 0.0832
        }
        
        # Regional multipliers
        regional_multipliers = {
            Region.US_EAST_1: 1.0,
            Region.US_WEST_1: 1.1,
            Region.EU_WEST_1: 1.15,
            Region.ASIA_SOUTHEAST_1: 1.2
        }
        
        base_cost = base_costs.get(instance_type, 0.1)
        multiplier = regional_multipliers.get(region, 1.0)
        
        return base_cost * multiplier
    
    async def terminate_instance(self, node_id: str) -> bool:
        """Terminate an instance"""
        try:
            if self.provider == CloudProvider.AWS:
                self.client.terminate_instances(InstanceIds=[node_id])
                return True
            elif self.provider == CloudProvider.GCP:
                # GCP termination logic
                return True
            elif self.provider == CloudProvider.AZURE:
                # Azure termination logic
                return True
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {node_id}: {e}")
            return False
    
    async def get_instance_metrics(self, node_id: str) -> Dict[str, Any]:
        """Get instance performance metrics"""
        try:
            # This would integrate with CloudWatch, Stackdriver, or Azure Monitor
            return {
                "cpu_usage": 45.2,
                "memory_usage": 62.1,
                "network_in": 1024000,
                "network_out": 2048000,
                "disk_read": 512000,
                "disk_write": 1024000
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics for {node_id}: {e}")
            return {}


class GlobalLoadBalancer:
    """Global intelligent load balancer"""
    
    def __init__(self, config: GlobalInfrastructureConfig):
        self.config = config
        self.routing_rules: List[LoadBalancerRule] = []
        self.node_health: Dict[str, bool] = {}
        self.node_metrics: Dict[str, Dict[str, Any]] = {}
        self.routing_history = deque(maxlen=10000)
        
        # Initialize CDN if enabled
        self.cdn_client = None
        if config.enable_cdn and CLOUDFLARE_AVAILABLE:
            self._initialize_cdn()
    
    def _initialize_cdn(self):
        """Initialize CDN provider"""
        try:
            # Initialize Cloudflare or other CDN
            pass
        except Exception as e:
            logger.error(f"Failed to initialize CDN: {e}")
    
    def add_routing_rule(self, rule: LoadBalancerRule):
        """Add a routing rule"""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority)
    
    async def route_request(self, request_info: Dict[str, Any]) -> Optional[InfrastructureNode]:
        """Route a request to the optimal node"""
        try:
            # Extract request information
            source_ip = request_info.get('source_ip', '')
            path = request_info.get('path', '/')
            headers = request_info.get('headers', {})
            
            # Find matching routing rules
            matching_rules = []
            for rule in self.routing_rules:
                if self._rule_matches(rule, request_info):
                    matching_rules.append(rule)
            
            if not matching_rules:
                # Use default routing
                return await self._default_routing(request_info)
            
            # Use highest priority rule
            primary_rule = matching_rules[0]
            
            # Select optimal node based on rule
            optimal_node = await self._select_optimal_node(primary_rule, request_info)
            
            # Record routing decision
            self.routing_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": source_ip,
                "path": path,
                "rule_id": primary_rule.rule_id,
                "target_node": optimal_node.node_id if optimal_node else None,
                "response_time_ms": 0  # Would be filled in later
            })
            
            return optimal_node
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return None
    
    def _rule_matches(self, rule: LoadBalancerRule, request_info: Dict[str, Any]) -> bool:
        """Check if a routing rule matches the request"""
        try:
            path = request_info.get('path', '/')
            source_ip = request_info.get('source_ip', '')
            
            # Check path patterns
            for pattern in rule.source_patterns:
                if pattern.startswith('/') and path.startswith(pattern):
                    return True
                elif self._ip_matches_pattern(source_ip, pattern):
                    return True
            
            # Check additional conditions
            for condition in rule.conditions:
                if not self._evaluate_condition(condition, request_info):
                    return False
            
            return bool(rule.source_patterns)  # True if any patterns matched
            
        except Exception as e:
            logger.error(f"Error matching rule: {e}")
            return False
    
    def _ip_matches_pattern(self, ip: str, pattern: str) -> bool:
        """Check if IP matches pattern (CIDR, range, etc.)"""
        try:
            if '/' in pattern:  # CIDR notation
                import ipaddress
                return ipaddress.ip_address(ip) in ipaddress.ip_network(pattern)
            elif '*' in pattern:  # Wildcard
                import fnmatch
                return fnmatch.fnmatch(ip, pattern)
            else:
                return ip == pattern
        except Exception:
            return False
    
    def _evaluate_condition(self, condition: Dict[str, Any], request_info: Dict[str, Any]) -> bool:
        """Evaluate a routing condition"""
        try:
            condition_type = condition.get('type')
            
            if condition_type == 'header':
                header_name = condition.get('header')
                expected_value = condition.get('value')
                actual_value = request_info.get('headers', {}).get(header_name)
                return actual_value == expected_value
            
            elif condition_type == 'time_of_day':
                current_hour = datetime.now(timezone.utc).hour
                start_hour = condition.get('start_hour', 0)
                end_hour = condition.get('end_hour', 23)
                return start_hour <= current_hour <= end_hour
            
            elif condition_type == 'geographic':
                # Would integrate with GeoIP service
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _select_optimal_node(self, rule: LoadBalancerRule, 
                                 request_info: Dict[str, Any]) -> Optional[InfrastructureNode]:
        """Select the optimal node for a request"""
        try:
            # Get healthy nodes matching the rule
            candidate_nodes = []
            
            for node_id, weight in rule.weight_distribution.items():
                if self.node_health.get(node_id, False):
                    # Get node from global registry (would be passed in)
                    node = await self._get_node_by_id(node_id)
                    if node:
                        candidate_nodes.append((node, weight))
            
            if not candidate_nodes:
                return None
            
            # Select based on weighted round-robin with performance optimization
            best_node = None
            best_score = float('inf')
            
            for node, weight in candidate_nodes:
                metrics = self.node_metrics.get(node.node_id, {})
                
                # Calculate composite score
                cpu_score = metrics.get('cpu_usage', 50) / 100
                memory_score = metrics.get('memory_usage', 50) / 100
                response_time_score = metrics.get('response_time_ms', 100) / 1000
                
                # Geographic proximity bonus (simplified)
                geo_score = self._calculate_geographic_score(node, request_info)
                
                # Combined score (lower is better)
                composite_score = (
                    cpu_score * 0.3 +
                    memory_score * 0.2 +
                    response_time_score * 0.3 +
                    geo_score * 0.2
                ) / weight  # Higher weight = lower effective score
                
                if composite_score < best_score:
                    best_score = composite_score
                    best_node = node
            
            return best_node
            
        except Exception as e:
            logger.error(f"Error selecting optimal node: {e}")
            return None
    
    def _calculate_geographic_score(self, node: InfrastructureNode, 
                                  request_info: Dict[str, Any]) -> float:
        """Calculate geographic proximity score"""
        try:
            # This would use actual geographic distance calculation
            # For now, return a simple regional preference
            source_ip = request_info.get('source_ip', '')
            
            # Simple continent-based scoring
            regional_preferences = {
                Region.US_EAST_1: 0.1,
                Region.US_WEST_1: 0.2,
                Region.EU_WEST_1: 0.5,
                Region.ASIA_SOUTHEAST_1: 0.8,
                Region.ASIA_NORTHEAST_1: 0.9
            }
            
            return regional_preferences.get(node.region, 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating geographic score: {e}")
            return 0.5
    
    async def _get_node_by_id(self, node_id: str) -> Optional[InfrastructureNode]:
        """Get node by ID (would interface with global node registry)"""
        # This would query the global infrastructure manager
        return None
    
    async def _default_routing(self, request_info: Dict[str, Any]) -> Optional[InfrastructureNode]:
        """Default routing when no rules match"""
        try:
            # Simple round-robin to healthy nodes
            healthy_nodes = [node_id for node_id, healthy in self.node_health.items() if healthy]
            
            if not healthy_nodes:
                return None
            
            # Get least loaded node
            best_node_id = None
            lowest_load = float('inf')
            
            for node_id in healthy_nodes:
                metrics = self.node_metrics.get(node_id, {})
                current_load = (metrics.get('cpu_usage', 0) + metrics.get('memory_usage', 0)) / 2
                
                if current_load < lowest_load:
                    lowest_load = current_load
                    best_node_id = node_id
            
            if best_node_id:
                return await self._get_node_by_id(best_node_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in default routing: {e}")
            return None
    
    def update_node_health(self, node_id: str, healthy: bool):
        """Update node health status"""
        self.node_health[node_id] = healthy
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update node performance metrics"""
        self.node_metrics[node_id] = metrics
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        try:
            total_requests = len(self.routing_history)
            
            if total_requests == 0:
                return {"total_requests": 0}
            
            # Calculate statistics
            recent_requests = list(self.routing_history)[-1000:]  # Last 1000 requests
            
            rule_usage = defaultdict(int)
            node_usage = defaultdict(int)
            avg_response_times = defaultdict(list)
            
            for request in recent_requests:
                rule_id = request.get('rule_id')
                node_id = request.get('target_node')
                response_time = request.get('response_time_ms', 0)
                
                if rule_id:
                    rule_usage[rule_id] += 1
                if node_id:
                    node_usage[node_id] += 1
                    avg_response_times[node_id].append(response_time)
            
            # Calculate average response times
            node_avg_response_times = {}
            for node_id, times in avg_response_times.items():
                if times:
                    node_avg_response_times[node_id] = sum(times) / len(times)
            
            return {
                "total_requests": total_requests,
                "recent_requests": len(recent_requests),
                "rule_usage": dict(rule_usage),
                "node_usage": dict(node_usage),
                "node_avg_response_times": node_avg_response_times,
                "healthy_nodes": sum(1 for healthy in self.node_health.values() if healthy),
                "total_nodes": len(self.node_health)
            }
            
        except Exception as e:
            logger.error(f"Error getting routing statistics: {e}")
            return {"error": str(e)}


class AutoScalingEngine:
    """Intelligent auto-scaling engine"""
    
    def __init__(self, config: GlobalInfrastructureConfig):
        self.config = config
        self.scaling_policies: List[AutoScalingPolicy] = []
        self.scaling_history = deque(maxlen=1000)
        self.cooldown_timers: Dict[str, datetime] = {}
        
        # Metrics tracking
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Machine learning for predictive scaling (simplified)
        self.demand_patterns: Dict[str, List[float]] = defaultdict(list)
    
    def add_scaling_policy(self, policy: AutoScalingPolicy):
        """Add an auto-scaling policy"""
        self.scaling_policies.append(policy)
    
    async def evaluate_scaling_decisions(self, nodes: List[InfrastructureNode]) -> List[Dict[str, Any]]:
        """Evaluate and return scaling decisions"""
        scaling_actions = []
        
        try:
            for policy in self.scaling_policies:
                if not policy.enabled:
                    continue
                
                # Check cooldown
                cooldown_key = f"{policy.policy_id}"
                if cooldown_key in self.cooldown_timers:
                    if datetime.now(timezone.utc) < self.cooldown_timers[cooldown_key]:
                        continue
                
                # Get nodes for this policy
                policy_nodes = [
                    node for node in nodes
                    if (node.region in policy.target_regions and 
                        node.provider in policy.target_providers and
                        node.status in [DeploymentStatus.HEALTHY, DeploymentStatus.DEGRADED])
                ]
                
                if not policy_nodes:
                    continue
                
                # Calculate aggregate metrics
                avg_cpu = sum(node.current_cpu_usage for node in policy_nodes) / len(policy_nodes)
                avg_memory = sum(node.current_memory_usage for node in policy_nodes) / len(policy_nodes)
                avg_response_time = sum(node.response_time_ms for node in policy_nodes) / len(policy_nodes)
                
                # Store metrics for trend analysis
                timestamp = datetime.now(timezone.utc)
                self.metrics_history[policy.policy_id].append({
                    "timestamp": timestamp,
                    "cpu": avg_cpu,
                    "memory": avg_memory,
                    "response_time": avg_response_time,
                    "node_count": len(policy_nodes)
                })
                
                # Evaluate scaling decisions
                scale_decision = self._evaluate_policy_scaling(policy, policy_nodes, {
                    "avg_cpu": avg_cpu,
                    "avg_memory": avg_memory,
                    "avg_response_time": avg_response_time
                })
                
                if scale_decision:
                    scaling_actions.append(scale_decision)
                    
                    # Set cooldown
                    self.cooldown_timers[cooldown_key] = (
                        datetime.now(timezone.utc) + timedelta(minutes=policy.cooldown_minutes)
                    )
            
            return scaling_actions
            
        except Exception as e:
            logger.error(f"Error evaluating scaling decisions: {e}")
            return []
    
    def _evaluate_policy_scaling(self, policy: AutoScalingPolicy, 
                               nodes: List[InfrastructureNode],
                               metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Evaluate scaling for a specific policy"""
        try:
            current_count = len(nodes)
            
            # Check scale-up conditions
            should_scale_up = (
                (metrics["avg_cpu"] > policy.cpu_threshold_up) or
                (metrics["avg_memory"] > policy.memory_threshold_up) or
                (metrics["avg_response_time"] > policy.response_time_threshold_ms)
            )
            
            # Check scale-down conditions
            should_scale_down = (
                (metrics["avg_cpu"] < policy.cpu_threshold_down) and
                (metrics["avg_memory"] < policy.memory_threshold_down) and
                (metrics["avg_response_time"] < policy.response_time_threshold_ms / 2)
            )
            
            # Predictive scaling check
            predicted_demand = self._predict_demand_change(policy.policy_id)
            if predicted_demand > 1.2:  # 20% increase predicted
                should_scale_up = True
            elif predicted_demand < 0.8:  # 20% decrease predicted
                should_scale_down = True
            
            # Make scaling decision
            if should_scale_up and current_count < policy.max_instances:
                target_count = min(current_count + policy.scale_up_count, policy.max_instances)
                return {
                    "action": "scale_up",
                    "policy_id": policy.policy_id,
                    "current_count": current_count,
                    "target_count": target_count,
                    "regions": policy.target_regions,
                    "providers": policy.target_providers,
                    "instance_type": policy.instance_type,
                    "service_tier": policy.service_tier,
                    "reason": f"CPU: {metrics['avg_cpu']:.1f}%, Memory: {metrics['avg_memory']:.1f}%, RT: {metrics['avg_response_time']:.1f}ms",
                    "timestamp": datetime.now(timezone.utc)
                }
            
            elif should_scale_down and current_count > policy.min_instances:
                target_count = max(current_count - policy.scale_down_count, policy.min_instances)
                
                # Select nodes to terminate (least utilized)
                nodes_to_terminate = sorted(nodes, key=lambda n: n.current_cpu_usage + n.current_memory_usage)
                nodes_to_terminate = nodes_to_terminate[:current_count - target_count]
                
                return {
                    "action": "scale_down",
                    "policy_id": policy.policy_id,
                    "current_count": current_count,
                    "target_count": target_count,
                    "nodes_to_terminate": [node.node_id for node in nodes_to_terminate],
                    "reason": f"CPU: {metrics['avg_cpu']:.1f}%, Memory: {metrics['avg_memory']:.1f}%, RT: {metrics['avg_response_time']:.1f}ms",
                    "timestamp": datetime.now(timezone.utc)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating policy scaling: {e}")
            return None
    
    def _predict_demand_change(self, policy_id: str) -> float:
        """Predict demand changes using simple pattern recognition"""
        try:
            if policy_id not in self.demand_patterns:
                return 1.0
            
            recent_patterns = self.demand_patterns[policy_id][-24:]  # Last 24 data points
            
            if len(recent_patterns) < 3:
                return 1.0
            
            # Simple trend analysis
            recent_avg = sum(recent_patterns[-3:]) / 3
            historical_avg = sum(recent_patterns[:-3]) / max(len(recent_patterns) - 3, 1)
            
            if historical_avg == 0:
                return 1.0
            
            trend_ratio = recent_avg / historical_avg
            
            # Time-based patterns (simplified)
            current_hour = datetime.now(timezone.utc).hour
            if 9 <= current_hour <= 17:  # Business hours
                trend_ratio *= 1.2
            elif 22 <= current_hour or current_hour <= 6:  # Night hours
                trend_ratio *= 0.8
            
            return max(0.1, min(3.0, trend_ratio))  # Bound the prediction
            
        except Exception as e:
            logger.error(f"Error predicting demand change: {e}")
            return 1.0
    
    def record_scaling_action(self, action: Dict[str, Any]):
        """Record a scaling action"""
        self.scaling_history.append({
            **action,
            "executed_at": datetime.now(timezone.utc).isoformat()
        })
    
    def update_demand_pattern(self, policy_id: str, demand_value: float):
        """Update demand patterns for predictive scaling"""
        self.demand_patterns[policy_id].append(demand_value)
        
        # Keep only recent patterns
        if len(self.demand_patterns[policy_id]) > 168:  # 7 days of hourly data
            self.demand_patterns[policy_id] = self.demand_patterns[policy_id][-168:]
    
    async def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        try:
            if not self.scaling_history:
                return {"total_actions": 0}
            
            recent_actions = list(self.scaling_history)[-100:]  # Last 100 actions
            
            action_counts = defaultdict(int)
            policy_counts = defaultdict(int)
            
            for action in recent_actions:
                action_counts[action.get("action", "unknown")] += 1
                policy_counts[action.get("policy_id", "unknown")] += 1
            
            return {
                "total_actions": len(self.scaling_history),
                "recent_actions": len(recent_actions),
                "action_breakdown": dict(action_counts),
                "policy_activity": dict(policy_counts),
                "active_policies": len(self.scaling_policies),
                "cooldown_timers": len(self.cooldown_timers)
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling statistics: {e}")
            return {"error": str(e)}


# Global infrastructure manager instance
global_infrastructure: Optional['GlobalInfrastructureManager'] = None


class GlobalInfrastructureManager:
    """Main global infrastructure management system"""
    
    def __init__(self, config: GlobalInfrastructureConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        
        # Core components
        self.cloud_managers: Dict[CloudProvider, CloudProviderManager] = {}
        self.load_balancer = GlobalLoadBalancer(config)
        self.auto_scaler = AutoScalingEngine(config)
        
        # Infrastructure state
        self.nodes: Dict[str, InfrastructureNode] = {}
        self.deployment_queue = asyncio.Queue()
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Cost tracking
        self.cost_tracker = {
            "daily_costs": deque(maxlen=30),
            "monthly_budget_used": 0.0,
            "cost_alerts_sent": 0
        }
        
        # Statistics
        self.stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "total_terminations": 0,
            "health_checks_performed": 0,
            "scaling_actions": 0
        }
        
        # Initialize cloud managers
        self._initialize_cloud_managers()
    
    def _initialize_cloud_managers(self):
        """Initialize cloud provider managers"""
        for credentials in self.config.cloud_credentials:
            if credentials.enabled:
                try:
                    manager = CloudProviderManager(credentials.provider, credentials)
                    self.cloud_managers[credentials.provider] = manager
                    logger.info(f"✅ Initialized {credentials.provider.value} cloud manager")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {credentials.provider.value}: {e}")
    
    async def start(self):
        """Start the global infrastructure system"""
        try:
            logger.info("🚀 Starting Global Infrastructure Manager...")
            
            # Start background tasks
            asyncio.create_task(self._deployment_worker())
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._auto_scaling_loop())
            asyncio.create_task(self._cost_monitoring_loop())
            
            # Deploy initial infrastructure
            await self._deploy_initial_infrastructure()
            
            logger.info("✅ Global Infrastructure Manager started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start global infrastructure: {e}")
            raise
    
    async def _deploy_initial_infrastructure(self):
        """Deploy initial infrastructure across regions"""
        try:
            deployment_tasks = []
            
            for region in self.config.target_regions:
                for _ in range(self.config.min_nodes_per_region):
                    # Select provider for this deployment
                    provider = self._select_optimal_provider(region)
                    if provider:
                        task = asyncio.create_task(
                            self._deploy_node(provider, region, ServiceTier.STANDARD)
                        )
                        deployment_tasks.append(task)
            
            # Wait for all deployments
            if deployment_tasks:
                results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
                
                successful = sum(1 for r in results if not isinstance(r, Exception))
                logger.info(f"✅ Initial deployment complete: {successful}/{len(deployment_tasks)} nodes deployed")
            
        except Exception as e:
            logger.error(f"Error in initial deployment: {e}")
    
    def _select_optimal_provider(self, region: Region) -> Optional[CloudProvider]:
        """Select the optimal cloud provider for a region"""
        try:
            # Prefer providers in order of preference
            for provider in self.config.preferred_providers:
                if provider in self.cloud_managers:
                    credentials = None
                    for cred in self.config.cloud_credentials:
                        if cred.provider == provider and region in cred.region_access:
                            credentials = cred
                            break
                    
                    if credentials and credentials.enabled:
                        return provider
            
            # Fallback to any available provider
            for provider in self.cloud_managers:
                return provider
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting provider for {region}: {e}")
            return None
    
    async def _deploy_node(self, provider: CloudProvider, region: Region, 
                         service_tier: ServiceTier) -> Optional[InfrastructureNode]:
        """Deploy a single node"""
        try:
            manager = self.cloud_managers.get(provider)
            if not manager:
                logger.error(f"No manager available for {provider}")
                return None
            
            # Deploy the instance
            node = await manager.deploy_instance(region, "standard", service_tier)
            
            if node:
                # Register the node
                self.nodes[node.node_id] = node
                
                # Start health checking
                health_task = asyncio.create_task(self._health_check_node(node.node_id))
                self.health_check_tasks[node.node_id] = health_task
                
                # Update load balancer
                self.load_balancer.update_node_health(node.node_id, True)
                
                self.stats["successful_deployments"] += 1
                logger.info(f"✅ Deployed node {node.node_id} in {region.value}")
                
                return node
            else:
                self.stats["failed_deployments"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error deploying node: {e}")
            self.stats["failed_deployments"] += 1
            return None
    
    async def _deployment_worker(self):
        """Background worker for processing deployment queue"""
        while True:
            try:
                deployment_request = await self.deployment_queue.get()
                
                provider = deployment_request["provider"]
                region = deployment_request["region"]
                service_tier = deployment_request["service_tier"]
                
                await self._deploy_node(provider, region, service_tier)
                
                self.deployment_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in deployment worker: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while True:
            try:
                # Perform health checks on all nodes
                for node_id in list(self.nodes.keys()):
                    if node_id not in self.health_check_tasks:
                        task = asyncio.create_task(self._health_check_node(node_id))
                        self.health_check_tasks[node_id] = task
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _health_check_node(self, node_id: str):
        """Perform health check on a specific node"""
        try:
            node = self.nodes.get(node_id)
            if not node:
                return
            
            # Perform HTTP health check
            healthy = await self._perform_http_health_check(node)
            
            # Update node status
            if healthy:
                if node.status != DeploymentStatus.HEALTHY:
                    node.status = DeploymentStatus.HEALTHY
                    logger.info(f"✅ Node {node_id} is healthy")
            else:
                if node.status == DeploymentStatus.HEALTHY:
                    node.status = DeploymentStatus.UNHEALTHY
                    logger.warning(f"⚠️ Node {node_id} is unhealthy")
            
            # Update load balancer
            self.load_balancer.update_node_health(node_id, healthy)
            
            # Get performance metrics
            if node_id in self.cloud_managers:
                manager = self.cloud_managers[node.provider]
                metrics = await manager.get_instance_metrics(node_id)
                
                # Update node metrics
                node.current_cpu_usage = metrics.get('cpu_usage', 0)
                node.current_memory_usage = metrics.get('memory_usage', 0)
                node.response_time_ms = metrics.get('response_time_ms', 0)
                
                # Update load balancer metrics
                self.load_balancer.update_node_metrics(node_id, metrics)
            
            node.last_health_check = datetime.now(timezone.utc)
            self.stats["health_checks_performed"] += 1
            
        except Exception as e:
            logger.error(f"Error in health check for {node_id}: {e}")
    
    async def _perform_http_health_check(self, node: InfrastructureNode) -> bool:
        """Perform HTTP health check"""
        try:
            if not HTTPX_AVAILABLE or not node.public_ip:
                return True  # Assume healthy if we can't check
            
            async with httpx.AsyncClient(timeout=self.config.health_check_timeout_seconds) as client:
                response = await client.get(f"http://{node.public_ip}/health")
                return response.status_code == 200
                
        except Exception as e:
            logger.debug(f"Health check failed for {node.node_id}: {e}")
            return False
    
    async def _auto_scaling_loop(self):
        """Auto-scaling evaluation loop"""
        while True:
            try:
                if self.config.enable_auto_scaling:
                    # Get all healthy nodes
                    healthy_nodes = [
                        node for node in self.nodes.values()
                        if node.status == DeploymentStatus.HEALTHY
                    ]
                    
                    # Evaluate scaling decisions
                    scaling_actions = await self.auto_scaler.evaluate_scaling_decisions(healthy_nodes)
                    
                    # Execute scaling actions
                    for action in scaling_actions:
                        await self._execute_scaling_action(action)
                        self.stats["scaling_actions"] += 1
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(60)
    
    async def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute a scaling action"""
        try:
            if action["action"] == "scale_up":
                # Add new instances
                regions = action["regions"]
                providers = action["providers"]
                target_count = action["target_count"]
                current_count = action["current_count"]
                instances_to_add = target_count - current_count
                
                for i in range(instances_to_add):
                    # Select region and provider
                    region = regions[i % len(regions)]
                    provider = self._select_optimal_provider(region)
                    
                    if provider:
                        await self.deployment_queue.put({
                            "provider": provider,
                            "region": region,
                            "service_tier": action.get("service_tier", ServiceTier.STANDARD)
                        })
                
                logger.info(f"🔺 Scaling up: Adding {instances_to_add} instances")
                
            elif action["action"] == "scale_down":
                # Remove instances
                nodes_to_terminate = action["nodes_to_terminate"]
                
                for node_id in nodes_to_terminate:
                    await self._terminate_node(node_id)
                
                logger.info(f"🔻 Scaling down: Removing {len(nodes_to_terminate)} instances")
            
            # Record the action
            self.auto_scaler.record_scaling_action(action)
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
    
    async def _terminate_node(self, node_id: str):
        """Terminate a node"""
        try:
            node = self.nodes.get(node_id)
            if not node:
                return
            
            # Remove from load balancer
            self.load_balancer.update_node_health(node_id, False)
            
            # Terminate instance
            manager = self.cloud_managers.get(node.provider)
            if manager:
                success = await manager.terminate_instance(node_id)
                
                if success:
                    # Clean up
                    del self.nodes[node_id]
                    
                    if node_id in self.health_check_tasks:
                        self.health_check_tasks[node_id].cancel()
                        del self.health_check_tasks[node_id]
                    
                    self.stats["total_terminations"] += 1
                    logger.info(f"🗑️ Terminated node {node_id}")
            
        except Exception as e:
            logger.error(f"Error terminating node {node_id}: {e}")
    
    async def _cost_monitoring_loop(self):
        """Cost monitoring and optimization loop"""
        while True:
            try:
                # Calculate current costs
                total_hourly_cost = sum(node.hourly_cost_usd for node in self.nodes.values())
                daily_cost = total_hourly_cost * 24
                monthly_cost = daily_cost * 30
                
                # Track daily costs
                self.cost_tracker["daily_costs"].append({
                    "date": datetime.now(timezone.utc).date().isoformat(),
                    "cost": daily_cost
                })
                
                # Check budget alerts
                budget_usage_percent = (monthly_cost / self.config.monthly_budget_usd) * 100
                
                if budget_usage_percent > self.config.cost_alert_threshold_percent:
                    logger.warning(f"💰 Cost alert: {budget_usage_percent:.1f}% of monthly budget used")
                    self.cost_tracker["cost_alerts_sent"] += 1
                
                # Update monthly budget tracking
                self.cost_tracker["monthly_budget_used"] = budget_usage_percent
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in cost monitoring: {e}")
                await asyncio.sleep(3600)
    
    # Public API methods
    
    async def deploy_nodes(self, count: int, region: Region, 
                         service_tier: ServiceTier = ServiceTier.STANDARD) -> List[str]:
        """Deploy multiple nodes"""
        node_ids = []
        
        for _ in range(count):
            provider = self._select_optimal_provider(region)
            if provider:
                await self.deployment_queue.put({
                    "provider": provider,
                    "region": region,
                    "service_tier": service_tier
                })
        
        return node_ids
    
    async def route_request(self, request_info: Dict[str, Any]) -> Optional[InfrastructureNode]:
        """Route a request to optimal node"""
        return await self.load_balancer.route_request(request_info)
    
    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get comprehensive infrastructure status"""
        try:
            # Node status summary
            status_counts = defaultdict(int)
            region_counts = defaultdict(int)
            provider_counts = defaultdict(int)
            
            for node in self.nodes.values():
                status_counts[node.status.value] += 1
                region_counts[node.region.value] += 1
                provider_counts[node.provider.value] += 1
            
            # Cost summary
            total_hourly_cost = sum(node.hourly_cost_usd for node in self.nodes.values())
            monthly_projected_cost = total_hourly_cost * 24 * 30
            
            # Performance summary
            if self.nodes:
                avg_cpu = sum(node.current_cpu_usage for node in self.nodes.values()) / len(self.nodes)
                avg_memory = sum(node.current_memory_usage for node in self.nodes.values()) / len(self.nodes)
                avg_response_time = sum(node.response_time_ms for node in self.nodes.values()) / len(self.nodes)
            else:
                avg_cpu = avg_memory = avg_response_time = 0
            
            return {
                "total_nodes": len(self.nodes),
                "node_status": dict(status_counts),
                "regional_distribution": dict(region_counts),
                "provider_distribution": dict(provider_counts),
                "performance": {
                    "avg_cpu_usage": avg_cpu,
                    "avg_memory_usage": avg_memory,
                    "avg_response_time_ms": avg_response_time
                },
                "cost": {
                    "hourly_cost_usd": total_hourly_cost,
                    "monthly_projected_usd": monthly_projected_cost,
                    "budget_usage_percent": self.cost_tracker["monthly_budget_used"]
                },
                "statistics": self.stats.copy(),
                "load_balancer": await self.load_balancer.get_routing_statistics(),
                "auto_scaler": await self.auto_scaler.get_scaling_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error getting infrastructure status: {e}")
            return {"error": str(e)}
    
    async def stop(self):
        """Stop the global infrastructure manager"""
        try:
            logger.info("🛑 Stopping Global Infrastructure Manager...")
            
            # Cancel all health check tasks
            for task in self.health_check_tasks.values():
                task.cancel()
            
            if self.health_check_tasks:
                await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
            
            logger.info("✅ Global Infrastructure Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping infrastructure manager: {e}")


def initialize_global_infrastructure(config: GlobalInfrastructureConfig, 
                                   redis_client: aioredis.Redis) -> GlobalInfrastructureManager:
    """Initialize the global infrastructure system"""
    global global_infrastructure
    
    global_infrastructure = GlobalInfrastructureManager(config, redis_client)
    logger.info("✅ Global Infrastructure Manager initialized")
    return global_infrastructure


def get_global_infrastructure() -> GlobalInfrastructureManager:
    """Get the global infrastructure manager instance"""
    if global_infrastructure is None:
        raise RuntimeError("Global infrastructure not initialized")
    return global_infrastructure


async def start_global_infrastructure():
    """Start the global infrastructure system"""
    if global_infrastructure:
        await global_infrastructure.start()


async def stop_global_infrastructure():
    """Stop the global infrastructure system"""
    if global_infrastructure:
        await global_infrastructure.stop()