#!/usr/bin/env python3
"""
Agent-Based Economic Model for PRSM Tokenomics
Phase 2 implementation using Mesa + NetworkX for network effects simulation

🎯 PURPOSE:
Comprehensive simulation of PRSM's token economy with 10K agents across 4 stakeholder types
to validate price discovery, incentive alignment, and network sustainability under various
economic scenarios and market conditions.

🔧 SIMULATION SCOPE:
- 10,000 agents across 4 stakeholder types
- Dynamic price discovery mechanisms
- Network effects modeling with NetworkX
- Quality-based reputation systems
- Bootstrap incentive mechanisms
- Economic sustainability validation

🚀 STAKEHOLDER TYPES:
- ContentCreators: Contribute models/data, earn royalties
- QueryUsers: Consume services, pay FTNS tokens
- NodeOperators: Provide compute, earn processing fees
- TokenHolders: Stake for returns, participate in governance

🎪 VALIDATION TARGETS:
- Price stability under load (±20% variance)
- Incentive alignment (>80% participant satisfaction)
- Network sustainability (positive cash flow)
- Bootstrap effectiveness (critical mass achievement)
"""

import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
import structlog
import asyncio
import json
from pathlib import Path

# Set high precision for financial calculations
getcontext().prec = 28

try:
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import NetworkGrid
    from mesa.datacollection import DataCollector
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False
    print("Warning: Mesa not available. Install with: pip install mesa")

logger = structlog.get_logger(__name__)

class StakeholderType(Enum):
    """Types of stakeholders in the PRSM economy"""
    CONTENT_CREATOR = "content_creator"
    QUERY_USER = "query_user"
    NODE_OPERATOR = "node_operator"
    TOKEN_HOLDER = "token_holder"

class EconomicAction(Enum):
    """Economic actions agents can take"""
    CREATE_CONTENT = "create_content"
    CONSUME_CONTENT = "consume_content"
    PROVIDE_COMPUTE = "provide_compute"
    STAKE_TOKENS = "stake_tokens"
    TRADE_TOKENS = "trade_tokens"
    VALIDATE_QUALITY = "validate_quality"

class MarketCondition(Enum):
    """Market conditions for simulation"""
    BEAR = "bear"
    BULL = "bull"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class EconomicMetrics:
    """Economic metrics for analysis"""
    timestamp: datetime
    total_supply: Decimal
    circulating_supply: Decimal
    average_price: Decimal
    transaction_volume: Decimal
    network_value: Decimal
    gini_coefficient: float
    stakeholder_satisfaction: Dict[str, float]
    network_growth_rate: float

@dataclass
class AgentProfile:
    """Profile configuration for different agent types"""
    agent_type: StakeholderType
    initial_balance: Decimal
    risk_tolerance: float  # 0.0 to 1.0
    activity_frequency: float  # actions per time step
    quality_threshold: float  # minimum quality acceptance
    network_connectivity: int  # number of connections
    economic_strategy: str  # strategy description

class PRSMEconomicAgent(Agent):
    """
    Base agent class for PRSM economic simulation
    
    Represents individual participants in the PRSM ecosystem with
    economic behavior patterns, decision-making capabilities, and
    network interaction dynamics.
    """
    
    def __init__(self, unique_id: int, model, agent_profile: AgentProfile):
        super().__init__(unique_id, model)
        self.profile = agent_profile
        
        # Economic state
        self.ftns_balance = agent_profile.initial_balance
        self.reputation_score = 50.0  # Start with neutral reputation
        self.transaction_history: List[Dict[str, Any]] = []
        
        # Behavioral parameters
        self.risk_tolerance = agent_profile.risk_tolerance
        self.activity_frequency = agent_profile.activity_frequency
        self.quality_threshold = agent_profile.quality_threshold
        
        # Network state
        self.connections: List[int] = []
        self.trust_scores: Dict[int, float] = {}
        
        # Performance tracking
        self.total_earnings = Decimal('0')
        self.total_spending = Decimal('0')
        self.satisfaction_score = 0.5  # 0.0 to 1.0
        
        # Strategy state
        self.current_strategy = agent_profile.economic_strategy
        self.strategy_performance = 0.0
        
        logger.debug(f"Created {agent_profile.agent_type.value} agent {unique_id}")
    
    def step(self):
        """Execute one step of agent behavior"""
        
        # Decide whether to act this step based on activity frequency
        if random.random() < self.activity_frequency:
            self._execute_economic_action()
        
        # Update reputation and satisfaction periodically
        if self.model.schedule.steps % 10 == 0:
            self._update_reputation()
            self._update_satisfaction()
    
    def _execute_economic_action(self):
        """Execute economic action based on agent type and strategy"""
        
        if self.profile.agent_type == StakeholderType.CONTENT_CREATOR:
            self._content_creator_action()
        elif self.profile.agent_type == StakeholderType.QUERY_USER:
            self._query_user_action()
        elif self.profile.agent_type == StakeholderType.NODE_OPERATOR:
            self._node_operator_action()
        elif self.profile.agent_type == StakeholderType.TOKEN_HOLDER:
            self._token_holder_action()
    
    def _content_creator_action(self):
        """Content creator specific actions"""
        
        # Decide whether to create new content
        if random.random() < 0.3:  # 30% chance to create content
            content_quality = random.uniform(0.3, 1.0)
            creation_cost = Decimal(str(random.uniform(1.0, 5.0)))
            
            if self.ftns_balance >= creation_cost:
                # Create content
                self.ftns_balance -= creation_cost
                self.total_spending += creation_cost
                
                # Register content creation with model
                revenue = self.model.register_content_creation(
                    self.unique_id, content_quality, creation_cost
                )
                
                self.ftns_balance += revenue
                self.total_earnings += revenue
                
                self._record_transaction(EconomicAction.CREATE_CONTENT, -creation_cost + revenue)
        
        # Participate in quality validation (earn small fees)
        if random.random() < 0.2:  # 20% chance to validate
            validation_fee = Decimal(str(random.uniform(0.1, 0.5)))
            self.ftns_balance += validation_fee
            self.total_earnings += validation_fee
            
            self._record_transaction(EconomicAction.VALIDATE_QUALITY, validation_fee)
    
    def _query_user_action(self):
        """Query user specific actions"""
        
        # Decide whether to make a query
        if random.random() < 0.4:  # 40% chance to query
            query_cost = Decimal(str(random.uniform(0.5, 3.0)))
            
            if self.ftns_balance >= query_cost:
                # Make query
                self.ftns_balance -= query_cost
                self.total_spending += query_cost
                
                # Register query with model
                query_quality = self.model.process_query(self.unique_id, query_cost)
                
                # Update satisfaction based on query quality
                if query_quality >= self.quality_threshold:
                    self.satisfaction_score += 0.01
                else:
                    self.satisfaction_score -= 0.02
                
                self.satisfaction_score = max(0.0, min(1.0, self.satisfaction_score))
                
                self._record_transaction(EconomicAction.CONSUME_CONTENT, -query_cost)
    
    def _node_operator_action(self):
        """Node operator specific actions"""
        
        # Provide compute resources (always active)
        compute_reward = Decimal(str(random.uniform(0.8, 2.5)))
        operational_cost = Decimal(str(random.uniform(0.3, 1.0)))
        
        net_reward = compute_reward - operational_cost
        
        self.ftns_balance += net_reward
        if net_reward > 0:
            self.total_earnings += net_reward
        else:
            self.total_spending += abs(net_reward)
        
        # Register compute provision with model
        self.model.register_compute_provision(self.unique_id, compute_reward)
        
        self._record_transaction(EconomicAction.PROVIDE_COMPUTE, net_reward)
        
        # Occasionally invest in infrastructure
        if random.random() < 0.1:  # 10% chance to invest
            investment_cost = Decimal(str(random.uniform(5.0, 20.0)))
            
            if self.ftns_balance >= investment_cost:
                self.ftns_balance -= investment_cost
                self.total_spending += investment_cost
                
                # Investment improves future rewards (simplified)
                self.activity_frequency *= 1.02
    
    def _token_holder_action(self):
        """Token holder specific actions"""
        
        # Staking decision
        if random.random() < 0.3:  # 30% chance to stake
            stake_amount = self.ftns_balance * Decimal(str(random.uniform(0.1, 0.3)))
            
            if stake_amount >= Decimal('1.0'):
                staking_reward = self.model.stake_tokens(self.unique_id, stake_amount)
                
                self.ftns_balance += staking_reward
                self.total_earnings += staking_reward
                
                self._record_transaction(EconomicAction.STAKE_TOKENS, staking_reward)
        
        # Trading decision
        if random.random() < 0.2:  # 20% chance to trade
            trade_amount = self.ftns_balance * Decimal(str(random.uniform(0.05, 0.15)))
            
            if trade_amount >= Decimal('0.5'):
                # Simulate trading (simplified)
                price_change = random.uniform(-0.1, 0.1)  # ±10% price change
                trade_result = trade_amount * Decimal(str(1 + price_change))
                
                profit_loss = trade_result - trade_amount
                self.ftns_balance += profit_loss
                
                if profit_loss > 0:
                    self.total_earnings += profit_loss
                else:
                    self.total_spending += abs(profit_loss)
                
                self._record_transaction(EconomicAction.TRADE_TOKENS, profit_loss)
    
    def _update_reputation(self):
        """Update agent reputation based on behavior"""
        
        # Base reputation update
        if len(self.transaction_history) > 0:
            recent_transactions = self.transaction_history[-10:]  # Last 10 transactions
            avg_transaction_value = sum(tx['amount'] for tx in recent_transactions) / len(recent_transactions)
            
            if avg_transaction_value > 0:
                self.reputation_score += 0.5
            else:
                self.reputation_score -= 0.2
        
        # Reputation bounds
        self.reputation_score = max(0.0, min(100.0, self.reputation_score))
    
    def _update_satisfaction(self):
        """Update satisfaction score based on economic performance"""
        
        if len(self.transaction_history) > 0:
            # Calculate net worth change
            initial_balance = self.profile.initial_balance
            current_net_worth = self.ftns_balance
            
            growth_rate = float((current_net_worth - initial_balance) / initial_balance) if initial_balance > 0 else 0
            
            # Update satisfaction based on growth
            if growth_rate > 0.1:  # 10% growth
                self.satisfaction_score += 0.01
            elif growth_rate < -0.1:  # 10% loss
                self.satisfaction_score -= 0.01
            
            self.satisfaction_score = max(0.0, min(1.0, self.satisfaction_score))
    
    def _record_transaction(self, action: EconomicAction, amount: Decimal):
        """Record transaction for analysis"""
        
        transaction = {
            'timestamp': self.model.current_time,
            'action': action.value,
            'amount': float(amount),
            'balance_after': float(self.ftns_balance),
            'step': self.model.schedule.steps
        }
        
        self.transaction_history.append(transaction)
        
        # Keep only recent transactions to manage memory
        if len(self.transaction_history) > 100:
            self.transaction_history = self.transaction_history[-100:]


class PRSMEconomicModel(Model):
    """
    PRSM Economic Model using Mesa framework
    
    Simulates the complete PRSM token economy with network effects,
    price discovery, incentive mechanisms, and stakeholder interactions.
    """
    
    def __init__(self, 
                 num_agents: int = 10000,
                 initial_token_supply: Decimal = Decimal('1000000'),
                 market_condition: MarketCondition = MarketCondition.STABLE):
        
        if not MESA_AVAILABLE:
            raise ImportError("Mesa framework required. Install with: pip install mesa")
        
        super().__init__()
        self.num_agents = num_agents
        self.initial_token_supply = initial_token_supply
        self.market_condition = market_condition
        
        # Economic state
        self.circulating_supply = initial_token_supply
        self.token_price = Decimal('1.0')  # Start at $1.00
        self.total_transaction_volume = Decimal('0')
        self.network_value = Decimal('0')
        
        # Network structure
        self.network = nx.barabasi_albert_graph(num_agents, 5)  # Scale-free network
        
        # Create Mesa model components
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(self.network)
        
        # Time tracking
        self.current_time = datetime.now(timezone.utc)
        self.step_duration = timedelta(hours=1)  # Each step = 1 hour
        
        # Data collection
        self.economic_metrics: List[EconomicMetrics] = []
        self.datacollector = DataCollector(
            model_reporters={
                "Token Price": lambda m: float(m.token_price),
                "Circulating Supply": lambda m: float(m.circulating_supply),
                "Transaction Volume": lambda m: float(m.total_transaction_volume),
                "Network Value": lambda m: float(m.network_value),
                "Average Satisfaction": lambda m: m.get_average_satisfaction()
            },
            agent_reporters={
                "Balance": lambda a: float(a.ftns_balance),
                "Reputation": lambda a: a.reputation_score,
                "Satisfaction": lambda a: a.satisfaction_score
            }
        )
        
        # Content and compute tracking
        self.content_registry: Dict[int, Dict[str, Any]] = {}
        self.compute_providers: Dict[int, Dict[str, Any]] = {}
        self.staking_pools: Dict[int, Decimal] = {}
        
        # Market dynamics
        self.supply_demand_ratio = 1.0
        self.market_volatility = 0.1
        
        # Create agents
        self._create_agents()
        
        logger.info(f"PRSM Economic Model initialized with {num_agents} agents")
    
    def _create_agents(self):
        """Create diverse agent population"""
        
        # Define stakeholder distribution
        stakeholder_distribution = {
            StakeholderType.QUERY_USER: 0.5,     # 50% query users
            StakeholderType.CONTENT_CREATOR: 0.2, # 20% content creators
            StakeholderType.NODE_OPERATOR: 0.2,   # 20% node operators
            StakeholderType.TOKEN_HOLDER: 0.1     # 10% token holders
        }
        
        agent_id = 0
        
        for stakeholder_type, proportion in stakeholder_distribution.items():
            num_stakeholder_agents = int(self.num_agents * proportion)
            
            for _ in range(num_stakeholder_agents):
                profile = self._generate_agent_profile(stakeholder_type)
                agent = PRSMEconomicAgent(agent_id, self, profile)
                
                self.schedule.add(agent)
                self.grid.place_agent(agent, agent_id)
                
                agent_id += 1
        
        # Fill remaining slots with query users
        while agent_id < self.num_agents:
            profile = self._generate_agent_profile(StakeholderType.QUERY_USER)
            agent = PRSMEconomicAgent(agent_id, self, profile)
            
            self.schedule.add(agent)
            self.grid.place_agent(agent, agent_id)
            
            agent_id += 1
    
    def _generate_agent_profile(self, stakeholder_type: StakeholderType) -> AgentProfile:
        """Generate profile for specific stakeholder type"""
        
        base_profiles = {
            StakeholderType.CONTENT_CREATOR: {
                "initial_balance": (50, 500),
                "risk_tolerance": (0.4, 0.8),
                "activity_frequency": (0.2, 0.5),
                "quality_threshold": (0.6, 0.9),
                "network_connectivity": (10, 30),
                "strategy": "quality_focused"
            },
            StakeholderType.QUERY_USER: {
                "initial_balance": (10, 100),
                "risk_tolerance": (0.2, 0.6),
                "activity_frequency": (0.3, 0.7),
                "quality_threshold": (0.5, 0.8),
                "network_connectivity": (5, 15),
                "strategy": "cost_conscious"
            },
            StakeholderType.NODE_OPERATOR: {
                "initial_balance": (100, 1000),
                "risk_tolerance": (0.3, 0.7),
                "activity_frequency": (0.8, 1.0),
                "quality_threshold": (0.4, 0.7),
                "network_connectivity": (15, 50),
                "strategy": "infrastructure_growth"
            },
            StakeholderType.TOKEN_HOLDER: {
                "initial_balance": (200, 2000),
                "risk_tolerance": (0.1, 0.9),
                "activity_frequency": (0.1, 0.3),
                "quality_threshold": (0.3, 0.6),
                "network_connectivity": (20, 100),
                "strategy": "value_accumulation"
            }
        }
        
        config = base_profiles[stakeholder_type]
        
        return AgentProfile(
            agent_type=stakeholder_type,
            initial_balance=Decimal(str(random.uniform(*config["initial_balance"]))),
            risk_tolerance=random.uniform(*config["risk_tolerance"]),
            activity_frequency=random.uniform(*config["activity_frequency"]),
            quality_threshold=random.uniform(*config["quality_threshold"]),
            network_connectivity=random.randint(*config["network_connectivity"]),
            economic_strategy=config["strategy"]
        )
    
    def step(self):
        """Execute one step of the simulation"""
        
        # Advance time
        self.current_time += self.step_duration
        
        # Execute agent steps
        self.schedule.step()
        
        # Update market dynamics
        self._update_market_dynamics()
        
        # Update network value
        self._calculate_network_value()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Record economic metrics
        self._record_economic_metrics()
        
        logger.debug(f"Step {self.schedule.steps} completed",
                    token_price=float(self.token_price),
                    network_value=float(self.network_value))
    
    def _update_market_dynamics(self):
        """Update token price and market dynamics"""
        
        # Calculate supply and demand pressures
        total_buy_pressure = Decimal('0')
        total_sell_pressure = Decimal('0')
        
        for agent in self.schedule.agents:
            # Simplified buy/sell pressure calculation
            if agent.satisfaction_score > 0.6:
                total_buy_pressure += agent.ftns_balance * Decimal('0.01')
            elif agent.satisfaction_score < 0.4:
                total_sell_pressure += agent.ftns_balance * Decimal('0.01')
        
        # Update supply/demand ratio
        if total_sell_pressure > 0:
            self.supply_demand_ratio = float(total_buy_pressure / total_sell_pressure)
        else:
            self.supply_demand_ratio = 2.0  # Default to buy pressure
        
        # Price update based on supply/demand
        price_change_factor = 1.0
        
        if self.supply_demand_ratio > 1.2:  # High demand
            price_change_factor = 1.0 + min(0.05, (self.supply_demand_ratio - 1.0) * 0.02)
        elif self.supply_demand_ratio < 0.8:  # High supply
            price_change_factor = 1.0 - min(0.05, (1.0 - self.supply_demand_ratio) * 0.02)
        
        # Apply market condition effects
        if self.market_condition == MarketCondition.BULL:
            price_change_factor *= 1.02
        elif self.market_condition == MarketCondition.BEAR:
            price_change_factor *= 0.98
        elif self.market_condition == MarketCondition.VOLATILE:
            volatility = random.uniform(-0.03, 0.03)
            price_change_factor *= (1.0 + volatility)
        
        # Update token price
        self.token_price *= Decimal(str(price_change_factor))
        
        # Ensure reasonable price bounds
        self.token_price = max(Decimal('0.10'), min(Decimal('10.0'), self.token_price))
    
    def _calculate_network_value(self):
        """Calculate total network value"""
        
        total_agent_value = sum(agent.ftns_balance for agent in self.schedule.agents)
        self.network_value = total_agent_value * self.token_price
    
    def _record_economic_metrics(self):
        """Record economic metrics for analysis"""
        
        # Calculate Gini coefficient for wealth distribution
        balances = [float(agent.ftns_balance) for agent in self.schedule.agents]
        gini = self._calculate_gini_coefficient(balances)
        
        # Calculate stakeholder satisfaction by type
        stakeholder_satisfaction = {}
        for stakeholder_type in StakeholderType:
            agents_of_type = [agent for agent in self.schedule.agents 
                            if agent.profile.agent_type == stakeholder_type]
            if agents_of_type:
                avg_satisfaction = sum(agent.satisfaction_score for agent in agents_of_type) / len(agents_of_type)
                stakeholder_satisfaction[stakeholder_type.value] = avg_satisfaction
        
        # Calculate network growth rate
        if len(self.economic_metrics) > 0:
            prev_value = self.economic_metrics[-1].network_value
            growth_rate = float((self.network_value - prev_value) / prev_value) if prev_value > 0 else 0
        else:
            growth_rate = 0.0
        
        metrics = EconomicMetrics(
            timestamp=self.current_time,
            total_supply=self.initial_token_supply,
            circulating_supply=self.circulating_supply,
            average_price=self.token_price,
            transaction_volume=self.total_transaction_volume,
            network_value=self.network_value,
            gini_coefficient=gini,
            stakeholder_satisfaction=stakeholder_satisfaction,
            network_growth_rate=growth_rate
        )
        
        self.economic_metrics.append(metrics)
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for wealth distribution"""
        if not values or all(v == 0 for v in values):
            return 0.0
        
        values = sorted([v for v in values if v >= 0])  # Remove negative values
        n = len(values)
        
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(values)
        total = cumsum[-1]
        
        if total == 0:
            return 0.0
        
        gini = (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, gini))
    
    def get_average_satisfaction(self) -> float:
        """Get average satisfaction across all agents"""
        if not self.schedule.agents:
            return 0.0
        
        total_satisfaction = sum(agent.satisfaction_score for agent in self.schedule.agents)
        return total_satisfaction / len(self.schedule.agents)
    
    # === Economic Actions ===
    
    def register_content_creation(self, creator_id: int, quality: float, cost: Decimal) -> Decimal:
        """Register content creation and calculate revenue"""
        
        # Store content information
        content_id = len(self.content_registry)
        self.content_registry[content_id] = {
            'creator_id': creator_id,
            'quality': quality,
            'creation_cost': cost,
            'created_at': self.current_time,
            'usage_count': 0,
            'total_revenue': Decimal('0')
        }
        
        # Calculate immediate reward (base + quality bonus)
        base_reward = cost * Decimal('0.5')  # 50% of cost back
        quality_bonus = base_reward * Decimal(str(quality))  # Quality multiplier
        
        total_reward = base_reward + quality_bonus
        
        # Update content registry
        self.content_registry[content_id]['total_revenue'] = total_reward
        
        return total_reward
    
    def process_query(self, user_id: int, cost: Decimal) -> float:
        """Process user query and return quality score"""
        
        # Select content based on availability (simplified)
        if self.content_registry:
            content_id = random.choice(list(self.content_registry.keys()))
            content = self.content_registry[content_id]
            
            # Update usage
            content['usage_count'] += 1
            
            # Distribute revenue to content creator
            creator_revenue = cost * Decimal('0.3')  # 30% to creator
            content['total_revenue'] += creator_revenue
            
            # Find creator and pay them
            for agent in self.schedule.agents:
                if agent.unique_id == content['creator_id']:
                    agent.ftns_balance += creator_revenue
                    agent.total_earnings += creator_revenue
                    break
            
            return content['quality']
        
        return 0.5  # Default quality if no content available
    
    def register_compute_provision(self, operator_id: int, reward: Decimal):
        """Register compute provision"""
        
        if operator_id not in self.compute_providers:
            self.compute_providers[operator_id] = {
                'total_compute': Decimal('0'),
                'total_rewards': Decimal('0'),
                'uptime': 0.0
            }
        
        provider = self.compute_providers[operator_id]
        provider['total_compute'] += Decimal('1.0')  # 1 unit of compute
        provider['total_rewards'] += reward
        provider['uptime'] += 1.0
    
    def stake_tokens(self, holder_id: int, amount: Decimal) -> Decimal:
        """Process token staking"""
        
        if holder_id not in self.staking_pools:
            self.staking_pools[holder_id] = Decimal('0')
        
        self.staking_pools[holder_id] += amount
        
        # Calculate staking reward (5% APY, simplified)
        annual_rate = Decimal('0.05')
        hourly_rate = annual_rate / Decimal('8760')  # Hours in a year
        
        reward = amount * hourly_rate
        
        return reward
    
    # === Analysis Methods ===
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """Get comprehensive economic summary"""
        
        if not self.economic_metrics:
            return {"error": "No metrics collected yet"}
        
        latest_metrics = self.economic_metrics[-1]
        
        # Price stability analysis
        if len(self.economic_metrics) >= 10:
            recent_prices = [m.average_price for m in self.economic_metrics[-10:]]
            price_variance = float(np.std([float(p) for p in recent_prices]) / np.mean([float(p) for p in recent_prices]))
        else:
            price_variance = 0.0
        
        # Network growth analysis
        if len(self.economic_metrics) >= 5:
            growth_rates = [m.network_growth_rate for m in self.economic_metrics[-5:]]
            avg_growth_rate = sum(growth_rates) / len(growth_rates)
        else:
            avg_growth_rate = 0.0
        
        # Stakeholder analysis
        stakeholder_counts = {}
        for stakeholder_type in StakeholderType:
            count = len([agent for agent in self.schedule.agents 
                        if agent.profile.agent_type == stakeholder_type])
            stakeholder_counts[stakeholder_type.value] = count
        
        return {
            "simulation_step": self.schedule.steps,
            "current_time": self.current_time.isoformat(),
            "token_economics": {
                "current_price": float(latest_metrics.average_price),
                "price_stability": 1.0 - min(1.0, price_variance),  # Higher = more stable
                "circulating_supply": float(latest_metrics.circulating_supply),
                "network_value": float(latest_metrics.network_value),
                "transaction_volume": float(latest_metrics.transaction_volume)
            },
            "network_health": {
                "gini_coefficient": latest_metrics.gini_coefficient,
                "average_growth_rate": avg_growth_rate,
                "stakeholder_satisfaction": latest_metrics.stakeholder_satisfaction,
                "supply_demand_ratio": self.supply_demand_ratio
            },
            "stakeholder_distribution": stakeholder_counts,
            "content_economy": {
                "total_content": len(self.content_registry),
                "active_creators": len(set(c['creator_id'] for c in self.content_registry.values())),
                "total_content_revenue": float(sum(c['total_revenue'] for c in self.content_registry.values()))
            },
            "compute_economy": {
                "active_operators": len(self.compute_providers),
                "total_compute_provided": float(sum(p['total_compute'] for p in self.compute_providers.values())),
                "total_compute_rewards": float(sum(p['total_rewards'] for p in self.compute_providers.values()))
            },
            "staking_economy": {
                "total_staked": float(sum(self.staking_pools.values())),
                "staking_participants": len(self.staking_pools),
                "staking_ratio": float(sum(self.staking_pools.values()) / self.circulating_supply) if self.circulating_supply > 0 else 0
            }
        }
    
    def validate_economic_targets(self) -> Dict[str, bool]:
        """Validate Phase 2 economic targets"""
        
        if len(self.economic_metrics) < 10:
            return {"insufficient_data": True}
        
        latest_metrics = self.economic_metrics[-1]
        recent_metrics = self.economic_metrics[-10:]
        
        # Price stability: ±20% variance target
        recent_prices = [float(m.average_price) for m in recent_metrics]
        price_variance = np.std(recent_prices) / np.mean(recent_prices)
        price_stable = price_variance <= 0.20
        
        # Incentive alignment: >80% participant satisfaction
        avg_satisfaction = sum(latest_metrics.stakeholder_satisfaction.values()) / len(latest_metrics.stakeholder_satisfaction)
        incentives_aligned = avg_satisfaction >= 0.80
        
        # Network sustainability: positive cash flow
        recent_growth = [m.network_growth_rate for m in recent_metrics[-5:]]
        avg_growth = sum(recent_growth) / len(recent_growth)
        network_sustainable = avg_growth >= 0.0
        
        # Economic distribution: Gini coefficient < 0.7
        wealth_distributed = latest_metrics.gini_coefficient <= 0.7
        
        return {
            "price_stability": price_stable,
            "incentive_alignment": incentives_aligned,
            "network_sustainability": network_sustainable,
            "wealth_distribution": wealth_distributed,
            "overall_success": all([price_stable, incentives_aligned, network_sustainable, wealth_distributed])
        }


# === Simulation Runner ===

class EconomicSimulationRunner:
    """
    Simulation runner for comprehensive economic validation
    
    Manages multiple simulation scenarios and provides analysis
    of tokenomics performance under different conditions.
    """
    
    def __init__(self):
        self.simulation_results: List[Dict[str, Any]] = []
        self.scenarios = [
            {"name": "stable_market", "condition": MarketCondition.STABLE, "steps": 168},  # 1 week
            {"name": "bull_market", "condition": MarketCondition.BULL, "steps": 168},
            {"name": "bear_market", "condition": MarketCondition.BEAR, "steps": 168},
            {"name": "volatile_market", "condition": MarketCondition.VOLATILE, "steps": 168}
        ]
        
        logger.info("Economic Simulation Runner initialized")
    
    async def run_comprehensive_simulation(self) -> Dict[str, Any]:
        """Run comprehensive economic simulation across scenarios"""
        
        logger.info("Starting comprehensive economic simulation")
        start_time = datetime.now(timezone.utc)
        
        simulation_report = {
            "simulation_id": str(random.randint(100000, 999999)),
            "start_time": start_time,
            "scenarios": [],
            "comparative_analysis": {},
            "validation_results": {},
            "recommendations": []
        }
        
        # Run each scenario
        for scenario in self.scenarios:
            logger.info(f"Running {scenario['name']} scenario")
            
            scenario_result = await self._run_scenario(scenario)
            simulation_report["scenarios"].append(scenario_result)
            
            # Brief pause between scenarios
            await asyncio.sleep(0.1)
        
        # Comparative analysis
        simulation_report["comparative_analysis"] = self._analyze_scenarios(simulation_report["scenarios"])
        
        # Validation against targets
        simulation_report["validation_results"] = self._validate_phase2_targets(simulation_report["scenarios"])
        
        # Generate recommendations
        simulation_report["recommendations"] = self._generate_recommendations(simulation_report)
        
        total_duration = datetime.now(timezone.utc) - start_time
        simulation_report["total_duration"] = total_duration.total_seconds()
        simulation_report["end_time"] = datetime.now(timezone.utc)
        
        logger.info("Comprehensive economic simulation completed",
                   duration=total_duration.total_seconds(),
                   scenarios=len(self.scenarios))
        
        return simulation_report
    
    async def _run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual scenario simulation"""
        
        scenario_start = datetime.now(timezone.utc)
        
        # Create model for this scenario
        model = PRSMEconomicModel(
            num_agents=1000,  # Reduced for faster testing
            market_condition=scenario["condition"]
        )
        
        # Run simulation steps
        for step in range(scenario["steps"]):
            model.step()
            
            # Progress logging
            if step % 24 == 0:  # Every 24 hours
                logger.debug(f"Scenario {scenario['name']} - Step {step}/{scenario['steps']}")
        
        # Collect final results
        economic_summary = model.get_economic_summary()
        validation_results = model.validate_economic_targets()
        
        scenario_duration = datetime.now(timezone.utc) - scenario_start
        
        return {
            "scenario_name": scenario["name"],
            "market_condition": scenario["condition"].value,
            "steps_completed": scenario["steps"],
            "duration_seconds": scenario_duration.total_seconds(),
            "economic_summary": economic_summary,
            "validation_results": validation_results,
            "final_metrics": model.economic_metrics[-1] if model.economic_metrics else None
        }
    
    def _analyze_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scenarios comparatively"""
        
        # Extract key metrics across scenarios
        scenario_comparison = {}
        
        for scenario in scenarios:
            name = scenario["scenario_name"]
            summary = scenario["economic_summary"]
            
            scenario_comparison[name] = {
                "final_price": summary["token_economics"]["current_price"],
                "price_stability": summary["token_economics"]["price_stability"],
                "network_value": summary["token_economics"]["network_value"],
                "satisfaction": sum(summary["network_health"]["stakeholder_satisfaction"].values()) / 4,
                "gini_coefficient": summary["network_health"]["gini_coefficient"],
                "validation_success": scenario["validation_results"].get("overall_success", False)
            }
        
        # Calculate best/worst performing scenarios
        best_scenario = max(scenario_comparison.items(), 
                          key=lambda x: x[1]["satisfaction"] * x[1]["price_stability"])
        worst_scenario = min(scenario_comparison.items(),
                           key=lambda x: x[1]["satisfaction"] * x[1]["price_stability"])
        
        return {
            "scenario_comparison": scenario_comparison,
            "best_performing": best_scenario[0],
            "worst_performing": worst_scenario[0],
            "average_satisfaction": sum(s["satisfaction"] for s in scenario_comparison.values()) / len(scenario_comparison),
            "average_price_stability": sum(s["price_stability"] for s in scenario_comparison.values()) / len(scenario_comparison)
        }
    
    def _validate_phase2_targets(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate Phase 2 economic targets across scenarios"""
        
        validation_summary = {
            "price_stability_scenarios": 0,
            "incentive_alignment_scenarios": 0,
            "sustainability_scenarios": 0,
            "overall_success_scenarios": 0,
            "total_scenarios": len(scenarios)
        }
        
        for scenario in scenarios:
            validation = scenario["validation_results"]
            
            if validation.get("price_stability", False):
                validation_summary["price_stability_scenarios"] += 1
            if validation.get("incentive_alignment", False):
                validation_summary["incentive_alignment_scenarios"] += 1
            if validation.get("network_sustainability", False):
                validation_summary["sustainability_scenarios"] += 1
            if validation.get("overall_success", False):
                validation_summary["overall_success_scenarios"] += 1
        
        # Calculate success rates
        total = validation_summary["total_scenarios"]
        validation_summary["success_rates"] = {
            "price_stability": validation_summary["price_stability_scenarios"] / total,
            "incentive_alignment": validation_summary["incentive_alignment_scenarios"] / total,
            "sustainability": validation_summary["sustainability_scenarios"] / total,
            "overall_success": validation_summary["overall_success_scenarios"] / total
        }
        
        # Phase 2 pass criteria: >75% scenarios must pass overall
        validation_summary["phase2_passed"] = validation_summary["success_rates"]["overall_success"] >= 0.75
        
        return validation_summary
    
    def _generate_recommendations(self, simulation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on simulation results"""
        
        recommendations = []
        validation = simulation_report["validation_results"]
        analysis = simulation_report["comparative_analysis"]
        
        # Price stability recommendations
        if validation["success_rates"]["price_stability"] < 0.75:
            recommendations.append(
                "Price stability below target (75% scenarios). Consider implementing: "
                "1) Automated market makers, 2) Token burn mechanisms, 3) Improved demand forecasting."
            )
        
        # Incentive alignment recommendations
        if validation["success_rates"]["incentive_alignment"] < 0.75:
            recommendations.append(
                "Incentive alignment needs improvement. Consider: "
                "1) Better reward distribution algorithms, 2) Quality-based bonuses, 3) Community governance."
            )
        
        # Sustainability recommendations
        if validation["success_rates"]["sustainability"] < 0.75:
            recommendations.append(
                "Network sustainability concerns. Implement: "
                "1) Fee optimization, 2) Infrastructure incentives, 3) Long-term value accrual mechanisms."
            )
        
        # Market condition specific recommendations
        worst_scenario = analysis["worst_performing"]
        if "bear" in worst_scenario:
            recommendations.append(
                "Poor performance in bear markets. Strengthen: "
                "1) Defensive tokenomics, 2) Counter-cyclical incentives, 3) Value storage mechanisms."
            )
        
        if analysis["average_satisfaction"] < 0.8:
            recommendations.append(
                f"Average stakeholder satisfaction ({analysis['average_satisfaction']:.1%}) below 80% target. "
                "Improve user experience and economic rewards."
            )
        
        return recommendations


# === Global instances and convenience functions ===

async def run_economic_simulation(steps: int = 168, num_agents: int = 1000) -> Dict[str, Any]:
    """Run a single economic simulation"""
    
    model = PRSMEconomicModel(num_agents=num_agents)
    
    logger.info(f"Running economic simulation: {steps} steps, {num_agents} agents")
    
    for step in range(steps):
        model.step()
        
        if step % 24 == 0:  # Log every 24 steps (1 day)
            summary = model.get_economic_summary()
            logger.info(f"Day {step//24}: Price ${summary['token_economics']['current_price']:.2f}, "
                       f"Satisfaction {summary['network_health']['stakeholder_satisfaction']}")
    
    return {
        "final_summary": model.get_economic_summary(),
        "validation_results": model.validate_economic_targets(),
        "total_steps": steps,
        "total_agents": num_agents
    }


async def run_comprehensive_economic_validation() -> Dict[str, Any]:
    """Run comprehensive economic validation for Phase 2"""
    
    runner = EconomicSimulationRunner()
    return await runner.run_comprehensive_simulation()


if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
            results = await run_comprehensive_economic_validation()
            print(f"Economic validation completed: {results['validation_results']['phase2_passed']}")
            return results['validation_results']['phase2_passed']
        else:
            results = await run_economic_simulation(steps=48, num_agents=500)  # 2 days, 500 agents
            print(f"Economic simulation completed: {results['validation_results']['overall_success']}")
            return results['validation_results']['overall_success']
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)