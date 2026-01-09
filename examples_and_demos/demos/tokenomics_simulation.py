#!/usr/bin/env python3
"""
PRSM Tokenomics Simulation & Stress Test
Enhanced economic simulation demonstrating FTNS token ecosystem viability

Features:
- 10-50 economic agents with diverse behaviors
- Dynamic reward distribution based on contribution quality
- Economic stress testing under various market conditions
- Gini coefficient and fairness analysis
- Interactive Jupyter notebook visualization

This simulation validates:
1. Token economy sustainability under various conditions
2. Fairness of reward distribution mechanisms  
3. Network resilience to economic attacks
4. Long-term incentive alignment
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
import json
import sys
from pathlib import Path

# Set high precision for financial calculations
getcontext().prec = 18

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Using matplotlib fallback.")

class AgentType(Enum):
    """Types of economic agents in the simulation"""
    DATA_CONTRIBUTOR = "data_contributor"
    MODEL_CREATOR = "model_creator"
    QUERY_USER = "query_user"
    VALIDATOR = "validator"
    FREELOADER = "freeloader"  # Bad actor type

class EconomicEvent(Enum):
    """Economic events that can occur during simulation"""
    CONTRIBUTE_DATA = "contribute_data"
    CREATE_MODEL = "create_model"
    VALIDATE_CONTENT = "validate_content"
    QUERY_SYSTEM = "query_system"
    STAKE_TOKENS = "stake_tokens"
    TRADE_TOKENS = "trade_tokens"
    SPAM_NETWORK = "spam_network"  # Bad actor behavior

class MarketCondition(Enum):
    """Market conditions for stress testing"""
    NORMAL = "normal"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    VOLATILITY_SPIKE = "volatility_spike"
    COMPUTE_SHORTAGE = "compute_shortage"
    DATA_FLOOD = "data_flood"

@dataclass
class ContributionQuality:
    """Quality metrics for contributions"""
    accuracy: float  # 0.0 to 1.0
    uniqueness: float  # 0.0 to 1.0
    usefulness: float  # 0.0 to 1.0
    size_mb: float
    citations: int
    validation_score: float
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score"""
        return (self.accuracy * 0.3 + 
                self.uniqueness * 0.25 + 
                self.usefulness * 0.25 + 
                self.validation_score * 0.2)

@dataclass
class EconomicAgent:
    """Economic agent in the FTNS ecosystem"""
    agent_id: str
    agent_type: AgentType
    ftns_balance: Decimal
    reputation_score: float
    
    # Behavioral parameters
    activity_level: float  # 0.0 to 1.0 - how often agent acts
    quality_focus: float   # 0.0 to 1.0 - focus on quality vs quantity
    risk_tolerance: float  # 0.0 to 1.0 - willingness to take risks
    social_factor: float   # 0.0 to 1.0 - influence of others' behavior
    
    # Performance tracking
    total_contributions: int = 0
    total_earnings: Decimal = field(default_factory=lambda: Decimal('0'))
    total_spending: Decimal = field(default_factory=lambda: Decimal('0'))
    contribution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Network effects
    trust_network: List[str] = field(default_factory=list)
    influence_score: float = 0.0

class FTNSEconomicSimulation:
    """
    FTNS Token Economy Simulation with stress testing capabilities
    """
    
    def __init__(self, 
                 num_agents: int = 30,
                 initial_token_supply: Decimal = Decimal('1000000'),
                 simulation_days: int = 30):
        
        self.num_agents = num_agents
        self.initial_token_supply = initial_token_supply
        self.simulation_days = simulation_days
        
        # Economic state
        self.current_day = 0
        self.circulating_supply = initial_token_supply
        self.token_price = Decimal('1.00')  # Start at $1.00
        self.market_condition = MarketCondition.NORMAL
        
        # Agents
        self.agents: Dict[str, EconomicAgent] = {}
        self.agent_types_distribution = {
            AgentType.DATA_CONTRIBUTOR: 0.3,    # 30%
            AgentType.MODEL_CREATOR: 0.2,       # 20%
            AgentType.QUERY_USER: 0.3,          # 30%
            AgentType.VALIDATOR: 0.15,          # 15%
            AgentType.FREELOADER: 0.05          # 5% bad actors
        }
        
        # Economic parameters
        self.base_reward_per_mb = Decimal('0.1')
        self.quality_multiplier_max = Decimal('5.0')
        self.validation_reward = Decimal('0.5')
        self.query_cost_base = Decimal('0.5')
        self.staking_apy = Decimal('0.12')  # 12% APY
        
        # Simulation data
        self.daily_metrics: List[Dict[str, Any]] = []
        self.transaction_log: List[Dict[str, Any]] = []
        self.stress_events: List[Dict[str, Any]] = []
        
        # Initialize agents
        self._create_agents()
        
        print(f"🎯 FTNS Economic Simulation initialized:")
        print(f"   Agents: {num_agents}")
        print(f"   Initial Supply: {initial_token_supply:,} FTNS")
        print(f"   Simulation Duration: {simulation_days} days")
    
    def _create_agents(self):
        """Create diverse population of economic agents"""
        
        agent_id = 0
        
        for agent_type, proportion in self.agent_types_distribution.items():
            num_type = int(self.num_agents * proportion)
            
            for _ in range(num_type):
                agent = self._generate_agent(agent_type, agent_id)
                self.agents[agent.agent_id] = agent
                agent_id += 1
        
        # Fill remaining slots with query users
        while agent_id < self.num_agents:
            agent = self._generate_agent(AgentType.QUERY_USER, agent_id)
            self.agents[agent.agent_id] = agent
            agent_id += 1
        
        # Create trust networks
        self._establish_trust_networks()
    
    def _generate_agent(self, agent_type: AgentType, agent_id: int) -> EconomicAgent:
        """Generate agent with type-specific characteristics"""
        
        # Base characteristics by type
        type_configs = {
            AgentType.DATA_CONTRIBUTOR: {
                "balance_range": (50, 500),
                "activity": (0.6, 0.9),
                "quality_focus": (0.7, 0.95),
                "risk_tolerance": (0.3, 0.7),
                "social_factor": (0.4, 0.8)
            },
            AgentType.MODEL_CREATOR: {
                "balance_range": (100, 1000),
                "activity": (0.3, 0.6),
                "quality_focus": (0.8, 0.98),
                "risk_tolerance": (0.4, 0.8),
                "social_factor": (0.5, 0.9)
            },
            AgentType.QUERY_USER: {
                "balance_range": (20, 200),
                "activity": (0.7, 1.0),
                "quality_focus": (0.3, 0.7),
                "risk_tolerance": (0.2, 0.6),
                "social_factor": (0.3, 0.7)
            },
            AgentType.VALIDATOR: {
                "balance_range": (200, 800),
                "activity": (0.8, 1.0),
                "quality_focus": (0.9, 1.0),
                "risk_tolerance": (0.2, 0.5),
                "social_factor": (0.6, 0.9)
            },
            AgentType.FREELOADER: {
                "balance_range": (10, 100),
                "activity": (0.9, 1.0),
                "quality_focus": (0.1, 0.3),
                "risk_tolerance": (0.8, 1.0),
                "social_factor": (0.1, 0.4)
            }
        }
        
        config = type_configs[agent_type]
        
        return EconomicAgent(
            agent_id=f"{agent_type.value}_{agent_id}",
            agent_type=agent_type,
            ftns_balance=Decimal(str(random.uniform(*config["balance_range"]))),
            reputation_score=random.uniform(40, 60),  # Start neutral
            activity_level=random.uniform(*config["activity"]),
            quality_focus=random.uniform(*config["quality_focus"]),
            risk_tolerance=random.uniform(*config["risk_tolerance"]),
            social_factor=random.uniform(*config["social_factor"])
        )
    
    def _establish_trust_networks(self):
        """Create trust networks between agents"""
        
        for agent in self.agents.values():
            # Each agent trusts 2-8 other agents
            num_connections = random.randint(2, min(8, len(self.agents) - 1))
            
            # Prefer similar agent types and higher reputation
            potential_connections = [
                other for other in self.agents.values() 
                if other.agent_id != agent.agent_id
            ]
            
            # Weight by reputation and type similarity
            weights = []
            for other in potential_connections:
                weight = other.reputation_score / 100.0
                if other.agent_type == agent.agent_type:
                    weight *= 1.5  # Prefer same type
                weights.append(weight)
            
            # Select weighted random connections
            if weights:
                connections = random.choices(
                    potential_connections, 
                    weights=weights, 
                    k=min(num_connections, len(potential_connections))
                )
                agent.trust_network = [conn.agent_id for conn in connections]
    
    def simulate_day(self, day: int):
        """Simulate one day of economic activity"""
        
        self.current_day = day
        daily_transactions = []
        daily_events = []
        
        # Process market condition effects
        self._apply_market_conditions()
        
        # Each agent has chance to act based on activity level
        for agent in self.agents.values():
            if random.random() < agent.activity_level:
                action, result = self._agent_daily_action(agent)
                
                if result:
                    daily_transactions.append({
                        'day': day,
                        'agent_id': agent.agent_id,
                        'agent_type': agent.agent_type.value,
                        'action': action.value,
                        'amount': float(result.get('amount', 0)),
                        'quality': result.get('quality', 0),
                        'balance_after': float(agent.ftns_balance)
                    })
        
        # Update token price based on supply/demand
        self._update_token_economics(daily_transactions)
        
        # Calculate daily metrics
        metrics = self._calculate_daily_metrics(day, daily_transactions)
        self.daily_metrics.append(metrics)
        
        # Store transactions
        self.transaction_log.extend(daily_transactions)
        
        if day % 5 == 0:  # Progress update every 5 days
            print(f"📅 Day {day}: Price ${float(self.token_price):.3f}, "
                  f"Avg Quality {metrics['avg_contribution_quality']:.2f}, "
                  f"Gini {metrics['gini_coefficient']:.3f}")
    
    def _agent_daily_action(self, agent: EconomicAgent) -> Tuple[EconomicEvent, Optional[Dict[str, Any]]]:
        """Determine and execute agent's daily action"""
        
        # Choose action based on agent type
        possible_actions = {
            AgentType.DATA_CONTRIBUTOR: [
                (EconomicEvent.CONTRIBUTE_DATA, 0.6),
                (EconomicEvent.VALIDATE_CONTENT, 0.3),
                (EconomicEvent.STAKE_TOKENS, 0.1)
            ],
            AgentType.MODEL_CREATOR: [
                (EconomicEvent.CREATE_MODEL, 0.5),
                (EconomicEvent.CONTRIBUTE_DATA, 0.3),
                (EconomicEvent.VALIDATE_CONTENT, 0.2)
            ],
            AgentType.QUERY_USER: [
                (EconomicEvent.QUERY_SYSTEM, 0.8),
                (EconomicEvent.STAKE_TOKENS, 0.2)
            ],
            AgentType.VALIDATOR: [
                (EconomicEvent.VALIDATE_CONTENT, 0.7),
                (EconomicEvent.STAKE_TOKENS, 0.3)
            ],
            AgentType.FREELOADER: [
                (EconomicEvent.SPAM_NETWORK, 0.6),
                (EconomicEvent.QUERY_SYSTEM, 0.4)
            ]
        }
        
        actions = possible_actions[agent.agent_type]
        weights = [prob for _, prob in actions]
        chosen_action = random.choices([action for action, _ in actions], weights=weights)[0]
        
        # Execute action
        result = self._execute_action(agent, chosen_action)
        
        return chosen_action, result
    
    def _execute_action(self, agent: EconomicAgent, action: EconomicEvent) -> Optional[Dict[str, Any]]:
        """Execute specific economic action"""
        
        if action == EconomicEvent.CONTRIBUTE_DATA:
            return self._contribute_data(agent)
        elif action == EconomicEvent.CREATE_MODEL:
            return self._create_model(agent)
        elif action == EconomicEvent.VALIDATE_CONTENT:
            return self._validate_content(agent)
        elif action == EconomicEvent.QUERY_SYSTEM:
            return self._query_system(agent)
        elif action == EconomicEvent.STAKE_TOKENS:
            return self._stake_tokens(agent)
        elif action == EconomicEvent.SPAM_NETWORK:
            return self._spam_network(agent)
        
        return None
    
    def _contribute_data(self, agent: EconomicAgent) -> Dict[str, Any]:
        """Agent contributes data to the network"""
        
        # Generate contribution quality based on agent characteristics
        quality = ContributionQuality(
            accuracy=min(1.0, np.random.normal(agent.quality_focus, 0.15)),
            uniqueness=random.uniform(0.3, 1.0),
            usefulness=min(1.0, np.random.normal(agent.quality_focus * 0.8, 0.2)),
            size_mb=random.uniform(1, 100),
            citations=0,  # Will be updated later
            validation_score=0.0  # Will be determined by validators
        )
        
        # Calculate reward based on quality and size
        base_reward = self.base_reward_per_mb * Decimal(str(quality.size_mb))
        quality_multiplier = Decimal(str(1.0 + (quality.overall_quality * float(self.quality_multiplier_max - 1))))
        total_reward = base_reward * quality_multiplier
        
        # Apply reputation multiplier
        reputation_multiplier = Decimal(str(0.5 + (agent.reputation_score / 100.0)))
        final_reward = total_reward * reputation_multiplier
        
        # Award tokens
        agent.ftns_balance += final_reward
        agent.total_earnings += final_reward
        agent.total_contributions += 1
        
        # Update reputation based on quality
        reputation_change = (quality.overall_quality - 0.5) * 2.0  # -1 to +1
        agent.reputation_score += reputation_change
        agent.reputation_score = max(0, min(100, agent.reputation_score))
        
        # Record contribution
        agent.contribution_history.append({
            'day': self.current_day,
            'type': 'data',
            'quality': quality.overall_quality,
            'reward': float(final_reward),
            'size_mb': quality.size_mb
        })
        
        return {
            'amount': final_reward,
            'quality': quality.overall_quality,
            'size_mb': quality.size_mb,
            'type': 'data_contribution'
        }
    
    def _create_model(self, agent: EconomicAgent) -> Dict[str, Any]:
        """Agent creates and contributes a model"""
        
        # Model creation is more complex and expensive
        creation_cost = Decimal(str(random.uniform(10, 100)))
        
        if agent.ftns_balance < creation_cost:
            return {'amount': 0, 'quality': 0, 'type': 'failed_model_creation'}
        
        # Deduct creation cost
        agent.ftns_balance -= creation_cost
        agent.total_spending += creation_cost
        
        # Generate model quality
        model_quality = min(1.0, np.random.normal(agent.quality_focus * 0.9, 0.1))
        
        # Calculate reward (higher than data contribution)
        base_reward = Decimal('50') + (Decimal(str(model_quality)) * Decimal('200'))
        reputation_multiplier = Decimal(str(0.5 + (agent.reputation_score / 100.0)))
        final_reward = base_reward * reputation_multiplier
        
        # Award tokens
        agent.ftns_balance += final_reward
        agent.total_earnings += final_reward
        agent.total_contributions += 1
        
        # Significant reputation boost for good models
        reputation_change = model_quality * 5.0
        agent.reputation_score += reputation_change
        agent.reputation_score = max(0, min(100, agent.reputation_score))
        
        # Record contribution
        agent.contribution_history.append({
            'day': self.current_day,
            'type': 'model',
            'quality': model_quality,
            'reward': float(final_reward),
            'creation_cost': float(creation_cost)
        })
        
        return {
            'amount': final_reward - creation_cost,
            'quality': model_quality,
            'type': 'model_creation'
        }
    
    def _validate_content(self, agent: EconomicAgent) -> Dict[str, Any]:
        """Agent validates content quality"""
        
        # Validators earn small but consistent rewards
        validation_reward = self.validation_reward * Decimal(str(0.8 + (agent.reputation_score / 500.0)))
        
        agent.ftns_balance += validation_reward
        agent.total_earnings += validation_reward
        
        # Small reputation boost for validation work
        agent.reputation_score += 0.2
        agent.reputation_score = min(100, agent.reputation_score)
        
        return {
            'amount': validation_reward,
            'quality': 0.8,  # Validation is generally high quality work
            'type': 'validation'
        }
    
    def _query_system(self, agent: EconomicAgent) -> Dict[str, Any]:
        """Agent queries the system (pays fees)"""
        
        # Calculate query cost based on market conditions
        query_cost = self.query_cost_base
        
        if self.market_condition == MarketCondition.COMPUTE_SHORTAGE:
            query_cost *= Decimal('1.5')
        elif self.market_condition == MarketCondition.BULL_MARKET:
            query_cost *= Decimal('1.2')
        
        if agent.ftns_balance < query_cost:
            return {'amount': 0, 'type': 'failed_query'}
        
        # Deduct cost
        agent.ftns_balance -= query_cost
        agent.total_spending += query_cost
        
        # Quality of service depends on network state
        service_quality = random.uniform(0.6, 0.9)
        
        # Update agent satisfaction (affects future behavior)
        if service_quality > 0.8:
            agent.reputation_score += 0.1
        elif service_quality < 0.6:
            agent.reputation_score -= 0.1
        
        return {
            'amount': -query_cost,
            'quality': service_quality,
            'type': 'query'
        }
    
    def _stake_tokens(self, agent: EconomicAgent) -> Dict[str, Any]:
        """Agent stakes tokens for rewards"""
        
        # Stake 10-30% of balance
        stake_percentage = random.uniform(0.1, 0.3)
        stake_amount = agent.ftns_balance * Decimal(str(stake_percentage))
        
        if stake_amount < Decimal('1'):
            return {'amount': 0, 'type': 'failed_staking'}
        
        # Calculate daily staking reward (APY / 365)
        daily_rate = self.staking_apy / Decimal('365')
        staking_reward = stake_amount * daily_rate
        
        agent.ftns_balance += staking_reward
        agent.total_earnings += staking_reward
        
        # Small reputation boost for long-term commitment
        agent.reputation_score += 0.05
        
        return {
            'amount': staking_reward,
            'quality': 0.5,  # Neutral quality for staking
            'type': 'staking'
        }
    
    def _spam_network(self, agent: EconomicAgent) -> Dict[str, Any]:
        """Bad actor spams the network (should be penalized)"""
        
        # Freeloaders try to game the system with low-quality contributions
        spam_quality = random.uniform(0.1, 0.3)
        size = random.uniform(0.1, 5)  # Small, low-value content
        
        # Very low reward due to poor quality
        reward = self.base_reward_per_mb * Decimal(str(size)) * Decimal(str(spam_quality))
        
        agent.ftns_balance += reward
        agent.total_earnings += reward
        
        # Significant reputation penalty for spam
        agent.reputation_score -= 2.0
        agent.reputation_score = max(0, agent.reputation_score)
        
        return {
            'amount': reward,
            'quality': spam_quality,
            'size_mb': size,
            'type': 'spam'
        }
    
    def _apply_market_conditions(self):
        """Apply market condition effects"""
        
        if self.market_condition == MarketCondition.BULL_MARKET:
            # Increase rewards and activity
            self.base_reward_per_mb *= Decimal('1.02')
            self.token_price *= Decimal('1.01')
            
        elif self.market_condition == MarketCondition.BEAR_MARKET:
            # Decrease rewards and activity
            self.base_reward_per_mb *= Decimal('0.98')
            self.token_price *= Decimal('0.99')
            
        elif self.market_condition == MarketCondition.VOLATILITY_SPIKE:
            # Random price changes
            price_change = Decimal(str(random.uniform(-0.05, 0.05)))
            self.token_price *= (Decimal('1') + price_change)
            
        elif self.market_condition == MarketCondition.COMPUTE_SHORTAGE:
            # Higher query costs, lower supply rewards
            self.query_cost_base *= Decimal('1.05')
            
        elif self.market_condition == MarketCondition.DATA_FLOOD:
            # Lower data rewards due to oversupply
            self.base_reward_per_mb *= Decimal('0.95')
    
    def _update_token_economics(self, daily_transactions: List[Dict[str, Any]]):
        """Update token price and supply based on activity"""
        
        # Calculate supply and demand pressures
        total_rewards = sum(tx['amount'] for tx in daily_transactions if tx['amount'] > 0)
        total_spending = sum(abs(tx['amount']) for tx in daily_transactions if tx['amount'] < 0)
        
        # Simple supply/demand price adjustment
        if total_spending > 0:
            demand_pressure = Decimal(str(total_spending)) / Decimal(str(max(1, total_rewards)))
            price_adjustment = (demand_pressure - Decimal('1')) * Decimal('0.01')  # 1% max daily change
            price_adjustment = max(Decimal('-0.01'), min(Decimal('0.01'), price_adjustment))
            
            self.token_price *= (Decimal('1') + price_adjustment)
        
        # Ensure reasonable price bounds
        self.token_price = max(Decimal('0.10'), min(Decimal('10.0'), self.token_price))
    
    def _calculate_daily_metrics(self, day: int, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive daily metrics"""
        
        # Basic metrics
        total_agents = len(self.agents)
        active_agents = len(set(tx['agent_id'] for tx in transactions))
        
        # Token distribution
        balances = [float(agent.ftns_balance) for agent in self.agents.values()]
        gini = self._calculate_gini_coefficient(balances)
        
        # Quality metrics
        quality_scores = [tx['quality'] for tx in transactions if 'quality' in tx and tx['quality'] > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Reputation distribution
        reputations = [agent.reputation_score for agent in self.agents.values()]
        avg_reputation = sum(reputations) / len(reputations)
        
        # Economic activity
        total_volume = sum(abs(tx['amount']) for tx in transactions)
        
        # Agent type performance
        type_performance = {}
        for agent_type in AgentType:
            type_agents = [agent for agent in self.agents.values() if agent.agent_type == agent_type]
            if type_agents:
                avg_balance = sum(float(agent.ftns_balance) for agent in type_agents) / len(type_agents)
                avg_earnings = sum(float(agent.total_earnings) for agent in type_agents) / len(type_agents)
                type_performance[agent_type.value] = {
                    'avg_balance': avg_balance,
                    'avg_earnings': avg_earnings,
                    'count': len(type_agents)
                }
        
        return {
            'day': day,
            'token_price': float(self.token_price),
            'circulating_supply': float(self.circulating_supply),
            'total_agents': total_agents,
            'active_agents': active_agents,
            'activity_rate': active_agents / total_agents,
            'gini_coefficient': gini,
            'avg_contribution_quality': avg_quality,
            'avg_reputation': avg_reputation,
            'total_transaction_volume': total_volume,
            'num_transactions': len(transactions),
            'market_condition': self.market_condition.value,
            'agent_type_performance': type_performance
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for wealth distribution"""
        if not values or all(v == 0 for v in values):
            return 0.0
        
        values = sorted([v for v in values if v >= 0])
        n = len(values)
        total = sum(values)
        
        if total == 0:
            return 0.0
        
        cumsum = np.cumsum(values)
        gini = (2 * sum((i + 1) * values[i] for i in range(n))) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, gini))
    
    def run_simulation(self, stress_scenarios: Optional[List[Dict[str, Any]]] = None):
        """Run complete simulation with optional stress scenarios"""
        
        print(f"🚀 Starting {self.simulation_days}-day FTNS economic simulation...")
        
        for day in range(self.simulation_days):
            # Apply stress scenarios if specified
            if stress_scenarios:
                for scenario in stress_scenarios:
                    if scenario['start_day'] <= day <= scenario['end_day']:
                        self.market_condition = MarketCondition(scenario['condition'])
                        if day == scenario['start_day']:
                            print(f"💥 Stress event: {scenario['condition']} (days {scenario['start_day']}-{scenario['end_day']})")
            
            # Run daily simulation
            self.simulate_day(day)
        
        print("✅ Simulation completed!")
        return self.get_simulation_results()
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results and analysis"""
        
        # Convert daily metrics to DataFrame for analysis
        df = pd.DataFrame(self.daily_metrics)
        
        # Calculate key performance indicators
        final_gini = df['gini_coefficient'].iloc[-1]
        avg_quality = df['avg_contribution_quality'].mean()
        price_stability = 1.0 - (df['token_price'].std() / df['token_price'].mean())
        avg_activity = df['activity_rate'].mean()
        
        # Agent performance analysis
        agent_analysis = {}
        for agent in self.agents.values():
            roi = float((agent.total_earnings - agent.total_spending) / max(agent.total_spending, Decimal('1')))
            agent_analysis[agent.agent_id] = {
                'type': agent.agent_type.value,
                'final_balance': float(agent.ftns_balance),
                'total_earnings': float(agent.total_earnings),
                'total_spending': float(agent.total_spending),
                'roi': roi,
                'reputation': agent.reputation_score,
                'contributions': agent.total_contributions
            }
        
        # Economic validation
        validation_results = {
            'wealth_distribution_fair': final_gini <= 0.7,  # Gini < 0.7 considered fair
            'quality_maintained': avg_quality >= 0.6,      # Quality > 60%
            'price_stable': price_stability >= 0.8,        # Price stability > 80%
            'high_participation': avg_activity >= 0.5,     # >50% daily activity
        }
        
        overall_success = all(validation_results.values())
        
        return {
            'simulation_parameters': {
                'num_agents': self.num_agents,
                'simulation_days': self.simulation_days,
                'initial_supply': float(self.initial_token_supply)
            },
            'key_metrics': {
                'final_token_price': float(self.token_price),
                'final_gini_coefficient': final_gini,
                'average_quality': avg_quality,
                'price_stability': price_stability,
                'average_activity_rate': avg_activity,
                'total_transactions': len(self.transaction_log)
            },
            'validation_results': validation_results,
            'overall_success': overall_success,
            'daily_metrics': df.to_dict('records'),
            'agent_analysis': agent_analysis,
            'transaction_log': self.transaction_log
        }

def run_stress_test_scenarios() -> Dict[str, Any]:
    """Run predefined stress test scenarios"""
    
    scenarios = [
        {
            'name': 'Normal Growth',
            'description': 'Baseline scenario with normal market conditions',
            'stress_events': []
        },
        {
            'name': 'Market Volatility',
            'description': 'High volatility market conditions',
            'stress_events': [
                {'start_day': 5, 'end_day': 10, 'condition': 'volatility_spike'},
                {'start_day': 20, 'end_day': 25, 'condition': 'volatility_spike'}
            ]
        },
        {
            'name': 'Economic Shock',
            'description': 'Bear market followed by compute shortage',
            'stress_events': [
                {'start_day': 3, 'end_day': 15, 'condition': 'bear_market'},
                {'start_day': 16, 'end_day': 25, 'condition': 'compute_shortage'}
            ]
        },
        {
            'name': 'Data Oversupply',
            'description': 'Market flooded with low-quality data',
            'stress_events': [
                {'start_day': 8, 'end_day': 22, 'condition': 'data_flood'}
            ]
        }
    ]
    
    all_results = {}
    
    print("🧪 Running FTNS Stress Test Scenarios...")
    print("=" * 50)
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"📝 {scenario['description']}")
        
        # Create simulation
        sim = FTNSEconomicSimulation(
            num_agents=40,
            simulation_days=30
        )
        
        # Run with stress events
        results = sim.run_simulation(stress_scenarios=scenario['stress_events'])
        
        # Store results
        all_results[scenario['name']] = results
        
        # Print summary
        metrics = results['key_metrics']
        validation = results['validation_results']
        
        print(f"✅ Final Price: ${metrics['final_token_price']:.3f}")
        print(f"✅ Gini Coefficient: {metrics['final_gini_coefficient']:.3f}")
        print(f"✅ Avg Quality: {metrics['average_quality']:.3f}")
        print(f"✅ Price Stability: {metrics['price_stability']:.3f}")
        print(f"✅ Success Rate: {sum(validation.values())}/{len(validation)} criteria met")
    
    return all_results

def create_analysis_charts(results: Dict[str, Any]):
    """Create comprehensive analysis charts"""
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Using matplotlib fallback.")
        return _create_matplotlib_charts(results)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Token Price Over Time',
            'Wealth Distribution (Gini Coefficient)', 
            'Quality Metrics',
            'Agent Type Performance'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Data preparation
    daily_data = pd.DataFrame(results['daily_metrics'])
    
    # 1. Token price chart
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['token_price'],
            mode='lines',
            name='Token Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 2. Gini coefficient
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['gini_coefficient'],
            mode='lines',
            name='Gini Coefficient',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )
    
    # Add fairness threshold line
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="Fairness Threshold", row=1, col=2)
    
    # 3. Quality metrics
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['avg_contribution_quality'],
            mode='lines',
            name='Avg Quality',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # 4. Agent type performance (final balances)
    agent_data = results['agent_analysis']
    type_balances = {}
    for agent_id, data in agent_data.items():
        agent_type = data['type']
        if agent_type not in type_balances:
            type_balances[agent_type] = []
        type_balances[agent_type].append(data['final_balance'])
    
    types = list(type_balances.keys())
    avg_balances = [np.mean(type_balances[t]) for t in types]
    
    fig.add_trace(
        go.Bar(
            x=types,
            y=avg_balances,
            name='Avg Final Balance',
            marker_color='purple'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="FTNS Tokenomics Simulation Analysis",
        showlegend=True,
        height=800
    )
    
    # Update axes
    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_xaxes(title_text="Day", row=1, col=2)
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_xaxes(title_text="Agent Type", row=2, col=2)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Gini Coefficient", row=1, col=2)
    fig.update_yaxes(title_text="Quality Score", row=2, col=1)
    fig.update_yaxes(title_text="FTNS Balance", row=2, col=2)
    
    return fig

def _create_matplotlib_charts(results: Dict[str, Any]):
    """Fallback matplotlib charts"""
    
    daily_data = pd.DataFrame(results['daily_metrics'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Token price
    ax1.plot(daily_data['day'], daily_data['token_price'], 'b-', linewidth=2)
    ax1.set_title('Token Price Over Time')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Gini coefficient
    ax2.plot(daily_data['day'], daily_data['gini_coefficient'], 'r-', linewidth=2)
    ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Fairness Threshold')
    ax2.set_title('Wealth Distribution (Gini Coefficient)')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Gini Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Quality metrics
    ax3.plot(daily_data['day'], daily_data['avg_contribution_quality'], 'g-', linewidth=2)
    ax3.set_title('Average Contribution Quality')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Quality Score')
    ax3.grid(True, alpha=0.3)
    
    # Agent type performance
    agent_data = results['agent_analysis']
    type_balances = {}
    for agent_id, data in agent_data.items():
        agent_type = data['type']
        if agent_type not in type_balances:
            type_balances[agent_type] = []
        type_balances[agent_type].append(data['final_balance'])
    
    types = list(type_balances.keys())
    avg_balances = [np.mean(type_balances[t]) for t in types]
    
    ax4.bar(types, avg_balances, color='purple', alpha=0.7)
    ax4.set_title('Average Final Balance by Agent Type')
    ax4.set_xlabel('Agent Type')
    ax4.set_ylabel('FTNS Balance')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Run stress test scenarios
    stress_results = run_stress_test_scenarios()
    
    # Analyze one scenario in detail
    normal_scenario = stress_results['Normal Growth']
    
    print("\n" + "="*60)
    print("📈 DETAILED ANALYSIS - Normal Growth Scenario")
    print("="*60)
    
    metrics = normal_scenario['key_metrics']
    validation = normal_scenario['validation_results']
    
    print(f"🎯 Final Token Price: ${metrics['final_token_price']:.3f}")
    print(f"🎯 Gini Coefficient: {metrics['final_gini_coefficient']:.3f}")
    print(f"🎯 Average Quality: {metrics['average_quality']:.3f}")
    print(f"🎯 Price Stability: {metrics['price_stability']:.1%}")
    print(f"🎯 Activity Rate: {metrics['average_activity_rate']:.1%}")
    
    print(f"\n✅ Validation Results:")
    for criterion, passed in validation.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion}: {status}")
    
    print(f"\n🏆 Overall Success: {'✅ PASS' if normal_scenario['overall_success'] else '❌ FAIL'}")
    
    # Create charts
    try:
        chart = create_analysis_charts(normal_scenario)
        if PLOTLY_AVAILABLE:
            chart.show()
        else:
            plt.show()
    except Exception as e:
        print(f"Error creating charts: {e}")
    
    # Save results
    output_file = Path(__file__).parent / "tokenomics_simulation_results.json"
    with open(output_file, 'w') as f:
        # Convert Decimal objects to float for JSON serialization
        import json
        json.dump(stress_results, f, indent=2, default=float)
    
    print(f"\n💾 Results saved to: {output_file}")