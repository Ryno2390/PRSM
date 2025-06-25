#!/usr/bin/env python3
"""
PRSM Cost Optimization & ROI Calculator

A comprehensive tool for analyzing costs, calculating ROI, and optimizing 
infrastructure spending for PRSM deployments.

This module provides detailed financial analysis capabilities including:
- Infrastructure cost modeling
- AI model usage cost analysis
- Performance vs cost optimization
- ROI calculations with various scenarios
- Cost prediction and optimization recommendations

Author: PRSM Platform Team
"""

import json
import math
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np


class DeploymentTier(Enum):
    """Deployment tier configurations"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class ProviderType(Enum):
    """AI service provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    SELF_HOSTED = "self_hosted"
    PRSM_NETWORK = "prsm_network"


@dataclass
class InfrastructureCosts:
    """Infrastructure cost configuration"""
    # Compute costs (per hour)
    cpu_cost_per_core_hour: float = 0.045  # AWS equivalent
    memory_cost_per_gb_hour: float = 0.012
    gpu_cost_per_hour: float = 2.50  # V100 equivalent
    
    # Storage costs (per GB per month)
    ssd_storage_cost: float = 0.08
    standard_storage_cost: float = 0.023
    
    # Network costs (per GB)
    data_transfer_out_cost: float = 0.09
    data_transfer_in_cost: float = 0.0
    
    # Database costs (per hour)
    database_cost_per_hour: float = 0.35
    cache_cost_per_hour: float = 0.15


@dataclass
class AIModelCosts:
    """AI model usage costs"""
    # OpenAI pricing (per 1k tokens)
    openai_gpt4_input: float = 0.01
    openai_gpt4_output: float = 0.03
    openai_gpt35_input: float = 0.0015
    openai_gpt35_output: float = 0.002
    
    # Anthropic pricing (per 1k tokens)
    anthropic_claude_input: float = 0.008
    anthropic_claude_output: float = 0.024
    
    # Hugging Face pricing (per hour)
    huggingface_inference_hour: float = 0.6
    
    # Self-hosted costs (per 1k tokens)
    self_hosted_inference_cost: float = 0.001
    
    # PRSM Network token costs
    prsm_token_cost_usd: float = 0.05  # per FTNS token


@dataclass
class UsageMetrics:
    """Usage pattern metrics"""
    daily_requests: int = 1000
    avg_input_tokens: int = 500
    avg_output_tokens: int = 200
    peak_concurrency: int = 50
    
    # Resource utilization
    cpu_cores: int = 8
    memory_gb: int = 32
    gpu_count: int = 1
    storage_gb: int = 100
    
    # Data transfer (GB per month)
    monthly_data_out: float = 50.0
    monthly_data_in: float = 20.0


@dataclass
class BusinessMetrics:
    """Business value metrics"""
    revenue_per_request: float = 0.25
    cost_per_engineer_hour: float = 150.0
    hours_saved_per_day: float = 2.0
    error_reduction_percentage: float = 15.0
    time_to_market_improvement_days: int = 30


class ROICalculator:
    """Comprehensive ROI and cost optimization calculator"""
    
    def __init__(self, infrastructure_costs: InfrastructureCosts = None,
                 ai_costs: AIModelCosts = None):
        self.infra_costs = infrastructure_costs or InfrastructureCosts()
        self.ai_costs = ai_costs or AIModelCosts()
        
    def calculate_monthly_infrastructure_cost(self, usage: UsageMetrics,
                                           tier: DeploymentTier) -> Dict[str, float]:
        """Calculate monthly infrastructure costs"""
        hours_per_month = 24 * 30
        
        # Base compute costs
        cpu_cost = usage.cpu_cores * self.infra_costs.cpu_cost_per_core_hour * hours_per_month
        memory_cost = usage.memory_gb * self.infra_costs.memory_cost_per_gb_hour * hours_per_month
        gpu_cost = usage.gpu_count * self.infra_costs.gpu_cost_per_hour * hours_per_month
        
        # Storage costs
        storage_cost = usage.storage_gb * self.infra_costs.ssd_storage_cost
        
        # Database and cache
        database_cost = self.infra_costs.database_cost_per_hour * hours_per_month
        cache_cost = self.infra_costs.cache_cost_per_hour * hours_per_month
        
        # Network costs
        network_cost = (usage.monthly_data_out * self.infra_costs.data_transfer_out_cost +
                       usage.monthly_data_in * self.infra_costs.data_transfer_in_cost)
        
        # Tier-based multipliers
        tier_multipliers = {
            DeploymentTier.DEVELOPMENT: 0.5,
            DeploymentTier.STAGING: 0.7,
            DeploymentTier.PRODUCTION: 1.0,
            DeploymentTier.ENTERPRISE: 1.5
        }
        
        multiplier = tier_multipliers[tier]
        
        costs = {
            "cpu": cpu_cost * multiplier,
            "memory": memory_cost * multiplier,
            "gpu": gpu_cost * multiplier,
            "storage": storage_cost * multiplier,
            "database": database_cost * multiplier,
            "cache": cache_cost * multiplier,
            "network": network_cost * multiplier
        }
        
        costs["total"] = sum(costs.values())
        return costs
    
    def calculate_ai_model_costs(self, usage: UsageMetrics,
                               provider_mix: Dict[ProviderType, float]) -> Dict[str, float]:
        """Calculate AI model usage costs based on provider mix"""
        monthly_requests = usage.daily_requests * 30
        total_input_tokens = monthly_requests * usage.avg_input_tokens
        total_output_tokens = monthly_requests * usage.avg_output_tokens
        
        costs = {}
        
        for provider, percentage in provider_mix.items():
            if percentage == 0:
                continue
                
            provider_requests = monthly_requests * percentage
            provider_input_tokens = total_input_tokens * percentage / 1000  # Convert to thousands
            provider_output_tokens = total_output_tokens * percentage / 1000
            
            if provider == ProviderType.OPENAI:
                # Assume 70% GPT-4, 30% GPT-3.5
                gpt4_cost = (provider_input_tokens * 0.7 * self.ai_costs.openai_gpt4_input +
                            provider_output_tokens * 0.7 * self.ai_costs.openai_gpt4_output)
                gpt35_cost = (provider_input_tokens * 0.3 * self.ai_costs.openai_gpt35_input +
                             provider_output_tokens * 0.3 * self.ai_costs.openai_gpt35_output)
                costs[f"openai"] = gpt4_cost + gpt35_cost
                
            elif provider == ProviderType.ANTHROPIC:
                costs["anthropic"] = (provider_input_tokens * self.ai_costs.anthropic_claude_input +
                                    provider_output_tokens * self.ai_costs.anthropic_claude_output)
                
            elif provider == ProviderType.HUGGINGFACE:
                # Estimate hours based on requests and processing time
                hours_needed = provider_requests * 0.002  # 2ms average per request
                costs["huggingface"] = hours_needed * self.ai_costs.huggingface_inference_hour
                
            elif provider == ProviderType.SELF_HOSTED:
                total_tokens = provider_input_tokens + provider_output_tokens
                costs["self_hosted"] = total_tokens * self.ai_costs.self_hosted_inference_cost
                
            elif provider == ProviderType.PRSM_NETWORK:
                # FTNS token costs
                estimated_tokens_needed = provider_requests * 0.1  # 0.1 FTNS per request
                costs["prsm_network"] = estimated_tokens_needed * self.ai_costs.prsm_token_cost_usd
        
        costs["total"] = sum(costs.values())
        return costs
    
    def calculate_business_value(self, usage: UsageMetrics, business: BusinessMetrics) -> Dict[str, float]:
        """Calculate business value generated"""
        monthly_requests = usage.daily_requests * 30
        
        # Direct revenue
        direct_revenue = monthly_requests * business.revenue_per_request
        
        # Cost savings from automation
        monthly_hours_saved = business.hours_saved_per_day * 30
        automation_savings = monthly_hours_saved * business.cost_per_engineer_hour
        
        # Error reduction savings (estimate 5% of revenue)
        error_reduction_savings = direct_revenue * (business.error_reduction_percentage / 100) * 0.05
        
        # Time to market improvement (one-time per month)
        ttm_value = business.time_to_market_improvement_days * business.cost_per_engineer_hour * 8  # 8 hours per day
        
        return {
            "direct_revenue": direct_revenue,
            "automation_savings": automation_savings,
            "error_reduction_savings": error_reduction_savings,
            "time_to_market_value": ttm_value,
            "total_value": direct_revenue + automation_savings + error_reduction_savings + ttm_value
        }
    
    def calculate_roi_scenarios(self, usage: UsageMetrics, business: BusinessMetrics,
                              tier: DeploymentTier) -> Dict[str, Any]:
        """Calculate ROI for different provider scenarios"""
        scenarios = {
            "all_openai": {ProviderType.OPENAI: 1.0},
            "all_anthropic": {ProviderType.ANTHROPIC: 1.0},
            "mixed_commercial": {
                ProviderType.OPENAI: 0.5,
                ProviderType.ANTHROPIC: 0.3,
                ProviderType.HUGGINGFACE: 0.2
            },
            "hybrid_self_hosted": {
                ProviderType.OPENAI: 0.3,
                ProviderType.SELF_HOSTED: 0.5,
                ProviderType.PRSM_NETWORK: 0.2
            },
            "prsm_optimized": {
                ProviderType.PRSM_NETWORK: 0.6,
                ProviderType.SELF_HOSTED: 0.3,
                ProviderType.OPENAI: 0.1
            }
        }
        
        results = {}
        business_value = self.calculate_business_value(usage, business)
        
        for scenario_name, provider_mix in scenarios.items():
            infra_costs = self.calculate_monthly_infrastructure_cost(usage, tier)
            ai_costs = self.calculate_ai_model_costs(usage, provider_mix)
            
            total_cost = infra_costs["total"] + ai_costs["total"]
            total_value = business_value["total_value"]
            
            roi = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            payback_months = total_cost / (total_value - total_cost) if total_value > total_cost else float('inf')
            
            results[scenario_name] = {
                "infrastructure_cost": infra_costs["total"],
                "ai_model_cost": ai_costs["total"],
                "total_cost": total_cost,
                "total_value": total_value,
                "monthly_profit": total_value - total_cost,
                "roi_percentage": roi,
                "payback_months": payback_months,
                "cost_breakdown": {**infra_costs, **ai_costs},
                "value_breakdown": business_value
            }
        
        return results
    
    def optimization_recommendations(self, scenarios: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Find best ROI scenario
        best_roi_scenario = max(scenarios.items(), key=lambda x: x[1]["roi_percentage"])
        recommendations.append(
            f"🏆 Best ROI scenario: '{best_roi_scenario[0]}' with {best_roi_scenario[1]['roi_percentage']:.1f}% ROI"
        )
        
        # Find lowest cost scenario
        lowest_cost_scenario = min(scenarios.items(), key=lambda x: x[1]["total_cost"])
        recommendations.append(
            f"💰 Lowest cost scenario: '{lowest_cost_scenario[0]}' at ${lowest_cost_scenario[1]['total_cost']:.2f}/month"
        )
        
        # Analyze cost drivers
        for scenario_name, data in scenarios.items():
            ai_cost_ratio = data["ai_model_cost"] / data["total_cost"]
            if ai_cost_ratio > 0.7:
                recommendations.append(
                    f"⚠️  '{scenario_name}': AI costs are {ai_cost_ratio:.1%} of total - consider self-hosting"
                )
        
        # PRSM Network benefits
        if "prsm_optimized" in scenarios:
            prsm_data = scenarios["prsm_optimized"]
            recommendations.append(
                f"🌐 PRSM Network optimized approach offers ${prsm_data['monthly_profit']:.2f}/month profit"
            )
        
        return recommendations
    
    def export_analysis(self, scenarios: Dict[str, Any], filename: str = None) -> str:
        """Export analysis to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prsm_cost_analysis_{timestamp}.json"
        
        analysis_data = {
            "analysis_date": datetime.now().isoformat(),
            "scenarios": scenarios,
            "recommendations": self.optimization_recommendations(scenarios),
            "summary": {
                "best_roi_scenario": max(scenarios.items(), key=lambda x: x[1]["roi_percentage"])[0],
                "lowest_cost_scenario": min(scenarios.items(), key=lambda x: x[1]["total_cost"])[0],
                "scenario_count": len(scenarios)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        return filename


class InteractiveCostAnalyzer:
    """Interactive command-line interface for cost analysis"""
    
    def __init__(self):
        self.calculator = ROICalculator()
        self.current_usage = UsageMetrics()
        self.current_business = BusinessMetrics()
        
    def run_interactive_session(self):
        """Run interactive cost analysis session"""
        print("🏦 PRSM Cost Optimization & ROI Calculator")
        print("=" * 50)
        print()
        
        while True:
            print("\n📊 Available Commands:")
            print("1. configure_usage - Set usage parameters")
            print("2. configure_business - Set business metrics")
            print("3. analyze_scenarios - Run ROI analysis")
            print("4. compare_tiers - Compare deployment tiers")
            print("5. optimize_costs - Get optimization recommendations")
            print("6. export_report - Export analysis report")
            print("7. show_current - Show current configuration")
            print("8. quit - Exit analyzer")
            
            choice = input("\n📝 Enter command (1-8): ").strip()
            
            if choice == "1":
                self._configure_usage()
            elif choice == "2":
                self._configure_business()
            elif choice == "3":
                self._analyze_scenarios()
            elif choice == "4":
                self._compare_tiers()
            elif choice == "5":
                self._optimize_costs()
            elif choice == "6":
                self._export_report()
            elif choice == "7":
                self._show_current_config()
            elif choice == "8":
                print("👋 Thank you for using PRSM Cost Analyzer!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
    
    def _configure_usage(self):
        """Configure usage parameters"""
        print("\n⚙️ Configure Usage Parameters")
        print("-" * 30)
        
        try:
            self.current_usage.daily_requests = int(input(f"Daily requests [{self.current_usage.daily_requests}]: ") or self.current_usage.daily_requests)
            self.current_usage.avg_input_tokens = int(input(f"Average input tokens [{self.current_usage.avg_input_tokens}]: ") or self.current_usage.avg_input_tokens)
            self.current_usage.avg_output_tokens = int(input(f"Average output tokens [{self.current_usage.avg_output_tokens}]: ") or self.current_usage.avg_output_tokens)
            self.current_usage.peak_concurrency = int(input(f"Peak concurrency [{self.current_usage.peak_concurrency}]: ") or self.current_usage.peak_concurrency)
            self.current_usage.cpu_cores = int(input(f"CPU cores [{self.current_usage.cpu_cores}]: ") or self.current_usage.cpu_cores)
            self.current_usage.memory_gb = int(input(f"Memory (GB) [{self.current_usage.memory_gb}]: ") or self.current_usage.memory_gb)
            self.current_usage.gpu_count = int(input(f"GPU count [{self.current_usage.gpu_count}]: ") or self.current_usage.gpu_count)
            
            print("✅ Usage parameters updated!")
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
    
    def _configure_business(self):
        """Configure business metrics"""
        print("\n💼 Configure Business Metrics")
        print("-" * 30)
        
        try:
            self.current_business.revenue_per_request = float(input(f"Revenue per request [$] [{self.current_business.revenue_per_request}]: ") or self.current_business.revenue_per_request)
            self.current_business.cost_per_engineer_hour = float(input(f"Cost per engineer hour [$] [{self.current_business.cost_per_engineer_hour}]: ") or self.current_business.cost_per_engineer_hour)
            self.current_business.hours_saved_per_day = float(input(f"Hours saved per day [{self.current_business.hours_saved_per_day}]: ") or self.current_business.hours_saved_per_day)
            self.current_business.error_reduction_percentage = float(input(f"Error reduction [%] [{self.current_business.error_reduction_percentage}]: ") or self.current_business.error_reduction_percentage)
            
            print("✅ Business metrics updated!")
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
    
    def _analyze_scenarios(self):
        """Analyze ROI scenarios"""
        print("\n📈 ROI Scenario Analysis")
        print("-" * 25)
        
        # Select deployment tier
        print("\nSelect deployment tier:")
        tiers = list(DeploymentTier)
        for i, tier in enumerate(tiers, 1):
            print(f"{i}. {tier.value.title()}")
        
        try:
            tier_choice = int(input("Enter tier (1-4): ")) - 1
            selected_tier = tiers[tier_choice]
        except (ValueError, IndexError):
            print("❌ Invalid tier selection. Using Production tier.")
            selected_tier = DeploymentTier.PRODUCTION
        
        scenarios = self.calculator.calculate_roi_scenarios(
            self.current_usage, self.current_business, selected_tier
        )
        
        print(f"\n📊 ROI Analysis Results - {selected_tier.value.title()} Tier")
        print("=" * 60)
        
        for scenario_name, data in scenarios.items():
            print(f"\n🔍 {scenario_name.replace('_', ' ').title()}")
            print(f"   💰 Total Cost: ${data['total_cost']:.2f}/month")
            print(f"   📈 Total Value: ${data['total_value']:.2f}/month")
            print(f"   💵 Monthly Profit: ${data['monthly_profit']:.2f}")
            print(f"   📊 ROI: {data['roi_percentage']:.1f}%")
            if data['payback_months'] != float('inf'):
                print(f"   ⏰ Payback: {data['payback_months']:.1f} months")
            else:
                print(f"   ⏰ Payback: Never (negative ROI)")
        
        # Show recommendations
        recommendations = self.calculator.optimization_recommendations(scenarios)
        print("\n💡 Optimization Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
    
    def _compare_tiers(self):
        """Compare different deployment tiers"""
        print("\n🏗️ Deployment Tier Comparison")
        print("-" * 30)
        
        tier_results = {}
        
        for tier in DeploymentTier:
            scenarios = self.calculator.calculate_roi_scenarios(
                self.current_usage, self.current_business, tier
            )
            # Use the best ROI scenario for comparison
            best_scenario = max(scenarios.items(), key=lambda x: x[1]["roi_percentage"])
            tier_results[tier.value] = best_scenario[1]
        
        print(f"\n📊 Best ROI Scenario by Tier")
        print("=" * 40)
        
        for tier_name, data in tier_results.items():
            print(f"\n🏷️ {tier_name.title()}")
            print(f"   💰 Monthly Cost: ${data['total_cost']:.2f}")
            print(f"   📈 Monthly Value: ${data['total_value']:.2f}")
            print(f"   💵 Monthly Profit: ${data['monthly_profit']:.2f}")
            print(f"   📊 ROI: {data['roi_percentage']:.1f}%")
    
    def _optimize_costs(self):
        """Provide cost optimization recommendations"""
        print("\n🎯 Cost Optimization Analysis")
        print("-" * 30)
        
        scenarios = self.calculator.calculate_roi_scenarios(
            self.current_usage, self.current_business, DeploymentTier.PRODUCTION
        )
        
        recommendations = self.calculator.optimization_recommendations(scenarios)
        
        print("\n💡 Detailed Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Detailed cost breakdown for best scenario
        best_scenario_name = max(scenarios.items(), key=lambda x: x[1]["roi_percentage"])[0]
        best_data = scenarios[best_scenario_name]
        
        print(f"\n📋 Detailed Cost Breakdown - {best_scenario_name.replace('_', ' ').title()}")
        print("=" * 60)
        
        print("Infrastructure Costs:")
        for item, cost in best_data["cost_breakdown"].items():
            if item != "total" and cost > 0:
                print(f"   {item.title()}: ${cost:.2f}")
        
        print("\nValue Sources:")
        for item, value in best_data["value_breakdown"].items():
            if item != "total_value" and value > 0:
                print(f"   {item.replace('_', ' ').title()}: ${value:.2f}")
    
    def _export_report(self):
        """Export analysis report"""
        print("\n📄 Export Analysis Report")
        print("-" * 25)
        
        scenarios = self.calculator.calculate_roi_scenarios(
            self.current_usage, self.current_business, DeploymentTier.PRODUCTION
        )
        
        filename = self.calculator.export_analysis(scenarios)
        print(f"✅ Report exported to: {filename}")
        
        # Also create a summary CSV
        csv_filename = filename.replace('.json', '_summary.csv')
        summary_data = []
        
        for scenario_name, data in scenarios.items():
            summary_data.append({
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Total Cost': data['total_cost'],
                'Total Value': data['total_value'],
                'Monthly Profit': data['monthly_profit'],
                'ROI Percentage': data['roi_percentage'],
                'Payback Months': data['payback_months'] if data['payback_months'] != float('inf') else 'Never'
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_filename, index=False)
        print(f"✅ Summary CSV exported to: {csv_filename}")
    
    def _show_current_config(self):
        """Show current configuration"""
        print("\n⚙️ Current Configuration")
        print("-" * 25)
        
        print("Usage Metrics:")
        for field, value in asdict(self.current_usage).items():
            print(f"   {field.replace('_', ' ').title()}: {value}")
        
        print("\nBusiness Metrics:")
        for field, value in asdict(self.current_business).items():
            print(f"   {field.replace('_', ' ').title()}: {value}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PRSM Cost Optimization & ROI Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run interactive analyzer
    python roi_calculator.py --interactive
    
    # Quick analysis with custom parameters
    python roi_calculator.py --daily-requests 5000 --revenue-per-request 0.50
    
    # Export analysis for production tier
    python roi_calculator.py --tier production --export analysis_report.json
        """
    )
    
    parser.add_argument("--interactive", action="store_true",
                      help="Run interactive cost analyzer")
    parser.add_argument("--daily-requests", type=int, default=1000,
                      help="Daily request volume")
    parser.add_argument("--revenue-per-request", type=float, default=0.25,
                      help="Revenue generated per request")
    parser.add_argument("--tier", choices=[t.value for t in DeploymentTier],
                      default="production", help="Deployment tier")
    parser.add_argument("--export", type=str,
                      help="Export analysis to file")
    
    args = parser.parse_args()
    
    if args.interactive:
        analyzer = InteractiveCostAnalyzer()
        analyzer.run_interactive_session()
    else:
        # Quick analysis mode
        calculator = ROICalculator()
        
        # Configure usage and business metrics
        usage = UsageMetrics(daily_requests=args.daily_requests)
        business = BusinessMetrics(revenue_per_request=args.revenue_per_request)
        tier = DeploymentTier(args.tier)
        
        # Calculate scenarios
        scenarios = calculator.calculate_roi_scenarios(usage, business, tier)
        
        # Display results
        print(f"🏦 PRSM ROI Analysis - {tier.value.title()} Tier")
        print("=" * 50)
        
        for scenario_name, data in scenarios.items():
            print(f"\n📊 {scenario_name.replace('_', ' ').title()}")
            print(f"   Cost: ${data['total_cost']:.2f}/month")
            print(f"   Value: ${data['total_value']:.2f}/month")
            print(f"   ROI: {data['roi_percentage']:.1f}%")
        
        # Show recommendations
        recommendations = calculator.optimization_recommendations(scenarios)
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
        
        # Export if requested
        if args.export:
            filename = calculator.export_analysis(scenarios, args.export)
            print(f"\n✅ Analysis exported to: {filename}")


if __name__ == "__main__":
    main()