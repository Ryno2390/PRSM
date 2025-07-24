#!/usr/bin/env python3
"""
PRSM Phase 7 Analytics Example
=============================

Demonstrates how to use PRSM's advanced analytics and dashboard features
for comprehensive business intelligence and system monitoring.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from prsm.analytics.dashboard_manager import DashboardManager
from prsm.enterprise.global_infrastructure import GlobalInfrastructure
from prsm.core.config import get_settings_safe


class AnalyticsExample:
    """Example class demonstrating Phase 7 analytics features"""
    
    def __init__(self):
        self.dashboard_manager = DashboardManager()
        self.global_infrastructure = GlobalInfrastructure()
        
    async def initialize(self):
        """Initialize analytics components"""
        print("🚀 Initializing PRSM Analytics System...")
        
        # Initialize components with safe configuration
        settings = get_settings_safe()
        if settings:
            print("✅ Configuration loaded successfully")
        else:
            print("⚠️  Using fallback configuration")
            
        await self.dashboard_manager.initialize()
        await self.global_infrastructure.initialize()
        
        print("✅ Analytics system initialized")

    async def create_executive_dashboard(self) -> str:
        """Create comprehensive executive dashboard"""
        print("\n📊 Creating Executive Dashboard...")
        
        dashboard_config = {
            'name': 'PRSM Executive Dashboard',
            'type': 'executive',
            'description': 'High-level KPIs and strategic metrics for executives',
            'widgets': [
                {
                    'type': 'metric_card',
                    'title': 'Active Users',
                    'data_source': 'user_metrics',
                    'position': {'row': 0, 'col': 0, 'width': 3, 'height': 2},
                    'config': {
                        'metric': 'active_users_24h',
                        'format': 'number',
                        'comparison': 'previous_day'
                    }
                },
                {
                    'type': 'metric_card',
                    'title': 'Query Success Rate',
                    'data_source': 'query_metrics',
                    'position': {'row': 0, 'col': 3, 'width': 3, 'height': 2},
                    'config': {
                        'metric': 'success_rate',
                        'format': 'percentage',
                        'target': 0.95
                    }
                },
                {
                    'type': 'line_chart',
                    'title': 'Daily Query Volume',
                    'data_source': 'query_metrics',
                    'position': {'row': 2, 'col': 0, 'width': 6, 'height': 4},
                    'config': {
                        'time_range': '7d',
                        'metrics': ['total_queries', 'successful_queries'],
                        'groupBy': 'day'
                    }
                },
                {
                    'type': 'donut_chart',
                    'title': 'Revenue by Source',
                    'data_source': 'revenue_metrics',
                    'position': {'row': 2, 'col': 6, 'width': 3, 'height': 4},
                    'config': {
                        'metric': 'revenue_by_source',
                        'time_range': '30d'
                    }
                },
                {
                    'type': 'table',
                    'title': 'Top Performing Integrations',
                    'data_source': 'marketplace_metrics',
                    'position': {'row': 6, 'col': 0, 'width': 9, 'height': 3},
                    'config': {
                        'columns': ['name', 'usage_count', 'revenue', 'rating'],
                        'sort_by': 'usage_count',
                        'limit': 10
                    }
                }
            ],
            'refresh_interval': 300,  # 5 minutes
            'auto_refresh': True
        }
        
        dashboard_id = await self.dashboard_manager.create_dashboard(dashboard_config)
        print(f"✅ Executive dashboard created: {dashboard_id}")
        
        return dashboard_id

    async def create_operational_dashboard(self) -> str:
        """Create operational dashboard for system monitoring"""
        print("\n🔧 Creating Operational Dashboard...")
        
        dashboard_config = {
            'name': 'PRSM Operations Dashboard',
            'type': 'operational',
            'description': 'Real-time system health and performance monitoring',
            'widgets': [
                {
                    'type': 'gauge',
                    'title': 'System Health Score',
                    'data_source': 'health_metrics',
                    'position': {'row': 0, 'col': 0, 'width': 3, 'height': 3},
                    'config': {
                        'metric': 'overall_health_score',
                        'min': 0,
                        'max': 100,
                        'thresholds': [
                            {'value': 70, 'color': 'red'},
                            {'value': 85, 'color': 'yellow'},
                            {'value': 95, 'color': 'green'}
                        ]
                    }
                },
                {
                    'type': 'real_time_chart',
                    'title': 'Response Time',
                    'data_source': 'performance_metrics',
                    'position': {'row': 0, 'col': 3, 'width': 6, 'height': 3},
                    'config': {
                        'metric': 'average_response_time',
                        'time_window': '1h',
                        'update_interval': 10,
                        'alert_threshold': 2000  # 2 seconds
                    }
                },
                {
                    'type': 'heatmap',
                    'title': 'Regional Load Distribution',
                    'data_source': 'infrastructure_metrics',
                    'position': {'row': 3, 'col': 0, 'width': 9, 'height': 4},
                    'config': {
                        'x_axis': 'region',
                        'y_axis': 'time',
                        'metric': 'cpu_utilization',
                        'time_range': '24h'
                    }
                },
                {
                    'type': 'alert_panel',
                    'title': 'Active Alerts',
                    'data_source': 'alert_metrics',
                    'position': {'row': 7, 'col': 0, 'width': 9, 'height': 2},
                    'config': {
                        'severity_filter': ['critical', 'warning'],
                        'max_alerts': 20,
                        'auto_acknowledge': False
                    }
                }
            ],
            'refresh_interval': 30,  # 30 seconds
            'auto_refresh': True
        }
        
        dashboard_id = await self.dashboard_manager.create_dashboard(dashboard_config)
        print(f"✅ Operational dashboard created: {dashboard_id}")
        
        return dashboard_id

    async def generate_sample_data(self):
        """Generate sample data for dashboard demonstration"""
        print("\n📈 Generating sample analytics data...")
        
        # Sample user metrics
        user_data = {
            'active_users_24h': 1247,
            'new_users_today': 89,
            'retention_rate': 0.85,
            'user_growth_rate': 0.12
        }
        
        # Sample query metrics
        query_data = {
            'total_queries': 15634,
            'successful_queries': 15012,
            'failed_queries': 622,
            'success_rate': 0.96,
            'average_response_time': 18.4,
            'median_response_time': 14.2,
            'p95_response_time': 45.7
        }
        
        # Sample revenue metrics
        revenue_data = {
            'total_revenue': 125000.00,
            'revenue_by_source': {
                'subscriptions': 85000.00,
                'marketplace': 25000.00,
                'enterprise': 15000.00
            },
            'monthly_recurring_revenue': 45000.00,
            'customer_lifetime_value': 2500.00
        }
        
        # Sample infrastructure metrics
        infrastructure_data = {
            'total_nodes': 50,
            'active_nodes': 48,
            'average_cpu_utilization': 0.67,
            'average_memory_utilization': 0.71,
            'network_throughput': 1.5e9,  # bytes per second
            'disk_usage': 0.45
        }
        
        return {
            'user_metrics': user_data,
            'query_metrics': query_data,
            'revenue_metrics': revenue_data,
            'infrastructure_metrics': infrastructure_data,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def update_dashboard_data(self, dashboard_id: str):
        """Update dashboard with real-time data"""
        print(f"\n🔄 Updating dashboard data: {dashboard_id}")
        
        # Generate sample data
        sample_data = await self.generate_sample_data()
        
        # Update dashboard
        await self.dashboard_manager.update_dashboard_data(dashboard_id, sample_data)
        print("✅ Dashboard data updated successfully")

    async def get_analytics_insights(self) -> Dict[str, Any]:
        """Generate automated analytics insights"""
        print("\n🧠 Generating analytics insights...")
        
        # This would typically analyze real data to generate insights
        insights = {
            'performance_insights': [
                {
                    'type': 'performance_improvement',
                    'message': 'Response times improved by 23% over the last week',
                    'confidence': 0.95,
                    'recommendation': 'Consider increasing this optimization across all regions'
                },
                {
                    'type': 'resource_optimization',
                    'message': 'US-West region showing 15% lower utilization than optimal',
                    'confidence': 0.87,
                    'recommendation': 'Consider redistributing workload or scaling down resources'
                }
            ],
            'business_insights': [
                {
                    'type': 'user_behavior',
                    'message': 'Enterprise users showing 45% higher retention than expected',
                    'confidence': 0.92,
                    'recommendation': 'Focus marketing efforts on similar customer segments'
                },
                {
                    'type': 'revenue_opportunity',
                    'message': 'Marketplace revenue grew 34% month-over-month',
                    'confidence': 0.89,
                    'recommendation': 'Accelerate marketplace feature development'
                }
            ],
            'predictive_insights': [
                {
                    'type': 'demand_forecast',
                    'message': 'Query volume expected to increase 28% next month',
                    'confidence': 0.84,
                    'recommendation': 'Prepare additional infrastructure capacity'
                }
            ],
            'generated_at': datetime.utcnow().isoformat()
        }
        
        print("✅ Analytics insights generated")
        return insights

    async def create_custom_report(self, report_type: str = "weekly_summary") -> Dict[str, Any]:
        """Create custom analytics report"""
        print(f"\n📄 Creating {report_type} report...")
        
        # Sample report data
        report_data = {
            'report_type': report_type,
            'period': {
                'start': (datetime.utcnow() - timedelta(days=7)).isoformat(),
                'end': datetime.utcnow().isoformat()
            },
            'summary': {
                'total_users': 1247,
                'new_users': 89,
                'total_queries': 15634,
                'success_rate': 0.96,
                'revenue': 125000.00
            },
            'highlights': [
                "Query success rate exceeded 96% target",
                "User growth rate increased 12% week-over-week",
                "New marketplace integration contributed $5,000 in revenue"
            ],
            'recommendations': [
                "Investigate and address the 4% of failed queries",
                "Scale infrastructure to handle projected growth",
                "Develop more marketplace integrations based on success"
            ],
            'detailed_metrics': {
                'performance': {
                    'average_response_time': 18.4,
                    'p95_response_time': 45.7,
                    'uptime_percentage': 99.95
                },
                'usage': {
                    'peak_concurrent_users': 1289,
                    'total_processing_time': 245678.9,
                    'most_popular_features': [
                        'Deep Reasoning', 'Content Search', 'Analytics Dashboard'
                    ]
                }
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        print("✅ Custom report generated")
        return report_data

    async def monitor_real_time_metrics(self, duration_seconds: int = 60):
        """Monitor real-time system metrics"""
        print(f"\n⚡ Monitoring real-time metrics for {duration_seconds} seconds...")
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=duration_seconds)
        
        while datetime.utcnow() < end_time:
            # Get current system health
            health_status = await self.global_infrastructure.get_performance_metrics()
            
            print(f"🔍 {datetime.utcnow().strftime('%H:%M:%S')} - "
                  f"CPU: {health_status.get('cpu_usage', 0.0):.1%}, "
                  f"Memory: {health_status.get('memory_usage', 0.0):.1%}, "
                  f"Latency: {health_status.get('network_latency', 0.0):.1f}ms")
            
            await asyncio.sleep(5)  # Update every 5 seconds
        
        print("✅ Real-time monitoring completed")

    async def run_complete_analytics_demo(self):
        """Run complete analytics demonstration"""
        print("🎯 Starting Complete PRSM Analytics Demo")
        print("=" * 50)
        
        try:
            # Initialize system
            await self.initialize()
            
            # Create dashboards
            exec_dashboard_id = await self.create_executive_dashboard()
            ops_dashboard_id = await self.create_operational_dashboard()
            
            # Update with sample data
            await self.update_dashboard_data(exec_dashboard_id)
            await self.update_dashboard_data(ops_dashboard_id)
            
            # Generate insights
            insights = await self.get_analytics_insights()
            print(f"\n💡 Key Insights Generated:")
            for category, insight_list in insights.items():
                if category != 'generated_at':
                    print(f"  {category.replace('_', ' ').title()}:")
                    for insight in insight_list:
                        print(f"    • {insight['message']}")
            
            # Create custom report
            report = await self.create_custom_report()
            print(f"\n📊 Weekly Summary Report:")
            print(f"  • Total Users: {report['summary']['total_users']:,}")
            print(f"  • Success Rate: {report['summary']['success_rate']:.1%}")
            print(f"  • Revenue: ${report['summary']['revenue']:,.2f}")
            
            # Brief real-time monitoring
            await self.monitor_real_time_metrics(30)
            
            print("\n🎉 Analytics demo completed successfully!")
            print(f"📈 Executive Dashboard: {exec_dashboard_id}")
            print(f"🔧 Operations Dashboard: {ops_dashboard_id}")
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise


async def main():
    """Main function to run the analytics example"""
    example = AnalyticsExample()
    await example.run_complete_analytics_demo()


if __name__ == "__main__":
    # Run the complete analytics demonstration
    asyncio.run(main())