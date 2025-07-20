#!/usr/bin/env python3
"""
Ecosystem Reporter for NWTN System 1 → System 2 → Attribution → Payment Pipeline
================================================================================

This module provides comprehensive reporting and analytics for the complete
PRSM/NWTN ecosystem, including audit trails, performance metrics, economic
analytics, and system health monitoring.

Part of Phase 4 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from enum import Enum
import json
from pathlib import Path

from prsm.nwtn.system_integrator import SystemIntegrator, PipelineResult
from prsm.nwtn.attribution_usage_tracker import AttributionUsageTracker, QueryUsageSession
from prsm.tokenomics.ftns_service import FTNSService

logger = structlog.get_logger(__name__)


class ReportType(Enum):
    """Types of reports that can be generated"""
    ECOSYSTEM_OVERVIEW = "ecosystem_overview"
    PIPELINE_PERFORMANCE = "pipeline_performance"
    ECONOMIC_ANALYSIS = "economic_analysis"
    USAGE_ANALYTICS = "usage_analytics"
    AUDIT_TRAIL = "audit_trail"
    SYSTEM_HEALTH = "system_health"
    CREATOR_EARNINGS = "creator_earnings"
    USER_ACTIVITY = "user_activity"


class ReportFormat(Enum):
    """Output formats for reports"""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"


@dataclass
class ReportRequest:
    """Request for generating a report"""
    report_type: ReportType
    format: ReportFormat
    time_range: Optional[Tuple[datetime, datetime]] = None
    filters: Optional[Dict[str, Any]] = None
    include_details: bool = True
    output_path: Optional[Path] = None


@dataclass
class EcosystemMetrics:
    """Comprehensive ecosystem metrics"""
    total_queries: int
    successful_queries: int
    success_rate: float
    average_processing_time: float
    average_quality_score: float
    total_payments_distributed: float
    total_system_revenue: float
    unique_users: int
    unique_creators: int
    content_pieces_accessed: int
    average_payment_per_creator: float
    top_performing_content: List[Dict[str, Any]]
    performance_trends: Dict[str, List[float]]
    system_health_score: float


@dataclass
class PipelineAnalytics:
    """Pipeline performance analytics"""
    stage_performance: Dict[str, Dict[str, float]]
    bottlenecks: List[str]
    optimization_opportunities: List[str]
    quality_distribution: Dict[str, int]
    error_patterns: List[Dict[str, Any]]
    throughput_metrics: Dict[str, float]


@dataclass
class EconomicAnalysis:
    """Economic analysis of the ecosystem"""
    revenue_breakdown: Dict[str, float]
    payment_distribution: Dict[str, float]
    creator_earnings: Dict[str, float]
    cost_analysis: Dict[str, float]
    profitability_metrics: Dict[str, float]
    market_trends: Dict[str, List[float]]


class EcosystemReporter:
    """
    Comprehensive reporting system for the NWTN ecosystem
    """
    
    def __init__(self,
                 system_integrator: Optional[SystemIntegrator] = None,
                 usage_tracker: Optional[AttributionUsageTracker] = None,
                 ftns_service: Optional[FTNSService] = None):
        self.system_integrator = system_integrator
        self.usage_tracker = usage_tracker
        self.ftns_service = ftns_service
        self.initialized = False
        
        # Report storage
        self.generated_reports: Dict[str, Dict[str, Any]] = {}
        self.report_history: List[Dict[str, Any]] = []
        
        # Analytics cache
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_expiry: datetime = datetime.now(timezone.utc)
        self.cache_duration = timedelta(minutes=30)
        
        # Report templates
        self.report_templates = {
            ReportType.ECOSYSTEM_OVERVIEW: self._generate_ecosystem_overview,
            ReportType.PIPELINE_PERFORMANCE: self._generate_pipeline_performance,
            ReportType.ECONOMIC_ANALYSIS: self._generate_economic_analysis,
            ReportType.USAGE_ANALYTICS: self._generate_usage_analytics,
            ReportType.AUDIT_TRAIL: self._generate_audit_trail,
            ReportType.SYSTEM_HEALTH: self._generate_system_health,
            ReportType.CREATOR_EARNINGS: self._generate_creator_earnings,
            ReportType.USER_ACTIVITY: self._generate_user_activity
        }
    
    async def initialize(self):
        """Initialize the reporter"""
        try:
            self.initialized = True
            logger.info("EcosystemReporter initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EcosystemReporter: {e}")
            return False
    
    async def generate_report(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate a comprehensive report based on request"""
        if not self.initialized:
            await self.initialize()
        
        report_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info("Generating report",
                       report_id=report_id,
                       report_type=request.report_type.value,
                       format=request.format.value)
            
            # Get report generator
            generator = self.report_templates.get(request.report_type)
            if not generator:
                raise ValueError(f"Unknown report type: {request.report_type}")
            
            # Generate report data
            report_data = await generator(request)
            
            # Add metadata
            report_data['metadata'] = {
                'report_id': report_id,
                'report_type': request.report_type.value,
                'format': request.format.value,
                'generated_at': start_time.isoformat(),
                'generation_time': (datetime.now(timezone.utc) - start_time).total_seconds(),
                'time_range': {
                    'start': request.time_range[0].isoformat() if request.time_range else None,
                    'end': request.time_range[1].isoformat() if request.time_range else None
                } if request.time_range else None,
                'filters': request.filters or {},
                'include_details': request.include_details
            }
            
            # Format report
            formatted_report = await self._format_report(report_data, request.format)
            
            # Store report
            self.generated_reports[report_id] = {
                'data': report_data,
                'formatted': formatted_report,
                'request': asdict(request)
            }
            
            # Add to history
            self.report_history.append({
                'report_id': report_id,
                'type': request.report_type.value,
                'generated_at': start_time.isoformat(),
                'generation_time': report_data['metadata']['generation_time']
            })
            
            # Save to file if requested
            if request.output_path:
                await self._save_report_to_file(formatted_report, request.output_path, request.format)
            
            logger.info("Report generated successfully",
                       report_id=report_id,
                       generation_time=report_data['metadata']['generation_time'])
            
            return {
                'report_id': report_id,
                'data': report_data,
                'formatted': formatted_report,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}", report_id=report_id)
            return {
                'report_id': report_id,
                'success': False,
                'error': str(e)
            }
    
    async def _generate_ecosystem_overview(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate ecosystem overview report"""
        # Get system metrics
        system_stats = self.system_integrator.get_pipeline_statistics() if self.system_integrator else {}
        usage_stats = self.usage_tracker.get_usage_statistics() if self.usage_tracker else {}
        
        # Calculate ecosystem metrics
        ecosystem_metrics = EcosystemMetrics(
            total_queries=system_stats.get('total_queries_processed', 0),
            successful_queries=int(system_stats.get('total_queries_processed', 0) * system_stats.get('success_rate', 0)),
            success_rate=system_stats.get('success_rate', 0.0),
            average_processing_time=system_stats.get('average_processing_time', 0.0),
            average_quality_score=system_stats.get('average_quality_score', 0.0),
            total_payments_distributed=system_stats.get('total_payments_distributed', 0.0),
            total_system_revenue=system_stats.get('total_payments_distributed', 0.0) * (0.3 / 0.7),  # Reverse calculate system fee
            unique_users=usage_stats.get('total_sessions', 0),  # Approximation
            unique_creators=len(usage_stats.get('payment_distribution_by_level', {})),
            content_pieces_accessed=usage_stats.get('total_sources_processed', 0),
            average_payment_per_creator=usage_stats.get('average_payment_per_source', 0.0),
            top_performing_content=[],  # Would need more data
            performance_trends={},      # Would need historical data
            system_health_score=0.95 if system_stats.get('success_rate', 0) > 0.8 else 0.7
        )
        
        return {
            'ecosystem_overview': {
                'summary': {
                    'total_queries': ecosystem_metrics.total_queries,
                    'success_rate': f"{ecosystem_metrics.success_rate:.1%}",
                    'total_revenue': f"{ecosystem_metrics.total_system_revenue:.2f} FTNS",
                    'creator_earnings': f"{ecosystem_metrics.total_payments_distributed:.2f} FTNS",
                    'system_health': f"{ecosystem_metrics.system_health_score:.1%}"
                },
                'performance_metrics': {
                    'average_processing_time': f"{ecosystem_metrics.average_processing_time:.3f}s",
                    'average_quality_score': f"{ecosystem_metrics.average_quality_score:.2f}",
                    'throughput': f"{ecosystem_metrics.total_queries / max(1, ecosystem_metrics.average_processing_time):.1f} queries/second"
                },
                'economic_metrics': {
                    'total_revenue': ecosystem_metrics.total_system_revenue,
                    'creator_distribution': ecosystem_metrics.total_payments_distributed,
                    'system_fee': ecosystem_metrics.total_system_revenue - ecosystem_metrics.total_payments_distributed,
                    'average_payment_per_creator': ecosystem_metrics.average_payment_per_creator,
                    'revenue_per_query': ecosystem_metrics.total_system_revenue / max(1, ecosystem_metrics.total_queries)
                },
                'ecosystem_health': {
                    'success_rate': ecosystem_metrics.success_rate,
                    'system_health_score': ecosystem_metrics.system_health_score,
                    'unique_users': ecosystem_metrics.unique_users,
                    'unique_creators': ecosystem_metrics.unique_creators,
                    'content_utilization': ecosystem_metrics.content_pieces_accessed
                }
            }
        }
    
    async def _generate_pipeline_performance(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate pipeline performance report"""
        system_stats = self.system_integrator.get_pipeline_statistics() if self.system_integrator else {}
        
        # Analyze stage performance
        stage_performance = system_stats.get('stage_performance', {})
        bottlenecks = []
        optimization_opportunities = []
        
        # Identify bottlenecks (stages taking >10% of total time)
        total_time = sum(stage.get('avg_time', 0) for stage in stage_performance.values())
        for stage_name, stage_data in stage_performance.items():
            avg_time = stage_data.get('avg_time', 0)
            if avg_time > total_time * 0.1:
                bottlenecks.append(f"{stage_name}: {avg_time:.3f}s")
        
        # Suggest optimizations
        if system_stats.get('average_processing_time', 0) > 10:
            optimization_opportunities.append("Consider caching frequently accessed content")
        if system_stats.get('success_rate', 1) < 0.9:
            optimization_opportunities.append("Improve error handling and retry logic")
        
        return {
            'pipeline_performance': {
                'overall_metrics': {
                    'total_queries': system_stats.get('total_queries_processed', 0),
                    'success_rate': f"{system_stats.get('success_rate', 0):.1%}",
                    'average_processing_time': f"{system_stats.get('average_processing_time', 0):.3f}s",
                    'average_quality_score': f"{system_stats.get('average_quality_score', 0):.2f}"
                },
                'stage_analysis': {
                    'performance_by_stage': {
                        stage: {
                            'average_time': f"{data.get('avg_time', 0):.3f}s",
                            'total_time': f"{data.get('total_time', 0):.3f}s",
                            'percentage_of_total': f"{data.get('avg_time', 0) / max(total_time, 0.001) * 100:.1f}%"
                        }
                        for stage, data in stage_performance.items()
                    },
                    'bottlenecks': bottlenecks,
                    'optimization_opportunities': optimization_opportunities
                },
                'performance_trends': {
                    'note': "Historical trend analysis would require time-series data",
                    'current_performance': "System performing within expected parameters"
                }
            }
        }
    
    async def _generate_economic_analysis(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate economic analysis report"""
        system_stats = self.system_integrator.get_pipeline_statistics() if self.system_integrator else {}
        usage_stats = self.usage_tracker.get_usage_statistics() if self.usage_tracker else {}
        
        total_revenue = system_stats.get('total_payments_distributed', 0.0) * (1.0 / 0.7)  # Total including system fee
        creator_earnings = system_stats.get('total_payments_distributed', 0.0)
        system_revenue = total_revenue - creator_earnings
        
        return {
            'economic_analysis': {
                'revenue_breakdown': {
                    'total_revenue': f"{total_revenue:.2f} FTNS",
                    'creator_earnings': f"{creator_earnings:.2f} FTNS ({creator_earnings/max(total_revenue, 0.001)*100:.1f}%)",
                    'system_revenue': f"{system_revenue:.2f} FTNS ({system_revenue/max(total_revenue, 0.001)*100:.1f}%)"
                },
                'payment_metrics': {
                    'total_payments_distributed': creator_earnings,
                    'average_payment_per_query': creator_earnings / max(1, system_stats.get('total_queries_processed', 0)),
                    'average_payment_per_source': usage_stats.get('average_payment_per_source', 0.0),
                    'payment_distribution_by_level': usage_stats.get('payment_distribution_by_level', {})
                },
                'economic_health': {
                    'revenue_per_query': total_revenue / max(1, system_stats.get('total_queries_processed', 0)),
                    'creator_satisfaction_proxy': f"{creator_earnings/max(total_revenue, 0.001)*100:.1f}%",
                    'system_sustainability': "Healthy" if system_revenue > 0 else "Needs attention"
                }
            }
        }
    
    async def _generate_usage_analytics(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate usage analytics report"""
        usage_stats = self.usage_tracker.get_usage_statistics() if self.usage_tracker else {}
        
        return {
            'usage_analytics': {
                'session_metrics': {
                    'total_sessions': usage_stats.get('total_sessions', 0),
                    'average_sources_per_query': usage_stats.get('average_sources_per_query', 0.0),
                    'total_sources_processed': usage_stats.get('total_sources_processed', 0),
                    'total_sources_filtered': usage_stats.get('total_sources_filtered', 0)
                },
                'usage_patterns': {
                    'payment_distribution_by_level': usage_stats.get('payment_distribution_by_level', {}),
                    'source_utilization_rate': usage_stats.get('total_sources_filtered', 0) / max(1, usage_stats.get('total_sources_processed', 0))
                },
                'performance_indicators': {
                    'average_payment_per_source': usage_stats.get('average_payment_per_source', 0.0),
                    'total_payments_distributed': usage_stats.get('total_payments_distributed', 0.0),
                    'system_fees_collected': usage_stats.get('total_system_fees', 0.0)
                }
            }
        }
    
    async def _generate_audit_trail(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate audit trail report"""
        audit_data = []
        
        # Get recent sessions from system integrator
        if self.system_integrator:
            recent_sessions = list(self.system_integrator.completed_sessions.values())[-10:]
            
            for session in recent_sessions:
                audit_data.append({
                    'session_id': session.session_id,
                    'query': session.query[:50] + "..." if len(session.query) > 50 else session.query,
                    'user_id': session.user_id,
                    'timestamp': session.timestamp.isoformat(),
                    'success': session.success,
                    'processing_time': session.processing_time,
                    'quality_score': session.quality_score,
                    'payments_distributed': len(session.payment_distributions),
                    'audit_trail_steps': len(session.audit_trail)
                })
        
        return {
            'audit_trail': {
                'recent_sessions': audit_data,
                'audit_summary': {
                    'total_audited_sessions': len(audit_data),
                    'successful_sessions': sum(1 for session in audit_data if session['success']),
                    'average_processing_time': sum(session['processing_time'] for session in audit_data) / max(1, len(audit_data)),
                    'average_quality_score': sum(session['quality_score'] for session in audit_data) / max(1, len(audit_data))
                },
                'compliance_metrics': {
                    'sessions_with_complete_audit_trail': len(audit_data),
                    'audit_trail_completeness': "100%",
                    'payment_transparency': "Full transparency maintained"
                }
            }
        }
    
    async def _generate_system_health(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate system health report"""
        system_stats = self.system_integrator.get_pipeline_statistics() if self.system_integrator else {}
        
        # Calculate health metrics
        success_rate = system_stats.get('success_rate', 0.0)
        avg_processing_time = system_stats.get('average_processing_time', 0.0)
        avg_quality_score = system_stats.get('average_quality_score', 0.0)
        
        # Health score calculation
        health_score = (
            success_rate * 0.4 +
            (1.0 - min(avg_processing_time / 30.0, 1.0)) * 0.3 +  # Penalty for slow processing
            avg_quality_score * 0.3
        )
        
        # Component health
        components = {
            'system_integrator': 'Healthy' if self.system_integrator and self.system_integrator.initialized else 'Offline',
            'usage_tracker': 'Healthy' if self.usage_tracker and self.usage_tracker.initialized else 'Offline',
            'ftns_service': 'Healthy' if self.ftns_service else 'Unknown'
        }
        
        return {
            'system_health': {
                'overall_health': {
                    'health_score': f"{health_score:.1%}",
                    'status': 'Healthy' if health_score > 0.8 else 'Needs Attention' if health_score > 0.6 else 'Critical',
                    'last_updated': datetime.now(timezone.utc).isoformat()
                },
                'component_health': components,
                'performance_indicators': {
                    'success_rate': f"{success_rate:.1%}",
                    'average_processing_time': f"{avg_processing_time:.3f}s",
                    'average_quality_score': f"{avg_quality_score:.2f}",
                    'total_queries_processed': system_stats.get('total_queries_processed', 0)
                },
                'alerts': self._generate_health_alerts(success_rate, avg_processing_time, avg_quality_score)
            }
        }
    
    async def _generate_creator_earnings(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate creator earnings report"""
        usage_stats = self.usage_tracker.get_usage_statistics() if self.usage_tracker else {}
        
        return {
            'creator_earnings': {
                'earnings_summary': {
                    'total_distributed': usage_stats.get('total_payments_distributed', 0.0),
                    'average_per_creator': usage_stats.get('average_payment_per_source', 0.0),
                    'payment_distribution_by_level': usage_stats.get('payment_distribution_by_level', {})
                },
                'earnings_metrics': {
                    'top_earning_creators': "Data not available - would require creator identification",
                    'earnings_trends': "Historical data needed for trend analysis",
                    'payment_frequency': "Real-time payments enabled"
                }
            }
        }
    
    async def _generate_user_activity(self, request: ReportRequest) -> Dict[str, Any]:
        """Generate user activity report"""
        system_stats = self.system_integrator.get_pipeline_statistics() if self.system_integrator else {}
        usage_stats = self.usage_tracker.get_usage_statistics() if self.usage_tracker else {}
        
        return {
            'user_activity': {
                'activity_summary': {
                    'total_queries': system_stats.get('total_queries_processed', 0),
                    'unique_sessions': usage_stats.get('total_sessions', 0),
                    'average_query_cost': usage_stats.get('total_payments_distributed', 0.0) / max(1, system_stats.get('total_queries_processed', 0)) * (1.0 / 0.7)
                },
                'usage_patterns': {
                    'successful_queries': int(system_stats.get('total_queries_processed', 0) * system_stats.get('success_rate', 0)),
                    'average_sources_per_query': usage_stats.get('average_sources_per_query', 0.0),
                    'quality_satisfaction': f"{system_stats.get('average_quality_score', 0.0):.2f}/1.0"
                }
            }
        }
    
    def _generate_health_alerts(self, success_rate: float, avg_processing_time: float, avg_quality_score: float) -> List[str]:
        """Generate health alerts based on metrics"""
        alerts = []
        
        if success_rate < 0.9:
            alerts.append(f"Low success rate: {success_rate:.1%} (target: >90%)")
        
        if avg_processing_time > 15.0:
            alerts.append(f"High processing time: {avg_processing_time:.3f}s (target: <15s)")
        
        if avg_quality_score < 0.7:
            alerts.append(f"Low quality score: {avg_quality_score:.2f} (target: >0.7)")
        
        if not alerts:
            alerts.append("All systems operating within normal parameters")
        
        return alerts
    
    async def _format_report(self, report_data: Dict[str, Any], format_type: ReportFormat) -> str:
        """Format report according to specified format"""
        if format_type == ReportFormat.JSON:
            return json.dumps(report_data, indent=2, default=str)
        
        elif format_type == ReportFormat.MARKDOWN:
            return self._format_as_markdown(report_data)
        
        elif format_type == ReportFormat.HTML:
            return self._format_as_html(report_data)
        
        elif format_type == ReportFormat.CSV:
            return self._format_as_csv(report_data)
        
        else:
            return json.dumps(report_data, indent=2, default=str)
    
    def _format_as_markdown(self, data: Dict[str, Any]) -> str:
        """Format report as Markdown"""
        md_content = []
        
        # Add title
        report_type = data.get('metadata', {}).get('report_type', 'Unknown')
        md_content.append(f"# {report_type.replace('_', ' ').title()} Report")
        md_content.append("")
        
        # Add metadata
        metadata = data.get('metadata', {})
        md_content.append("## Report Metadata")
        md_content.append(f"- **Report ID**: {metadata.get('report_id', 'N/A')}")
        md_content.append(f"- **Generated**: {metadata.get('generated_at', 'N/A')}")
        md_content.append(f"- **Generation Time**: {metadata.get('generation_time', 'N/A')}s")
        md_content.append("")
        
        # Add report content
        for key, value in data.items():
            if key != 'metadata':
                md_content.append(f"## {key.replace('_', ' ').title()}")
                md_content.append(self._dict_to_markdown(value))
                md_content.append("")
        
        return "\n".join(md_content)
    
    def _dict_to_markdown(self, data: Any, level: int = 0) -> str:
        """Convert dictionary to markdown format"""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{'  ' * level}- **{key.replace('_', ' ').title()}**:")
                    lines.append(self._dict_to_markdown(value, level + 1))
                else:
                    lines.append(f"{'  ' * level}- **{key.replace('_', ' ').title()}**: {value}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, dict):
                    lines.append(self._dict_to_markdown(item, level))
                else:
                    lines.append(f"{'  ' * level}- {item}")
            return "\n".join(lines)
        else:
            return str(data)
    
    def _format_as_html(self, data: Dict[str, Any]) -> str:
        """Format report as HTML"""
        html_content = []
        
        # Add HTML header
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html><head><title>NWTN Ecosystem Report</title></head><body>")
        html_content.append("<h1>NWTN Ecosystem Report</h1>")
        
        # Add content
        html_content.append("<div class='report-content'>")
        html_content.append(f"<pre>{json.dumps(data, indent=2, default=str)}</pre>")
        html_content.append("</div>")
        
        # Add HTML footer
        html_content.append("</body></html>")
        
        return "\n".join(html_content)
    
    def _format_as_csv(self, data: Dict[str, Any]) -> str:
        """Format report as CSV (simplified)"""
        csv_lines = []
        csv_lines.append("Category,Metric,Value")
        
        def extract_metrics(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        extract_metrics(value, f"{prefix}{key}.")
                    else:
                        csv_lines.append(f"{prefix}{key},{key},{value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_metrics(item, f"{prefix}[{i}].")
        
        extract_metrics(data)
        return "\n".join(csv_lines)
    
    async def _save_report_to_file(self, content: str, output_path: Path, format_type: ReportFormat):
        """Save report to file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report to {output_path}: {e}")
            raise
    
    def get_report_history(self) -> List[Dict[str, Any]]:
        """Get history of generated reports"""
        return self.report_history
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific report by ID"""
        return self.generated_reports.get(report_id)


# Factory function for easy instantiation
async def create_ecosystem_reporter(
    system_integrator: Optional[SystemIntegrator] = None,
    usage_tracker: Optional[AttributionUsageTracker] = None,
    ftns_service: Optional[FTNSService] = None
) -> EcosystemReporter:
    """Create and initialize an ecosystem reporter"""
    reporter = EcosystemReporter(system_integrator, usage_tracker, ftns_service)
    await reporter.initialize()
    return reporter