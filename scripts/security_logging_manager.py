#!/usr/bin/env python3
"""
PRSM Security Logging Management CLI
===================================

Command-line interface for managing and monitoring the comprehensive
security logging system with real-time metrics and administrative controls.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import click
import structlog
from tabulate import tabulate

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.security.comprehensive_logging import (
    get_security_logger, 
    LogLevel, 
    EventCategory, 
    AlertSeverity
)

# Set up logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """PRSM Security Logging Management Tool"""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--event-type', '-t', required=True, help='Type of security event')
@click.option('--level', '-l', type=click.Choice(['debug', 'info', 'warning', 'error', 'critical', 'audit']), 
              default='info', help='Log level')
@click.option('--category', '-c', type=click.Choice([
    'authentication', 'authorization', 'data_access', 'api_security',
    'network_security', 'crypto_operations', 'governance', 'marketplace',
    'web3_operations', 'user_activity', 'system_integrity', 'compliance',
    'threat_detection', 'incident_response'
]), required=True, help='Event category')
@click.option('--message', '-m', required=True, help='Event message')
@click.option('--user-id', '-u', help='User ID associated with event')
@click.option('--ip-address', '-i', help='IP address')
@click.option('--risk-score', '-r', type=int, default=0, help='Risk score (0-100)')
@click.option('--threat-indicators', help='Comma-separated list of threat indicators')
@click.option('--compliance-flags', help='Comma-separated list of compliance flags')
@click.option('--metadata', help='JSON metadata string')
def log_event(event_type, level, category, message, user_id, ip_address, 
              risk_score, threat_indicators, compliance_flags, metadata):
    """Log a security event"""
    
    async def _log_event():
        try:
            click.echo(f"📝 Logging security event: {event_type}")
            
            # Parse optional parameters
            threat_list = []
            if threat_indicators:
                threat_list = [t.strip() for t in threat_indicators.split(',')]
            
            compliance_list = []
            if compliance_flags:
                compliance_list = [c.strip() for c in compliance_flags.split(',')]
            
            metadata_dict = {}
            if metadata:
                metadata_dict = json.loads(metadata)
            
            # Get security logger
            security_logger = await get_security_logger()
            
            # Convert string enums
            log_level = LogLevel(level)
            event_category = EventCategory(category)
            
            # Log the event
            await security_logger.log_security_event(
                event_type=event_type,
                level=log_level,
                category=event_category,
                message=message,
                user_id=user_id,
                ip_address=ip_address,
                risk_score=risk_score,
                threat_indicators=threat_list,
                compliance_flags=compliance_list,
                metadata=metadata_dict,
                component="cli_tool"
            )
            
            click.echo("✅ Security event logged successfully")
            click.echo(f"   Event Type: {event_type}")
            click.echo(f"   Level: {level}")
            click.echo(f"   Category: {category}")
            click.echo(f"   Risk Score: {risk_score}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to log event: {e}")
            return False
    
    success = asyncio.run(_log_event())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--hours', '-h', type=int, default=24, help='Time period in hours')
@click.option('--category', '-c', type=click.Choice([
    'authentication', 'authorization', 'data_access', 'api_security',
    'network_security', 'crypto_operations', 'governance', 'marketplace',
    'web3_operations', 'user_activity', 'system_integrity', 'compliance',
    'threat_detection', 'incident_response'
]), help='Filter by category')
def metrics(hours, category):
    """Get security metrics and statistics"""
    
    async def _metrics():
        try:
            click.echo(f"📊 Security Metrics ({hours} hours)")
            click.echo("=" * 50)
            
            security_logger = await get_security_logger()
            metrics_data = await security_logger.get_security_metrics(hours=hours)
            
            # Display overview
            click.echo(f"Time Period: {hours} hours")
            click.echo(f"Total Logs: {metrics_data.get('total_logs', 0):,}")
            click.echo(f"Alerts Triggered: {metrics_data.get('alerts_triggered', 0)}")
            click.echo(f"Errors Encountered: {metrics_data.get('errors_encountered', 0)}")
            click.echo(f"System Uptime: {metrics_data.get('uptime_hours', 0):.1f} hours")
            click.echo(f"Queue Size: {metrics_data.get('queue_size', 0)}")
            click.echo()
            
            # Display category breakdown
            categories = metrics_data.get('categories', {})
            if categories:
                click.echo("📋 Events by Category:")
                category_table = []
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    category_table.append([cat.replace('_', ' ').title(), count])
                
                click.echo(tabulate(category_table, headers=['Category', 'Count'], tablefmt='grid'))
                click.echo()
            
            # Display risk distribution
            risk_dist = metrics_data.get('risk_distribution', {})
            if risk_dist:
                click.echo("⚠️ Risk Distribution:")
                risk_table = []
                for risk_level, count in risk_dist.items():
                    risk_table.append([risk_level.title(), count])
                
                click.echo(tabulate(risk_table, headers=['Risk Level', 'Count'], tablefmt='grid'))
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get metrics: {e}")
            return False
    
    success = asyncio.run(_metrics())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--limit', '-l', type=int, default=20, help='Maximum number of alerts')
@click.option('--severity', '-s', type=click.Choice(['low', 'medium', 'high', 'critical']), 
              help='Filter by severity')
def alerts(limit, severity):
    """Get recent security alerts"""
    
    async def _alerts():
        try:
            click.echo(f"🚨 Recent Security Alerts (limit: {limit})")
            click.echo("=" * 50)
            
            security_logger = await get_security_logger()
            
            # Get alerts from alert manager
            all_alerts = security_logger.alert_manager.alert_history
            
            # Filter by severity if specified
            if severity:
                filtered_alerts = [
                    alert for alert in all_alerts
                    if alert["severity"] == AlertSeverity(severity).value
                ]
            else:
                filtered_alerts = all_alerts
            
            # Sort by timestamp and limit
            recent_alerts = sorted(
                filtered_alerts,
                key=lambda x: x["triggered_at"],
                reverse=True
            )[:limit]
            
            if not recent_alerts:
                click.echo("ℹ️ No recent alerts found")
                return True
            
            # Display alerts table
            alert_table = []
            for alert in recent_alerts:
                alert_table.append([
                    alert["alert_id"][:8],
                    alert["rule_id"],
                    alert["severity"],
                    alert["triggered_at"][:19],  # Remove microseconds
                    alert["log_entry"]["event_type"],
                    "✅" if not alert.get("suppressed", False) else "🔇"
                ])
            
            headers = ['Alert ID', 'Rule', 'Severity', 'Triggered At', 'Event Type', 'Status']
            click.echo(tabulate(alert_table, headers=headers, tablefmt='grid'))
            click.echo()
            click.echo(f"📈 Total Alerts: {len(all_alerts)}")
            click.echo(f"🔍 Filtered Results: {len(recent_alerts)}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get alerts: {e}")
            return False
    
    success = asyncio.run(_alerts())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--rule-id', '-r', required=True, help='Unique rule identifier')
@click.option('--description', '-d', required=True, help='Rule description')
@click.option('--severity', '-s', type=click.Choice(['low', 'medium', 'high', 'critical']), 
              required=True, help='Alert severity')
@click.option('--category', '-c', type=click.Choice([
    'authentication', 'authorization', 'data_access', 'api_security',
    'network_security', 'crypto_operations', 'governance', 'marketplace',
    'web3_operations', 'user_activity', 'system_integrity', 'compliance',
    'threat_detection', 'incident_response'
]), help='Category filter')
@click.option('--level', '-l', type=click.Choice(['debug', 'info', 'warning', 'error', 'critical', 'audit']), 
              help='Log level filter')
@click.option('--min-risk-score', type=int, help='Minimum risk score threshold')
@click.option('--keywords', help='Comma-separated list of keywords to match')
@click.option('--channels', default='log', help='Comma-separated notification channels')
def add_alert_rule(rule_id, description, severity, category, level, min_risk_score, keywords, channels):
    """Add a new security alert rule"""
    
    async def _add_rule():
        try:
            click.echo(f"⚙️ Adding alert rule: {rule_id}")
            click.echo(f"Description: {description}")
            click.echo()
            
            # Parse channels
            channel_list = [c.strip() for c in channels.split(',')]
            
            # Build condition configuration
            condition_config = {}
            if category:
                condition_config["category"] = category
            if level:
                condition_config["level"] = level
            if min_risk_score is not None:
                condition_config["min_risk_score"] = min_risk_score
            if keywords:
                condition_config["keywords"] = [k.strip() for k in keywords.split(',')]
            
            click.echo("📋 Rule Configuration:")
            click.echo(f"   Rule ID: {rule_id}")
            click.echo(f"   Severity: {severity}")
            click.echo(f"   Channels: {channel_list}")
            click.echo(f"   Conditions: {json.dumps(condition_config, indent=2)}")
            click.echo()
            
            if not click.confirm("Create this alert rule?"):
                click.echo("❌ Rule creation cancelled")
                return False
            
            # Get security logger
            security_logger = await get_security_logger()
            
            # Create condition function
            def condition_func(log_entry):
                # Check category match
                if "category" in condition_config and log_entry.category.value != condition_config["category"]:
                    return False
                
                # Check level match
                if "level" in condition_config and log_entry.level.value != condition_config["level"]:
                    return False
                
                # Check risk score threshold
                if "min_risk_score" in condition_config and log_entry.risk_score < condition_config["min_risk_score"]:
                    return False
                
                # Check for keywords in message
                if "keywords" in condition_config:
                    keywords_list = condition_config["keywords"]
                    message_lower = log_entry.message.lower()
                    if not any(keyword.lower() in message_lower for keyword in keywords_list):
                        return False
                
                return True
            
            # Add the alert rule
            security_logger.alert_manager.add_alert_rule(
                rule_id=rule_id,
                condition=condition_func,
                severity=AlertSeverity(severity),
                notification_channels=channel_list
            )
            
            click.echo("✅ Alert rule created successfully")
            click.echo(f"   Rule ID: {rule_id}")
            click.echo(f"   Severity: {severity}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to create alert rule: {e}")
            return False
    
    success = asyncio.run(_add_rule())
    sys.exit(0 if success else 1)


@cli.command()
def list_rules():
    """List all configured alert rules"""
    
    async def _list_rules():
        try:
            click.echo("📋 Configured Alert Rules")
            click.echo("=" * 40)
            
            security_logger = await get_security_logger()
            
            rules_info = []
            for rule_id, rule_data in security_logger.alert_manager.alert_rules.items():
                rules_info.append([
                    rule_id,
                    rule_data["severity"].value,
                    ", ".join(rule_data["channels"]),
                    rule_data["trigger_count"],
                    rule_data["last_triggered"].isoformat()[:19] if rule_data["last_triggered"] else "Never"
                ])
            
            if not rules_info:
                click.echo("ℹ️ No alert rules configured")
                return True
            
            headers = ['Rule ID', 'Severity', 'Channels', 'Triggers', 'Last Triggered']
            click.echo(tabulate(rules_info, headers=headers, tablefmt='grid'))
            click.echo()
            click.echo(f"📊 Total Rules: {len(rules_info)}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to list rules: {e}")
            return False
    
    success = asyncio.run(_list_rules())
    sys.exit(0 if success else 1)


@cli.command()
def status():
    """Get security logging system status"""
    
    async def _status():
        try:
            click.echo("🏥 Security Logging System Status")
            click.echo("=" * 45)
            
            security_logger = await get_security_logger()
            
            uptime = (datetime.now() - security_logger.stats["start_time"]).total_seconds() / 3600
            queue_util = (security_logger.log_queue.qsize() / security_logger.log_queue.maxsize) * 100
            
            # System status overview
            status_table = [
                ["System Status", "🟢 Healthy"],
                ["Uptime", f"{uptime:.1f} hours"],
                ["Logs Written", f"{security_logger.stats['logs_written']:,}"],
                ["Errors Encountered", f"{security_logger.stats['errors_encountered']}"],
                ["Queue Size", f"{security_logger.log_queue.qsize():,}"],
                ["Queue Utilization", f"{queue_util:.1f}%"],
                ["Log Directory", str(security_logger.log_dir)],
                ["Alert Rules", f"{len(security_logger.alert_manager.alert_rules)}"],
                ["Total Alerts", f"{len(security_logger.alert_manager.alert_history)}"]
            ]
            
            click.echo(tabulate(status_table, headers=['Metric', 'Value'], tablefmt='grid'))
            click.echo()
            
            # Configuration details
            click.echo("⚙️ Configuration:")
            config_table = [
                ["Max Log Size", f"{security_logger.config['max_log_size_mb']} MB"],
                ["Retention Days", f"{security_logger.config['retention_days']} days"],
                ["Alerts Enabled", "✅" if security_logger.config['enable_real_time_alerts'] else "❌"],
                ["Metrics Enabled", "✅" if security_logger.config['enable_metrics'] else "❌"],
                ["Compression", "✅" if security_logger.config['enable_log_compression'] else "❌"]
            ]
            
            click.echo(tabulate(config_table, headers=['Setting', 'Value'], tablefmt='grid'))
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get status: {e}")
            return False
    
    success = asyncio.run(_status())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--category', '-c', type=click.Choice([
    'authentication', 'authorization', 'data_access', 'api_security',
    'network_security', 'crypto_operations', 'governance', 'marketplace',
    'web3_operations', 'user_activity', 'system_integrity', 'compliance',
    'threat_detection', 'incident_response'
]), help='Test category')
@click.option('--count', '-n', type=int, default=5, help='Number of test events')
def test_logging(category, count):
    """Generate test security events for system validation"""
    
    async def _test():
        try:
            click.echo(f"🧪 Generating {count} test security events")
            if category:
                click.echo(f"Category: {category}")
            click.echo()
            
            security_logger = await get_security_logger()
            
            for i in range(count):
                test_category = EventCategory(category) if category else EventCategory.USER_ACTIVITY
                
                await security_logger.log_security_event(
                    event_type=f"test_event_{i+1}",
                    level=LogLevel.INFO,
                    category=test_category,
                    message=f"Test security event {i+1} for CLI validation",
                    user_id="test_user",
                    risk_score=10 + (i * 5),
                    metadata={"test": True, "event_number": i+1},
                    component="cli_test"
                )
                
                click.echo(f"✅ Test event {i+1} logged")
            
            click.echo()
            click.echo(f"🎉 Successfully generated {count} test events")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Test failed: {e}")
            return False
    
    success = asyncio.run(_test())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    cli()