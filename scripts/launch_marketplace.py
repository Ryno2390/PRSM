#!/usr/bin/env python3
"""
PRSM Marketplace Launch Script
==============================

Command-line tool for launching the PRSM marketplace with initial
model listings and validating marketplace readiness.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

import click
import structlog

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.marketplace.initial_listings import launch_marketplace_with_initial_listings, initial_listings_creator

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
    """PRSM Marketplace Launch Tool"""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
def launch():
    """Launch the marketplace with initial model listings"""
    
    async def _launch():
        try:
            click.echo("🚀 Launching PRSM Marketplace...")
            click.echo("=" * 60)
            
            # Launch marketplace
            result = await launch_marketplace_with_initial_listings()
            
            if result.get("success", False):
                click.echo("✅ Marketplace launched successfully!")
                
                # Display launch summary
                summary = result.get("launch_summary", {})
                click.echo(f"\n📊 Launch Summary:")
                click.echo(f"   Total Listings Created: {summary.get('total_initial_listings', 0)}")
                click.echo(f"   Categories Covered: {summary.get('categories_covered', 0)}")
                click.echo(f"   Providers Included: {summary.get('providers_included', 0)}")
                click.echo(f"   Featured Models: {summary.get('featured_models', 0)}")
                click.echo(f"   Launch Ready: {'✅ Yes' if summary.get('launch_ready', False) else '❌ No'}")
                
                # Display marketplace stats
                stats = summary.get('marketplace_stats', {})
                if stats:
                    click.echo(f"\n📈 Marketplace Statistics:")
                    click.echo(f"   Total Models: {stats.get('total_models', 0)}")
                    click.echo(f"   Verified Models: {stats.get('verified_models', 0)}")
                    click.echo(f"   Featured Models: {stats.get('featured_models', 0)}")
                
                click.echo(f"\n🎉 Marketplace is now live and ready for users!")
                return True
            else:
                click.echo("❌ Marketplace launch failed!")
                click.echo(f"Error: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error("Marketplace launch failed", error=str(e))
            click.echo(f"❌ Launch failed: {e}")
            return False
    
    success = asyncio.run(_launch())
    sys.exit(0 if success else 1)


@cli.command()
def status():
    """Check marketplace launch readiness status"""
    
    async def _status():
        try:
            click.echo("📊 PRSM Marketplace Launch Status")
            click.echo("=" * 40)
            
            # Get launch status
            summary = await initial_listings_creator.get_launch_summary()
            
            # Display status
            click.echo(f"Launch Ready: {'✅ Yes' if summary.get('launch_ready', False) else '❌ No'}")
            click.echo(f"Initial Listings: {summary.get('total_initial_listings', 0)}")
            click.echo(f"Categories Covered: {summary.get('categories_covered', 0)}")
            click.echo(f"Providers Included: {summary.get('providers_included', 0)}")
            click.echo(f"Featured Models: {summary.get('featured_models', 0)}")
            
            # Display marketplace stats
            stats = summary.get('marketplace_stats', {})
            if stats:
                click.echo(f"\n📈 Current Marketplace:")
                click.echo(f"   Total Models: {stats.get('total_models', 0)}")
                click.echo(f"   Total Providers: {stats.get('total_providers', 0)}")
                click.echo(f"   Total Categories: {stats.get('total_categories', 0)}")
                click.echo(f"   Verified Models: {stats.get('verified_models', 0)}")
                click.echo(f"   Featured Models: {stats.get('featured_models', 0)}")
            
            return summary.get('launch_ready', False)
            
        except Exception as e:
            logger.error("Failed to get launch status", error=str(e))
            click.echo(f"❌ Status check failed: {e}")
            return False
    
    success = asyncio.run(_status())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
def preview(json_output):
    """Preview initial listings that would be created"""
    
    async def _preview():
        try:
            if not json_output:
                click.echo("👀 PRSM Marketplace Initial Listings Preview")
                click.echo("=" * 50)
            
            # Get preview data
            preview_data = []
            categories = {}
            providers = set()
            pricing_tiers = set()
            
            for listing_data in initial_listings_creator.initial_listings:
                preview_item = {
                    "name": listing_data["name"],
                    "provider": listing_data["provider"].value,
                    "category": listing_data["category"].value,
                    "pricing_tier": listing_data["pricing_tier"].value,
                    "description": listing_data["description"][:100] + "..." if len(listing_data["description"]) > 100 else listing_data["description"]
                }
                preview_data.append(preview_item)
                
                # Group data
                category = preview_item["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(preview_item)
                providers.add(preview_item["provider"])
                pricing_tiers.add(preview_item["pricing_tier"])
            
            if json_output:
                # JSON output
                output = {
                    "total_listings": len(preview_data),
                    "categories_count": len(categories),
                    "providers": sorted(list(providers)),
                    "pricing_tiers": sorted(list(pricing_tiers)),
                    "listings": preview_data,
                    "by_category": categories
                }
                click.echo(json.dumps(output, indent=2, default=str))
            else:
                # Human-readable output
                click.echo(f"📋 Summary:")
                click.echo(f"   Total Listings: {len(preview_data)}")
                click.echo(f"   Categories: {len(categories)}")
                click.echo(f"   Providers: {', '.join(sorted(providers))}")
                click.echo(f"   Pricing Tiers: {', '.join(sorted(pricing_tiers))}")
                
                click.echo(f"\n📂 Models by Category:")
                for category, models in categories.items():
                    click.echo(f"\n   {category.replace('_', ' ').title()}:")
                    for model in models:
                        price_info = f" ({model['pricing_tier']})" if model['pricing_tier'] != 'free' else " (Free)"
                        click.echo(f"     • {model['name']} - {model['provider']}{price_info}")
                
                click.echo(f"\n💡 Use 'launch' command to create these listings in the marketplace")
            
            return True
            
        except Exception as e:
            logger.error("Failed to preview listings", error=str(e))
            if json_output:
                click.echo(json.dumps({"error": str(e)}))
            else:
                click.echo(f"❌ Preview failed: {e}")
            return False
    
    success = asyncio.run(_preview())
    sys.exit(0 if success else 1)


@cli.command()
def validate():
    """Validate marketplace readiness and configuration"""
    
    async def _validate():
        try:
            click.echo("🔍 Validating Marketplace Configuration")
            click.echo("=" * 45)
            
            validation_results = []
            
            # Check initial listings data
            listings_count = len(initial_listings_creator.initial_listings)
            if listings_count >= 5:
                validation_results.append(("✅", f"Sufficient initial listings ({listings_count})"))
            else:
                validation_results.append(("❌", f"Insufficient initial listings ({listings_count} < 5)"))
            
            # Check category coverage
            categories = set(listing["category"] for listing in initial_listings_creator.initial_listings)
            if len(categories) >= 3:
                validation_results.append(("✅", f"Good category coverage ({len(categories)} categories)"))
            else:
                validation_results.append(("❌", f"Poor category coverage ({len(categories)} < 3)"))
            
            # Check provider diversity
            providers = set(listing["provider"] for listing in initial_listings_creator.initial_listings)
            if len(providers) >= 3:
                validation_results.append(("✅", f"Good provider diversity ({len(providers)} providers)"))
            else:
                validation_results.append(("❌", f"Poor provider diversity ({len(providers)} < 3)"))
            
            # Check pricing tier mix
            pricing_tiers = set(listing["pricing_tier"] for listing in initial_listings_creator.initial_listings)
            if len(pricing_tiers) >= 2:
                validation_results.append(("✅", f"Mixed pricing tiers ({len(pricing_tiers)} tiers)"))
            else:
                validation_results.append(("❌", f"Limited pricing options ({len(pricing_tiers)} tier)"))
            
            # Check for required categories
            required_categories = ["language_model", "image_generation"]
            missing_required = [cat for cat in required_categories if cat not in [c.value for c in categories]]
            if not missing_required:
                validation_results.append(("✅", "All required categories present"))
            else:
                validation_results.append(("❌", f"Missing required categories: {missing_required}"))
            
            # Display results
            all_passed = True
            for status, message in validation_results:
                click.echo(f"   {status} {message}")
                if status == "❌":
                    all_passed = False
            
            click.echo(f"\n{'✅ Marketplace ready for launch!' if all_passed else '❌ Marketplace not ready - fix issues above'}")
            
            return all_passed
            
        except Exception as e:
            logger.error("Validation failed", error=str(e))
            click.echo(f"❌ Validation failed: {e}")
            return False
    
    success = asyncio.run(_validate())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    cli()