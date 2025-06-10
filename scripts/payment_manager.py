#!/usr/bin/env python3
"""
PRSM Payment Management CLI
==========================

Command-line interface for managing payment processing, viewing transactions,
testing payment flows, and administering the fiat-to-crypto conversion system.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List

import click
import structlog
from tabulate import tabulate

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.payments import (
    get_payment_processor, get_fiat_gateway, get_crypto_exchange,
    PaymentMethod, PaymentStatus, FiatCurrency, CryptoCurrency,
    PaymentRequest, TransactionQuery
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
    """PRSM Payment Management Tool"""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--user-id', '-u', required=True, help='User ID for the payment')
@click.option('--amount', '-a', type=float, required=True, help='Fiat amount')
@click.option('--fiat-currency', '-f', type=click.Choice(['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']), 
              default='USD', help='Fiat currency')
@click.option('--crypto-currency', '-c', type=click.Choice(['MATIC', 'ETH', 'USDC', 'USDT', 'FTNS']), 
              default='FTNS', help='Target cryptocurrency')
@click.option('--payment-method', '-m', type=click.Choice([
    'credit_card', 'debit_card', 'bank_transfer', 'paypal', 'mock'
]), default='mock', help='Payment method')
@click.option('--metadata', help='JSON metadata string')
def create_payment(user_id, amount, fiat_currency, crypto_currency, payment_method, metadata):
    """Create a test payment transaction"""
    
    async def _create_payment():
        try:
            click.echo(f"💳 Creating payment for {amount} {fiat_currency} → {crypto_currency}")
            click.echo(f"User: {user_id}, Method: {payment_method}")
            click.echo()
            
            # Parse metadata
            metadata_dict = {}
            if metadata:
                metadata_dict = json.loads(metadata)
            
            # Get payment processor
            payment_processor = await get_payment_processor()
            
            # Create payment request
            request = PaymentRequest(
                user_id=user_id,
                fiat_amount=Decimal(str(amount)),
                fiat_currency=FiatCurrency(fiat_currency),
                crypto_currency=CryptoCurrency(crypto_currency),
                payment_method=PaymentMethod(payment_method),
                metadata=metadata_dict
            )
            
            # Create payment
            response = await payment_processor.create_payment(request)
            
            if response.success:
                click.echo("✅ Payment created successfully!")
                click.echo(f"   Transaction ID: {response.transaction_id}")
                click.echo(f"   Status: {response.status.value}")
                click.echo(f"   Fiat Amount: {response.fiat_amount} {response.fiat_currency.value}")
                if response.crypto_amount:
                    click.echo(f"   Crypto Amount: {response.crypto_amount} {response.crypto_currency.value}")
                if response.exchange_rate:
                    click.echo(f"   Exchange Rate: {response.exchange_rate}")
                click.echo(f"   Processing Fee: {response.processing_fee}")
                
                if response.payment_url:
                    click.echo(f"   Payment URL: {response.payment_url}")
                if response.requires_action:
                    click.echo(f"   Requires Action: {response.next_action}")
                if response.expires_at:
                    click.echo(f"   Expires At: {response.expires_at}")
                
            else:
                click.echo("❌ Payment creation failed")
                click.echo(f"   Error: {response.message}")
            
            return response.success
            
        except Exception as e:
            click.echo(f"❌ Payment creation failed: {e}")
            return False
    
    success = asyncio.run(_create_payment())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--transaction-id', '-t', required=True, help='Transaction ID to check')
def check_status(transaction_id):
    """Check payment transaction status"""
    
    async def _check_status():
        try:
            click.echo(f"📊 Checking status for transaction: {transaction_id}")
            click.echo()
            
            payment_processor = await get_payment_processor()
            response = await payment_processor.get_payment_status(transaction_id)
            
            if response.success:
                click.echo("✅ Transaction status retrieved")
                click.echo(f"   Transaction ID: {response.transaction_id}")
                click.echo(f"   Status: {response.status.value}")
                click.echo(f"   Fiat Amount: {response.fiat_amount} {response.fiat_currency.value}")
                if response.crypto_amount:
                    click.echo(f"   Crypto Amount: {response.crypto_amount} {response.crypto_currency.value}")
                if response.exchange_rate:
                    click.echo(f"   Exchange Rate: {response.exchange_rate}")
                click.echo(f"   Processing Fee: {response.processing_fee}")
                if response.network_fee:
                    click.echo(f"   Network Fee: {response.network_fee}")
                if response.provider_reference:
                    click.echo(f"   Provider Reference: {response.provider_reference}")
                if response.completed_at:
                    click.echo(f"   Completed At: {response.completed_at}")
                
            else:
                click.echo("❌ Failed to get transaction status")
                click.echo(f"   Error: {response.message}")
            
            return response.success
            
        except Exception as e:
            click.echo(f"❌ Status check failed: {e}")
            return False
    
    success = asyncio.run(_check_status())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--user-id', '-u', help='Filter by user ID')
@click.option('--status', '-s', type=click.Choice(['pending', 'processing', 'completed', 'failed', 'cancelled']), 
              help='Filter by status')
@click.option('--payment-method', '-m', type=click.Choice([
    'credit_card', 'debit_card', 'bank_transfer', 'paypal', 'mock'
]), help='Filter by payment method')
@click.option('--limit', '-l', type=int, default=20, help='Maximum number of transactions')
@click.option('--days', '-d', type=int, default=30, help='Number of days to look back')
def list_transactions(user_id, status, payment_method, limit, days):
    """List recent payment transactions"""
    
    async def _list_transactions():
        try:
            click.echo(f"📋 Listing transactions (last {days} days, limit: {limit})")
            if user_id:
                click.echo(f"User: {user_id}")
            if status:
                click.echo(f"Status: {status}")
            if payment_method:
                click.echo(f"Payment Method: {payment_method}")
            click.echo()
            
            payment_processor = await get_payment_processor()
            
            # Build query
            start_date = datetime.now() - timedelta(days=days)
            query = TransactionQuery(
                user_id=user_id,
                status=PaymentStatus(status) if status else None,
                payment_method=PaymentMethod(payment_method) if payment_method else None,
                start_date=start_date,
                limit=limit,
                sort_by="created_at",
                sort_order="desc"
            )
            
            transaction_list = await payment_processor.list_transactions(query)
            
            if not transaction_list.transactions:
                click.echo("ℹ️ No transactions found")
                return True
            
            # Create table
            table_data = []
            for tx in transaction_list.transactions:
                table_data.append([
                    tx.transaction_id[:8],  # Shortened ID
                    tx.fiat_amount,
                    tx.fiat_currency.value,
                    tx.crypto_currency.value,
                    tx.status.value,
                    tx.created_at.strftime("%Y-%m-%d %H:%M") if tx.created_at else "N/A"
                ])
            
            headers = ['TX ID', 'Amount', 'Fiat', 'Crypto', 'Status', 'Created']
            click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
            click.echo()
            click.echo(f"📊 Total: {transaction_list.total_count} transactions")
            click.echo(f"📄 Page: {transaction_list.page}/{transaction_list.total_pages}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to list transactions: {e}")
            return False
    
    success = asyncio.run(_list_transactions())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--user-id', '-u', help='Get stats for specific user')
@click.option('--days', '-d', type=int, default=30, help='Number of days to analyze')
def payment_stats(user_id, days):
    """Get payment statistics"""
    
    async def _payment_stats():
        try:
            click.echo(f"📈 Payment Statistics (last {days} days)")
            if user_id:
                click.echo(f"User: {user_id}")
            click.echo("=" * 50)
            
            payment_processor = await get_payment_processor()
            
            start_date = datetime.now() - timedelta(days=days)
            stats = await payment_processor.get_payment_stats(
                user_id=user_id,
                start_date=start_date
            )
            
            # Overview stats
            success_rate = (stats.successful_transactions / stats.total_transactions * 100) if stats.total_transactions > 0 else 0
            
            overview_table = [
                ["Total Transactions", f"{stats.total_transactions:,}"],
                ["Successful", f"{stats.successful_transactions:,}"],
                ["Failed", f"{stats.failed_transactions:,}"],
                ["Success Rate", f"{success_rate:.1f}%"],
                ["Total Volume (USD)", f"${stats.total_volume_usd:,.2f}"],
                ["Average Transaction", f"${stats.average_transaction_size:.2f}"]
            ]
            
            click.echo("📊 Overview:")
            click.echo(tabulate(overview_table, headers=['Metric', 'Value'], tablefmt='grid'))
            click.echo()
            
            # Volume by fiat currency
            if stats.volume_by_fiat_currency:
                click.echo("💰 Volume by Fiat Currency:")
                fiat_table = []
                for currency, volume in stats.volume_by_fiat_currency.items():
                    fiat_table.append([currency, f"${volume:,.2f}"])
                
                click.echo(tabulate(fiat_table, headers=['Currency', 'Volume'], tablefmt='grid'))
                click.echo()
            
            # Volume by crypto currency
            if stats.volume_by_crypto_currency:
                click.echo("⚡ Volume by Crypto Currency:")
                crypto_table = []
                for currency, volume in stats.volume_by_crypto_currency.items():
                    crypto_table.append([currency, f"{volume:,.8f}"])
                
                click.echo(tabulate(crypto_table, headers=['Currency', 'Volume'], tablefmt='grid'))
                click.echo()
            
            # Transactions by payment method
            if stats.transactions_by_method:
                click.echo("💳 Transactions by Payment Method:")
                method_table = []
                for method, count in stats.transactions_by_method.items():
                    percentage = (count / stats.total_transactions * 100) if stats.total_transactions > 0 else 0
                    method_table.append([method.replace('_', ' ').title(), count, f"{percentage:.1f}%"])
                
                click.echo(tabulate(method_table, headers=['Method', 'Count', 'Percentage'], tablefmt='grid'))
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get payment stats: {e}")
            return False
    
    success = asyncio.run(_payment_stats())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--from-currency', '-f', required=True, help='Source currency code')
@click.option('--to-currency', '-t', required=True, help='Target currency code')
@click.option('--amount', '-a', type=float, help='Amount for conversion calculation')
def exchange_rate(from_currency, to_currency, amount):
    """Get current exchange rate"""
    
    async def _exchange_rate():
        try:
            click.echo(f"💱 Exchange Rate: {from_currency} → {to_currency}")
            if amount:
                click.echo(f"Amount: {amount} {from_currency}")
            click.echo()
            
            crypto_exchange = await get_crypto_exchange()
            
            rate = await crypto_exchange.get_exchange_rate(from_currency, to_currency)
            
            if not rate:
                click.echo(f"❌ Exchange rate not available for {from_currency}/{to_currency}")
                return False
            
            click.echo("✅ Exchange rate retrieved")
            click.echo(f"   Rate: 1 {from_currency} = {rate.rate} {to_currency}")
            click.echo(f"   Source: {rate.source}")
            click.echo(f"   Timestamp: {rate.timestamp}")
            
            if rate.volume_24h:
                click.echo(f"   24h Volume: {rate.volume_24h:,.2f}")
            if rate.price_change_24h:
                click.echo(f"   24h Change: {rate.price_change_24h:+.2f}%")
            
            # Calculate conversion if amount provided
            if amount:
                conversion = await crypto_exchange.calculate_conversion(
                    Decimal(str(amount)), from_currency, to_currency
                )
                
                if conversion["success"]:
                    click.echo()
                    click.echo("🔄 Conversion Calculation:")
                    click.echo(f"   Input: {conversion['input_amount']} {conversion['input_currency']}")
                    click.echo(f"   Output: {conversion['output_amount']:.8f} {conversion['output_currency']}")
                    if conversion['slippage'] > 0:
                        click.echo(f"   Slippage: {conversion['slippage']:.8f} ({conversion['slippage_percent']:.2f}%)")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get exchange rate: {e}")
            return False
    
    success = asyncio.run(_exchange_rate())
    sys.exit(0 if success else 1)


@cli.command()
def system_status():
    """Get payment system status"""
    
    async def _system_status():
        try:
            click.echo("🏥 Payment System Status")
            click.echo("=" * 40)
            
            fiat_gateway = await get_fiat_gateway()
            crypto_exchange = await get_crypto_exchange()
            
            # Fiat gateway status
            click.echo("🏦 Fiat Gateway:")
            providers = fiat_gateway.get_supported_providers()
            for provider in providers:
                click.echo(f"   ✅ {provider.title()}")
            click.echo()
            
            # Exchange provider status
            click.echo("⚡ Exchange Providers:")
            exchange_status = crypto_exchange.get_provider_status()
            for provider, status in exchange_status.items():
                status_icon = "✅" if status["enabled"] else "❌"
                click.echo(f"   {status_icon} {provider.title()}")
                if status["cache_size"] > 0:
                    click.echo(f"      Cache: {status['cache_size']} rates")
            click.echo()
            
            # Supported currencies
            click.echo("🌍 Supported Currencies:")
            
            fiat_currencies = [currency.value for currency in FiatCurrency]
            crypto_currencies = [currency.value for currency in CryptoCurrency]
            payment_methods = [method.value.replace('_', ' ').title() for method in PaymentMethod]
            
            click.echo(f"   Fiat: {', '.join(fiat_currencies)}")
            click.echo(f"   Crypto: {', '.join(crypto_currencies)}")
            click.echo(f"   Payment Methods: {', '.join(payment_methods)}")
            click.echo()
            
            # System health
            click.echo("💚 System Health: All systems operational")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get system status: {e}")
            return False
    
    success = asyncio.run(_system_status())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--count', '-n', type=int, default=3, help='Number of test payments to create')
@click.option('--user-id', '-u', default='test_user', help='Test user ID')
def test_payments(count, user_id):
    """Create multiple test payments for system validation"""
    
    async def _test_payments():
        try:
            click.echo(f"🧪 Creating {count} test payments")
            click.echo(f"Test User: {user_id}")
            click.echo()
            
            payment_processor = await get_payment_processor()
            
            test_cases = [
                {"amount": Decimal("100"), "fiat": FiatCurrency.USD, "crypto": CryptoCurrency.FTNS},
                {"amount": Decimal("50"), "fiat": FiatCurrency.EUR, "crypto": CryptoCurrency.MATIC},
                {"amount": Decimal("200"), "fiat": FiatCurrency.GBP, "crypto": CryptoCurrency.USDC},
                {"amount": Decimal("75"), "fiat": FiatCurrency.CAD, "crypto": CryptoCurrency.ETH},
                {"amount": Decimal("150"), "fiat": FiatCurrency.AUD, "crypto": CryptoCurrency.USDT}
            ]
            
            successful_payments = 0
            
            for i in range(count):
                test_case = test_cases[i % len(test_cases)]
                
                click.echo(f"Creating test payment {i+1}/{count}...")
                
                request = PaymentRequest(
                    user_id=user_id,
                    fiat_amount=test_case["amount"],
                    fiat_currency=test_case["fiat"],
                    crypto_currency=test_case["crypto"],
                    payment_method=PaymentMethod.MOCK,
                    metadata={"test": True, "test_number": i+1}
                )
                
                response = await payment_processor.create_payment(request)
                
                if response.success:
                    click.echo(f"✅ Payment {i+1}: {response.transaction_id[:8]} - {response.status.value}")
                    successful_payments += 1
                else:
                    click.echo(f"❌ Payment {i+1}: {response.message}")
                
                # Small delay between payments
                await asyncio.sleep(0.1)
            
            click.echo()
            click.echo(f"🎉 Test completed: {successful_payments}/{count} payments successful")
            
            return successful_payments == count
            
        except Exception as e:
            click.echo(f"❌ Test payments failed: {e}")
            return False
    
    success = asyncio.run(_test_payments())
    sys.exit(0 if success else 1)


@cli.command()
def cleanup():
    """Cleanup payment processor resources"""
    
    async def _cleanup():
        try:
            click.echo("🧹 Cleaning up payment processor resources...")
            
            payment_processor = await get_payment_processor()
            await payment_processor.cleanup()
            
            click.echo("✅ Cleanup completed")
            return True
            
        except Exception as e:
            click.echo(f"❌ Cleanup failed: {e}")
            return False
    
    success = asyncio.run(_cleanup())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    cli()