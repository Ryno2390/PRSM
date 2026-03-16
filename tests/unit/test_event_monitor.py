"""
Unit tests for Web3 event monitoring.

Tests the event monitor with mocked web3 interactions — no live RPC needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from decimal import Decimal

from prsm.economy.web3.event_monitor import (
    EventProcessor,
    TransferEventProcessor,
    ApprovalEventProcessor,
    Web3EventMonitor,
    EventFilter,
    ProcessedEvent,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_db_service():
    """Create a mock database service."""
    db_service = MagicMock()
    db_service.create_ftns_transaction = AsyncMock()
    db_service.get_ftns_wallet_by_address = AsyncMock(return_value=None)
    db_service.create_ftns_wallet = AsyncMock()
    db_service.update_ftns_wallet = AsyncMock()
    return db_service


@pytest.fixture
def mock_wallet_connector():
    """Create a mock wallet connector with web3 instance."""
    connector = MagicMock()
    connector.w3 = MagicMock()
    connector.w3.eth = MagicMock()
    connector.w3.eth.block_number = 100
    connector.w3.eth.get_block = MagicMock(return_value={'timestamp': 1700000000})
    return connector


@pytest.fixture
def mock_contract_interface():
    """Create a mock contract interface."""
    interface = MagicMock()
    interface.contracts = {}
    return interface


@pytest.fixture
def processed_event():
    """Create a sample processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSToken",
        event_name="Transfer",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "from": "0xSender0000000000000000000000000000000",
            "to": "0xReceiver00000000000000000000000000000",
            "value": 1000000000000000000,  # 1 FTNS with 18 decimals
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.fixture
def approval_processed_event():
    """Create a sample approval processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSToken",
        event_name="Approval",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=1,
        args={
            "owner": "0xOwner00000000000000000000000000000000",
            "spender": "0xSpender00000000000000000000000000000",
            "value": 5000000000000000000,  # 5 FTNS with 18 decimals
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


# =============================================================================
# Test 1: Transfer Processor Records Transaction
# =============================================================================

@pytest.mark.asyncio
async def test_transfer_processor_records_transaction(mock_db_service, processed_event):
    """Test that TransferEventProcessor creates transaction and updates wallet balances."""
    processor = TransferEventProcessor(mock_db_service)
    
    # Mock wallet retrieval to return None (wallet doesn't exist yet)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    result = await processor.process(processed_event)
    
    assert result is True
    # Verify transaction was created
    mock_db_service.create_ftns_transaction.assert_called_once()
    # Verify wallet balance updates were attempted for both sender and receiver
    assert mock_db_service.get_ftns_wallet_by_address.call_count == 2


# =============================================================================
# Test 2: Transfer Processor Handles Missing Fields
# =============================================================================

@pytest.mark.asyncio
async def test_transfer_processor_handles_missing_fields(mock_db_service):
    """Test that TransferEventProcessor gracefully handles events with missing value field."""
    processor = TransferEventProcessor(mock_db_service)
    
    # Create event with missing 'value' field - defaults to 0 per implementation
    event_missing_value = ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSToken",
        event_name="Transfer",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "from": "0xSender0000000000000000000000000000000",
            "to": "0xReceiver00000000000000000000000000000",
            # 'value' is missing - defaults to 0
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )
    
    result = await processor.process(event_missing_value)
    
    # Should return True gracefully (value defaults to 0, not an error)
    assert result is True
    # Transaction should still be created with 0 amount
    mock_db_service.create_ftns_transaction.assert_called_once()


# =============================================================================
# Test 3: Approval Processor Records Approval
# =============================================================================

@pytest.mark.asyncio
async def test_approval_processor_records_approval(mock_db_service, approval_processed_event):
    """Test that ApprovalEventProcessor creates approval transaction record."""
    processor = ApprovalEventProcessor(mock_db_service)
    
    result = await processor.process(approval_processed_event)
    
    assert result is True
    # Verify transaction was created
    mock_db_service.create_ftns_transaction.assert_called_once()
    
    # Verify the transaction was created with correct type
    call_args = mock_db_service.create_ftns_transaction.call_args
    transaction = call_args[0][0]
    assert transaction.transaction_type == "approval"


# =============================================================================
# Test 4: Event Processor Base Raises NotImplementedError
# =============================================================================

@pytest.mark.asyncio
async def test_event_processor_base_raises(mock_db_service, processed_event):
    """Test that base EventProcessor.process() raises NotImplementedError (abstract class)."""
    processor = EventProcessor(mock_db_service)
    
    with pytest.raises(NotImplementedError):
        await processor.process(processed_event)


# =============================================================================
# Test 5: Add Contract Monitor Appends Filters
# =============================================================================

@pytest.mark.asyncio
async def test_add_contract_monitor_appends_filters(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that add_contract_monitor creates and appends event filters."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    # Mock the executor to return a block number
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(return_value=100)
        
        result = await monitor.add_contract_monitor("FTNSToken", ["Transfer"])
    
    assert result is True
    assert len(monitor.event_filters) == 1
    assert monitor.event_filters[0].contract_name == "FTNSToken"
    assert monitor.event_filters[0].event_name == "Transfer"


# =============================================================================
# Test 6: Start Stop Monitoring Lifecycle
# =============================================================================

@pytest.mark.asyncio
async def test_start_stop_monitoring_lifecycle(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that start_monitoring and stop_monitoring correctly manage is_running state."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    # Initially not running
    assert monitor.is_running is False
    
    # Start monitoring
    await monitor.start_monitoring()
    assert monitor.is_running is True
    
    # Stop monitoring
    await monitor.stop_monitoring()
    assert monitor.is_running is False


# =============================================================================
# Test 7: Duplicate Event Deduplication
# =============================================================================

@pytest.mark.asyncio
async def test_duplicate_event_deduplication(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that processing the same event_id twice only counts as one processed event."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    # Create a mock log entry
    mock_log = {
        'transactionHash': MagicMock(hex=lambda: '0xabcdef123456'),
        'logIndex': 0,
        'blockNumber': 100,
        'address': '0x1234567890abcdef',
        'event': 'Transfer',
        'args': {
            'from': '0xSender',
            'to': '0xReceiver',
            'value': 1000000000000000000
        }
    }
    
    # Mock the block retrieval
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            side_effect=[
                {'timestamp': 1700000000},  # get_block call
            ]
        )
        
        # Process the same event twice
        await monitor._process_event_log(mock_log, "FTNSToken")
        initial_count = monitor.events_processed
        
        await monitor._process_event_log(mock_log, "FTNSToken")
    
    # Should only be processed once due to deduplication
    assert monitor.events_processed == initial_count
    assert len(monitor.processed_events) == 1


# =============================================================================
# Test 8: Get Monitoring Status Shape
# =============================================================================

def test_get_monitoring_status_shape(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that get_monitoring_status returns dict with all required keys."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    status = monitor.get_monitoring_status()
    
    # Assert required keys are present
    required_keys = [
        "is_running",
        "active_filters",
        "total_filters",
        "events_processed",
        "errors_encountered",
        "last_processed_block",
        "last_activity",
        "filters"
    ]
    
    for key in required_keys:
        assert key in status, f"Missing required key: {key}"


# =============================================================================
# Test 9: Pause Resume Filter
# =============================================================================

@pytest.mark.asyncio
async def test_pause_resume_filter(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that pause_monitoring and resume_monitoring correctly toggle filter active state."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    # Add a filter
    monitor.event_filters.append(EventFilter(
        contract_name="FTNSToken",
        event_name="Transfer",
        from_block=100
    ))
    
    # Initially active
    assert monitor.event_filters[0].active is True
    
    # Pause the filter
    await monitor.pause_monitoring(contract_name="FTNSToken", event_name="Transfer")
    assert monitor.event_filters[0].active is False
    
    # Resume the filter
    await monitor.resume_monitoring(contract_name="FTNSToken", event_name="Transfer")
    assert monitor.event_filters[0].active is True


# =============================================================================
# Test 10: Get Event Logs Uses Executor
# =============================================================================

@pytest.mark.asyncio
async def test_get_event_logs_uses_executor(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that _get_event_logs uses executor for web3 calls (create_filter and get_all_entries)."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    # Create a mock contract with mock event
    mock_contract = MagicMock()
    mock_event = MagicMock()
    mock_filter = MagicMock()
    mock_filter.get_all_entries = MagicMock(return_value=[])
    mock_event.create_filter = MagicMock(return_value=mock_filter)
    mock_contract.events.Transfer = mock_event
    
    # Track executor calls
    executor_calls = []
    
    async def mock_run_in_executor(None_arg, func):
        executor_calls.append(func)
        return func()
    
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor
        
        logs = await monitor._get_event_logs(
            contract=mock_contract,
            event_name="Transfer",
            from_block=100,
            to_block=200
        )
    
    # Verify both create_filter and get_all_entries were called via executor
    assert len(executor_calls) == 2
    assert mock_event.create_filter.called
    assert mock_filter.get_all_entries.called
    assert logs == []
