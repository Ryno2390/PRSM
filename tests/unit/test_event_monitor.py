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
    MintEventProcessor,
    BurnEventProcessor,
    StakedEventProcessor,
    UnstakedEventProcessor,
    RewardsClaimedEventProcessor,
    PurchaseEventProcessor,
    ListingCreatedEventProcessor,
    BridgeOutEventProcessor,
    BridgeInEventProcessor,
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


def make_mock_wallet(address: str, balance: Decimal, locked_balance: Decimal = Decimal('0'), staked_balance: Decimal = Decimal('0')):
    """Create a mock wallet object with the necessary attributes for testing."""
    wallet = MagicMock()
    wallet.blockchain_address = address
    wallet.balance = balance
    wallet.locked_balance = locked_balance
    wallet.staked_balance = staked_balance
    return wallet


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
async def test_event_processor_base_raises(mock_db_service):
    """Test that EventProcessor cannot be instantiated directly (raises TypeError for abstract class)."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        EventProcessor(mock_db_service)


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


# =============================================================================
# Base Class Tests
# =============================================================================

def test_base_processor_is_abstract(mock_db_service):
    """Test that EventProcessor cannot be instantiated directly (raises TypeError)."""
    with pytest.raises(TypeError):
        EventProcessor(mock_db_service)


@pytest.mark.asyncio
async def test_update_wallet_balance_creates_wallet_if_missing(mock_db_service):
    """Test that _update_wallet_balance creates a wallet if it doesn't exist."""
    # Create a concrete processor to test the base class method
    processor = MintEventProcessor(mock_db_service)
    
    # Mock wallet retrieval to return None (wallet doesn't exist)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    await processor._update_wallet_balance("0xnewwallet", balance_delta=Decimal('100'))
    
    # Verify wallet was created
    mock_db_service.create_ftns_wallet.assert_called_once()
    created_wallet = mock_db_service.create_ftns_wallet.call_args[0][0]
    assert created_wallet.blockchain_address == "0xnewwallet"
    assert created_wallet.balance == Decimal('100')


@pytest.mark.asyncio
async def test_update_wallet_balance_clamps_to_zero(mock_db_service):
    """Test that negative delta that would produce negative balance is clamped to 0."""
    processor = BurnEventProcessor(mock_db_service)
    
    # Create existing wallet with small balance
    existing_wallet = make_mock_wallet("0xtestwallet", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    # Try to subtract more than balance
    await processor._update_wallet_balance("0xtestwallet", balance_delta=-Decimal('100'))
    
    # Verify balance was clamped to 0
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    assert updated_wallet.balance == Decimal('0')


@pytest.mark.asyncio
async def test_update_wallet_balance_updates_staked_balance(mock_db_service):
    """Test that staked_delta is applied independently."""
    processor = StakedEventProcessor(mock_db_service)
    
    existing_wallet = make_mock_wallet("0xstaker", Decimal('100'), staked_balance=Decimal('50'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor._update_wallet_balance("0xstaker", balance_delta=-Decimal('30'), staked_delta=Decimal('30'))
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    assert updated_wallet.balance == Decimal('70')
    assert updated_wallet.staked_balance == Decimal('80')


# =============================================================================
# MintEventProcessor Tests
# =============================================================================

@pytest.fixture
def mint_processed_event():
    """Create a sample mint processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSToken",
        event_name="Mint",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "to": "0xRecipient0000000000000000000000000000",
            "value": 2000000000000000000,  # 2 FTNS
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_mint_processor_process_success(mock_db_service, mint_processed_event):
    """Test that MintEventProcessor processes successfully."""
    processor = MintEventProcessor(mock_db_service)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    result = await processor.process(mint_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_mint_processor_balance_change_correct(mock_db_service, mint_processed_event):
    """Test that MintEventProcessor applies correct balance deltas."""
    processor = MintEventProcessor(mock_db_service)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    await processor.process(mint_processed_event)
    
    # Verify wallet was created with correct balance
    created_wallet = mock_db_service.create_ftns_wallet.call_args[0][0]
    assert created_wallet.balance == Decimal('2')


@pytest.mark.asyncio
async def test_mint_processor_transaction_type_correct(mock_db_service, mint_processed_event):
    """Test that MintEventProcessor creates transaction with correct transaction_type."""
    processor = MintEventProcessor(mock_db_service)
    
    await processor.process(mint_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "mint"


@pytest.mark.asyncio
async def test_mint_processor_process_exception_returns_false(mock_db_service, mint_processed_event):
    """Test that MintEventProcessor returns False on exception."""
    processor = MintEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(mint_processed_event)
    
    assert result is False


# =============================================================================
# BurnEventProcessor Tests
# =============================================================================

@pytest.fixture
def burn_processed_event():
    """Create a sample burn processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSToken",
        event_name="Burn",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "from": "0xBurner0000000000000000000000000000000",
            "value": 500000000000000000,  # 0.5 FTNS
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_burn_processor_process_success(mock_db_service, burn_processed_event):
    """Test that BurnEventProcessor processes successfully."""
    processor = BurnEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xburner0000000000000000000000000000000", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    result = await processor.process(burn_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_burn_processor_balance_change_correct(mock_db_service, burn_processed_event):
    """Test that BurnEventProcessor applies correct balance deltas."""
    processor = BurnEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xburner0000000000000000000000000000000", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(burn_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    assert updated_wallet.balance == Decimal('9.5')


@pytest.mark.asyncio
async def test_burn_processor_transaction_type_correct(mock_db_service, burn_processed_event):
    """Test that BurnEventProcessor creates transaction with correct transaction_type."""
    processor = BurnEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xburner0000000000000000000000000000000", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(burn_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "burn"


@pytest.mark.asyncio
async def test_burn_processor_process_exception_returns_false(mock_db_service, burn_processed_event):
    """Test that BurnEventProcessor returns False on exception."""
    processor = BurnEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(burn_processed_event)
    
    assert result is False


# =============================================================================
# StakedEventProcessor Tests
# =============================================================================

@pytest.fixture
def staked_processed_event():
    """Create a sample staked processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSStaking",
        event_name="Staked",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "user": "0xStaker0000000000000000000000000000000",
            "poolId": 1,
            "stakeId": 42,
            "amount": 3000000000000000000,  # 3 FTNS
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_staked_processor_process_success(mock_db_service, staked_processed_event):
    """Test that StakedEventProcessor processes successfully."""
    processor = StakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xstaker0000000000000000000000000000000", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    result = await processor.process(staked_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_staked_processor_balance_change_correct(mock_db_service, staked_processed_event):
    """Test that StakedEventProcessor applies correct balance deltas."""
    processor = StakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xstaker0000000000000000000000000000000", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(staked_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    assert updated_wallet.balance == Decimal('7')  # 10 - 3
    assert updated_wallet.staked_balance == Decimal('3')


@pytest.mark.asyncio
async def test_staked_processor_transaction_type_correct(mock_db_service, staked_processed_event):
    """Test that StakedEventProcessor creates transaction with correct transaction_type."""
    processor = StakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xstaker0000000000000000000000000000000", Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(staked_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "staking"


@pytest.mark.asyncio
async def test_staked_processor_process_exception_returns_false(mock_db_service, staked_processed_event):
    """Test that StakedEventProcessor returns False on exception."""
    processor = StakedEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(staked_processed_event)
    
    assert result is False


# =============================================================================
# UnstakedEventProcessor Tests
# =============================================================================

@pytest.fixture
def unstaked_processed_event():
    """Create a sample unstaked processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSStaking",
        event_name="Unstaked",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "user": "0xUnstaker0000000000000000000000000000",
            "poolId": 1,
            "stakeId": 42,
            "amount": 2000000000000000000,  # 2 FTNS principal
            "rewards": 500000000000000000,  # 0.5 FTNS rewards
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_unstaked_processor_process_success(mock_db_service, unstaked_processed_event):
    """Test that UnstakedEventProcessor processes successfully."""
    processor = UnstakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xunstaker0000000000000000000000000000", Decimal('5'), staked_balance=Decimal('2'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    result = await processor.process(unstaked_processed_event)
    
    assert result is True
    # Should create 2 transactions: one for principal, one for rewards
    assert mock_db_service.create_ftns_transaction.call_count == 2


@pytest.mark.asyncio
async def test_unstaked_processor_balance_change_correct(mock_db_service, unstaked_processed_event):
    """Test that UnstakedEventProcessor applies correct balance deltas."""
    processor = UnstakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xunstaker0000000000000000000000000000", Decimal('5'), staked_balance=Decimal('2'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(unstaked_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    # balance = 5 + 2 (principal) + 0.5 (rewards) = 7.5
    assert updated_wallet.balance == Decimal('7.5')
    # staked_balance = 2 - 2 (principal only, not rewards) = 0
    assert updated_wallet.staked_balance == Decimal('0')


@pytest.mark.asyncio
async def test_unstaked_processor_transaction_type_correct(mock_db_service, unstaked_processed_event):
    """Test that UnstakedEventProcessor creates transactions with correct transaction_types."""
    processor = UnstakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xunstaker0000000000000000000000000000", Decimal('5'), staked_balance=Decimal('2'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(unstaked_processed_event)
    
    calls = mock_db_service.create_ftns_transaction.call_args_list
    transaction_types = [call[0][0].transaction_type for call in calls]
    assert "unstaking" in transaction_types
    assert "staking_reward" in transaction_types


@pytest.mark.asyncio
async def test_unstaked_processor_process_exception_returns_false(mock_db_service, unstaked_processed_event):
    """Test that UnstakedEventProcessor returns False on exception."""
    processor = UnstakedEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(unstaked_processed_event)
    
    assert result is False


@pytest.mark.asyncio
async def test_unstaked_records_both_principal_and_rewards(mock_db_service, unstaked_processed_event):
    """Test that UnstakedEventProcessor creates two FTNSTransaction records."""
    processor = UnstakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xunstaker0000000000000000000000000000", Decimal('5'), staked_balance=Decimal('2'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(unstaked_processed_event)
    
    assert mock_db_service.create_ftns_transaction.call_count == 2


@pytest.mark.asyncio
async def test_staked_balance_decrements_on_unstake(mock_db_service, unstaked_processed_event):
    """Test that staked_delta is -principal, not -total."""
    processor = UnstakedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xunstaker0000000000000000000000000000", Decimal('5'), staked_balance=Decimal('2'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(unstaked_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    # staked_delta should be -2 (principal), not -2.5 (total)
    assert updated_wallet.staked_balance == Decimal('0')


# =============================================================================
# RewardsClaimedEventProcessor Tests
# =============================================================================

@pytest.fixture
def rewards_claimed_processed_event():
    """Create a sample rewards claimed processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSStaking",
        event_name="RewardsClaimed",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "user": "0xClaimer00000000000000000000000000000",
            "poolId": 1,
            "stakeId": 42,
            "rewards": 1500000000000000000,  # 1.5 FTNS
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_rewards_claimed_processor_process_success(mock_db_service, rewards_claimed_processed_event):
    """Test that RewardsClaimedEventProcessor processes successfully."""
    processor = RewardsClaimedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xclaimer00000000000000000000000000000", Decimal('5'), staked_balance=Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    result = await processor.process(rewards_claimed_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_rewards_claimed_processor_balance_change_correct(mock_db_service, rewards_claimed_processed_event):
    """Test that RewardsClaimedEventProcessor applies correct balance deltas."""
    processor = RewardsClaimedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xclaimer00000000000000000000000000000", Decimal('5'), staked_balance=Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(rewards_claimed_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    assert updated_wallet.balance == Decimal('6.5')  # 5 + 1.5


@pytest.mark.asyncio
async def test_rewards_claimed_processor_transaction_type_correct(mock_db_service, rewards_claimed_processed_event):
    """Test that RewardsClaimedEventProcessor creates transaction with correct transaction_type."""
    processor = RewardsClaimedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xclaimer00000000000000000000000000000", Decimal('5'), staked_balance=Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(rewards_claimed_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "staking_reward"


@pytest.mark.asyncio
async def test_rewards_claimed_processor_process_exception_returns_false(mock_db_service, rewards_claimed_processed_event):
    """Test that RewardsClaimedEventProcessor returns False on exception."""
    processor = RewardsClaimedEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(rewards_claimed_processed_event)
    
    assert result is False


@pytest.mark.asyncio
async def test_rewards_claimed_does_not_change_staked_balance(mock_db_service, rewards_claimed_processed_event):
    """Test that staked_delta=0 (stake continues)."""
    processor = RewardsClaimedEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xclaimer00000000000000000000000000000", Decimal('5'), staked_balance=Decimal('10'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(rewards_claimed_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    assert updated_wallet.staked_balance == Decimal('10')  # unchanged


# =============================================================================
# PurchaseEventProcessor Tests
# =============================================================================

@pytest.fixture
def purchase_processed_event():
    """Create a sample purchase processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSMarketplace",
        event_name="Purchase",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "purchaseId": "0xpurchase1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "listingId": 5,
            "buyer": "0xBuyer0000000000000000000000000000000",
            "seller": "0xSeller000000000000000000000000000000",
            "quantity": 1,
            "totalPrice": 4000000000000000000,  # 4 FTNS
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_purchase_processor_process_success(mock_db_service, purchase_processed_event):
    """Test that PurchaseEventProcessor processes successfully."""
    processor = PurchaseEventProcessor(mock_db_service)
    mock_db_service.store_royalty_distribution_record = AsyncMock()
    
    result = await processor.process(purchase_processed_event)
    
    assert result is True
    # Should create 2 transactions: buyer debit and seller credit
    assert mock_db_service.create_ftns_transaction.call_count == 2


@pytest.mark.asyncio
async def test_purchase_processor_balance_change_correct(mock_db_service, purchase_processed_event):
    """Test that PurchaseEventProcessor applies correct balance deltas."""
    processor = PurchaseEventProcessor(mock_db_service)
    mock_db_service.store_royalty_distribution_record = AsyncMock()
    
    buyer_wallet = make_mock_wallet("0xbuyer0000000000000000000000000000000", Decimal('10'))
    seller_wallet = make_mock_wallet("0xseller000000000000000000000000000000", Decimal('5'))
    
    def get_wallet(address):
        if "buyer" in address.lower():
            return buyer_wallet
        elif "seller" in address.lower():
            return seller_wallet
        return None
    
    mock_db_service.get_ftns_wallet_by_address.side_effect = get_wallet
    
    await processor.process(purchase_processed_event)
    
    # Verify both wallets were updated
    assert mock_db_service.update_ftns_wallet.call_count == 2


@pytest.mark.asyncio
async def test_purchase_processor_transaction_type_correct(mock_db_service, purchase_processed_event):
    """Test that PurchaseEventProcessor creates transactions with correct transaction_types."""
    processor = PurchaseEventProcessor(mock_db_service)
    mock_db_service.store_royalty_distribution_record = AsyncMock()
    
    await processor.process(purchase_processed_event)
    
    calls = mock_db_service.create_ftns_transaction.call_args_list
    transaction_types = [call[0][0].transaction_type for call in calls]
    assert "marketplace_purchase" in transaction_types
    assert "marketplace_sale" in transaction_types


@pytest.mark.asyncio
async def test_purchase_processor_process_exception_returns_false(mock_db_service, purchase_processed_event):
    """Test that PurchaseEventProcessor returns False on exception."""
    processor = PurchaseEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(purchase_processed_event)
    
    assert result is False


@pytest.mark.asyncio
async def test_purchase_triggers_royalty_record(mock_db_service, purchase_processed_event):
    """Test that store_royalty_distribution_record is called."""
    processor = PurchaseEventProcessor(mock_db_service)
    mock_db_service.store_royalty_distribution_record = AsyncMock()
    
    await processor.process(purchase_processed_event)
    
    mock_db_service.store_royalty_distribution_record.assert_called_once()
    royalty_record = mock_db_service.store_royalty_distribution_record.call_args[0][0]
    assert royalty_record["buyer"] == "0xbuyer0000000000000000000000000000000"
    assert royalty_record["seller"] == "0xseller000000000000000000000000000000"
    assert royalty_record["amount"] == 4.0


# =============================================================================
# ListingCreatedEventProcessor Tests
# =============================================================================

@pytest.fixture
def listing_created_processed_event():
    """Create a sample listing created processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSMarketplace",
        event_name="ListingCreated",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "listingId": 10,
            "seller": "0xLister000000000000000000000000000000",
            "assetType": 1,
            "price": 2500000000000000000,  # 2.5 FTNS
            "quantity": 3,
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_listing_created_processor_process_success(mock_db_service, listing_created_processed_event):
    """Test that ListingCreatedEventProcessor processes successfully."""
    processor = ListingCreatedEventProcessor(mock_db_service)
    
    result = await processor.process(listing_created_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_listing_created_processor_transaction_type_correct(mock_db_service, listing_created_processed_event):
    """Test that ListingCreatedEventProcessor creates transaction with correct transaction_type."""
    processor = ListingCreatedEventProcessor(mock_db_service)
    
    await processor.process(listing_created_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "marketplace_listing"


@pytest.mark.asyncio
async def test_listing_created_processor_process_exception_returns_false(mock_db_service, listing_created_processed_event):
    """Test that ListingCreatedEventProcessor returns False on exception."""
    processor = ListingCreatedEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(listing_created_processed_event)
    
    assert result is False


@pytest.mark.asyncio
async def test_listing_created_no_balance_change(mock_db_service, listing_created_processed_event):
    """Test that _update_wallet_balance is not called for listing creation."""
    processor = ListingCreatedEventProcessor(mock_db_service)
    
    await processor.process(listing_created_processed_event)
    
    # Wallet balance should not be updated
    mock_db_service.update_ftns_wallet.assert_not_called()


# =============================================================================
# BridgeOutEventProcessor Tests
# =============================================================================

@pytest.fixture
def bridge_out_processed_event():
    """Create a sample bridge out processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSBridge",
        event_name="BridgeOut",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "sender": "0xBridger00000000000000000000000000000",
            "amount": 10000000000000000000,  # 10 FTNS
            "fee": 100000000000000000,  # 0.1 FTNS
            "destinationChain": 5,
            "nonce": 12345,
            "transactionId": "0xbridge1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_bridge_out_processor_process_success(mock_db_service, bridge_out_processed_event):
    """Test that BridgeOutEventProcessor processes successfully."""
    processor = BridgeOutEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xbridger00000000000000000000000000000", Decimal('20'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    result = await processor.process(bridge_out_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_bridge_out_processor_balance_change_correct(mock_db_service, bridge_out_processed_event):
    """Test that BridgeOutEventProcessor applies correct balance deltas (amount + fee)."""
    processor = BridgeOutEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xbridger00000000000000000000000000000", Decimal('20'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(bridge_out_processed_event)
    
    updated_wallet = mock_db_service.update_ftns_wallet.call_args[0][0]
    # 20 - 10 (amount) - 0.1 (fee) = 9.9
    assert updated_wallet.balance == Decimal('9.9')


@pytest.mark.asyncio
async def test_bridge_out_processor_transaction_type_correct(mock_db_service, bridge_out_processed_event):
    """Test that BridgeOutEventProcessor creates transaction with correct transaction_type."""
    processor = BridgeOutEventProcessor(mock_db_service)
    existing_wallet = make_mock_wallet("0xbridger00000000000000000000000000000", Decimal('20'))
    mock_db_service.get_ftns_wallet_by_address.return_value = existing_wallet
    
    await processor.process(bridge_out_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "bridge_out"


@pytest.mark.asyncio
async def test_bridge_out_processor_process_exception_returns_false(mock_db_service, bridge_out_processed_event):
    """Test that BridgeOutEventProcessor returns False on exception."""
    processor = BridgeOutEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(bridge_out_processed_event)
    
    assert result is False


# =============================================================================
# BridgeInEventProcessor Tests
# =============================================================================

@pytest.fixture
def bridge_in_processed_event():
    """Create a sample bridge in processed event for testing."""
    return ProcessedEvent(
        contract_address="0x1234567890abcdef1234567890abcdef12345678",
        contract_name="FTNSBridge",
        event_name="BridgeIn",
        block_number=100,
        transaction_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        log_index=0,
        args={
            "user": "0xReceiver0000000000000000000000000000",
            "amount": 5000000000000000000,  # 5 FTNS
            "sourceChain": 1,
            "sourceTransactionId": "0xsource1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "transactionId": "0xbridgein1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        },
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        processed_at=datetime(2024, 1, 1, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_bridge_in_processor_process_success(mock_db_service, bridge_in_processed_event):
    """Test that BridgeInEventProcessor processes successfully."""
    processor = BridgeInEventProcessor(mock_db_service)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    result = await processor.process(bridge_in_processed_event)
    
    assert result is True
    mock_db_service.create_ftns_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_bridge_in_processor_balance_change_correct(mock_db_service, bridge_in_processed_event):
    """Test that BridgeInEventProcessor applies correct balance deltas."""
    processor = BridgeInEventProcessor(mock_db_service)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    await processor.process(bridge_in_processed_event)
    
    created_wallet = mock_db_service.create_ftns_wallet.call_args[0][0]
    assert created_wallet.balance == Decimal('5')


@pytest.mark.asyncio
async def test_bridge_in_processor_transaction_type_correct(mock_db_service, bridge_in_processed_event):
    """Test that BridgeInEventProcessor creates transaction with correct transaction_type."""
    processor = BridgeInEventProcessor(mock_db_service)
    mock_db_service.get_ftns_wallet_by_address.return_value = None
    
    await processor.process(bridge_in_processed_event)
    
    transaction = mock_db_service.create_ftns_transaction.call_args[0][0]
    assert transaction.transaction_type == "bridge_in"


@pytest.mark.asyncio
async def test_bridge_in_processor_process_exception_returns_false(mock_db_service, bridge_in_processed_event):
    """Test that BridgeInEventProcessor returns False on exception."""
    processor = BridgeInEventProcessor(mock_db_service)
    mock_db_service.create_ftns_transaction.side_effect = Exception("DB error")
    
    result = await processor.process(bridge_in_processed_event)
    
    assert result is False


# =============================================================================
# Monitor Registration Tests
# =============================================================================

def test_setup_default_processors_registers_all_11_events(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that _setup_default_processors registers all 11 event processors."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    assert len(monitor.event_processors) == 11
    expected_events = [
        "Transfer", "Approval", "Mint", "Burn",
        "Staked", "Unstaked", "RewardsClaimed",
        "Purchase", "ListingCreated",
        "BridgeOut", "BridgeIn"
    ]
    for event in expected_events:
        assert event in monitor.event_processors


@pytest.mark.asyncio
async def test_unknown_event_logs_debug_not_error(mock_wallet_connector, mock_contract_interface, mock_db_service):
    """Test that no processor for event logs debug, not error, and no crash."""
    monitor = Web3EventMonitor(
        wallet_connector=mock_wallet_connector,
        contract_interface=mock_contract_interface,
        db_service=mock_db_service
    )
    
    # Create a mock log entry for an unknown event
    mock_log = {
        'transactionHash': MagicMock(hex=lambda: '0xabcdef123456'),
        'logIndex': 0,
        'blockNumber': 100,
        'address': '0x1234567890abcdef',
        'event': 'UnknownEvent',
        'args': {}
    }
    
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            return_value={'timestamp': 1700000000}
        )
        
        # Should not raise an exception
        await monitor._process_event_log(mock_log, "FTNSToken")
    
    # Event should not be added to processed_events
    assert len(monitor.processed_events) == 0
    # errors_encountered should still be 0 (debug log, not error)
    assert monitor.errors_encountered == 0
