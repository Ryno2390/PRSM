#!/usr/bin/env python3
"""
Enterprise Integration Suite
============================

Comprehensive enterprise integration platform for connecting with external
systems, databases, APIs, and data sources with ETL/ELT capabilities.

This extends the existing collaborative coding integrations with enterprise-grade
data connectivity and processing capabilities.
"""

from .data_connectors import (
    DataConnector,
    SQLConnector, 
    NoSQLConnector,
    RestAPIConnector,
    FileConnector,
    StreamConnector
)

from .etl_pipeline import (
    ETLPipeline,
    ETLTask,
    PipelineManager,
    DataProcessor
)

from .transformation_engine import (
    TransformationEngine,
    DataTransformer,
    TransformationRule,
    ValidationRule
)

from .sync_manager import (
    DataSyncManager,
    SyncConfiguration,
    RealtimeSync,
    BatchSync
)

from .integration_manager import (
    EnterpriseIntegrationManager,
    Integration,
    IntegrationStatus,
    IntegrationType
)

__version__ = "1.0.0"

__all__ = [
    # Data Connectors
    'DataConnector',
    'SQLConnector', 
    'NoSQLConnector',
    'RestAPIConnector',
    'FileConnector',
    'StreamConnector',
    
    # ETL Pipeline
    'ETLPipeline',
    'ETLTask', 
    'PipelineManager',
    'DataProcessor',
    
    # Transformation Engine
    'TransformationEngine',
    'DataTransformer',
    'TransformationRule',
    'ValidationRule',
    
    # Synchronization
    'DataSyncManager',
    'SyncConfiguration',
    'RealtimeSync',
    'BatchSync',
    
    # Integration Management
    'EnterpriseIntegrationManager',
    'Integration',
    'IntegrationStatus',
    'IntegrationType'
]