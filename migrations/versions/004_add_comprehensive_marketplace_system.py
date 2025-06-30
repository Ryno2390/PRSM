"""Add comprehensive marketplace system

Revision ID: 004_add_comprehensive_marketplace_system
Revises: 003_add_ftns_token_system
Create Date: 2025-06-30 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY

# revision identifiers, used by Alembic.
revision: str = '004_add_comprehensive_marketplace_system'
down_revision: Union[str, None] = '003_add_ftns_token_system'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add comprehensive marketplace system tables"""
    
    # Create marketplace_resources base table
    op.create_table(
        'marketplace_resources',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('resource_type', sa.String(50), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('short_description', sa.String(500)),
        sa.Column('owner_user_id', UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('provider_name', sa.String(255)),
        sa.Column('provider_verified', sa.Boolean, default=False),
        sa.Column('status', sa.String(50), nullable=False, default='draft', index=True),
        sa.Column('quality_grade', sa.String(50), nullable=False, default='experimental'),
        sa.Column('pricing_model', sa.String(50), nullable=False, default='free'),
        sa.Column('base_price', sa.DECIMAL(10, 2), default=0),
        sa.Column('subscription_price', sa.DECIMAL(10, 2), default=0),
        sa.Column('enterprise_price', sa.DECIMAL(10, 2), default=0),
        sa.Column('download_count', sa.Integer, default=0),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('rating_average', sa.DECIMAL(3, 2), default=0),
        sa.Column('rating_count', sa.Integer, default=0),
        sa.Column('version', sa.String(50), default='1.0.0'),
        sa.Column('documentation_url', sa.String(1000)),
        sa.Column('source_url', sa.String(1000)),
        sa.Column('license_type', sa.String(100)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('published_at', sa.DateTime(timezone=True))
    )
    
    # Create indexes for marketplace_resources
    op.create_index('idx_marketplace_resources_type_status', 'marketplace_resources', ['resource_type', 'status'])
    op.create_index('idx_marketplace_resources_owner_type', 'marketplace_resources', ['owner_user_id', 'resource_type'])
    op.create_index('idx_marketplace_resources_rating', 'marketplace_resources', ['rating_average'])
    op.create_index('idx_marketplace_resources_created', 'marketplace_resources', ['created_at'])
    
    # Create AI model listings table
    op.create_table(
        'ai_model_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('model_category', sa.String(50), nullable=False, index=True),
        sa.Column('model_provider', sa.String(100)),
        sa.Column('model_architecture', sa.String(100)),
        sa.Column('parameter_count', sa.String(50)),
        sa.Column('context_length', sa.Integer),
        sa.Column('capabilities', sa.JSON),
        sa.Column('languages_supported', sa.JSON),
        sa.Column('modalities', sa.JSON),
        sa.Column('benchmark_scores', sa.JSON),
        sa.Column('latency_ms', sa.Integer),
        sa.Column('throughput_tokens_per_second', sa.Integer),
        sa.Column('memory_requirements_gb', sa.Integer),
        sa.Column('is_fine_tuned', sa.Boolean, default=False),
        sa.Column('base_model_id', UUID(as_uuid=True), sa.ForeignKey('ai_model_listings.id')),
        sa.Column('fine_tuning_dataset', sa.String(500)),
        sa.Column('fine_tuning_task', sa.String(200)),
        sa.Column('api_endpoint', sa.String(1000)),
        sa.Column('api_key_required', sa.Boolean, default=True),
        sa.Column('rate_limits', sa.JSON)
    )
    
    # Create dataset listings table
    op.create_table(
        'dataset_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('dataset_category', sa.String(50), nullable=False, index=True),
        sa.Column('data_format', sa.String(50), nullable=False),
        sa.Column('size_bytes', sa.BigInteger, nullable=False),
        sa.Column('record_count', sa.Integer, nullable=False),
        sa.Column('feature_count', sa.Integer),
        sa.Column('schema_definition', sa.JSON),
        sa.Column('sample_data', sa.JSON),
        sa.Column('column_descriptions', sa.JSON),
        sa.Column('completeness_score', sa.DECIMAL(3, 2), default=0),
        sa.Column('accuracy_score', sa.DECIMAL(3, 2), default=0),
        sa.Column('consistency_score', sa.DECIMAL(3, 2), default=0),
        sa.Column('freshness_date', sa.DateTime(timezone=True)),
        sa.Column('ethical_review_status', sa.String(50), default='pending'),
        sa.Column('privacy_compliance', sa.JSON),
        sa.Column('data_lineage', sa.JSON),
        sa.Column('bias_assessment', sa.JSON),
        sa.Column('access_url', sa.String(1000)),
        sa.Column('sample_data_url', sa.String(1000)),
        sa.Column('preprocessing_scripts', sa.JSON),
        sa.Column('data_loader_code', sa.Text)
    )
    
    # Create agent workflow listings table
    op.create_table(
        'agent_workflow_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('agent_type', sa.String(50), nullable=False, index=True),
        sa.Column('agent_capabilities', sa.JSON),
        sa.Column('required_models', sa.JSON),
        sa.Column('required_tools', sa.JSON),
        sa.Column('environment_requirements', sa.JSON),
        sa.Column('default_configuration', sa.JSON),
        sa.Column('customization_options', sa.JSON),
        sa.Column('workflow_definition', sa.JSON),
        sa.Column('success_rate', sa.DECIMAL(3, 2), default=0),
        sa.Column('average_execution_time', sa.Integer),
        sa.Column('resource_usage', sa.JSON),
        sa.Column('api_endpoints', sa.JSON),
        sa.Column('webhook_support', sa.Boolean, default=False),
        sa.Column('integration_examples', sa.JSON)
    )
    
    # Create MCP tool listings table
    op.create_table(
        'mcp_tool_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('tool_category', sa.String(50), nullable=False, index=True),
        sa.Column('protocol_version', sa.String(20), default='1.0'),
        sa.Column('functions_provided', sa.JSON),
        sa.Column('input_schema', sa.JSON),
        sa.Column('output_schema', sa.JSON),
        sa.Column('installation_method', sa.String(50)),
        sa.Column('package_name', sa.String(200)),
        sa.Column('container_image', sa.String(500)),
        sa.Column('configuration_schema', sa.JSON),
        sa.Column('security_requirements', sa.JSON),
        sa.Column('sandboxing_enabled', sa.Boolean, default=False),
        sa.Column('permission_requirements', sa.JSON),
        sa.Column('compatible_models', sa.JSON),
        sa.Column('platform_support', sa.JSON),
        sa.Column('dependencies', sa.JSON)
    )
    
    # Create compute resource listings table
    op.create_table(
        'compute_resource_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('cpu_cores', sa.Integer),
        sa.Column('memory_gb', sa.Integer),
        sa.Column('storage_gb', sa.Integer),
        sa.Column('gpu_count', sa.Integer),
        sa.Column('gpu_type', sa.String(100)),
        sa.Column('geographic_regions', sa.JSON),
        sa.Column('availability_schedule', sa.JSON),
        sa.Column('uptime_percentage', sa.DECIMAL(5, 2), default=99.9),
        sa.Column('benchmark_scores', sa.JSON),
        sa.Column('network_bandwidth_gbps', sa.Integer),
        sa.Column('latency_ms', sa.Integer),
        sa.Column('price_per_hour', sa.DECIMAL(10, 4)),
        sa.Column('price_per_compute_unit', sa.DECIMAL(10, 4)),
        sa.Column('minimum_rental_hours', sa.Integer, default=1),
        sa.Column('available_frameworks', sa.JSON),
        sa.Column('container_support', sa.Boolean, default=True),
        sa.Column('custom_images_allowed', sa.Boolean, default=False)
    )
    
    # Create knowledge resource listings table
    op.create_table(
        'knowledge_resource_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('knowledge_type', sa.String(50), nullable=False, index=True),
        sa.Column('domain_specialization', sa.String(100)),
        sa.Column('entity_count', sa.Integer),
        sa.Column('relationship_count', sa.Integer),
        sa.Column('concept_hierarchy_depth', sa.Integer),
        sa.Column('query_languages', sa.JSON),
        sa.Column('api_endpoints', sa.JSON),
        sa.Column('export_formats', sa.JSON),
        sa.Column('expert_validation_score', sa.DECIMAL(3, 2), default=0),
        sa.Column('coverage_completeness', sa.DECIMAL(3, 2), default=0),
        sa.Column('update_frequency', sa.String(50)),
        sa.Column('embedding_models_supported', sa.JSON),
        sa.Column('rag_integration_examples', sa.JSON)
    )
    
    # Create evaluation service listings table
    op.create_table(
        'evaluation_service_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('evaluation_type', sa.String(50), nullable=False, index=True),
        sa.Column('supported_model_types', sa.JSON),
        sa.Column('evaluation_metrics', sa.JSON),
        sa.Column('benchmark_datasets', sa.JSON),
        sa.Column('evaluation_methodology', sa.Text),
        sa.Column('reproducibility_score', sa.DECIMAL(3, 2), default=0),
        sa.Column('peer_review_status', sa.String(50), default='pending'),
        sa.Column('typical_evaluation_time', sa.Integer),
        sa.Column('result_formats', sa.JSON),
        sa.Column('comparison_baselines', sa.JSON),
        sa.Column('standard_compliance', sa.JSON),
        sa.Column('certification_authority', sa.String(200))
    )
    
    # Create training service listings table
    op.create_table(
        'training_service_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('training_type', sa.String(50), nullable=False, index=True),
        sa.Column('supported_frameworks', sa.JSON),
        sa.Column('supported_model_architectures', sa.JSON),
        sa.Column('optimization_techniques', sa.JSON),
        sa.Column('available_compute', sa.JSON),
        sa.Column('maximum_model_size', sa.String(50)),
        sa.Column('distributed_training', sa.Boolean, default=False),
        sa.Column('automated_hyperparameter_tuning', sa.Boolean, default=False),
        sa.Column('early_stopping', sa.Boolean, default=True),
        sa.Column('checkpointing_enabled', sa.Boolean, default=True),
        sa.Column('typical_training_time', sa.Integer),
        sa.Column('model_compression_ratio', sa.DECIMAL(3, 2)),
        sa.Column('performance_retention', sa.DECIMAL(3, 2))
    )
    
    # Create safety tool listings table
    op.create_table(
        'safety_tool_listings',
        sa.Column('id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('safety_category', sa.String(50), nullable=False, index=True),
        sa.Column('supported_model_types', sa.JSON),
        sa.Column('detection_capabilities', sa.JSON),
        sa.Column('prevention_mechanisms', sa.JSON),
        sa.Column('regulatory_compliance', sa.JSON),
        sa.Column('certification_status', sa.String(50)),
        sa.Column('third_party_validated', sa.Boolean, default=False),
        sa.Column('real_time_monitoring', sa.Boolean, default=False),
        sa.Column('batch_processing', sa.Boolean, default=True),
        sa.Column('api_integration', sa.Boolean, default=True),
        sa.Column('false_positive_rate', sa.DECIMAL(5, 4)),
        sa.Column('false_negative_rate', sa.DECIMAL(5, 4)),
        sa.Column('processing_latency_ms', sa.Integer)
    )
    
    # Create marketplace tags table
    op.create_table(
        'marketplace_tags',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('category', sa.String(50)),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )
    
    # Create marketplace reviews table
    op.create_table(
        'marketplace_reviews',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('rating', sa.Integer, nullable=False),
        sa.Column('title', sa.String(200)),
        sa.Column('content', sa.Text),
        sa.Column('verified_purchase', sa.Boolean, default=False),
        sa.Column('helpful_count', sa.Integer, default=0),
        sa.Column('usage_duration_days', sa.Integer),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create marketplace orders table
    op.create_table(
        'marketplace_orders',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('resource_id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), nullable=False, index=True),
        sa.Column('buyer_user_id', UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('order_type', sa.String(50), nullable=False),
        sa.Column('quantity', sa.Integer, default=1),
        sa.Column('unit_price', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('total_price', sa.DECIMAL(10, 2), nullable=False),
        sa.Column('currency', sa.String(10), default='FTNS'),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('payment_status', sa.String(50), nullable=False, default='pending'),
        sa.Column('fulfillment_status', sa.String(50), nullable=False, default='pending'),
        sa.Column('subscription_start_date', sa.DateTime(timezone=True)),
        sa.Column('subscription_end_date', sa.DateTime(timezone=True)),
        sa.Column('auto_renewal', sa.Boolean, default=False),
        sa.Column('rental_start_time', sa.DateTime(timezone=True)),
        sa.Column('rental_end_time', sa.DateTime(timezone=True)),
        sa.Column('transaction_id', sa.String(255)),
        sa.Column('access_granted_at', sa.DateTime(timezone=True)),
        sa.Column('access_expires_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create marketplace analytics table
    op.create_table(
        'marketplace_analytics',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('resource_id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), nullable=False, index=True),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('period_type', sa.String(20), nullable=False),
        sa.Column('views', sa.Integer, default=0),
        sa.Column('downloads', sa.Integer, default=0),
        sa.Column('purchases', sa.Integer, default=0),
        sa.Column('revenue', sa.DECIMAL(10, 2), default=0),
        sa.Column('average_rating', sa.DECIMAL(3, 2)),
        sa.Column('review_count', sa.Integer, default=0),
        sa.Column('conversion_rate', sa.DECIMAL(5, 4)),
        sa.Column('unique_viewers', sa.Integer, default=0),
        sa.Column('return_users', sa.Integer, default=0),
        sa.Column('average_session_duration', sa.Integer),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )
    
    # Create association tables
    op.create_table(
        'marketplace_resource_tags',
        sa.Column('resource_id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('tag_id', UUID(as_uuid=True), sa.ForeignKey('marketplace_tags.id'), primary_key=True)
    )
    
    op.create_table(
        'marketplace_resource_reviews',
        sa.Column('resource_id', UUID(as_uuid=True), sa.ForeignKey('marketplace_resources.id'), primary_key=True),
        sa.Column('review_id', UUID(as_uuid=True), sa.ForeignKey('marketplace_reviews.id'), primary_key=True)
    )
    
    # Create additional indexes for performance
    op.create_index('idx_ai_models_category_provider', 'ai_model_listings', ['model_category', 'model_provider'])
    op.create_index('idx_datasets_category_format', 'dataset_listings', ['dataset_category', 'data_format'])
    op.create_index('idx_agents_type', 'agent_workflow_listings', ['agent_type'])
    op.create_index('idx_tools_category', 'mcp_tool_listings', ['tool_category'])
    op.create_index('idx_compute_type', 'compute_resource_listings', ['resource_type'])
    op.create_index('idx_knowledge_type_domain', 'knowledge_resource_listings', ['knowledge_type', 'domain_specialization'])
    op.create_index('idx_evaluation_type', 'evaluation_service_listings', ['evaluation_type'])
    op.create_index('idx_training_type', 'training_service_listings', ['training_type'])
    op.create_index('idx_safety_category', 'safety_tool_listings', ['safety_category'])
    
    # Indexes for orders and analytics
    op.create_index('idx_marketplace_orders_buyer_status', 'marketplace_orders', ['buyer_user_id', 'status'])
    op.create_index('idx_marketplace_orders_resource_status', 'marketplace_orders', ['resource_id', 'status'])
    op.create_index('idx_marketplace_orders_created', 'marketplace_orders', ['created_at'])
    
    op.create_index('idx_marketplace_analytics_date', 'marketplace_analytics', ['date'])
    op.create_index('idx_marketplace_analytics_resource_date', 'marketplace_analytics', ['resource_id', 'date'])
    
    op.create_index('idx_marketplace_reviews_rating', 'marketplace_reviews', ['rating'])
    op.create_index('idx_marketplace_reviews_created', 'marketplace_reviews', ['created_at'])
    
    # Create unique constraints
    op.create_unique_constraint(
        'unique_analytics_per_period', 
        'marketplace_analytics', 
        ['resource_id', 'date', 'period_type']
    )


def downgrade() -> None:
    """Remove comprehensive marketplace system tables"""
    
    # Drop association tables first
    op.drop_table('marketplace_resource_reviews')
    op.drop_table('marketplace_resource_tags')
    
    # Drop specific resource type tables
    op.drop_table('safety_tool_listings')
    op.drop_table('training_service_listings')
    op.drop_table('evaluation_service_listings')
    op.drop_table('knowledge_resource_listings')
    op.drop_table('compute_resource_listings')
    op.drop_table('mcp_tool_listings')
    op.drop_table('agent_workflow_listings')
    op.drop_table('dataset_listings')
    op.drop_table('ai_model_listings')
    
    # Drop supporting tables
    op.drop_table('marketplace_analytics')
    op.drop_table('marketplace_orders')
    op.drop_table('marketplace_reviews')
    op.drop_table('marketplace_tags')
    
    # Drop base table last
    op.drop_table('marketplace_resources')