# Data Management API

Handle datasets, embeddings, storage, and data processing operations within the PRSM ecosystem.

## 🎯 Overview

The Data Management API provides comprehensive data handling capabilities including dataset storage, vector embeddings, data processing pipelines, and distributed data synchronization across the PRSM network.

## 📋 Base URL

```
https://api.prsm.ai/v1/data
```

## 🔐 Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/data
```

## 🚀 Quick Start

### Upload Dataset

```python
import prsm

client = prsm.Client(api_key="your-api-key")

# Upload a dataset
dataset = client.data.upload_dataset(
    name="research_papers",
    description="Collection of AI research papers",
    file_path="./papers.jsonl",
    format="jsonl",
    metadata={
        "domain": "artificial_intelligence",
        "language": "english",
        "size_mb": 150
    }
)

print(f"Dataset uploaded: {dataset.id}")
```

## 📊 Endpoints

### POST /data/datasets
Create and upload a new dataset.

**Request Body (Multipart Form):**
```
Content-Type: multipart/form-data

file: [binary data]
metadata: {
  "name": "scientific_papers",
  "description": "Collection of scientific papers for analysis",
  "format": "jsonl",
  "schema": {
    "title": "string",
    "abstract": "string", 
    "authors": "array",
    "published_date": "datetime"
  },
  "tags": ["research", "nlp", "scientific"],
  "access_level": "private"
}
```

**Response:**
```json
{
  "id": "ds_abc123",
  "name": "scientific_papers",
  "status": "processing",
  "size_bytes": 157286400,
  "record_count": 50000,
  "upload_progress": 1.0,
  "processing_progress": 0.3,
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:45:00Z",
  "storage_location": "s3://prsm-datasets/ds_abc123/",
  "schema_validation": "passed"
}
```

### GET /data/datasets
List all datasets.

**Query Parameters:**
- `limit`: Maximum number of datasets to return (default: 50)
- `offset`: Number of datasets to skip
- `tags`: Filter by tags (comma-separated)
- `format`: Filter by format (json, jsonl, csv, parquet)
- `access_level`: Filter by access level (public, private, shared)

**Response:**
```json
{
  "datasets": [
    {
      "id": "ds_abc123",
      "name": "scientific_papers",
      "description": "Collection of scientific papers",
      "size_bytes": 157286400,
      "record_count": 50000,
      "format": "jsonl",
      "created_at": "2024-01-15T10:30:00Z",
      "last_accessed": "2024-01-16T14:20:00Z",
      "access_count": 25,
      "tags": ["research", "nlp", "scientific"]
    }
  ],
  "total_count": 127,
  "storage_used_bytes": 2147483648
}
```

### GET /data/datasets/{dataset_id}
Get dataset details and metadata.

**Response:**
```json
{
  "id": "ds_abc123",
  "name": "scientific_papers",
  "description": "Collection of scientific papers for analysis",
  "size_bytes": 157286400,
  "record_count": 50000,
  "format": "jsonl",
  "schema": {
    "title": {"type": "string", "required": true},
    "abstract": {"type": "string", "required": true},
    "authors": {"type": "array", "items": {"type": "string"}},
    "published_date": {"type": "datetime"}
  },
  "statistics": {
    "avg_record_size": 3145,
    "min_record_size": 512,
    "max_record_size": 8192,
    "null_value_percentage": 0.02
  },
  "sample_records": [
    {
      "title": "Attention Is All You Need",
      "abstract": "The dominant sequence transduction models...",
      "authors": ["Ashish Vaswani", "Noam Shazeer"],
      "published_date": "2017-06-12T00:00:00Z"
    }
  ],
  "storage_info": {
    "location": "s3://prsm-datasets/ds_abc123/",
    "compression": "gzip",
    "encryption": "AES-256",
    "backup_count": 3
  }
}
```

### POST /data/datasets/{dataset_id}/query
Query dataset with filters and aggregations.

**Request Body:**
```json
{
  "filters": {
    "published_date": {
      "gte": "2020-01-01T00:00:00Z",
      "lte": "2023-12-31T23:59:59Z"
    },
    "authors": {
      "contains": "Transformer"
    }
  },
  "aggregations": {
    "year_distribution": {
      "type": "date_histogram",
      "field": "published_date",
      "interval": "year"
    },
    "author_count": {
      "type": "cardinality",
      "field": "authors"
    }
  },
  "limit": 100,
  "offset": 0,
  "sort": [
    {"field": "published_date", "order": "desc"}
  ]
}
```

**Response:**
```json
{
  "records": [
    {
      "title": "Recent Advances in Transformer Models",
      "abstract": "This paper reviews recent developments...",
      "authors": ["Jane Doe", "John Smith"],
      "published_date": "2023-11-15T00:00:00Z"
    }
  ],
  "total_matches": 1247,
  "aggregations": {
    "year_distribution": {
      "buckets": [
        {"key": "2023", "doc_count": 450},
        {"key": "2022", "doc_count": 380},
        {"key": "2021", "doc_count": 320}
      ]
    },
    "author_count": {
      "value": 2847
    }
  },
  "query_time_ms": 45
}
```

### POST /data/embeddings
Generate embeddings for datasets or text.

**Request Body:**
```json
{
  "input_type": "dataset",
  "dataset_id": "ds_abc123",
  "field": "abstract",
  "embedding_model": "text-embedding-ada-002",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "batch_size": 100,
  "output_format": "numpy"
}
```

**Response:**
```json
{
  "embedding_job_id": "emb_xyz789",
  "status": "processing",
  "progress": 0.15,
  "estimated_completion": "2024-01-15T11:30:00Z",
  "embedding_config": {
    "model": "text-embedding-ada-002",
    "dimensions": 1536,
    "chunk_count": 75000
  },
  "cost_estimate": 25.50
}
```

### GET /data/embeddings/{embedding_job_id}
Get embedding generation status and results.

**Response:**
```json
{
  "embedding_job_id": "emb_xyz789",
  "status": "completed",
  "progress": 1.0,
  "started_at": "2024-01-15T10:45:00Z",
  "completed_at": "2024-01-15T11:25:00Z",
  "embedding_config": {
    "model": "text-embedding-ada-002",
    "dimensions": 1536,
    "chunk_count": 75000
  },
  "results": {
    "embedding_file_url": "https://storage.prsm.ai/embeddings/emb_xyz789.npy",
    "metadata_file_url": "https://storage.prsm.ai/embeddings/emb_xyz789_meta.json",
    "index_file_url": "https://storage.prsm.ai/embeddings/emb_xyz789.index"
  },
  "statistics": {
    "total_tokens": 2500000,
    "avg_embedding_norm": 0.85,
    "cost": 25.00
  }
}
```

### POST /data/search
Perform semantic search across embeddings.

**Request Body:**
```json
{
  "query": "machine learning transformer architecture",
  "embedding_job_id": "emb_xyz789",
  "top_k": 10,
  "similarity_threshold": 0.7,
  "filters": {
    "published_date": {"gte": "2020-01-01"}
  },
  "include_metadata": true,
  "include_similarity_scores": true
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "chunk_12345",
      "content": "Transformer architectures have revolutionized...",
      "similarity_score": 0.92,
      "metadata": {
        "source_record_id": "record_456",
        "chunk_index": 2,
        "title": "Attention Mechanisms in Deep Learning",
        "authors": ["Alice Johnson"]
      }
    }
  ],
  "total_results": 10,
  "query_time_ms": 25,
  "embedding_model": "text-embedding-ada-002"
}
```

## 🗂️ Data Processing

### Data Transformation

```python
# Transform dataset format
transformation = client.data.transform(
    dataset_id="ds_abc123",
    transformations=[
        {
            "type": "format_conversion",
            "from_format": "jsonl",
            "to_format": "parquet"
        },
        {
            "type": "column_selection",
            "columns": ["title", "abstract", "authors"]
        },
        {
            "type": "text_cleaning",
            "fields": ["title", "abstract"],
            "operations": ["remove_html", "normalize_whitespace"]
        }
    ]
)
```

### Data Validation

```python
# Validate dataset quality
validation = client.data.validate(
    dataset_id="ds_abc123",
    validation_rules=[
        {
            "field": "title",
            "rule": "not_null",
            "severity": "error"
        },
        {
            "field": "abstract", 
            "rule": "min_length",
            "value": 50,
            "severity": "warning"
        },
        {
            "field": "published_date",
            "rule": "date_range",
            "min_date": "1900-01-01",
            "max_date": "2024-12-31",
            "severity": "error"
        }
    ]
)
```

### Data Enrichment

```python
# Enrich dataset with additional information
enrichment = client.data.enrich(
    dataset_id="ds_abc123",
    enrichment_config={
        "sentiment_analysis": {
            "field": "abstract",
            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        },
        "entity_extraction": {
            "field": "abstract",
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english"
        },
        "topic_modeling": {
            "field": "abstract",
            "num_topics": 50
        }
    }
)
```

## 📊 Vector Database Operations

### Create Vector Index

```python
# Create optimized vector index
index = client.data.create_vector_index(
    embedding_job_id="emb_xyz789",
    index_config={
        "algorithm": "hnsw",
        "parameters": {
            "ef_construction": 200,
            "m": 16
        },
        "distance_metric": "cosine"
    }
)
```

### Hybrid Search

```python
# Combine semantic and keyword search
results = client.data.hybrid_search(
    query="transformer attention mechanism",
    embedding_job_id="emb_xyz789",
    search_config={
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "keyword_fields": ["title", "authors"],
        "boost_recent": True,
        "date_decay": 0.1
    }
)
```

### Clustering and Analysis

```python
# Perform clustering on embeddings
clustering = client.data.cluster_embeddings(
    embedding_job_id="emb_xyz789",
    algorithm="kmeans",
    num_clusters=20,
    include_cluster_analysis=True
)
```

## 🔄 Data Synchronization

### Cross-Network Sync

```python
# Synchronize data across network nodes
sync_job = client.data.sync_across_network(
    dataset_id="ds_abc123",
    target_nodes=["node_1", "node_2", "node_3"],
    sync_strategy="eventual_consistency",
    compression=True
)
```

### Real-time Updates

```python
# Set up real-time data streaming
stream = client.data.create_stream(
    dataset_id="ds_abc123",
    stream_config={
        "format": "jsonl",
        "compression": "gzip",
        "batch_size": 1000,
        "flush_interval": 10
    }
)

# Subscribe to data updates
@stream.on_new_data
def handle_new_data(batch):
    print(f"Received {len(batch)} new records")
```

## 🔐 Access Control and Security

### Dataset Permissions

```python
# Set fine-grained access permissions
permissions = client.data.set_permissions(
    dataset_id="ds_abc123",
    permissions={
        "read": ["user1", "user2", "team:researchers"],
        "write": ["user1"],
        "admin": ["user1"],
        "embedding_generation": ["team:researchers"],
        "export": ["user1", "user2"]
    }
)
```

### Data Encryption

```python
# Configure encryption for sensitive data
encryption_config = {
    "encryption_at_rest": True,
    "encryption_in_transit": True,
    "key_management": "customer_managed",
    "encryption_algorithm": "AES-256-GCM"
}

dataset = client.data.upload_dataset(
    name="sensitive_data",
    file_path="./data.jsonl",
    encryption_config=encryption_config
)
```

### Data Anonymization

```python
# Anonymize sensitive fields
anonymization = client.data.anonymize(
    dataset_id="ds_abc123",
    anonymization_rules=[
        {
            "field": "author_email",
            "method": "hash",
            "algorithm": "sha256"
        },
        {
            "field": "institution",
            "method": "generalize",
            "mapping": {
                "Stanford University": "Major University",
                "MIT": "Major University"
            }
        }
    ]
)
```

## 📈 Analytics and Monitoring

### Dataset Usage Analytics

```python
# Get comprehensive usage analytics
analytics = client.data.analytics(
    dataset_id="ds_abc123",
    timeframe="last_30_days",
    metrics=[
        "access_count",
        "query_frequency",
        "embedding_usage",
        "storage_growth"
    ]
)
```

### Performance Monitoring

```python
# Monitor data operation performance
performance = client.data.performance_metrics(
    operations=["query", "embedding", "search"],
    timeframe="last_24_hours",
    group_by="operation_type"
)
```

### Cost Analysis

```python
# Analyze data storage and processing costs
costs = client.data.cost_analysis(
    timeframe="last_month",
    breakdown_by=["dataset", "operation", "storage_tier"]
)
```

## 🛠️ Advanced Features

### Data Versioning

```python
# Create dataset version
version = client.data.create_version(
    dataset_id="ds_abc123",
    version_name="v2.0",
    changes=[
        "Added 10,000 new records",
        "Updated schema to include DOI field",
        "Improved data quality validation"
    ]
)

# Compare versions
diff = client.data.compare_versions(
    dataset_id="ds_abc123",
    version_a="v1.0",
    version_b="v2.0"
)
```

### Data Lineage Tracking

```python
# Track data lineage and transformations
lineage = client.data.get_lineage(
    dataset_id="ds_abc123",
    include_transformations=True,
    include_derivations=True
)
```

### Automated Quality Monitoring

```python
# Set up automated quality monitoring
monitor = client.data.create_quality_monitor(
    dataset_id="ds_abc123",
    monitoring_rules=[
        {
            "metric": "completeness",
            "threshold": 0.95,
            "alert_on_breach": True
        },
        {
            "metric": "consistency",
            "threshold": 0.90,
            "alert_on_breach": True
        }
    ],
    check_frequency="daily"
)
```

## 🧪 Testing and Development

### Synthetic Data Generation

```python
# Generate synthetic data for testing
synthetic_data = client.data.generate_synthetic(
    schema={
        "title": {"type": "text", "style": "academic"},
        "abstract": {"type": "text", "length": 200},
        "authors": {"type": "list", "min_items": 1, "max_items": 5}
    },
    count=1000,
    seed=42
)
```

### Data Sampling

```python
# Create representative samples
sample = client.data.create_sample(
    dataset_id="ds_abc123",
    sampling_method="stratified",
    sample_size=10000,
    strata_field="publication_year"
)
```

## 📞 Support

- **Data Issues**: data-support@prsm.ai
- **Storage Problems**: storage@prsm.ai  
- **Performance**: performance@prsm.ai
- **Security**: security@prsm.ai