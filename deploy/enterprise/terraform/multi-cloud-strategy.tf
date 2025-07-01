# PRSM Multi-Cloud Infrastructure Strategy
# ==========================================
# Production-Ready Multi-Cloud Deployment for Phase 2+ Expansion
# 
# STRATEGIC EVOLUTION: AWS-First → Multi-Cloud Excellence
# ======================================================
# 
# Phase 1 (Months 1-18): AWS-First Foundation (COMPLETED)
# - Single-cloud operational excellence established
# - Production workloads proven on AWS
# - Team expertise and operational processes matured
# 
# Phase 2 (Months 19-30): Strategic Multi-Cloud Expansion
# - Customer-driven geographic requirements
# - Risk mitigation through provider diversification  
# - Regulatory compliance for global markets
# - Enhanced disaster recovery capabilities
# 
# Phase 3 (Months 31+): Global Multi-Cloud Optimization
# - Advanced cross-cloud orchestration
# - Cost optimization across providers
# - Edge computing integration
# - Advanced AI/ML workload placement

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

# ==========================================
# Multi-Cloud Configuration Variables
# ==========================================

variable "multi_cloud_enabled" {
  description = "Enable multi-cloud deployment (Phase 2+)"
  type        = bool
  default     = false
}

variable "multi_cloud_strategy" {
  description = "Multi-cloud deployment strategy"
  type        = string
  default     = "aws-primary-gcp-secondary"
  validation {
    condition = contains([
      "aws-primary-gcp-secondary",
      "aws-primary-azure-secondary", 
      "aws-gcp-azure-balanced",
      "aws-only"
    ], var.multi_cloud_strategy)
    error_message = "Multi-cloud strategy must be one of: aws-primary-gcp-secondary, aws-primary-azure-secondary, aws-gcp-azure-balanced, aws-only."
  }
}

variable "geographic_regions" {
  description = "Geographic regions for global deployment"
  type = object({
    primary = object({
      aws_region   = string
      gcp_region   = string
      azure_region = string
    })
    secondary = list(object({
      aws_region   = string
      gcp_region   = string  
      azure_region = string
      enabled      = bool
    }))
  })
  default = {
    primary = {
      aws_region   = "us-west-2"
      gcp_region   = "us-west1"
      azure_region = "West US 2"
    }
    secondary = [
      {
        aws_region   = "us-east-1"
        gcp_region   = "us-east1"
        azure_region = "East US"
        enabled      = true
      },
      {
        aws_region   = "eu-west-1"
        gcp_region   = "europe-west1"
        azure_region = "West Europe"
        enabled      = true
      },
      {
        aws_region   = "ap-southeast-1"
        gcp_region   = "asia-southeast1"
        azure_region = "Southeast Asia"
        enabled      = false  # Phase 3+
      }
    ]
  }
}

variable "workload_distribution" {
  description = "Workload distribution across cloud providers"
  type = object({
    compute_primary_provider     = string  # aws, gcp, azure
    database_primary_provider    = string
    cache_primary_provider       = string
    ai_ml_preferred_provider     = string
    storage_preferred_provider   = string
    cdn_preferred_provider       = string
  })
  default = {
    compute_primary_provider     = "aws"
    database_primary_provider    = "aws"
    cache_primary_provider       = "aws"
    ai_ml_preferred_provider     = "gcp"    # GCP's AI/ML capabilities
    storage_preferred_provider   = "aws"    # S3's ecosystem
    cdn_preferred_provider       = "aws"    # CloudFront integration
  }
}

variable "disaster_recovery_config" {
  description = "Disaster recovery configuration across clouds"
  type = object({
    enable_cross_cloud_backup    = bool
    backup_retention_days        = number
    rpo_hours                   = number  # Recovery Point Objective
    rto_hours                   = number  # Recovery Time Objective
    primary_to_secondary_sync   = bool
  })
  default = {
    enable_cross_cloud_backup    = true
    backup_retention_days        = 30
    rpo_hours                   = 1
    rto_hours                   = 4
    primary_to_secondary_sync   = true
  }
}

variable "compliance_requirements" {
  description = "Compliance and regulatory requirements"
  type = object({
    data_residency_required     = bool
    gdpr_compliance_required    = bool
    hipaa_compliance_required   = bool
    soc2_type2_required        = bool
    data_sovereignty_regions    = list(string)
  })
  default = {
    data_residency_required     = true
    gdpr_compliance_required    = true
    hipaa_compliance_required   = false
    soc2_type2_required        = true
    data_sovereignty_regions    = ["eu-west-1", "europe-west1", "West Europe"]
  }
}

# ==========================================
# Local Values for Multi-Cloud Strategy
# ==========================================

locals {
  # Determine active providers based on strategy
  aws_enabled   = true  # Always enabled as primary
  gcp_enabled   = var.multi_cloud_enabled && contains(["aws-primary-gcp-secondary", "aws-gcp-azure-balanced"], var.multi_cloud_strategy)
  azure_enabled = var.multi_cloud_enabled && contains(["aws-primary-azure-secondary", "aws-gcp-azure-balanced"], var.multi_cloud_strategy)
  
  # Multi-cloud naming convention
  multi_cloud_prefix = var.multi_cloud_enabled ? "prsm-mc-${var.environment}" : "prsm-${var.environment}"
  
  # Common tags for all resources
  multi_cloud_tags = {
    Environment      = var.environment
    Project         = "PRSM"
    ManagedBy       = "Terraform"
    Owner           = "PRSM-Team"
    CloudStrategy   = var.multi_cloud_strategy
    MultiCloudPhase = var.multi_cloud_enabled ? "Phase-2-Multi-Cloud" : "Phase-1-AWS-First"
    ComplianceLevel = "SOC2-GDPR-Ready"
  }
  
  # Workload placement logic
  workload_placement = {
    for region in var.geographic_regions.secondary : region.aws_region => {
      compute_provider = var.workload_distribution.compute_primary_provider
      database_provider = var.workload_distribution.database_primary_provider
      cache_provider = var.workload_distribution.cache_primary_provider
      enabled = region.enabled && var.multi_cloud_enabled
    } if region.enabled
  }
}

# ==========================================
# Provider Configurations (Conditional)
# ==========================================

# AWS Provider (Primary - Always Active)
provider "aws" {
  alias  = "primary"
  region = var.geographic_regions.primary.aws_region
  
  default_tags {
    tags = local.multi_cloud_tags
  }
}

# AWS Secondary Regions
provider "aws" {
  alias  = "us_east"
  region = "us-east-1"
  
  default_tags {
    tags = merge(local.multi_cloud_tags, {
      Region = "us-east-1"
      Role   = "secondary"
    })
  }
}

provider "aws" {
  alias  = "eu_west"
  region = "eu-west-1"
  
  default_tags {
    tags = merge(local.multi_cloud_tags, {
      Region = "eu-west-1" 
      Role   = "secondary"
      Compliance = "GDPR"
    })
  }
}

# GCP Provider (Phase 2+)
provider "google" {
  count   = local.gcp_enabled ? 1 : 0
  project = "prsm-enterprise-${var.environment}"
  region  = var.geographic_regions.primary.gcp_region
}

provider "google" {
  alias   = "gcp_secondary" 
  count   = local.gcp_enabled ? 1 : 0
  project = "prsm-enterprise-${var.environment}"
  region  = "us-east1"
}

# Azure Provider (Phase 2+)
provider "azurerm" {
  count = local.azure_enabled ? 1 : 0
  features {
    resource_group {
      prevent_deletion_if_contains_resources = var.environment == "production"
    }
    key_vault {
      purge_soft_delete_on_destroy    = var.environment != "production"
      recover_soft_deleted_key_vaults = true
    }
  }
}

# ==========================================
# Multi-Cloud Infrastructure Modules
# ==========================================

# AWS Infrastructure (Always Active)
module "aws_primary" {
  source = "./modules/aws"
  
  environment = var.environment
  region      = var.geographic_regions.primary.aws_region
  name_prefix = local.multi_cloud_prefix
  
  # Workload-specific configuration
  enable_ai_ml_workloads = var.workload_distribution.ai_ml_preferred_provider == "aws"
  enable_primary_database = var.workload_distribution.database_primary_provider == "aws"
  enable_primary_cache = var.workload_distribution.cache_primary_provider == "aws"
  
  # Multi-cloud coordination
  multi_cloud_enabled = var.multi_cloud_enabled
  cross_cloud_backup_enabled = var.disaster_recovery_config.enable_cross_cloud_backup
  
  # Compliance
  compliance_requirements = var.compliance_requirements
  
  tags = local.multi_cloud_tags
  
  providers = {
    aws = aws.primary
  }
}

# AWS Secondary Regions (Disaster Recovery)
module "aws_secondary_us_east" {
  source = "./modules/aws"
  count  = var.multi_cloud_enabled ? 1 : 0
  
  environment = var.environment
  region      = "us-east-1"
  name_prefix = "${local.multi_cloud_prefix}-us-east"
  
  # Disaster recovery configuration
  is_disaster_recovery_region = true
  primary_region_backup_source = module.aws_primary.backup_source_config
  rpo_hours = var.disaster_recovery_config.rpo_hours
  rto_hours = var.disaster_recovery_config.rto_hours
  
  tags = merge(local.multi_cloud_tags, {
    Region = "us-east-1"
    Role   = "disaster-recovery"
  })
  
  providers = {
    aws = aws.us_east
  }
}

module "aws_secondary_eu_west" {
  source = "./modules/aws"
  count  = var.multi_cloud_enabled && var.compliance_requirements.gdpr_compliance_required ? 1 : 0
  
  environment = var.environment
  region      = "eu-west-1"
  name_prefix = "${local.multi_cloud_prefix}-eu-west"
  
  # GDPR compliance configuration
  is_gdpr_region = true
  data_residency_required = var.compliance_requirements.data_residency_required
  
  # Disaster recovery configuration
  is_disaster_recovery_region = true
  primary_region_backup_source = module.aws_primary.backup_source_config
  
  tags = merge(local.multi_cloud_tags, {
    Region = "eu-west-1"
    Role   = "gdpr-compliant-secondary"
    Compliance = "GDPR"
  })
  
  providers = {
    aws = aws.eu_west
  }
}

# GCP Infrastructure (Phase 2+)
module "gcp_primary" {
  source = "./modules/gcp"
  count  = local.gcp_enabled ? 1 : 0
  
  environment = var.environment
  region      = var.geographic_regions.primary.gcp_region
  project_id  = "prsm-enterprise-${var.environment}"
  name_prefix = local.multi_cloud_prefix
  
  # Workload-specific configuration  
  enable_ai_ml_workloads = var.workload_distribution.ai_ml_preferred_provider == "gcp"
  enable_bigquery_analytics = true
  enable_vertex_ai = true
  
  # Multi-cloud coordination
  aws_primary_vpc_id = module.aws_primary.vpc_id
  cross_cloud_networking_enabled = true
  
  tags = local.multi_cloud_tags
  
  providers = {
    google = google[0]
  }
  
  depends_on = [module.aws_primary]
}

# Azure Infrastructure (Phase 2+)
module "azure_primary" {
  source = "./modules/azure"
  count  = local.azure_enabled ? 1 : 0
  
  environment = var.environment
  location    = var.geographic_regions.primary.azure_region
  name_prefix = local.multi_cloud_prefix
  
  # Workload-specific configuration
  enable_cognitive_services = true
  enable_synapse_analytics = var.environment == "production"
  
  # Multi-cloud coordination
  aws_primary_vpc_id = module.aws_primary.vpc_id
  cross_cloud_networking_enabled = true
  
  tags = local.multi_cloud_tags
  
  providers = {
    azurerm = azurerm[0]
  }
  
  depends_on = [module.aws_primary]
}

# ==========================================
# Cross-Cloud Networking (Phase 2+)
# ==========================================

module "multi_cloud_networking" {
  source = "./modules/multi-cloud-networking"
  count  = var.multi_cloud_enabled ? 1 : 0
  
  environment = var.environment
  
  # Provider endpoints
  aws_vpc_id     = module.aws_primary.vpc_id
  aws_region     = var.geographic_regions.primary.aws_region
  gcp_vpc_id     = local.gcp_enabled ? module.gcp_primary[0].vpc_id : null
  gcp_project_id = local.gcp_enabled ? "prsm-enterprise-${var.environment}" : null
  azure_vnet_id  = local.azure_enabled ? module.azure_primary[0].vnet_id : null
  
  # Cross-cloud connectivity
  enable_aws_gcp_vpn     = local.gcp_enabled
  enable_aws_azure_vpn   = local.azure_enabled
  enable_gcp_azure_vpn   = local.gcp_enabled && local.azure_enabled
  
  # Security configuration
  cross_cloud_encryption_enabled = true
  network_segmentation_enabled = true
  
  tags = local.multi_cloud_tags
}

# ==========================================
# Multi-Cloud Data Replication (Phase 2+)
# ==========================================

module "multi_cloud_data_replication" {
  source = "./modules/data-replication"
  count  = var.multi_cloud_enabled && var.disaster_recovery_config.enable_cross_cloud_backup ? 1 : 0
  
  environment = var.environment
  
  # Primary data sources
  aws_primary_database_endpoint = module.aws_primary.database_endpoint
  aws_primary_redis_endpoint   = module.aws_primary.redis_endpoint
  aws_primary_s3_bucket       = module.aws_primary.s3_bucket_name
  
  # Replication targets
  gcp_enabled = local.gcp_enabled
  gcp_sql_instance = local.gcp_enabled ? module.gcp_primary[0].sql_instance_name : null
  gcp_redis_instance = local.gcp_enabled ? module.gcp_primary[0].redis_instance_name : null
  gcp_storage_bucket = local.gcp_enabled ? module.gcp_primary[0].storage_bucket_name : null
  
  azure_enabled = local.azure_enabled  
  azure_sql_server = local.azure_enabled ? module.azure_primary[0].sql_server_name : null
  azure_redis_cache = local.azure_enabled ? module.azure_primary[0].redis_cache_name : null
  azure_storage_account = local.azure_enabled ? module.azure_primary[0].storage_account_name : null
  
  # Replication configuration
  rpo_hours = var.disaster_recovery_config.rpo_hours
  backup_retention_days = var.disaster_recovery_config.backup_retention_days
  enable_real_time_sync = var.disaster_recovery_config.primary_to_secondary_sync
  
  tags = local.multi_cloud_tags
}

# ==========================================
# Multi-Cloud Monitoring & Observability
# ==========================================

module "multi_cloud_monitoring" {
  source = "./modules/monitoring"
  count  = var.multi_cloud_enabled ? 1 : 0
  
  environment = var.environment
  
  # Provider configurations
  aws_enabled   = true
  gcp_enabled   = local.gcp_enabled
  azure_enabled = local.azure_enabled
  
  # Monitoring targets
  aws_resources = {
    cluster_name = module.aws_primary.cluster_name
    database_id  = module.aws_primary.database_identifier
    redis_id     = module.aws_primary.redis_cluster_id
  }
  
  gcp_resources = local.gcp_enabled ? {
    cluster_name = module.gcp_primary[0].cluster_name
    sql_instance = module.gcp_primary[0].sql_instance_name
    redis_instance = module.gcp_primary[0].redis_instance_name
  } : {}
  
  azure_resources = local.azure_enabled ? {
    cluster_name = module.azure_primary[0].cluster_name
    sql_server   = module.azure_primary[0].sql_server_name
    redis_cache  = module.azure_primary[0].redis_cache_name
  } : {}
  
  # Unified monitoring configuration
  enable_cross_cloud_alerting = true
  enable_cost_optimization_alerts = true
  enable_performance_comparison = true
  
  tags = local.multi_cloud_tags
}

# ==========================================
# Multi-Cloud Security & Compliance
# ==========================================

module "multi_cloud_security" {
  source = "./modules/security"
  count  = var.multi_cloud_enabled ? 1 : 0
  
  environment = var.environment
  
  # Compliance requirements
  compliance_requirements = var.compliance_requirements
  
  # Provider security configurations
  aws_security_config = {
    enable_guardduty = true
    enable_security_hub = true
    enable_config_rules = true
  }
  
  gcp_security_config = local.gcp_enabled ? {
    enable_security_command_center = true
    enable_cloud_armor = true
    enable_iam_recommender = true
  } : {}
  
  azure_security_config = local.azure_enabled ? {
    enable_security_center = true
    enable_sentinel = true
    enable_defender = true
  } : {}
  
  # Cross-cloud security features
  enable_unified_identity_management = true
  enable_cross_cloud_audit_logging = true
  enable_centralized_secret_management = true
  
  tags = local.multi_cloud_tags
}

# ==========================================
# Cost Optimization & Management
# ==========================================

module "multi_cloud_cost_optimization" {
  source = "./modules/cost-optimization"
  count  = var.multi_cloud_enabled ? 1 : 0
  
  environment = var.environment
  
  # Provider cost configurations
  aws_cost_config = {
    enable_cost_anomaly_detection = true
    enable_trusted_advisor = var.environment == "production"
    budget_threshold_usd = 10000
  }
  
  gcp_cost_config = local.gcp_enabled ? {
    enable_billing_alerts = true
    budget_threshold_usd = 5000
    enable_recommender = true
  } : {}
  
  azure_cost_config = local.azure_enabled ? {
    enable_cost_alerts = true
    budget_threshold_usd = 5000
    enable_advisor = true
  } : {}
  
  # Cross-cloud cost optimization
  enable_workload_cost_comparison = true
  enable_resource_rightsizing = true
  enable_spot_instance_optimization = true
  
  tags = local.multi_cloud_tags
}

# ==========================================
# Outputs
# ==========================================

output "multi_cloud_strategy" {
  description = "Active multi-cloud strategy"
  value = {
    strategy_name = var.multi_cloud_strategy
    enabled = var.multi_cloud_enabled
    aws_enabled = local.aws_enabled
    gcp_enabled = local.gcp_enabled
    azure_enabled = local.azure_enabled
    phase = var.multi_cloud_enabled ? "Phase-2-Multi-Cloud" : "Phase-1-AWS-First"
  }
}

output "active_regions" {
  description = "Active deployment regions across all clouds"
  value = {
    primary = var.geographic_regions.primary
    secondary = [for region in var.geographic_regions.secondary : region if region.enabled]
  }
}

output "workload_distribution" {
  description = "Current workload distribution across providers"
  value = var.workload_distribution
}

output "disaster_recovery_endpoints" {
  description = "Disaster recovery endpoints across clouds"
  value = var.multi_cloud_enabled ? {
    aws_secondary_endpoints = {
      us_east = length(module.aws_secondary_us_east) > 0 ? module.aws_secondary_us_east[0].cluster_endpoint : null
      eu_west = length(module.aws_secondary_eu_west) > 0 ? module.aws_secondary_eu_west[0].cluster_endpoint : null
    }
    gcp_endpoints = local.gcp_enabled ? {
      primary = module.gcp_primary[0].cluster_endpoint
    } : {}
    azure_endpoints = local.azure_enabled ? {
      primary = module.azure_primary[0].cluster_endpoint  
    } : {}
  } : {}
  sensitive = true
}

output "compliance_status" {
  description = "Multi-cloud compliance status"
  value = {
    gdpr_regions_active = var.compliance_requirements.gdpr_compliance_required
    data_residency_enforced = var.compliance_requirements.data_residency_required
    soc2_compliant = var.compliance_requirements.soc2_type2_required
    multi_cloud_audit_ready = var.multi_cloud_enabled
  }
}

output "cost_optimization_status" {
  description = "Cost optimization configuration across clouds"
  value = var.multi_cloud_enabled ? {
    cross_cloud_monitoring_enabled = true
    workload_cost_comparison_enabled = true
    automated_rightsizing_enabled = true
    estimated_monthly_savings_percent = "15-25"
  } : {
    single_cloud_optimization_enabled = true
    estimated_monthly_savings_percent = "10-15"
  }
}

# ==========================================
# Migration Roadmap & Documentation
# ==========================================

# Outputs for migration planning
output "migration_readiness" {
  description = "Multi-cloud migration readiness assessment"
  value = {
    aws_foundation_ready = true
    team_operational_maturity = var.multi_cloud_enabled ? "Advanced" : "Building"
    recommended_next_steps = var.multi_cloud_enabled ? [
      "Execute Phase 2 multi-cloud deployment",
      "Implement cross-cloud networking",
      "Enable disaster recovery automation",
      "Deploy unified monitoring"
    ] : [
      "Continue AWS operational excellence",
      "Build team multi-cloud expertise", 
      "Evaluate customer multi-cloud requirements",
      "Plan Phase 2 provider selection"
    ]
    estimated_migration_timeline_weeks = var.multi_cloud_enabled ? 8 : 16
  }
}