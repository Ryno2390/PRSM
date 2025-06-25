# AWS Integration Guide

Complete guide for deploying and integrating PRSM with Amazon Web Services (AWS) infrastructure.

## Overview

This guide covers deploying PRSM on AWS using various services including ECS, Lambda, RDS, ElastiCache, and more. We'll explore different deployment patterns from simple single-instance setups to highly available, auto-scaling architectures.

## Architecture Options

### Option 1: ECS with Fargate (Recommended)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │    │   PRSM Services  │    │   Data Layer    │
│   Load Balancer │◄──►│   ECS Fargate    │◄──►│   RDS + Redis   │
│                 │    │                  │    │                 │
│ • ALB           │    │ • PRSM Core      │    │ • PostgreSQL    │
│ • Auto Scaling  │    │ • API Gateway    │    │ • ElastiCache   │
│ • SSL Termination│   │ • Task Scaling   │    │ • S3 Storage    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Option 2: Lambda + API Gateway (Serverless)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   Lambda Functions│   │   Data Layer    │
│                 │◄──►│                  │◄──►│                 │
│ • REST/GraphQL  │    │ • PRSM Handlers  │    │ • DynamoDB      │
│ • Authentication│    │ • Auto Scaling   │    │ • S3 Storage    │
│ • Rate Limiting │    │ • Event-driven   │    │ • ElastiCache   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed for container deployments
- Terraform or CDK for infrastructure as code

## ECS Fargate Deployment

### 1. Infrastructure Setup with Terraform

```hcl
# infrastructure/main.tf
provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "prsm_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "prsm-vpc"
  }
}

# Subnets
resource "aws_subnet" "prsm_subnet_public" {
  count             = 2
  vpc_id            = aws_vpc.prsm_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "prsm-subnet-public-${count.index + 1}"
  }
}

resource "aws_subnet" "prsm_subnet_private" {
  count             = 2
  vpc_id            = aws_vpc.prsm_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "prsm-subnet-private-${count.index + 1}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "prsm_igw" {
  vpc_id = aws_vpc.prsm_vpc.id
  
  tags = {
    Name = "prsm-igw"
  }
}

# NAT Gateway
resource "aws_eip" "prsm_nat_eip" {
  count  = 2
  domain = "vpc"
  
  tags = {
    Name = "prsm-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "prsm_nat" {
  count         = 2
  allocation_id = aws_eip.prsm_nat_eip[count.index].id
  subnet_id     = aws_subnet.prsm_subnet_public[count.index].id
  
  tags = {
    Name = "prsm-nat-${count.index + 1}"
  }
}

# Route Tables
resource "aws_route_table" "prsm_rt_public" {
  vpc_id = aws_vpc.prsm_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.prsm_igw.id
  }
  
  tags = {
    Name = "prsm-rt-public"
  }
}

resource "aws_route_table" "prsm_rt_private" {
  count  = 2
  vpc_id = aws_vpc.prsm_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.prsm_nat[count.index].id
  }
  
  tags = {
    Name = "prsm-rt-private-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "prsm_rta_public" {
  count          = 2
  subnet_id      = aws_subnet.prsm_subnet_public[count.index].id
  route_table_id = aws_route_table.prsm_rt_public.id
}

resource "aws_route_table_association" "prsm_rta_private" {
  count          = 2
  subnet_id      = aws_subnet.prsm_subnet_private[count.index].id
  route_table_id = aws_route_table.prsm_rt_private[count.index].id
}

# Security Groups
resource "aws_security_group" "prsm_alb_sg" {
  name_prefix = "prsm-alb-sg"
  vpc_id      = aws_vpc.prsm_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "prsm-alb-sg"
  }
}

resource "aws_security_group" "prsm_ecs_sg" {
  name_prefix = "prsm-ecs-sg"
  vpc_id      = aws_vpc.prsm_vpc.id
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.prsm_alb_sg.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "prsm-ecs-sg"
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "prsm_db_subnet_group" {
  name       = "prsm-db-subnet-group"
  subnet_ids = aws_subnet.prsm_subnet_private[*].id
  
  tags = {
    Name = "prsm-db-subnet-group"
  }
}

# RDS Security Group
resource "aws_security_group" "prsm_rds_sg" {
  name_prefix = "prsm-rds-sg"
  vpc_id      = aws_vpc.prsm_vpc.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.prsm_ecs_sg.id]
  }
  
  tags = {
    Name = "prsm-rds-sg"
  }
}

# RDS Instance
resource "aws_db_instance" "prsm_postgres" {
  identifier = "prsm-postgres"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "prsm"
  username = "prsm_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.prsm_rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.prsm_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "07:00-09:00"
  maintenance_window     = "sun:06:00-sun:07:00"
  
  skip_final_snapshot = true
  
  tags = {
    Name = "prsm-postgres"
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "prsm_cache_subnet_group" {
  name       = "prsm-cache-subnet-group"
  subnet_ids = aws_subnet.prsm_subnet_private[*].id
}

# ElastiCache Security Group
resource "aws_security_group" "prsm_redis_sg" {
  name_prefix = "prsm-redis-sg"
  vpc_id      = aws_vpc.prsm_vpc.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.prsm_ecs_sg.id]
  }
  
  tags = {
    Name = "prsm-redis-sg"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "prsm_redis" {
  replication_group_id       = "prsm-redis"
  description                = "Redis cluster for PRSM"
  
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  
  subnet_group_name          = aws_elasticache_subnet_group.prsm_cache_subnet_group.name
  security_group_ids         = [aws_security_group.prsm_redis_sg.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "prsm-redis"
  }
}
```

### 2. ECS Configuration

```hcl
# ecs.tf
# ECS Cluster
resource "aws_ecs_cluster" "prsm_cluster" {
  name = "prsm-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name = "prsm-cluster"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "prsm_task" {
  family                   = "prsm-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "prsm-api"
      image = "${aws_ecr_repository.prsm_repo.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "DATABASE_URL"
          value = "postgresql://${aws_db_instance.prsm_postgres.username}:${var.db_password}@${aws_db_instance.prsm_postgres.endpoint}/${aws_db_instance.prsm_postgres.db_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://${aws_elasticache_replication_group.prsm_redis.primary_endpoint_address}:6379"
        },
        {
          name  = "PRSM_ENV"
          value = "production"
        }
      ]
      
      secrets = [
        {
          name      = "PRSM_API_KEY"
          valueFrom = aws_ssm_parameter.prsm_api_key.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.prsm_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
  
  tags = {
    Name = "prsm-task"
  }
}

# ECS Service
resource "aws_ecs_service" "prsm_service" {
  name            = "prsm-service"
  cluster         = aws_ecs_cluster.prsm_cluster.id
  task_definition = aws_ecs_task_definition.prsm_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.prsm_subnet_private[*].id
    security_groups  = [aws_security_group.prsm_ecs_sg.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.prsm_tg.arn
    container_name   = "prsm-api"
    container_port   = 8000
  }
  
  depends_on = [aws_lb_listener.prsm_listener]
  
  tags = {
    Name = "prsm-service"
  }
}

# Application Load Balancer
resource "aws_lb" "prsm_alb" {
  name               = "prsm-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.prsm_alb_sg.id]
  subnets            = aws_subnet.prsm_subnet_public[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name = "prsm-alb"
  }
}

# Target Group
resource "aws_lb_target_group" "prsm_tg" {
  name     = "prsm-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.prsm_vpc.id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }
  
  tags = {
    Name = "prsm-tg"
  }
}

# Load Balancer Listener
resource "aws_lb_listener" "prsm_listener" {
  load_balancer_arn = aws_lb.prsm_alb.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.prsm_tg.arn
  }
}
```

### 3. IAM Roles and Policies

```hcl
# iam.tf
# ECS Execution Role
resource "aws_iam_role" "ecs_execution_role" {
  name = "prsm-ecs-execution-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_ssm_policy" {
  name = "prsm-ecs-execution-ssm-policy"
  role = aws_iam_role.ecs_execution_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameters",
          "ssm:GetParameter"
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/prsm/*"
      }
    ]
  })
}

# ECS Task Role
resource "aws_iam_role" "ecs_task_role" {
  name = "prsm-ecs-task-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_s3_policy" {
  name = "prsm-ecs-task-s3-policy"
  role = aws_iam_role.ecs_task_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.prsm_storage.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.prsm_storage.arn
      }
    ]
  })
}
```

### 4. Container Registry and Storage

```hcl
# ecr.tf
# ECR Repository
resource "aws_ecr_repository" "prsm_repo" {
  name                 = "prsm"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = {
    Name = "prsm-repo"
  }
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "prsm_repo_policy" {
  repository = aws_ecr_repository.prsm_repo.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# S3 Bucket for PRSM Storage
resource "aws_s3_bucket" "prsm_storage" {
  bucket = "prsm-storage-${random_string.bucket_suffix.result}"
  
  tags = {
    Name = "prsm-storage"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "prsm_storage_versioning" {
  bucket = aws_s3_bucket.prsm_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "prsm_storage_encryption" {
  bucket = aws_s3_bucket.prsm_storage.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}
```

### 5. Monitoring and Logging

```hcl
# monitoring.tf
# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "prsm_logs" {
  name              = "/ecs/prsm"
  retention_in_days = 7
  
  tags = {
    Name = "prsm-logs"
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "prsm_cpu_high" {
  alarm_name          = "prsm-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ECS CPU utilization"
  
  dimensions = {
    ServiceName = aws_ecs_service.prsm_service.name
    ClusterName = aws_ecs_cluster.prsm_cluster.name
  }
  
  alarm_actions = [aws_sns_topic.prsm_alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "prsm_memory_high" {
  alarm_name          = "prsm-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ECS memory utilization"
  
  dimensions = {
    ServiceName = aws_ecs_service.prsm_service.name
    ClusterName = aws_ecs_cluster.prsm_cluster.name
  }
  
  alarm_actions = [aws_sns_topic.prsm_alerts.arn]
}

# SNS Topic for Alerts
resource "aws_sns_topic" "prsm_alerts" {
  name = "prsm-alerts"
  
  tags = {
    Name = "prsm-alerts"
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "prsm_ecs_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.prsm_cluster.name}/${aws_ecs_service.prsm_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "prsm_ecs_policy_cpu" {
  name               = "prsm-ecs-policy-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.prsm_ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.prsm_ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.prsm_ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

## Lambda Serverless Deployment

### 1. Lambda Function for PRSM API

```python
# lambda/prsm_handler.py
import json
import asyncio
import os
from typing import Dict, Any
from prsm_sdk import PRSMClient, PRSMError

# Initialize PRSM client
prsm_client = PRSMClient(
    base_url=os.getenv('PRSM_CORE_URL'),
    api_key=os.getenv('PRSM_API_KEY')
)

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Lambda handler for PRSM API requests"""
    try:
        # Parse request
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Extract parameters
        prompt = body.get('prompt')
        user_id = body.get('user_id')
        context_allocation = body.get('context_allocation', 50)
        
        if not prompt or not user_id:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required parameters: prompt, user_id'
                })
            }
        
        # Process query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                prsm_client.query(
                    prompt=prompt,
                    user_id=user_id,
                    context_allocation=context_allocation
                )
            )
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'query_id': response.query_id,
                    'answer': response.final_answer,
                    'cost': response.ftns_charged,
                    'processing_time': response.processing_time,
                    'quality_score': response.quality_score
                })
            }
        finally:
            loop.close()
    
    except PRSMError as e:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'PRSM Error: {str(e)}'
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Internal error: {str(e)}'
            })
        }
```

### 2. Serverless Framework Configuration

```yaml
# serverless.yml
service: prsm-serverless

provider:
  name: aws
  runtime: python3.11
  region: ${opt:region, 'us-east-1'}
  stage: ${opt:stage, 'dev'}
  
  environment:
    PRSM_CORE_URL: ${ssm:/prsm/${self:provider.stage}/core-url}
    PRSM_API_KEY: ${ssm:/prsm/${self:provider.stage}/api-key~true}
  
  iamRoleStatements:
    - Effect: Allow
      Action:
        - dynamodb:Query
        - dynamodb:Scan
        - dynamodb:GetItem
        - dynamodb:PutItem
        - dynamodb:UpdateItem
        - dynamodb:DeleteItem
      Resource:
        - "arn:aws:dynamodb:${self:provider.region}:*:table/prsm-queries-${self:provider.stage}"
        - "arn:aws:dynamodb:${self:provider.region}:*:table/prsm-sessions-${self:provider.stage}"
    
    - Effect: Allow
      Action:
        - s3:GetObject
        - s3:PutObject
      Resource:
        - "arn:aws:s3:::prsm-storage-${self:provider.stage}/*"

functions:
  prsmQuery:
    handler: prsm_handler.lambda_handler
    timeout: 30
    memorySize: 512
    events:
      - http:
          path: /query
          method: post
          cors: true
  
  prsmHealth:
    handler: health_handler.lambda_handler
    timeout: 10
    events:
      - http:
          path: /health
          method: get
          cors: true

resources:
  Resources:
    # DynamoDB Tables
    PRSMQueriesTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: prsm-queries-${self:provider.stage}
        AttributeDefinitions:
          - AttributeName: user_id
            AttributeType: S
          - AttributeName: query_id
            AttributeType: S
        KeySchema:
          - AttributeName: user_id
            KeyType: HASH
          - AttributeName: query_id
            KeyType: RANGE
        BillingMode: PAY_PER_REQUEST
    
    PRSMSessionsTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: prsm-sessions-${self:provider.stage}
        AttributeDefinitions:
          - AttributeName: session_id
            AttributeType: S
        KeySchema:
          - AttributeName: session_id
            KeyType: HASH
        BillingMode: PAY_PER_REQUEST
        TimeToLiveSpecification:
          AttributeName: ttl
          Enabled: true

plugins:
  - serverless-python-requirements
  - serverless-plugin-warmup

custom:
  pythonRequirements:
    dockerizePip: non-linux
  warmup:
    enabled: true
    prewarm: true
```

## CDK Deployment Alternative

### CDK Stack Example

```python
# infrastructure/prsm_stack.py
from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_rds as rds,
    aws_elasticache as elasticache,
    aws_elbv2 as elbv2,
    aws_logs as logs,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ecr as ecr,
    Duration
)
from constructs import Construct

class PRSMStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # VPC
        vpc = ec2.Vpc(
            self, "PRSMVPC",
            cidr="10.0.0.0/16",
            max_azs=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public-subnet",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private-subnet",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )
        
        # ECS Cluster
        cluster = ecs.Cluster(
            self, "PRSMCluster",
            vpc=vpc,
            container_insights=True
        )
        
        # RDS Database
        database = rds.DatabaseInstance(
            self, "PRSMDatabase",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_15_4
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T3,
                ec2.InstanceSize.MICRO
            ),
            vpc=vpc,
            credentials=rds.Credentials.from_generated_secret("prsm_user"),
            multi_az=False,
            allocated_storage=20,
            storage_encrypted=True,
            deletion_protection=False,
            backup_retention=Duration.days(7)
        )
        
        # Redis Cache
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self, "RedisSubnetGroup",
            description="Subnet group for Redis",
            subnet_ids=[subnet.subnet_id for subnet in vpc.private_subnets]
        )
        
        redis_security_group = ec2.SecurityGroup(
            self, "RedisSecurityGroup",
            vpc=vpc,
            description="Security group for Redis"
        )
        
        redis = elasticache.CfnReplicationGroup(
            self, "PRSMRedis",
            description="Redis cluster for PRSM",
            cache_node_type="cache.t3.micro",
            engine="redis",
            num_cache_clusters=2,
            cache_subnet_group_name=redis_subnet_group.ref,
            security_group_ids=[redis_security_group.security_group_id],
            at_rest_encryption_enabled=True,
            transit_encryption_enabled=True
        )
        
        # ECR Repository
        repository = ecr.Repository(
            self, "PRSMRepository",
            repository_name="prsm",
            image_scan_on_push=True
        )
        
        # ECS Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self, "PRSMTaskDefinition",
            memory_limit_mib=1024,
            cpu=512
        )
        
        # Add container to task definition
        container = task_definition.add_container(
            "prsm-api",
            image=ecs.ContainerImage.from_ecr_repository(repository, "latest"),
            environment={
                "DATABASE_URL": f"postgresql://{database.secret.secret_value_from_json('username')}:{database.secret.secret_value_from_json('password')}@{database.instance_endpoint.hostname}/{database.instance_endpoint.database_name}",
                "REDIS_URL": f"redis://{redis.attr_primary_end_point_address}:6379",
                "PRSM_ENV": "production"
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ecs",
                log_group=logs.LogGroup(
                    self, "PRSMLogGroup",
                    log_group_name="/ecs/prsm",
                    retention=logs.RetentionDays.ONE_WEEK
                )
            )
        )
        
        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8000,
                protocol=ecs.Protocol.TCP
            )
        )
        
        # ECS Service
        service = ecs.FargateService(
            self, "PRSMService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            assign_public_ip=False
        )
        
        # Application Load Balancer
        alb = elbv2.ApplicationLoadBalancer(
            self, "PRSMALB",
            vpc=vpc,
            internet_facing=True
        )
        
        listener = alb.add_listener(
            "PRSMListener",
            port=80,
            default_targets=[service]
        )
        
        # Allow ALB to reach ECS service
        service.connections.allow_from(
            alb,
            ec2.Port.tcp(8000),
            "Allow ALB to reach ECS service"
        )
        
        # Allow ECS service to reach RDS
        database.connections.allow_from(
            service,
            ec2.Port.tcp(5432),
            "Allow ECS to reach RDS"
        )
        
        # Allow ECS service to reach Redis
        redis_security_group.add_ingress_rule(
            service.connections.security_groups[0],
            ec2.Port.tcp(6379),
            "Allow ECS to reach Redis"
        )
```

## Best Practices and Security

### 1. Security Hardening

```yaml
# security-config.yml
security_measures:
  network:
    - "Use VPC with private subnets for services"
    - "Implement security groups with least privilege"
    - "Enable VPC Flow Logs for monitoring"
    - "Use NAT Gateway for outbound internet access"
  
  encryption:
    - "Enable encryption at rest for RDS and ElastiCache"
    - "Use SSL/TLS for data in transit"
    - "Encrypt S3 buckets with default encryption"
    - "Store secrets in AWS Systems Manager Parameter Store"
  
  access_control:
    - "Use IAM roles with minimal required permissions"
    - "Enable MFA for AWS console access"
    - "Implement API Gateway authentication"
    - "Use AWS WAF for application protection"
  
  monitoring:
    - "Enable CloudTrail for API logging"
    - "Set up CloudWatch alarms for anomalies"
    - "Use AWS Config for compliance monitoring"
    - "Implement centralized logging with CloudWatch Logs"
```

### 2. Cost Optimization

```python
# cost-optimization.py
cost_optimization_strategies = {
    "compute": {
        "fargate": "Use Fargate Spot for non-critical workloads",
        "auto_scaling": "Implement CPU and memory-based auto scaling",
        "right_sizing": "Monitor and adjust task CPU/memory allocation"
    },
    "storage": {
        "s3_lifecycle": "Implement S3 lifecycle policies for log rotation",
        "rds_optimization": "Use appropriate RDS instance size and storage type",
        "backup_retention": "Optimize backup retention periods"
    },
    "networking": {
        "nat_gateway": "Consider NAT instances for lower traffic",
        "cloudfront": "Use CloudFront for static content delivery",
        "vpc_endpoints": "Use VPC endpoints for AWS service access"
    }
}
```

### 3. Deployment Automation

```bash
#!/bin/bash
# deploy.sh
set -e

echo "🚀 Deploying PRSM to AWS"

# Variables
REGION=${AWS_REGION:-us-east-1}
ENVIRONMENT=${ENVIRONMENT:-production}
DOCKER_TAG=${DOCKER_TAG:-latest}

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t prsm:$DOCKER_TAG .

# Get ECR login token
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Tag and push image
docker tag prsm:$DOCKER_TAG $ECR_REPOSITORY_URI:$DOCKER_TAG
docker push $ECR_REPOSITORY_URI:$DOCKER_TAG

# Deploy infrastructure
echo "🏗️ Deploying infrastructure..."
cd infrastructure
terraform init
terraform plan -var="docker_tag=$DOCKER_TAG" -var="environment=$ENVIRONMENT"
terraform apply -auto-approve -var="docker_tag=$DOCKER_TAG" -var="environment=$ENVIRONMENT"

# Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --region $REGION \
    --cluster prsm-cluster-$ENVIRONMENT \
    --service prsm-service-$ENVIRONMENT \
    --force-new-deployment

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
    --region $REGION \
    --cluster prsm-cluster-$ENVIRONMENT \
    --services prsm-service-$ENVIRONMENT

echo "✅ Deployment completed successfully!"

# Get ALB endpoint
ALB_ENDPOINT=$(aws elbv2 describe-load-balancers \
    --region $REGION \
    --names prsm-alb-$ENVIRONMENT \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

echo "🌐 PRSM is available at: http://$ALB_ENDPOINT"
```

---

**Next Steps:**
- [Google Cloud Integration](./gcp-integration.md)
- [Azure Integration](./azure-integration.md)
- [Kubernetes Integration](./kubernetes-integration.md)
- [Monitoring and Observability](../devops-integration/monitoring-integration.md)
