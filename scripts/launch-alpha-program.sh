#!/bin/bash
set -euo pipefail

# PRSM Alpha User Testing Program Launch Script
# Deploys the complete alpha testing infrastructure for 100+ technical users

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/alpha-program-launch-$(date +%Y%m%d-%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Alpha program configuration
ALPHA_ENVIRONMENT="alpha"
ALPHA_SUBDOMAIN="alpha-api"
TARGET_ALPHA_USERS=100
INITIAL_FTNS_GRANT=1000
ALPHA_API_VERSION="v0.9.0"

# Alpha infrastructure specifications
ALPHA_API_REPLICAS=3
ALPHA_WORKER_REPLICAS=2
ALPHA_DATABASE_SIZE="50Gi"
ALPHA_REDIS_MEMORY="4Gi"

print_status() {
    local color=$1
    local icon=$2
    local message=$3
    echo -e "${color}${icon} [$(date '+%H:%M:%S')] $message${NC}"
    echo "[$TIMESTAMP] [INFO] $message" >> "$LOG_FILE"
}

error_exit() {
    local error_msg=$1
    print_status "$RED" "❌" "ERROR: $error_msg"
    exit 1
}

# Function to check prerequisites
check_alpha_prerequisites() {
    print_status "$BLUE" "🔍" "Checking alpha program prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "docker" "python3" "pip" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done
    
    # Check Python dependencies
    local python_deps=("fastapi" "pydantic" "structlog" "asyncio")
    for dep in "${python_deps[@]}"; do
        if ! python3 -c "import $dep" &> /dev/null; then
            print_status "$YELLOW" "⚠️" "Installing Python dependency: $dep"
            pip install "$dep" || error_exit "Failed to install $dep"
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if alpha namespace exists
    if ! kubectl get namespace alpha-system &> /dev/null; then
        print_status "$BLUE" "🔧" "Creating alpha-system namespace"
        kubectl create namespace alpha-system
    fi
    
    print_status "$GREEN" "✅" "Prerequisites check completed"
}

# Function to deploy alpha database
deploy_alpha_database() {
    print_status "$BLUE" "🗄️" "Deploying alpha program database..."
    
    # Deploy PostgreSQL for alpha users
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install alpha-postgres bitnami/postgresql \
        --namespace alpha-system \
        --set auth.database=prsm_alpha \
        --set auth.username=alpha_user \
        --set auth.password="$(openssl rand -base64 32)" \
        --set primary.persistence.size="${ALPHA_DATABASE_SIZE}" \
        --set primary.resources.requests.memory=1Gi \
        --set primary.resources.requests.cpu=500m \
        --set primary.resources.limits.memory=2Gi \
        --set primary.resources.limits.cpu=1000m \
        --wait
    
    print_status "$GREEN" "✅" "Alpha database deployed"
}

# Function to deploy alpha Redis
deploy_alpha_redis() {
    print_status "$BLUE" "💾" "Deploying alpha program Redis cache..."
    
    helm upgrade --install alpha-redis bitnami/redis \
        --namespace alpha-system \
        --set auth.password="$(openssl rand -base64 32)" \
        --set master.persistence.size=10Gi \
        --set master.resources.requests.memory="${ALPHA_REDIS_MEMORY}" \
        --set master.resources.requests.cpu=250m \
        --set master.resources.limits.memory="${ALPHA_REDIS_MEMORY}" \
        --set master.resources.limits.cpu=500m \
        --wait
    
    print_status "$GREEN" "✅" "Alpha Redis deployed"
}

# Function to create alpha API deployment
create_alpha_api_deployment() {
    print_status "$BLUE" "🚀" "Creating alpha API deployment..."
    
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-alpha-api
  namespace: alpha-system
  labels:
    app: prsm-alpha-api
    version: ${ALPHA_API_VERSION}
spec:
  replicas: ${ALPHA_API_REPLICAS}
  selector:
    matchLabels:
      app: prsm-alpha-api
  template:
    metadata:
      labels:
        app: prsm-alpha-api
        version: ${ALPHA_API_VERSION}
    spec:
      containers:
      - name: prsm-alpha-api
        image: prsm-api:${ALPHA_API_VERSION}
        ports:
        - containerPort: 8000
        env:
        - name: PRSM_ENV
          value: "alpha"
        - name: PRSM_LOG_LEVEL
          value: "INFO"
        - name: ALPHA_PROGRAM_ENABLED
          value: "true"
        - name: MAX_ALPHA_USERS
          value: "${TARGET_ALPHA_USERS}"
        - name: INITIAL_FTNS_GRANT
          value: "${INITIAL_FTNS_GRANT}"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: alpha-postgres-postgresql
              key: postgres-password
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: alpha-redis
              key: redis-password
        resources:
          requests:
            memory: 512Mi
            cpu: 250m
          limits:
            memory: 1Gi
            cpu: 500m
        livenessProbe:
          httpGet:
            path: /alpha/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /alpha/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: prsm-alpha-api-service
  namespace: alpha-system
  labels:
    app: prsm-alpha-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: prsm-alpha-api
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prsm-alpha-ingress
  namespace: alpha-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - ${ALPHA_SUBDOMAIN}.prsm.network
    secretName: alpha-api-tls
  rules:
  - host: ${ALPHA_SUBDOMAIN}.prsm.network
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prsm-alpha-api-service
            port:
              number: 80
EOF
    
    print_status "$GREEN" "✅" "Alpha API deployment created"
}

# Function to deploy monitoring for alpha program
deploy_alpha_monitoring() {
    print_status "$BLUE" "📊" "Deploying alpha program monitoring..."
    
    # Create alpha-specific Grafana dashboard
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: alpha-dashboard-config
  namespace: alpha-system
data:
  alpha-dashboard.json: |
    {
      "dashboard": {
        "title": "PRSM Alpha Program Dashboard",
        "panels": [
          {
            "title": "Alpha User Registration",
            "type": "stat",
            "targets": [
              {"expr": "prsm_alpha_users_total"}
            ]
          },
          {
            "title": "Daily Active Users",
            "type": "graph",
            "targets": [
              {"expr": "prsm_alpha_active_users_daily"}
            ]
          },
          {
            "title": "Query Volume",
            "type": "graph",
            "targets": [
              {"expr": "rate(prsm_alpha_queries_total[5m])"}
            ]
          },
          {
            "title": "Feedback Submissions",
            "type": "stat",
            "targets": [
              {"expr": "prsm_alpha_feedback_total"}
            ]
          },
          {
            "title": "Engagement Score Distribution",
            "type": "histogram",
            "targets": [
              {"expr": "prsm_alpha_engagement_score"}
            ]
          }
        ]
      }
    }
EOF
    
    print_status "$GREEN" "✅" "Alpha monitoring deployed"
}

# Function to create alpha program documentation site
create_alpha_docs() {
    print_status "$BLUE" "📚" "Creating alpha program documentation site..."
    
    # Deploy documentation server
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alpha-docs
  namespace: alpha-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: alpha-docs
  template:
    metadata:
      labels:
        app: alpha-docs
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: docs-volume
          mountPath: /usr/share/nginx/html
      volumes:
      - name: docs-volume
        configMap:
          name: alpha-docs-content
---
apiVersion: v1
kind: Service
metadata:
  name: alpha-docs-service
  namespace: alpha-system
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 80
  selector:
    app: alpha-docs
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: alpha-docs-ingress
  namespace: alpha-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - alpha-docs.prsm.network
    secretName: alpha-docs-tls
  rules:
  - host: alpha-docs.prsm.network
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: alpha-docs-service
            port:
              number: 80
EOF
    
    print_status "$GREEN" "✅" "Alpha documentation site created"
}

# Function to setup alpha user analytics
setup_alpha_analytics() {
    print_status "$BLUE" "📈" "Setting up alpha user analytics..."
    
    # Create analytics service
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alpha-analytics
  namespace: alpha-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alpha-analytics
  template:
    metadata:
      labels:
        app: alpha-analytics
    spec:
      containers:
      - name: analytics
        image: prsm-analytics:latest
        env:
        - name: ALPHA_MODE
          value: "true"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: alpha-postgres-postgresql
              key: postgres-password
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
            cpu: 200m
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: alpha-analytics-report
  namespace: alpha-system
spec:
  schedule: "0 9 * * *"  # Daily at 9 AM UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: report-generator
            image: prsm-analytics:latest
            command:
            - python
            - -c
            - |
              import asyncio
              from prsm.alpha.user_management import get_alpha_manager
              
              async def generate_daily_report():
                  manager = get_alpha_manager()
                  analytics = await manager.get_program_analytics()
                  
                  # Generate and send daily report
                  print("Daily Alpha Program Report:")
                  print(f"Total Users: {analytics['program_overview']['total_registered_users']}")
                  print(f"Active Users (7d): {analytics['program_overview']['active_users_7d']}")
                  print(f"Total Queries: {analytics['usage_metrics']['total_queries']}")
                  print(f"Feedback Entries: {analytics['usage_metrics']['total_feedback_entries']}")
                  
                  # Export detailed analytics
                  report = await manager.export_analytics(format="json")
                  with open("/tmp/alpha_report.json", "w") as f:
                      f.write(report)
              
              asyncio.run(generate_daily_report())
          restartPolicy: OnFailure
EOF
    
    print_status "$GREEN" "✅" "Alpha analytics setup completed"
}

# Function to verify alpha program deployment
verify_alpha_deployment() {
    print_status "$BLUE" "🔍" "Verifying alpha program deployment..."
    
    # Wait for all deployments to be ready
    print_status "$BLUE" "⏳" "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available deployment/prsm-alpha-api -n alpha-system --timeout=300s
    kubectl wait --for=condition=available deployment/alpha-docs -n alpha-system --timeout=300s
    
    # Check service endpoints
    print_status "$BLUE" "🌐" "Checking service endpoints..."
    
    # Get API service external IP
    local api_ip
    api_ip=$(kubectl get service prsm-alpha-api-service -n alpha-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [[ "$api_ip" != "pending" && -n "$api_ip" ]]; then
        print_status "$GREEN" "✅" "Alpha API available at: http://$api_ip"
    else
        print_status "$YELLOW" "⏳" "Alpha API LoadBalancer IP still pending..."
    fi
    
    # Test API health endpoint
    local api_health
    if kubectl exec -n alpha-system deployment/prsm-alpha-api -- curl -f http://localhost:8000/alpha/health &>/dev/null; then
        print_status "$GREEN" "✅" "Alpha API health check passed"
    else
        print_status "$YELLOW" "⚠️" "Alpha API health check pending..."
    fi
    
    # Check database connectivity
    if kubectl exec -n alpha-system deployment/alpha-postgres-postgresql -- pg_isready -U alpha_user &>/dev/null; then
        print_status "$GREEN" "✅" "Alpha database connectivity verified"
    else
        print_status "$RED" "❌" "Alpha database connectivity failed"
    fi
    
    # Check Redis connectivity
    if kubectl exec -n alpha-system deployment/alpha-redis-master -- redis-cli ping &>/dev/null; then
        print_status "$GREEN" "✅" "Alpha Redis connectivity verified"
    else
        print_status "$RED" "❌" "Alpha Redis connectivity failed"
    fi
    
    print_status "$GREEN" "✅" "Alpha program deployment verification completed"
}

# Function to create alpha user recruitment materials
create_recruitment_materials() {
    print_status "$BLUE" "📢" "Creating alpha user recruitment materials..."
    
    # Create recruitment email template
    cat > "${PROJECT_ROOT}/docs/alpha/recruitment_email_template.md" << 'EOF'
# PRSM Alpha Testing Program Invitation

Subject: Exclusive Invitation: Join the PRSM Alpha Testing Program

Dear [NAME],

We're excited to invite you to participate in the **PRSM Alpha Testing Program** - an exclusive opportunity to test the future of decentralized AI infrastructure.

## What is PRSM?
PRSM (Protocol for Recursive Scientific Modeling) is a revolutionary platform that:
- Provides intelligent routing between local and cloud AI models
- Optimizes costs while maintaining quality
- Ensures privacy through local processing of sensitive data
- Enables collaboration in a federated AI network

## Why We're Inviting You
Based on your background in [AI/ML/Data Science/Software Engineering], we believe you'd provide valuable insights to help shape PRSM's development.

## Alpha Program Benefits
✅ **Early Access**: Be among the first 100 users to experience PRSM
✅ **Free Credits**: 1,000 FTNS tokens for extensive testing ($100 value)
✅ **Direct Influence**: Your feedback directly shapes product development
✅ **Community Access**: Join our exclusive Discord with other technical users
✅ **Recognition**: Certificate of participation and potential beta access

## What We're Looking For
- 2-3 hours of testing over 2-3 weeks
- Feedback on usability, performance, and features
- Bug reports and feature suggestions
- Participation in community discussions

## Get Started
1. Register at: https://alpha.prsm.network/register
2. Use invitation code: [INVITATION_CODE]
3. Complete onboarding in under 5 minutes
4. Start testing immediately

## Questions?
- Technical: alpha-support@prsm.network
- General: [YOUR_EMAIL]
- Discord: https://discord.gg/prsm-alpha

We're building the future of AI infrastructure, and we'd love your help making it amazing.

Best regards,
The PRSM Team

---
*This invitation is valid for 7 days. Limited to 100 alpha users.*
EOF
    
    # Create social media recruitment posts
    cat > "${PROJECT_ROOT}/docs/alpha/social_media_posts.md" << 'EOF'
# PRSM Alpha Program Social Media Posts

## Twitter/X Posts

### Post 1
🚀 Excited to announce the PRSM Alpha Testing Program! 

Looking for 100 technical users to test our decentralized AI infrastructure.

✅ Free access + $100 credits
✅ Direct impact on development
✅ Exclusive community access

Apply: https://alpha.prsm.network
#AI #MachineLearning #Decentralized

### Post 2
🧪 PRSM Alpha is LIVE! 

Revolutionary features:
🔄 Intelligent local/cloud routing
💰 Cost optimization
🔒 Privacy-first design
🌐 Federated AI network

Join 100 alpha testers: https://alpha.prsm.network
#AlphaTesting #AIInfrastructure

## LinkedIn Post
We're launching the PRSM Alpha Testing Program and looking for exceptional technical professionals to help shape the future of AI infrastructure.

PRSM introduces intelligent routing between local and cloud models, optimizing for cost, performance, and privacy. Our federated approach enables collaborative AI while maintaining data sovereignty.

Alpha Program Highlights:
• Exclusive access for 100 technical users
• $100 worth of free testing credits
• Direct influence on product development
• Community of AI researchers and engineers
• Certification upon completion

Ideal candidates:
• AI/ML researchers and engineers
• Data scientists and analysts
• Software engineers working with AI
• Technical leaders evaluating AI solutions

The program runs for 3 weeks with flexible testing schedules. Your feedback will directly influence our roadmap and public launch.

Apply now: https://alpha.prsm.network

#ArtificialIntelligence #MachineLearning #AlphaTesting #TechInnovation

## Reddit Posts

### r/MachineLearning
**[Alpha] PRSM - Decentralized AI Infrastructure Testing Program**

Hey r/MachineLearning! We're launching our alpha testing program for PRSM, a protocol that intelligently routes between local and cloud AI models.

**What makes PRSM different:**
- Automatic cost optimization 
- Privacy-preserving local routing for sensitive data
- Federated network for collaborative AI
- Real-time performance benchmarking

**Alpha Program:**
- 100 spots available
- 1,000 free tokens ($100 value)
- 2-3 week commitment
- Direct developer access

Looking for researchers, engineers, and data scientists who want early access to cutting-edge AI infrastructure.

Register: https://alpha.prsm.network
Questions welcome in comments!

### r/ArtificialIntelligence
**Join the PRSM Alpha: Testing the Future of AI Infrastructure**

PRSM introduces intelligent routing between local and cloud AI models, optimizing for cost, performance, and privacy needs.

Perfect for:
- Researchers with proprietary datasets
- Companies balancing cost and performance  
- Anyone interested in federated AI systems

Alpha includes free credits, exclusive community access, and direct impact on development.

Limited to 100 technical users. Apply: https://alpha.prsm.network
EOF
    
    # Create recruitment tracking spreadsheet
    cat > "${PROJECT_ROOT}/docs/alpha/recruitment_tracking.csv" << 'EOF'
Date,Channel,Invitations_Sent,Registrations,Conversion_Rate,Notes
2024-06-14,Email,0,0,0%,Program launch
2024-06-14,Twitter,0,0,0%,Initial posts
2024-06-14,LinkedIn,0,0,0%,Professional network
2024-06-14,Reddit,0,0,0%,Community outreach
2024-06-14,Direct_Outreach,0,0,0%,Personal invitations
EOF
    
    print_status "$GREEN" "✅" "Recruitment materials created"
}

# Function to launch community Discord
setup_community_discord() {
    print_status "$BLUE" "💬" "Setting up alpha community Discord..."
    
    # Create Discord setup instructions
    cat > "${PROJECT_ROOT}/docs/alpha/discord_setup.md" << 'EOF'
# PRSM Alpha Community Discord Setup

## Server Structure

### Channels

**📢 Announcements**
- #welcome - Welcome message and program overview
- #announcements - Important updates and news
- #weekly-updates - Weekly progress reports

**💬 General Discussion**
- #general-chat - Open discussion about PRSM
- #introductions - New member introductions
- #off-topic - Non-PRSM related chat

**🔧 Technical Discussion**
- #api-questions - API usage and integration help
- #performance-testing - Performance benchmarks and optimization
- #routing-strategies - Discussion of routing approaches
- #model-integration - Custom model integration topics

**🐛 Feedback & Support**
- #bug-reports - Bug reports and issues
- #feature-requests - New feature suggestions
- #support - General help and support
- #feedback-general - General feedback and suggestions

**🎯 Testing Focus**
- #daily-challenges - Daily testing challenges
- #use-case-sharing - Share interesting use cases
- #benchmarking - Performance testing results
- #collaboration - Cross-user collaboration

**📊 Analytics & Insights**
- #program-stats - Weekly program statistics
- #leaderboard - Community engagement leaderboard
- #achievements - User milestones and achievements

### Roles

**@Alpha Team** - PRSM development team
**@Alpha User** - Verified alpha program participants
**@High Contributor** - Users with >50 queries and 5+ feedback items
**@Community Helper** - Users helping others with questions
**@Beta Candidate** - Users eligible for beta program

### Bots

**PRSM Bot** - Integration with alpha program
- User verification
- Stats tracking
- Automated announcements
- Query assistance

**MEE6** - Moderation and engagement
- Welcome messages
- Role management
- Leaderboards

### Rules

1. **Be Respectful** - Treat all members with respect
2. **Stay On Topic** - Keep discussions relevant to channels
3. **No Spam** - Avoid repetitive or promotional content
4. **Share Constructively** - Provide helpful feedback and insights
5. **Respect Privacy** - Don't share others' data or results without permission
6. **Use Appropriate Channels** - Post in the most relevant channel
7. **Search Before Asking** - Check if your question has been answered
8. **Follow Discord ToS** - Adhere to Discord's Terms of Service

### Alpha Program Integration

**Verification Process:**
1. Register for alpha program
2. Receive Discord invite link
3. Join server and verify with alpha user ID
4. Get @Alpha User role automatically

**Features:**
- Real-time stats updates
- Automated milestone celebrations
- Direct feedback submission
- Weekly challenge notifications
EOF
    
    print_status "$GREEN" "✅" "Discord community setup guide created"
}

# Function to display alpha program summary
display_alpha_summary() {
    print_status "$CYAN" "📊" "PRSM Alpha Program Launch Summary"
    echo "=============================================="
    echo "Program: PRSM Alpha Testing"
    echo "Version: $ALPHA_API_VERSION"
    echo "Target Users: $TARGET_ALPHA_USERS technical users"
    echo "Launch Date: $(date)"
    echo "=============================================="
    
    print_status "$BLUE" "🏗️" "Infrastructure Status:"
    echo "  API Replicas: $ALPHA_API_REPLICAS"
    echo "  Worker Replicas: $ALPHA_WORKER_REPLICAS"
    echo "  Database Size: $ALPHA_DATABASE_SIZE"
    echo "  Redis Memory: $ALPHA_REDIS_MEMORY"
    echo "  Namespace: alpha-system"
    
    print_status "$BLUE" "🎯" "Program Features:"
    echo "  ✅ User registration and authentication"
    echo "  ✅ Real-time usage tracking and analytics"
    echo "  ✅ Comprehensive feedback collection"
    echo "  ✅ Community collaboration tools"
    echo "  ✅ Automated reporting and monitoring"
    echo "  ✅ Onboarding documentation"
    
    print_status "$BLUE" "🌐" "Access Points:"
    echo "  Alpha API: https://${ALPHA_SUBDOMAIN}.prsm.network"
    echo "  Documentation: https://alpha-docs.prsm.network"
    echo "  Registration: https://alpha.prsm.network/register"
    echo "  Community: https://discord.gg/prsm-alpha"
    
    print_status "$BLUE" "📈" "Success Metrics:"
    echo "  Target Registrations: $TARGET_ALPHA_USERS users"
    echo "  Expected Queries: 10,000+ total"
    echo "  Feedback Target: 1,000+ entries"
    echo "  Completion Rate: >80%"
    
    print_status "$GREEN" "✅" "Alpha program launched successfully!"
    print_status "$CYAN" "🎉" "Ready to onboard 100+ technical users for comprehensive testing!"
}

# Main alpha program launch
main() {
    print_status "$PURPLE" "🚀" "Launching PRSM Alpha User Testing Program"
    print_status "$BLUE" "⚙️" "Target: $TARGET_ALPHA_USERS technical users"
    print_status "$BLUE" "🎯" "Initial FTNS Grant: $INITIAL_FTNS_GRANT tokens per user"
    
    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/docs/alpha"
    
    # Launch sequence
    check_alpha_prerequisites
    deploy_alpha_database
    deploy_alpha_redis
    create_alpha_api_deployment
    deploy_alpha_monitoring
    create_alpha_docs
    setup_alpha_analytics
    create_recruitment_materials
    setup_community_discord
    verify_alpha_deployment
    display_alpha_summary
    
    print_status "$GREEN" "🎉" "PRSM Alpha Program launched successfully!"
    print_status "$CYAN" "📢" "Ready to recruit and onboard alpha users!"
}

# Run main function
main "$@"