# CI/CD Integration Guide

Integrate PRSM into your CI/CD pipelines for automated testing, deployment, and quality assurance.

## 🎯 Overview

This guide covers integrating PRSM into various CI/CD platforms including GitHub Actions, GitLab CI, Jenkins, and Azure DevOps for automated deployment and testing workflows.

## 📋 Prerequisites

- Git repository with PRSM code
- CI/CD platform access
- Docker registry access
- Basic knowledge of CI/CD concepts
- Target deployment environment

## 🚀 GitHub Actions Integration

### Basic Workflow

```yaml
# .github/workflows/prsm-ci.yml
name: PRSM CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/prsm-api

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: prsm_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        black --check .
        flake8 .
        mypy prsm/
    
    - name: Run security checks
      run: |
        bandit -r prsm/
        safety check
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/prsm_test
        REDIS_URL: redis://localhost:6379/0
        ENVIRONMENT: test
      run: |
        pytest tests/ -v --cov=prsm --cov-report=xml --cov-report=term
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILDTIME=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.created'] }}
          VERSION=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.version'] }}
          REVISION=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.revision'] }}

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      env:
        KUBECONFIG_DATA: ${{ secrets.STAGING_KUBECONFIG }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        echo "$KUBECONFIG_DATA" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        
        # Update image in deployment
        kubectl set image deployment/prsm-api \
          prsm-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop-${{ github.sha }} \
          -n prsm-staging
        
        # Wait for rollout
        kubectl rollout status deployment/prsm-api -n prsm-staging --timeout=300s
    
    - name: Run staging tests
      run: |
        # Wait for staging to be ready
        sleep 30
        
        # Run integration tests against staging
        export STAGING_URL=$(kubectl get service prsm-api-service -n prsm-staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        python -m pytest tests/integration/ --staging-url=http://$STAGING_URL

  deploy-production:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      env:
        KUBECONFIG_DATA: ${{ secrets.PRODUCTION_KUBECONFIG }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        echo "$KUBECONFIG_DATA" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        
        # Blue-green deployment strategy
        kubectl set image deployment/prsm-api \
          prsm-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }} \
          -n prsm-production
        
        # Wait for rollout
        kubectl rollout status deployment/prsm-api -n prsm-production --timeout=600s
    
    - name: Health check
      run: |
        export PRODUCTION_URL=$(kubectl get service prsm-api-service -n prsm-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        # Wait for service to be ready
        for i in {1..30}; do
          if curl -f http://$PRODUCTION_URL/health; then
            echo "Production deployment successful!"
            break
          fi
          echo "Waiting for production to be ready... ($i/30)"
          sleep 10
        done
    
    - name: Create GitHub release
      if: github.ref == 'refs/heads/main'
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: v${{ github.run_number }}
        release_name: Release v${{ github.run_number }}
        body: |
          Automated release from commit ${{ github.sha }}
          
          ## Changes
          ${{ github.event.head_commit.message }}
          
          ## Docker Image
          `${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }}`
        draft: false
        prerelease: false
```

### Advanced GitHub Actions

```yaml
# .github/workflows/advanced-prsm-pipeline.yml
name: Advanced PRSM Pipeline

on:
  push:
    branches: [main, develop, feature/*]
  pull_request:
    branches: [main, develop]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      backend: ${{ steps.changes.outputs.backend }}
      frontend: ${{ steps.changes.outputs.frontend }}
      docs: ${{ steps.changes.outputs.docs }}
      infrastructure: ${{ steps.changes.outputs.infrastructure }}
    steps:
    - uses: actions/checkout@v4
    - uses: dorny/paths-filter@v2
      id: changes
      with:
        filters: |
          backend:
            - 'prsm/**'
            - 'requirements*.txt'
            - 'Dockerfile'
            - 'alembic/**'
          frontend:
            - 'frontend/**'
            - 'package*.json'
          docs:
            - 'docs/**'
            - '*.md'
          infrastructure:
            - 'k8s/**'
            - 'terraform/**'
            - 'helm/**'

  backend-tests:
    runs-on: ubuntu-latest
    needs: detect-changes
    if: needs.detect-changes.outputs.backend == 'true'
    
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
        test-type: ['unit', 'integration', 'performance']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run ${{ matrix.test-type }} tests
      env:
        TEST_TYPE: ${{ matrix.test-type }}
      run: |
        case $TEST_TYPE in
          unit)
            pytest tests/unit/ -v --junitxml=test-results-unit.xml
            ;;
          integration)
            pytest tests/integration/ -v --junitxml=test-results-integration.xml
            ;;
          performance)
            pytest tests/performance/ -v --junitxml=test-results-performance.xml
            ;;
        esac
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}-${{ matrix.test-type }}
        path: test-results-*.xml

  quality-gates:
    runs-on: ubuntu-latest
    needs: backend-tests
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for SonarQube
    
    - name: SonarQube Scan
      uses: sonarqube-quality-gate-action@master
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
    
    - name: Quality Gate Check
      run: |
        # Check if quality gate passed
        if [[ "${{ env.SONAR_QUALITY_GATE_STATUS }}" != "PASSED" ]]; then
          echo "Quality gate failed!"
          exit 1
        fi

  infrastructure-validation:
    runs-on: ubuntu-latest
    needs: detect-changes
    if: needs.detect-changes.outputs.infrastructure == 'true'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Validate Kubernetes manifests
      run: |
        # Install kubeval
        wget https://github.com/instrumenta/kubeval/releases/latest/download/kubeval-linux-amd64.tar.gz
        tar xf kubeval-linux-amd64.tar.gz
        sudo mv kubeval /usr/local/bin
        
        # Validate manifests
        find k8s/ -name "*.yaml" -exec kubeval {} \;
    
    - name: Terraform validation
      if: hashFiles('terraform/**/*.tf') != ''
      uses: hashicorp/setup-terraform@v2
      with:
        terraform_version: 1.5.0
    
    - name: Terraform fmt check
      if: hashFiles('terraform/**/*.tf') != ''
      run: terraform fmt -check -recursive terraform/
    
    - name: Terraform validate
      if: hashFiles('terraform/**/*.tf') != ''
      run: |
        cd terraform/
        terraform init -backend=false
        terraform validate

  build-matrix:
    runs-on: ubuntu-latest
    needs: [backend-tests, quality-gates]
    if: needs.detect-changes.outputs.backend == 'true'
    
    strategy:
      matrix:
        include:
        - platform: linux/amd64
          arch: amd64
        - platform: linux/arm64
          arch: arm64
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build for ${{ matrix.arch }}
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: ${{ matrix.platform }}
        push: false
        tags: prsm-api:${{ matrix.arch }}
        cache-from: type=gha,scope=${{ matrix.arch }}
        cache-to: type=gha,mode=max,scope=${{ matrix.arch }}

  deploy-preview:
    runs-on: ubuntu-latest
    needs: [backend-tests, quality-gates]
    if: github.event_name == 'pull_request'
    environment: 
      name: preview-pr-${{ github.event.number }}
      url: https://prsm-pr-${{ github.event.number }}.preview.example.com
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy preview environment
      env:
        PR_NUMBER: ${{ github.event.number }}
        COMMIT_SHA: ${{ github.sha }}
      run: |
        # Deploy to preview namespace
        envsubst < k8s/preview/deployment-template.yaml | kubectl apply -f -
        
        # Wait for deployment
        kubectl wait --for=condition=available --timeout=300s \
          deployment/prsm-api-pr-$PR_NUMBER -n preview
    
    - name: Comment PR with preview URL
      uses: actions/github-script@v6
      with:
        script: |
          const prNumber = context.issue.number;
          const previewUrl = `https://prsm-pr-${prNumber}.preview.example.com`;
          
          github.rest.issues.createComment({
            issue_number: prNumber,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `🚀 Preview deployment ready at: ${previewUrl}`
          });

  cleanup-preview:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request' && github.event.action == 'closed'
    
    steps:
    - name: Cleanup preview environment
      env:
        PR_NUMBER: ${{ github.event.number }}
      run: |
        # Delete preview deployment
        kubectl delete namespace preview-pr-$PR_NUMBER --ignore-not-found=true
```

## 🦊 GitLab CI Integration

### GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - test
  - security
  - build
  - deploy-staging
  - deploy-production

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  POSTGRES_DB: prsm_test
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: test_password

# Templates
.docker-login: &docker-login
  - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $CI_REGISTRY

.kubectl-setup: &kubectl-setup
  - curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  - chmod +x kubectl
  - mv kubectl /usr/local/bin/

# Validation stage
lint:
  stage: validate
  image: python:3.11
  before_script:
    - pip install black flake8 mypy
  script:
    - black --check .
    - flake8 .
    - mypy prsm/
  rules:
    - changes:
        - "**/*.py"

validate-yaml:
  stage: validate
  image: alpine:latest
  before_script:
    - apk add --no-cache yamllint
  script:
    - yamllint k8s/ helm/
  rules:
    - changes:
        - "k8s/**/*.yaml"
        - "helm/**/*.yaml"

# Test stage
unit-tests:
  stage: test
  image: python:3.11
  services:
    - postgres:14
    - redis:7
  variables:
    DATABASE_URL: postgresql://postgres:test_password@postgres:5432/prsm_test
    REDIS_URL: redis://redis:6379/0
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - pytest tests/unit/ -v --cov=prsm --cov-report=xml --cov-report=term
  coverage: '/TOTAL.+ ([0-9]{1,3}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
      junit: test-results.xml
    paths:
      - coverage.xml
    expire_in: 1 week

integration-tests:
  stage: test
  image: python:3.11
  services:
    - postgres:14
    - redis:7
  variables:
    DATABASE_URL: postgresql://postgres:test_password@postgres:5432/prsm_test
    REDIS_URL: redis://redis:6379/0
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - pytest tests/integration/ -v
  artifacts:
    reports:
      junit: integration-test-results.xml
    expire_in: 1 week

performance-tests:
  stage: test
  image: python:3.11
  services:
    - postgres:14
    - redis:7
  variables:
    DATABASE_URL: postgresql://postgres:test_password@postgres:5432/prsm_test
    REDIS_URL: redis://redis:6379/0
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pip install locust
  script:
    - pytest tests/performance/ -v
    - locust -f tests/load/locustfile.py --headless --users 10 --spawn-rate 2 --run-time 1m --host http://localhost:8000
  artifacts:
    paths:
      - performance-results/
    expire_in: 1 week
  allow_failure: true

# Security stage
security-scan:
  stage: security
  image: python:3.11
  before_script:
    - pip install bandit safety
  script:
    - bandit -r prsm/ -f json -o bandit-report.json
    - safety check --json --output safety-report.json
  artifacts:
    reports:
      sast: bandit-report.json
    paths:
      - bandit-report.json
      - safety-report.json
    expire_in: 1 week
  allow_failure: true

container-scan:
  stage: security
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - *docker-login
  script:
    - docker build -t $IMAGE_TAG .
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock 
        -v $PWD:/tmp/.cache/ aquasec/trivy:latest image 
        --exit-code 0 --no-progress --format template --template "@contrib/sarif.tpl" 
        -o /tmp/.cache/trivy-results.sarif $IMAGE_TAG
  artifacts:
    reports:
      sast: trivy-results.sarif
    expire_in: 1 week

# Build stage
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - *docker-login
  script:
    - docker build 
        --build-arg VERSION=$CI_COMMIT_TAG 
        --build-arg COMMIT_SHA=$CI_COMMIT_SHA 
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") 
        -t $IMAGE_TAG 
        -t $CI_REGISTRY_IMAGE:latest .
    - docker push $IMAGE_TAG
    - docker push $CI_REGISTRY_IMAGE:latest
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_COMMIT_BRANCH == "develop"
    - if: $CI_COMMIT_TAG

# Deploy staging
deploy-staging:
  stage: deploy-staging
  image: alpine/k8s:latest
  environment:
    name: staging
    url: https://prsm-staging.example.com
  before_script:
    - *kubectl-setup
    - echo $STAGING_KUBECONFIG | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
  script:
    - kubectl set image deployment/prsm-api prsm-api=$IMAGE_TAG -n prsm-staging
    - kubectl rollout status deployment/prsm-api -n prsm-staging --timeout=300s
    
    # Health check
    - |
      for i in {1..30}; do
        if curl -f https://prsm-staging.example.com/health; then
          echo "Staging deployment successful!"
          break
        fi
        echo "Waiting for staging to be ready... ($i/30)"
        sleep 10
      done
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"

# Deploy production
deploy-production:
  stage: deploy-production
  image: alpine/k8s:latest
  environment:
    name: production
    url: https://prsm.example.com
  before_script:
    - *kubectl-setup
    - echo $PRODUCTION_KUBECONFIG | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
  script:
    # Blue-green deployment
    - kubectl set image deployment/prsm-api prsm-api=$IMAGE_TAG -n prsm-production
    - kubectl rollout status deployment/prsm-api -n prsm-production --timeout=600s
    
    # Health check
    - |
      for i in {1..60}; do
        if curl -f https://prsm.example.com/health; then
          echo "Production deployment successful!"
          break
        fi
        echo "Waiting for production to be ready... ($i/60)"
        sleep 10
      done
    
    # Create Git tag for successful production deployment
    - |
      if [ "$CI_COMMIT_BRANCH" == "main" ]; then
        git tag "prod-$(date +%Y%m%d-%H%M%S)-$CI_COMMIT_SHORT_SHA"
        git push origin --tags
      fi
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  when: manual  # Require manual approval for production
```

## 🔧 Jenkins Integration

### Jenkinsfile

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        IMAGE_NAME = 'prsm/api'
        IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_COMMIT.take(8)}"
        KUBECONFIG_STAGING = credentials('kubeconfig-staging')
        KUBECONFIG_PRODUCTION = credentials('kubeconfig-production')
    }
    
    tools {
        python '3.11'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT = sh(
                        script: 'git rev-parse HEAD',
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Setup') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                '''
            }
        }
        
        stage('Lint') {
            parallel {
                stage('Black') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            black --check .
                        '''
                    }
                }
                stage('Flake8') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            flake8 . --output-file=flake8-report.txt
                        '''
                        publishHTML([
                            allowMissing: false,
                            alwaysLinkToLastBuild: true,
                            keepAll: true,
                            reportDir: '.',
                            reportFiles: 'flake8-report.txt',
                            reportName: 'Flake8 Report'
                        ])
                    }
                }
                stage('MyPy') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            mypy prsm/ --xml-report mypy-report
                        '''
                        publishHTML([
                            allowMissing: false,
                            alwaysLinkToLastBuild: true,
                            keepAll: true,
                            reportDir: 'mypy-report',
                            reportFiles: 'index.html',
                            reportName: 'MyPy Report'
                        ])
                    }
                }
            }
        }
        
        stage('Security') {
            parallel {
                stage('Bandit') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            bandit -r prsm/ -f json -o bandit-report.json
                        '''
                        archiveArtifacts artifacts: 'bandit-report.json'
                    }
                }
                stage('Safety') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            safety check --json --output safety-report.json
                        '''
                        archiveArtifacts artifacts: 'safety-report.json'
                    }
                }
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            pytest tests/unit/ -v --junitxml=unit-test-results.xml --cov=prsm --cov-report=xml:coverage.xml
                        '''
                        publishTestResults testResultsPattern: 'unit-test-results.xml'
                        publishCoverage adapters: [
                            coberturaAdapter('coverage.xml')
                        ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh '''
                            docker-compose -f docker-compose.test.yml up -d
                            sleep 30
                            . venv/bin/activate
                            pytest tests/integration/ -v --junitxml=integration-test-results.xml
                            docker-compose -f docker-compose.test.yml down
                        '''
                        publishTestResults testResultsPattern: 'integration-test-results.xml'
                    }
                }
            }
        }
        
        stage('Build Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    buildingTag()
                }
            }
            steps {
                script {
                    def image = docker.build("${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.IMAGE_TAG}")
                    docker.withRegistry("https://${env.DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                script {
                    withKubeConfig([credentialsId: 'kubeconfig-staging']) {
                        sh """
                            kubectl set image deployment/prsm-api prsm-api=${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.IMAGE_TAG} -n prsm-staging
                            kubectl rollout status deployment/prsm-api -n prsm-staging --timeout=300s
                        """
                    }
                }
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                script {
                    withKubeConfig([credentialsId: 'kubeconfig-production']) {
                        sh """
                            kubectl set image deployment/prsm-api prsm-api=${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.IMAGE_TAG} -n prsm-production
                            kubectl rollout status deployment/prsm-api -n prsm-production --timeout=600s
                        """
                    }
                }
            }
        }
        
        stage('Health Check') {
            when {
                anyOf {
                    branch 'develop'
                    branch 'main'
                }
            }
            steps {
                script {
                    def environment = env.BRANCH_NAME == 'main' ? 'production' : 'staging'
                    def url = env.BRANCH_NAME == 'main' ? 'https://prsm.example.com' : 'https://prsm-staging.example.com'
                    
                    timeout(time: 5, unit: 'MINUTES') {
                        waitUntil {
                            script {
                                def response = sh(
                                    script: "curl -s -o /dev/null -w '%{http_code}' ${url}/health",
                                    returnStdout: true
                                ).trim()
                                return response == '200'
                            }
                        }
                    }
                    echo "Health check passed for ${environment}"
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            script {
                if (env.BRANCH_NAME == 'main') {
                    // Create Git tag for successful production deployment
                    sh """
                        git tag prod-\$(date +%Y%m%d-%H%M%S)-${env.GIT_COMMIT.take(8)}
                        git push origin --tags
                    """
                }
            }
        }
        failure {
            emailext (
                subject: "PRSM Pipeline Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## 🔄 Azure DevOps Integration

### Azure Pipeline

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
    - main
    - develop
  tags:
    include:
    - v*

pr:
  branches:
    include:
    - main
    - develop

variables:
  containerRegistry: 'prsmregistry.azurecr.io'
  imageRepository: 'prsm-api'
  dockerfilePath: 'Dockerfile'
  tag: '$(Build.BuildId)'
  pythonVersion: '3.11'

stages:
- stage: Validate
  displayName: 'Code Validation'
  jobs:
  - job: Lint
    displayName: 'Code Linting'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
      displayName: 'Use Python $(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install black flake8 mypy
      displayName: 'Install linting tools'
    
    - script: |
        black --check .
        flake8 .
        mypy prsm/
      displayName: 'Run linting'

- stage: Test
  displayName: 'Testing'
  dependsOn: Validate
  jobs:
  - job: UnitTests
    displayName: 'Unit Tests'
    pool:
      vmImage: 'ubuntu-latest'
    services:
      postgres: postgres:14
      redis: redis:7
    variables:
      DATABASE_URL: 'postgresql://postgres:postgres@localhost:5432/prsm_test'
      REDIS_URL: 'redis://localhost:6379/0'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
      displayName: 'Use Python $(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/unit/ -v --junitxml=test-results.xml --cov=prsm --cov-report=xml
      displayName: 'Run unit tests'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: 'test-results.xml'
        testRunTitle: 'Unit Tests'
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: 'coverage.xml'

  - job: SecurityScan
    displayName: 'Security Scanning'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
    
    - script: |
        pip install bandit safety
        bandit -r prsm/ -f json -o bandit-report.json
        safety check --json --output safety-report.json
      displayName: 'Run security scans'
    
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: 'bandit-report.json'
        artifactName: 'security-reports'

- stage: Build
  displayName: 'Build and Push'
  dependsOn: Test
  condition: and(succeeded(), or(eq(variables['Build.SourceBranch'], 'refs/heads/main'), eq(variables['Build.SourceBranch'], 'refs/heads/develop')))
  jobs:
  - job: Build
    displayName: 'Build Docker Image'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: Docker@2
      displayName: 'Build and push image'
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: 'azure-container-registry'
        tags: |
          $(tag)
          latest

- stage: DeployStaging
  displayName: 'Deploy to Staging'
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
  jobs:
  - deployment: DeployStaging
    displayName: 'Deploy to Staging'
    pool:
      vmImage: 'ubuntu-latest'
    environment: 'staging'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: 'Deploy to Kubernetes'
            inputs:
              action: deploy
              kubernetesServiceConnection: 'staging-k8s'
              namespace: 'prsm-staging'
              manifests: |
                k8s/staging/deployment.yaml
                k8s/staging/service.yaml
              containers: '$(containerRegistry)/$(imageRepository):$(tag)'

- stage: DeployProduction
  displayName: 'Deploy to Production'
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: DeployProduction
    displayName: 'Deploy to Production'
    pool:
      vmImage: 'ubuntu-latest'
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: 'Deploy to Kubernetes'
            inputs:
              action: deploy
              kubernetesServiceConnection: 'production-k8s'
              namespace: 'prsm-production'
              manifests: |
                k8s/production/deployment.yaml
                k8s/production/service.yaml
              containers: '$(containerRegistry)/$(imageRepository):$(tag)'
          
          - task: Bash@3
            displayName: 'Health Check'
            inputs:
              targetType: 'inline'
              script: |
                for i in {1..30}; do
                  if curl -f https://prsm.example.com/health; then
                    echo "Production deployment successful!"
                    exit 0
                  fi
                  echo "Waiting for production to be ready... ($i/30)"
                  sleep 10
                done
                echo "Health check failed!"
                exit 1
```

## 📊 Quality Gates and Testing

### SonarQube Integration

```yaml
# sonar-project.properties
sonar.projectKey=prsm-api
sonar.projectName=PRSM API
sonar.projectVersion=1.0
sonar.sources=prsm/
sonar.tests=tests/
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.xunit.reportPath=test-results.xml
sonar.exclusions=**/*_pb2.py,**/migrations/**
sonar.coverage.exclusions=tests/**,**/migrations/**
```

### Testing Strategy

```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prsm.core.database import Base
from prsm.api.main import create_app

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///./test.db")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_app(test_db):
    """Create test application."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["DATABASE_URL"] = "sqlite:///./test.db"
    return app

@pytest.fixture
def client(test_app):
    """Create test client."""
    return test_app.test_client()

# tests/test_health_check.py
def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

# tests/performance/test_load.py
import asyncio
import aiohttp
import time
from statistics import mean, median

async def load_test_query_endpoint():
    """Load test the query endpoint."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            task = asyncio.create_task(
                make_request(session, f"Test query {i}")
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_requests = [r for r in results if not isinstance(r, Exception)]
        response_times = [r['response_time'] for r in successful_requests]
        
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Successful requests: {len(successful_requests)}/100")
        print(f"Average response time: {mean(response_times):.2f}s")
        print(f"Median response time: {median(response_times):.2f}s")
        
        assert len(successful_requests) >= 95  # 95% success rate
        assert mean(response_times) < 2.0  # Average response time under 2s

async def make_request(session, query):
    """Make a single request."""
    start_time = time.time()
    try:
        async with session.post(
            'http://localhost:8000/api/v1/query',
            json={'prompt': query, 'user_id': 'test-user'}
        ) as response:
            await response.json()
            return {
                'status_code': response.status,
                'response_time': time.time() - start_time
            }
    except Exception as e:
        return e
```

## 📋 Best Practices

### Environment Configuration

```yaml
# .env.ci
# CI/CD environment variables
DATABASE_URL=postgresql://postgres:test_password@localhost:5432/prsm_test
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=test
LOG_LEVEL=DEBUG
TESTING=true

# Security
JWT_SECRET=test-jwt-secret
API_KEY=test-api-key

# Features
ENABLE_METRICS=false
ENABLE_TRACING=false
```

### Deployment Scripts

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-staging}
IMAGE_TAG=${2:-latest}
NAMESPACE="prsm-${ENVIRONMENT}"

echo "Deploying PRSM to ${ENVIRONMENT}..."
echo "Image tag: ${IMAGE_TAG}"
echo "Namespace: ${NAMESPACE}"

# Validate inputs
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    echo "Error: Environment must be 'staging' or 'production'"
    exit 1
fi

# Set kubeconfig based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    export KUBECONFIG="$HOME/.kube/production-config"
else
    export KUBECONFIG="$HOME/.kube/staging-config"
fi

# Update deployment with new image
kubectl set image deployment/prsm-api \
    prsm-api="prsm/api:${IMAGE_TAG}" \
    -n "${NAMESPACE}"

# Wait for rollout
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/prsm-api \
    -n "${NAMESPACE}" \
    --timeout=600s

# Health check
echo "Performing health check..."
if [ "$ENVIRONMENT" = "production" ]; then
    HEALTH_URL="https://prsm.example.com/health"
else
    HEALTH_URL="https://prsm-staging.example.com/health"
fi

for i in {1..30}; do
    if curl -f "$HEALTH_URL" >/dev/null 2>&1; then
        echo "Health check passed!"
        echo "Deployment successful!"
        exit 0
    fi
    echo "Waiting for service to be ready... ($i/30)"
    sleep 10
done

echo "Health check failed!"
exit 1
```

---

**Need help with CI/CD integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).