#!/bin/bash
set -euo pipefail

# Production Deployment Script with Immutable Git SHA Tags
# This script demonstrates proper image tagging for production deployments

# Get current Git SHA
GIT_SHA=$(git rev-parse --short HEAD)
GIT_TAG=$(git describe --tags --always)
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION="v0.1.0"

echo "🚀 PRSM Production Deployment"
echo "=================================="
echo "Version: ${VERSION}"
echo "Git SHA: ${GIT_SHA}"
echo "Git Tag: ${GIT_TAG}"
echo "Build Date: ${BUILD_DATE}"
echo "=================================="

# Validate environment
if [[ -z "${ENVIRONMENT:-}" ]]; then
    echo "❌ ERROR: ENVIRONMENT variable must be set (staging/production)"
    exit 1
fi

if [[ "${ENVIRONMENT}" != "staging" && "${ENVIRONMENT}" != "production" ]]; then
    echo "❌ ERROR: ENVIRONMENT must be 'staging' or 'production'"
    exit 1
fi

# Build images with immutable tags
IMMUTABLE_TAG="${VERSION}-sha-${GIT_SHA}"
REGISTRY="${REGISTRY:-gcr.io/prsm-platform}"

echo "📦 Building images with immutable tags..."
echo "Tag: ${IMMUTABLE_TAG}"

# Build main API image
docker build \
    --tag "${REGISTRY}/prsm-api:${IMMUTABLE_TAG}" \
    --tag "${REGISTRY}/prsm-api:${VERSION}" \
    --build-arg GIT_SHA="${GIT_SHA}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg VERSION="${VERSION}" \
    -f docker/Dockerfile.api .

# Build worker image
docker build \
    --tag "${REGISTRY}/prsm-worker:${IMMUTABLE_TAG}" \
    --tag "${REGISTRY}/prsm-worker:${VERSION}" \
    --build-arg GIT_SHA="${GIT_SHA}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg VERSION="${VERSION}" \
    -f docker/Dockerfile.worker .

echo "✅ Images built successfully"

# Push images to registry
echo "📤 Pushing images to registry..."
docker push "${REGISTRY}/prsm-api:${IMMUTABLE_TAG}"
docker push "${REGISTRY}/prsm-api:${VERSION}"
docker push "${REGISTRY}/prsm-worker:${IMMUTABLE_TAG}"
docker push "${REGISTRY}/prsm-worker:${VERSION}"

echo "✅ Images pushed to registry"

# Generate kustomization with immutable tags
echo "🔧 Generating deployment manifests..."
KUSTOMIZE_DIR="deploy/kubernetes/overlays/${ENVIRONMENT}"
TEMP_DIR=$(mktemp -d)

# Copy kustomization files
cp -r "${KUSTOMIZE_DIR}" "${TEMP_DIR}/"

# Replace placeholder Git SHA in patches
if [[ -f "${TEMP_DIR}/${ENVIRONMENT}/production-patches.yaml" ]]; then
    sed -i.bak "s/PLACEHOLDER_GIT_SHA/${GIT_SHA}/g" "${TEMP_DIR}/${ENVIRONMENT}/production-patches.yaml"
fi

# Update kustomization.yaml with immutable tags
cat > "${TEMP_DIR}/${ENVIRONMENT}/image-patches.yaml" << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

images:
  - name: prsm-api
    newName: ${REGISTRY}/prsm-api
    newTag: ${IMMUTABLE_TAG}
  - name: prsm-worker
    newName: ${REGISTRY}/prsm-worker
    newTag: ${IMMUTABLE_TAG}
EOF

# Add image patches to kustomization
echo "  - image-patches.yaml" >> "${TEMP_DIR}/${ENVIRONMENT}/kustomization.yaml"

# Apply deployment
echo "🚀 Deploying to ${ENVIRONMENT}..."
kubectl apply -k "${TEMP_DIR}/${ENVIRONMENT}"

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/prsm-api -n "prsm-${ENVIRONMENT}" --timeout=600s
kubectl rollout status deployment/prsm-worker -n "prsm-${ENVIRONMENT}" --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n "prsm-${ENVIRONMENT}" -l app.kubernetes.io/name=prsm-api
kubectl get pods -n "prsm-${ENVIRONMENT}" -l app.kubernetes.io/name=prsm-worker

# Health check
echo "🏥 Running health checks..."
API_POD=$(kubectl get pods -n "prsm-${ENVIRONMENT}" -l app.kubernetes.io/name=prsm-api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n "prsm-${ENVIRONMENT}" "${API_POD}" -- curl -f http://localhost:8000/health

echo "✅ Deployment completed successfully!"
echo "🏷️  Deployed image: ${REGISTRY}/prsm-api:${IMMUTABLE_TAG}"
echo "📝 Git SHA: ${GIT_SHA}"
echo "🕐 Deployed at: ${BUILD_DATE}"

# Clean up
rm -rf "${TEMP_DIR}"

# Optional: Tag Git commit with deployment info
if [[ "${ENVIRONMENT}" == "production" ]]; then
    echo "🏷️  Tagging Git commit for production deployment..."
    git tag -a "deploy-prod-${GIT_SHA}" -m "Production deployment ${IMMUTABLE_TAG} at ${BUILD_DATE}"
    echo "✅ Git tag created: deploy-prod-${GIT_SHA}"
fi