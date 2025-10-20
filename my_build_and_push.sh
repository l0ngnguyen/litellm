#!/bin/bash

# Custom Docker build and push script
# Builds LiteLLM with custom UI and pushes to GitHub Container Registry
# Image tags include date information for versioning

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LiteLLM Custom Docker Build and Push ===${NC}"
echo ""

# Configuration
GITHUB_USERNAME="${GITHUB_USERNAME:-$(git config user.name | tr '[:upper:]' '[:lower:]' | tr ' ' '-')}"
REGISTRY="ghcr.io"
IMAGE_NAME="litellm"
DOCKERFILE="my_Dockerfile"

# Generate date-based tag
DATE_TAG=$(date +%Y-%m-%d)
DATETIME_TAG=$(date +%Y-%m-%d-%H%M)

# Full image names
FULL_IMAGE_NAME="${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Registry: ${REGISTRY}"
echo "  Username: ${GITHUB_USERNAME}"
echo "  Image: ${IMAGE_NAME}"
echo "  Dockerfile: ${DOCKERFILE}"
echo "  Date Tag: ${DATE_TAG}"
echo "  DateTime Tag: ${DATETIME_TAG}"
echo ""

# Check if GitHub username is set
if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${RED}Error: GITHUB_USERNAME is not set.${NC}"
    echo "Please set it via environment variable:"
    echo "  export GITHUB_USERNAME=your-github-username"
    echo "Or it will be auto-detected from git config user.name"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: ${DOCKERFILE} not found.${NC}"
    exit 1
fi

# Check if logged into GitHub Container Registry
echo -e "${YELLOW}Checking GitHub Container Registry authentication...${NC}"

# Try to login if CR_PAT is set
if [ -n "$CR_PAT" ]; then
    echo "Found CR_PAT environment variable, attempting login..."
    if echo "$CR_PAT" | docker login ghcr.io -u "${GITHUB_USERNAME}" --password-stdin 2>/dev/null; then
        echo -e "${GREEN}✓ Successfully logged in to GitHub Container Registry${NC}"
    else
        echo -e "${RED}✗ Login failed with provided CR_PAT${NC}"
        echo "Please check your token and username"
        exit 1
    fi
else
    # Check if already logged in
    if docker login ghcr.io --username "${GITHUB_USERNAME}" --password-stdin < /dev/null 2>&1 | grep -q "Login Succeeded"; then
        echo -e "${GREEN}✓ Already logged in to GitHub Container Registry${NC}"
    else
        echo -e "${YELLOW}Not logged in to GitHub Container Registry.${NC}"
        echo ""
        echo "Please login using one of these methods:"
        echo ""
        echo "Method 1: Set CR_PAT environment variable and re-run"
        echo "  export CR_PAT=your_github_token"
        echo "  ./my_build_and_push.sh"
        echo ""
        echo "Method 2: Login manually first"
        echo "  echo \$CR_PAT | docker login ghcr.io -u ${GITHUB_USERNAME} --password-stdin"
        echo "  ./my_build_and_push.sh"
        echo ""
        echo "To create a token:"
        echo "  https://github.com/settings/tokens"
        echo "  Required scopes: write:packages, read:packages, delete:packages"
        echo ""
        exit 1
    fi
fi

echo -e "${GREEN}✓ Authenticated with GitHub Container Registry${NC}"
echo ""

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
echo "This may take several minutes..."
echo ""

docker build \
    -f "${DOCKERFILE}" \
    -t "${FULL_IMAGE_NAME}:${DATE_TAG}" \
    -t "${FULL_IMAGE_NAME}:${DATETIME_TAG}" \
    -t "${FULL_IMAGE_NAME}:latest" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Push images to registry
echo ""
echo -e "${YELLOW}Pushing images to GitHub Container Registry...${NC}"
echo ""

# Push date tag
echo "Pushing ${FULL_IMAGE_NAME}:${DATE_TAG}..."
docker push "${FULL_IMAGE_NAME}:${DATE_TAG}"

# Push datetime tag
echo "Pushing ${FULL_IMAGE_NAME}:${DATETIME_TAG}..."
docker push "${FULL_IMAGE_NAME}:${DATETIME_TAG}"

# Push latest tag
echo "Pushing ${FULL_IMAGE_NAME}:latest..."
docker push "${FULL_IMAGE_NAME}:latest"

echo ""
echo -e "${GREEN}=== Build and Push Completed Successfully ===${NC}"
echo ""
echo "Available images:"
echo "  - ${FULL_IMAGE_NAME}:${DATE_TAG}"
echo "  - ${FULL_IMAGE_NAME}:${DATETIME_TAG}"
echo "  - ${FULL_IMAGE_NAME}:latest"
echo ""
echo "To pull and run:"
echo "  docker pull ${FULL_IMAGE_NAME}:${DATE_TAG}"
echo "  docker run -p 4000:4000 ${FULL_IMAGE_NAME}:${DATE_TAG}"
echo ""
echo "View on GitHub:"
echo "  https://github.com/${GITHUB_USERNAME}?tab=packages"
