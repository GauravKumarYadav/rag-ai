#!/bin/bash
# Run live endpoint tests against running containers
#
# Usage:
#   ./scripts/run_live_tests.sh              # Run all tests
#   ./scripts/run_live_tests.sh containers   # Test only container health
#   ./scripts/run_live_tests.sh auth         # Test only authentication
#   ./scripts/run_live_tests.sh quick        # Quick smoke test of all endpoints

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
CHROMADB_URL="${CHROMADB_URL:-http://localhost:8020}"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Live Endpoint Integration Tests${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "API URL: $API_BASE_URL"
echo "Ollama URL: $OLLAMA_URL"
echo "ChromaDB URL: $CHROMADB_URL"
echo ""

# Check if containers are running first
echo -e "${YELLOW}Checking container status...${NC}"
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo -e "${YELLOW}Warning: Neither docker nor podman found. Skipping container check.${NC}"
    CONTAINER_CMD=""
fi

if [ -n "$CONTAINER_CMD" ]; then
    REQUIRED_CONTAINERS=("rag-nginx" "rag-chat-api" "rag-ollama" "rag-chromadb" "rag-redis" "rag-mysql")
    MISSING_CONTAINERS=()
    
    for container in "${REQUIRED_CONTAINERS[@]}"; do
        if ! $CONTAINER_CMD ps --format "{{.Names}}" | grep -q "^${container}$"; then
            MISSING_CONTAINERS+=("$container")
        fi
    done
    
    if [ ${#MISSING_CONTAINERS[@]} -gt 0 ]; then
        echo -e "${RED}ERROR: The following containers are not running:${NC}"
        for container in "${MISSING_CONTAINERS[@]}"; do
            echo -e "  - $container"
        done
        echo ""
        echo "Start containers with: podman-compose -f docker-compose.microservices.yml up -d"
        exit 1
    fi
    echo -e "${GREEN}All required containers are running.${NC}"
fi

echo ""

# Run tests based on argument
TEST_PATTERN=""
case "${1:-all}" in
    containers)
        TEST_PATTERN="TestContainerHealth"
        echo -e "${YELLOW}Running container health tests...${NC}"
        ;;
    auth)
        TEST_PATTERN="TestAuthentication"
        echo -e "${YELLOW}Running authentication tests...${NC}"
        ;;
    status)
        TEST_PATTERN="TestHealthAndStatus"
        echo -e "${YELLOW}Running health and status tests...${NC}"
        ;;
    conversations)
        TEST_PATTERN="TestConversations"
        echo -e "${YELLOW}Running conversation tests...${NC}"
        ;;
    clients)
        TEST_PATTERN="TestClients"
        echo -e "${YELLOW}Running client tests...${NC}"
        ;;
    documents)
        TEST_PATTERN="TestDocuments"
        echo -e "${YELLOW}Running document tests...${NC}"
        ;;
    chat)
        TEST_PATTERN="TestChat"
        echo -e "${YELLOW}Running chat tests...${NC}"
        ;;
    models)
        TEST_PATTERN="TestModels"
        echo -e "${YELLOW}Running model tests...${NC}"
        ;;
    admin)
        TEST_PATTERN="TestAdmin"
        echo -e "${YELLOW}Running admin tests...${NC}"
        ;;
    evaluation)
        TEST_PATTERN="TestEvaluation"
        echo -e "${YELLOW}Running evaluation tests...${NC}"
        ;;
    quick)
        TEST_PATTERN="TestEndpointSummary"
        echo -e "${YELLOW}Running quick smoke tests...${NC}"
        ;;
    all)
        TEST_PATTERN=""
        echo -e "${YELLOW}Running all tests...${NC}"
        ;;
    *)
        echo "Usage: $0 [containers|auth|status|conversations|clients|documents|chat|models|admin|evaluation|quick|all]"
        exit 1
        ;;
esac

echo ""

# Export environment variables
export API_BASE_URL
export OLLAMA_URL
export CHROMADB_URL

# Run pytest
cd "$(dirname "$0")/.."

if [ -n "$TEST_PATTERN" ]; then
    python -m pytest tests/test_live_endpoints.py -v -k "$TEST_PATTERN" --tb=short
else
    python -m pytest tests/test_live_endpoints.py -v --tb=short
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Some tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE
