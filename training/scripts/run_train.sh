#!/bin/bash
# Railway Object Detection Training Script
# Usage: ./run_train.sh [MODEL] [EPOCHS] [BATCH_SIZE]

set -e  # Exit on any error

# Default parameters
MODEL=${1:-"yolo11n.pt"}
EPOCHS=${2:-100}
BATCH_SIZE=${3:-16}
DEVICE=${4:-"0"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚂 Railway Object Detection Training${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "📋 Model: ${GREEN}${MODEL}${NC}"
echo -e "🔄 Epochs: ${GREEN}${EPOCHS}${NC}"
echo -e "📦 Batch Size: ${GREEN}${BATCH_SIZE}${NC}"
echo -e "🎯 Device: ${GREEN}${DEVICE}${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "train.py" ]]; then
    echo -e "${RED}❌ Error: train.py not found. Please run from training/scripts/ directory${NC}"
    exit 1
fi

# Check if data exists
PROJECT_ROOT="../../"
DATA_PATH="${PROJECT_ROOT}data/yolo_dataset/data.yaml"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo -e "${RED}❌ Error: Dataset not found at ${DATA_PATH}${NC}"
    echo -e "${YELLOW}💡 Please ensure the dataset is properly prepared${NC}"
    exit 1
fi

# Create result directory if it doesn't exist
RESULT_DIR="${PROJECT_ROOT}result"
mkdir -p "${RESULT_DIR}"

echo -e "${GREEN}✅ All checks passed. Starting training...${NC}"
echo ""

# Run training
python train.py \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --verbose \
    "$@"

echo ""
echo -e "${GREEN}🎉 Training script completed!${NC}"
echo -e "${BLUE}📁 Check results in: ${RESULT_DIR}${NC}"