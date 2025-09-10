#!/bin/bash
# Jetson Orin Nano Setup Script for Railway Detection
# Run this script on your Jetson Orin Nano

echo "🚂 Setting up Railway Detection on Jetson Orin Nano"
echo "================================================="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-opencv python3-numpy

# Install PyTorch for Jetson (ARM64)
# Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 for latest version
wget https://nvidia.box.com/shared/static/mp164o2ox7tg3f930qdf8321s9v3rr26.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.1.0-cp310-cp310-linux_aarch64.whl

# Install TorchVision
sudo apt install -y libjpeg-dev zlib1g-dev
pip3 install torchvision

# Install Ultralytics YOLO
pip3 install ultralytics

# Install additional dependencies
pip3 install opencv-python-headless pillow

# Set up CUDA environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Test TensorRT installation
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# Test CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "✅ Setup complete! You can now run TensorRT inference."
echo ""
echo "📋 Next steps:"
echo "1. Copy your .engine files to this device"
echo "2. Run: python3 jetson_tensorrt_inference.py --engine model.engine --source 0"

