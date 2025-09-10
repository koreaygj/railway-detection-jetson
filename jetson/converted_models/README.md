
# Jetson Nano Deployment Instructions

## 1. Copy converted models to Jetson Nano
scp -r converted_models/* jetson@<jetson-ip>:~/railway-detection/

## 2. Install dependencies on Jetson Nano
sudo apt update
sudo apt install python3-pip python3-opencv
pip3 install numpy onnxruntime-gpu

## 3. Run inference
python3 jetson_inference.py --model <model_path> --source 0

## 4. Benchmark performance
python3 jetson_inference.py --model <model_path> --benchmark

## 5. Available models:

### yolo11n_railway:
   - onnx_fp16: converted_models/best/best_onnx_fp16.onnx

### yolo11s_railway:
   - onnx_fp16: converted_models/best/best_onnx_fp16.onnx

### yolo11n_20250909_141751:
   - onnx_fp16: converted_models/best/best_onnx_fp16.onnx
