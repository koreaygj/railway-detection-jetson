# Jetson Orin Nano TensorRT Deployment Guide

## 🚂 Railway Object Detection on Jetson Orin Nano

### Model Performance Summary
- **YOLOv11n (yolo11n_20250909_141751)**: mAP@0.5 **94.6%**, mAP@0.5:0.95 **86.4%**
- **YOLOv11n (yolo11n_railway)**: mAP@0.5 **94.5%**, mAP@0.5:0.95 **84.9%**  
- **YOLOv11s (yolo11s_railway)**: mAP@0.5 **94.4%**, mAP@0.5:0.95 **86.3%**

---

## 📋 Step 1: Prepare Jetson Orin Nano

### System Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-opencv python3-numpy

# Check JetPack version
sudo apt show nvidia-jetpack
```

### Install PyTorch for Jetson
```bash
# Download PyTorch wheel for Jetson (adjust version as needed)
wget https://nvidia.box.com/shared/static/mp164o2ox7tg3f930qdf8321s9v3rr26.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.1.0-cp310-cp310-linux_aarch64.whl

# Install torchvision
sudo apt install -y libjpeg-dev zlib1g-dev
git clone https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
```

### Install Ultralytics
```bash
pip3 install ultralytics
pip3 install onnx onnxruntime-gpu
```

---

## 📦 Step 2: Transfer Models to Jetson

### Copy trained models
```bash
# From your development machine
scp result/trained_models/*/weights/best.pt jetson@<jetson-ip>:~/railway-detection/models/

# Or create directory structure on Jetson:
mkdir -p ~/railway-detection/models
cd ~/railway-detection/models
```

### Copy scripts
```bash
# Copy all conversion and inference scripts
scp jetson/*.py jetson@<jetson-ip>:~/railway-detection/
```

---

## ⚡ Step 3: Convert to TensorRT on Jetson

### Method 1: Using Ultralytics directly
```bash
# Convert individual models
yolo export model=models/yolo11n_railway_best.pt format=engine imgsz=640 half=True device=0
yolo export model=models/yolo11s_railway_best.pt format=engine imgsz=640 half=True device=0
```

### Method 2: Using conversion script
```bash
# Edit convert_on_jetson.py to set correct model paths
python3 convert_on_jetson.py
```

### Method 3: Manual conversion with custom settings
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolo11n_railway_best.pt')

# Export to TensorRT FP16 (recommended)
model.export(
    format='engine',
    imgsz=640,
    batch=1,
    device=0,
    half=True,        # FP16 for best performance
    workspace=4       # 4GB workspace
)
```

---

## 🏃 Step 4: Run Inference

### Real-time Camera Inference
```bash
python3 jetson_tensorrt_inference.py --engine yolo11n_railway.engine --source 0
```

### Video File Inference  
```bash
python3 jetson_tensorrt_inference.py --engine yolo11n_railway.engine --source video.mp4 --save-video output.mp4
```

### Benchmark Performance
```bash
python3 jetson_tensorrt_inference.py --engine yolo11n_railway.engine --benchmark
```

---

## 📊 Expected Performance on Jetson Orin Nano

### Hardware Specs:
- **GPU**: NVIDIA Ampere (1024 CUDA cores)  
- **Memory**: 8GB LPDDR5
- **AI Performance**: 40 TOPS (INT8)

### Expected FPS @ 640x640:
| Model | TensorRT FP16 | TensorRT INT8 |
|-------|---------------|---------------|
| YOLOv11n | **25-30 FPS** | **35-45 FPS** |
| YOLOv11s | **15-20 FPS** | **20-25 FPS** |

### Optimization Commands:
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check current performance mode
sudo nvpmodel -q
```

---

## 🔧 Advanced Optimizations

### 1. INT8 Quantization (for maximum speed)
```python
model.export(
    format='engine',
    imgsz=640,
    batch=1,
    device=0,
    int8=True,
    data='path/to/calibration/data.yaml'
)
```

### 2. Dynamic Batch Size
```python
model.export(
    format='engine',
    imgsz=640,
    batch=[1, 2, 4],  # Dynamic batching
    device=0,
    half=True
)
```

### 3. Memory Optimization
```bash
# Reduce system memory usage
sudo systemctl disable nvargus-daemon
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Monitor memory usage
watch nvidia-smi
```

---

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   ```bash
   # Reduce batch size or image resolution
   # Check memory usage: nvidia-smi
   ```

2. **Low FPS performance**
   ```bash
   # Check if using GPU
   nvidia-smi
   
   # Enable max performance
   sudo nvpmodel -m 0 && sudo jetson_clocks
   ```

3. **TensorRT build fails**
   ```bash
   # Increase workspace memory
   # Check available disk space
   df -h
   ```

4. **Model accuracy drops**
   - Use FP16 instead of INT8
   - Verify input preprocessing matches training

### Monitor Performance:
```bash
# GPU utilization
nvidia-smi -l 1

# System resources  
htop

# Power consumption
sudo jtop
```

---

## 📈 Performance Tuning Tips

1. **Use FP16 for best balance** (speed vs accuracy)
2. **Enable MAX performance mode** before inference
3. **Optimize input pipeline** (use GPU memory pinning)
4. **Batch processing** for multiple images
5. **Profile your pipeline** to find bottlenecks

### Sample Performance Profiling:
```python
import time
import numpy as np

# Profile each step
times = {'preprocess': [], 'inference': [], 'postprocess': []}

for _ in range(100):
    start = time.time()
    # ... preprocessing
    times['preprocess'].append(time.time() - start)
    
    start = time.time()
    # ... inference  
    times['inference'].append(time.time() - start)
    
    start = time.time()
    # ... postprocessing
    times['postprocess'].append(time.time() - start)

# Print average times
for stage, timings in times.items():
    print(f"{stage}: {np.mean(timings)*1000:.2f}ms ± {np.std(timings)*1000:.2f}ms")
```

---

## 🎯 Target Performance Goals

For railway safety applications:

- **Minimum FPS**: 15 FPS (for real-time monitoring)
- **Optimal FPS**: 25+ FPS (for responsive detection)  
- **Accuracy**: mAP@0.5 > 90% (safety critical)
- **Latency**: < 50ms total pipeline time

**Recommended Model**: YOLOv11n with TensorRT FP16 - best balance of speed and accuracy.

---

## 📞 Support

If you encounter issues:
1. Check Jetson community forums
2. Verify JetPack and TensorRT versions
3. Test with smaller input resolution first
4. Monitor system resources during conversion/inference

**Hardware Requirements Met**: ✅ Jetson Orin Nano specs are sufficient for real-time railway detection at target performance levels.