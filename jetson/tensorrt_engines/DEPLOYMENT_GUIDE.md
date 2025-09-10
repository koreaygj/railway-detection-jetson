# TensorRT Deployment Guide for Jetson Orin Nano

## Prerequisites
1. Jetson Orin Nano with JetPack 5.x or later
2. TensorRT 8.x or later
3. CUDA 11.x or later

## Setup Steps

### 1. Prepare Jetson Orin Nano
```bash
# Run the setup script on your Jetson
chmod +x setup_jetson.sh
./setup_jetson.sh
```

### 2. Transfer Files
```bash
# Copy TensorRT engines and scripts to Jetson
scp -r tensorrt_engines/ jetson@<jetson-ip>:~/railway-detection/
scp jetson_tensorrt_inference.py jetson@<jetson-ip>:~/railway-detection/
```

### 3. Run Inference

#### Camera Inference:
```bash
python3 jetson_tensorrt_inference.py --engine model_fp16.engine --source 0
```

#### Video File Inference:
```bash
python3 jetson_tensorrt_inference.py --engine model_fp16.engine --source video.mp4
```

#### Benchmark:
```bash
python3 jetson_tensorrt_inference.py --engine model_fp16.engine --benchmark
```

## Performance Expectations

### Jetson Orin Nano Specifications:
- GPU: NVIDIA Ampere (1024 CUDA cores)
- Memory: 8GB LPDDR5
- AI Performance: 40 TOPS (INT8)

### Expected Performance:
- **YOLOv11n + TensorRT FP16**: ~25-30 FPS at 640x640
- **YOLOv11s + TensorRT FP16**: ~15-20 FPS at 640x640
- **YOLOv11n + TensorRT INT8**: ~35-45 FPS at 640x640

## Optimization Tips

1. **Use FP16 for best balance of speed/accuracy**
2. **Enable MAX performance mode**:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```
3. **Optimize memory usage**:
   ```bash
   sudo systemctl disable nvargus-daemon
   echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
   ```

## Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Low FPS**: Check if model is using GPU (nvidia-smi)
3. **TensorRT build fails**: Ensure sufficient workspace memory

### Monitor Performance:
```bash
# Check GPU utilization
nvidia-smi

# Check system resources
htop

# Check power consumption
sudo jetson_clocks --show
```

## Model Performance Summary

Successfully converted models: 0/3

