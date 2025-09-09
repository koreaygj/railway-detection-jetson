# Railway Object Detection on NVIDIA Jetson: Real-time Performance Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![CUDA](https://img.shields.io/badge/CUDA-11.4+-green.svg)](https://developer.nvidia.com/cuda-downloads)

## Abstract

Railway safety monitoring through real-time object detection is critical for preventing accidents caused by foreign objects on railway infrastructure. This repository presents a comprehensive benchmark study of lightweight object detection models optimized for deployment on NVIDIA Jetson Orin Nano, focusing on foreign object detection (FOD) in railway environments. We evaluate various YOLO-series models to determine the optimal performance-accuracy trade-off for real-time railway applications, considering computational constraints, power consumption, and deployment feasibility.

## 🚂 Motivation

### Application Necessity
- **Safety Critical**: Detection of foreign objects (plastic bags, debris, bird nests, balloons) on railway power lines and tracks
- **Real-time Requirements**: Railway environments demand immediate response for safety interventions
- **Cost Efficiency**: Embedded solutions reduce Total Cost of Ownership (TCO) and power consumption
- **Scalability**: Edge deployment enables distributed monitoring across extensive railway networks

### Research Objectives
1. **Performance Trade-off Analysis**: Identify optimal balance between detection accuracy and inference speed
2. **Embedded Optimization**: Implement lightweight models suitable for resource-constrained environments
3. **Quantization Strategies**: Evaluate INT8 and FP16 quantization effects on model performance
4. **Real-world Applicability**: Validate feasibility for actual railway monitoring systems

## 📊 Dataset

### RailFOD23 Dataset
- **Source**: [RailFOD23 on Figshare](https://figshare.com/articles/figure/RailFOD23_zip/24180738/3)
- **Size**: 14,615 images with 40,541 object annotations
- **Format**: COCO format annotations
- **Categories**: Foreign objects on railway infrastructure including:
  - Plastic bags and debris
  - Bird nests
  - Balloons
  - Other foreign materials

### Data Characteristics
- **Collection Method**: Manual collection, AI-generated images, and automatic synthesis
- **Annotation Quality**: Professional manual annotation with quality validation
- **Use Case**: Railway safety monitoring and AI research applications

## 🔧 Experimental Setup

### Hardware Platform
- **Primary Target**: NVIDIA Jetson Orin Nano
  - **Compute Capability**: 2.56 TFLOPS (FP16)
  - **Target Performance**: 20-30 GFLOPS (25 FPS baseline)
  - **Power Efficiency**: Optimized for railway field deployment

### Software Environment
- **Framework**: Ultralytics YOLOv11
- **Inference Engine**: TensorRT optimization
- **Quantization**: FP16 and INT8 precision evaluation
- **Development**: Python 3.8+, CUDA 11.4+

### Training Configuration
- **Epochs**: 50-100 (ablation study)
- **Evaluation Metrics**: COCO-style mAP, FPS, memory usage, power consumption
- **Optimization**: Model pruning, quantization, and TensorRT acceleration

## 🎯 Key Contributions

### 1. Application Necessity Analysis
- Comprehensive analysis of railway object detection requirements
- Justification for embedded deployment in railway environments
- Cost-benefit analysis of edge vs. cloud processing

### 2. Performance Trade-off Identification
- Systematic evaluation of accuracy vs. speed trade-offs
- Identification of acceptable performance thresholds for railway safety
- Real-time processing requirements analysis (target: 25+ FPS)

### 3. Optimization Approaches
- **Quantization Strategies**: Comparative analysis of FP16 vs. INT8
- **Model Architecture**: YOLO series selection rationale for railway applications
- **Hardware Acceleration**: TensorRT optimization for Jetson platform
- **Power Optimization**: Energy-efficient inference strategies

### 4. Deployment Feasibility
- Demonstration of practical railway environment deployment
- Integration guidelines for existing railway monitoring systems
- Performance validation under real-world constraints

## 📈 Benchmark Results

*[Results will be updated upon completion of experiments]*

### Model Performance Comparison
| Model | mAP@0.5 | mAP@0.5:0.95 | FPS (FP16) | FPS (INT8) | Memory (MB) | Power (W) |
|-------|---------|--------------|------------|------------|-------------|-----------|
| YOLOv11n | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11s | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11m | TBD | TBD | TBD | TBD | TBD | TBD |

### Optimization Impact
- **Quantization Effects**: Performance vs. accuracy analysis
- **TensorRT Acceleration**: Inference speed improvements
- **Memory Optimization**: Resource usage reduction

## 🚀 Installation

### Prerequisites
```bash
# NVIDIA Jetson Orin Nano setup
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-dev

# CUDA and cuDNN (pre-installed on JetPack)
# TensorRT (included in JetPack)
```

### Dependencies
```bash
# Clone repository
git clone https://github.com/koreaygj/railway-detection-jetson.git
cd railway-detection-jetson

# Install requirements
pip3 install -r requirements.txt

# Install additional Jetson-specific packages
pip3 install jetson-stats
sudo -H pip3 install jetson-inference jetson-utils
```

## 📖 Usage

### Training
```bash
# Train model on RailFOD23 dataset
python train.py --data railfod23.yaml --model yolo11n.pt --epochs 100 --device 0

# Multi-GPU training (if available)
python -m torch.distributed.launch --nproc_per_node 2 train.py --data railfod23.yaml --model yolo11s.pt
```

### Inference
```bash
# Run inference on Jetson
python detect.py --model best.pt --source /path/to/images --device 0 --half

# Real-time camera inference
python detect.py --model best.pt --source 0 --device 0 --half --save
```

### Benchmarking
```bash
# Performance evaluation
python benchmark.py --model best.pt --data test_images/ --device 0

# TensorRT conversion and benchmarking
python export.py --model best.pt --format engine --half
python benchmark_trt.py --engine best.engine --data test_images/
```

## 📊 Evaluation Metrics

### Primary Metrics
- **mAP@0.5**: COCO-style mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: COCO-style mAP across IoU thresholds 0.5-0.95
- **FPS**: Frames per second (inference speed)
- **Latency**: End-to-end processing time per frame

### Efficiency Metrics
- **Memory Usage**: Peak GPU memory consumption
- **Power Consumption**: Average power draw during inference
- **Model Size**: Disk storage requirements
- **FLOPS**: Floating point operations count

## 🔬 Research Direction

### Future Work
1. **Advanced Quantization**: Exploration of mixed-precision and dynamic quantization
2. **Model Compression**: Investigation of pruning, distillation, and neural architecture search
3. **Multi-Modal Integration**: Fusion with other sensor data (LiDAR, thermal)
4. **Edge Deployment**: Real-world railway infrastructure testing

### Expected Outcomes
- Establishment of performance benchmarks for railway object detection
- Guidelines for embedded AI deployment in railway environments
- Foundation for advanced lightweight detection research
- Practical deployment framework for railway safety systems

## 📚 Citation

*[To be updated upon paper publication]*

```bibtex
@article{yang2024railway,
  title={Railway Object Detection on NVIDIA Jetson: Real-time Performance Optimization},
  author={Yang, Gyeongjin and [Other Authors]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]},
  note={Paper in preparation}
}
```

## 🏢 Affiliation

**AVLab, Chungbuk National University**
- Advanced Vision Laboratory
- Department of [Department Name]
- Cheongju, South Korea

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RailFOD23 dataset authors for providing comprehensive railway object detection data
- Ultralytics team for YOLOv11 implementation
- NVIDIA for Jetson platform and development tools
- Chungbuk National University AVLab for research support

## 📞 Contact

For questions about this research, please contact:
- **Primary Researcher**: [Your Name] - [your.email@domain.com]
- **Lab**: AVLab, Chungbuk National University

---

**Note**: This repository contains benchmark code and experimental results for an ongoing research project. Results and methodologies are subject to updates as the research progresses.