# Railway Object Detection - Training Guide

This directory contains all training-related scripts, configurations, and utilities for the Railway Object Detection project using YOLOv11.

## 📂 Directory Structure

```
training/
├── README.md                 # This file - comprehensive training guide
├── scripts/                  # Training and evaluation scripts
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Model evaluation and benchmarking
│   └── quick_start.py       # Interactive quick start script
├── configs/                  # Configuration files and templates
└── utils/                   # Utility scripts and helpers
```

## 🚀 Quick Start

### Option 1: Interactive Mode (Recommended for beginners)
```bash
cd training/scripts
python quick_start.py
```
This interactive script will guide you through:
- Environment validation
- Dataset verification  
- Training configuration
- Model evaluation

### Option 2: Direct Training
```bash
cd training/scripts
python train.py --model yolo11n.pt --epochs 100 --batch-size 16
```

## 📊 Dataset Information

- **Location**: `../data/yolo_dataset/`
- **Training Images**: 11,691
- **Validation Images**: 2,924
- **Classes**: 4 (niaocao, suliaodai, piaofuwu, qiqiu)
- **Format**: YOLO format with COCO-style evaluation

### Class Descriptions
| Class ID | Chinese Name | English Name | Description |
|----------|--------------|--------------|-------------|
| 0 | niaocao | Bird Nest | Bird nests on railway infrastructure |
| 1 | suliaodai | Plastic Bag | Plastic bags and debris |
| 2 | piaofuwu | Flag/Flutter | Fluttering objects and flags |
| 3 | qiqiu | Balloon | Balloons and similar objects |

## 🔧 Training Scripts

### 1. train.py - Main Training Script

**Purpose**: Complete training pipeline with advanced configuration options.

**Key Features**:
- Multiple model variants support (YOLOv11n/s/m)
- Weights & Biases integration
- Resume training capability
- Optimized hyperparameters for railway detection
- Automatic model validation

**Usage**:
```bash
# Basic training
python train.py --model yolo11n.pt --epochs 100

# Advanced training with logging
python train.py \
    --model yolo11n.pt \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.01 \
    --wandb \
    --name "railway_experiment_1"

# Resume training
python train.py --resume ./railway-detection/train/weights/last.pt
```

**Key Parameters**:
- `--model`: Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt)
- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Batch size (default: 16, reduce if GPU memory limited)
- `--lr`: Learning rate (default: 0.01)
- `--device`: GPU device (0, 1, 2, etc. or cpu)
- `--wandb`: Enable Weights & Biases logging
- `--name`: Experiment name

### 2. evaluate.py - Model Evaluation & Benchmarking

**Purpose**: Comprehensive model evaluation with performance metrics and visualization.

**Key Features**:
- COCO-style mAP evaluation
- Inference speed benchmarking
- Confusion matrix generation
- Performance visualization plots
- JSON results export

**Usage**:
```bash
# Basic evaluation
python evaluate.py --model ./railway-detection/train/weights/best.pt

# Full evaluation with plots and benchmarking
python evaluate.py \
    --model ./railway-detection/train/weights/best.pt \
    --save-plots \
    --benchmark \
    --save-dir ../results/evaluation_yolo11n
```

**Output**:
- Validation metrics (mAP@0.5, mAP@0.5:0.95, etc.)
- Speed benchmarking results
- Confusion matrix plot
- Performance distribution plots
- JSON summary file

### 3. quick_start.py - Interactive Guide

**Purpose**: User-friendly interactive script for beginners.

**Features**:
- Environment validation
- Dataset verification
- Guided training configuration
- Automatic model discovery for evaluation
- Step-by-step instructions

**Menu Options**:
1. Quick training (50 epochs)
2. Full training (100 epochs) 
3. Model evaluation
4. Custom training parameters
5. Exit

## 📈 Training Configuration

### Recommended Settings

#### For Development/Testing:
```bash
python train.py \
    --model yolo11n.pt \
    --epochs 50 \
    --batch-size 16 \
    --name "dev_test"
```

#### For Full Training:
```bash
python train.py \
    --model yolo11n.pt \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.01 \
    --weight-decay 0.0005 \
    --wandb \
    --name "railway_yolo11n_full"
```

#### For Resource-Constrained Systems:
```bash
python train.py \
    --model yolo11n.pt \
    --epochs 100 \
    --batch-size 8 \
    --workers 4 \
    --name "railway_low_resource"
```

### Hyperparameter Tuning

The training script includes railway-optimized augmentation settings:

```python
# Optimized for railway detection
'hsv_h': 0.015,      # Hue augmentation
'hsv_s': 0.7,        # Saturation augmentation  
'hsv_v': 0.4,        # Value augmentation
'degrees': 0.0,      # Rotation (disabled for railway)
'translate': 0.1,    # Translation
'scale': 0.5,        # Scaling
'shear': 0.0,        # Shear (disabled)
'perspective': 0.0,  # Perspective (disabled)
'flipud': 0.0,       # Vertical flip (disabled)
'fliplr': 0.5,       # Horizontal flip
'mosaic': 1.0,       # Mosaic augmentation
'mixup': 0.0,        # Mixup (disabled)
```

## 🎯 Model Variants Comparison

| Model | Parameters | Size (MB) | Speed (ms) | mAP@0.5 | Recommended Use |
|-------|------------|-----------|------------|---------|-----------------|
| YOLOv11n | 2.6M | 6.0 | ~10 | TBD | Jetson deployment, real-time |
| YOLOv11s | 9.4M | 22.5 | ~15 | TBD | Balanced accuracy/speed |
| YOLOv11m | 20.1M | 49.7 | ~25 | TBD | High accuracy applications |

## 🔍 Evaluation Metrics

### Primary Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: COCO-style mAP across IoU thresholds 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Performance Metrics
- **FPS**: Frames per second (inference speed)
- **Inference Time**: Average time per image (milliseconds)
- **Memory Usage**: Peak GPU memory consumption
- **Model Size**: Disk storage requirements

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python train.py --model yolo11n.pt --batch-size 8

# Use CPU training (slower)
python train.py --model yolo11n.pt --device cpu
```

#### 2. Dataset Not Found
```bash
# Verify dataset structure
ls -la ../data/yolo_dataset/
ls -la ../data/yolo_dataset/train/images/
ls -la ../data/yolo_dataset/val/images/
```

#### 3. Dependencies Missing
```bash
# Install requirements
pip install -r ../requirements.txt

# Check specific package
python -c "import ultralytics; print('OK')"
```

#### 4. Slow Training
- Reduce `--workers` parameter
- Use smaller batch size
- Check GPU utilization: `nvidia-smi`

### Performance Tips

1. **GPU Optimization**:
   - Use `--half` for FP16 training (faster, less memory)
   - Monitor GPU usage: `watch -n 1 nvidia-smi`
   - Close unnecessary applications

2. **Storage Optimization**:
   - Use SSD for dataset storage
   - Ensure sufficient disk space (>10GB recommended)

3. **Memory Management**:
   - Start with batch-size 8, increase if possible
   - Use `--cache ram` for small datasets

## 📊 Expected Training Time

| Configuration | Hardware | Approximate Time |
|---------------|----------|------------------|
| YOLOv11n, 50 epochs | RTX 3080 | 2-3 hours |
| YOLOv11n, 100 epochs | RTX 3080 | 4-6 hours |
| YOLOv11n, 50 epochs | GTX 1660 | 4-6 hours |
| YOLOv11n, 100 epochs | CPU | 24-48 hours |

## 📝 Training Checklist

Before starting training:

- [ ] Verify dataset integrity (11,691 train + 2,924 val images)
- [ ] Check GPU availability and memory
- [ ] Install all required dependencies  
- [ ] Set appropriate batch size for your hardware
- [ ] Choose experiment name and logging preferences
- [ ] Ensure sufficient disk space for results

## 🔗 Integration with Main Project

After training, models can be:

1. **Evaluated**: Use `evaluate.py` for comprehensive analysis
2. **Deployed**: Convert to TensorRT for Jetson deployment
3. **Optimized**: Apply quantization for edge deployment
4. **Benchmarked**: Compare different variants and configurations

## 📚 Additional Resources

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Weights & Biases Setup Guide](https://docs.wandb.ai/quickstart)
- [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)
- [Model Optimization Techniques](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

For questions or issues, please refer to the main project README or contact the AVLab team.