# Training Parameters Reference Guide

## 📋 Quick Reference

This document provides a comprehensive reference for all training parameters used in the railway object detection project, including detailed explanations and rationale for each setting.

---

## 🎯 Core Training Parameters

### **Basic Configuration**
```python
# Model and Data
model: 'yolo11n.pt'              # Pre-trained model weights
data: 'data.yaml'                # Dataset configuration file
epochs: 100                      # Total training epochs
batch: 16                        # Batch size
imgsz: 640                       # Input image size (640x640)
device: '0'                      # GPU device (0, 1, 2, etc. or 'cpu')
workers: 8                       # Number of dataloader workers
```

### **Optimization Parameters**
```python
# Learning Rate and Optimization
lr0: 0.01                        # Initial learning rate
lrf: 0.01                        # Final learning rate (lr0 * lrf)
momentum: 0.937                  # SGD momentum factor
weight_decay: 0.0005             # L2 regularization coefficient
warmup_epochs: 3.0               # Number of warmup epochs
warmup_momentum: 0.8             # Warmup momentum
warmup_bias_lr: 0.1              # Warmup bias learning rate
```

### **Training Control**
```python
# Training Behavior
seed: 42                         # Random seed for reproducibility
deterministic: True              # Use deterministic algorithms
save_period: 10                  # Save checkpoint every N epochs
patience: 100                    # Early stopping patience
verbose: True                    # Verbose output
amp: True                        # Automatic Mixed Precision
```

---

## 🔄 Data Augmentation Parameters

### **Spatial Augmentations**
```python
# Geometric Transformations
degrees: 0.0                     # Image rotation (+/- deg)
translate: 0.1                   # Image translation (+/- fraction)
scale: 0.5                       # Image scale (+/- gain)
shear: 0.0                       # Image shear (+/- deg)
perspective: 0.0                 # Image perspective (+/- fraction)
flipud: 0.0                      # Vertical flip probability
fliplr: 0.5                      # Horizontal flip probability
```

**Rationale for Spatial Settings:**
- **degrees: 0.0** - No rotation to preserve railway line orientation
- **translate: 0.1** - Moderate translation to simulate camera position variations
- **scale: 0.5** - Scale variation to handle different object sizes
- **shear: 0.0** - No shearing to maintain object shape integrity
- **perspective: 0.0** - No perspective to avoid unrealistic distortions
- **flipud: 0.0** - No vertical flip to maintain gravity orientation
- **fliplr: 0.5** - Horizontal flip for left-right symmetry

### **Color Augmentations**
```python
# Color Space Modifications
hsv_h: 0.015                     # Hue augmentation (fraction)
hsv_s: 0.7                       # Saturation augmentation (fraction)
hsv_v: 0.4                       # Value augmentation (fraction)
```

**Rationale for Color Settings:**
- **hsv_h: 0.015** - Minimal hue variation (±1.5%) to preserve object colors
- **hsv_s: 0.7** - Strong saturation variation (±70%) for lighting changes
- **hsv_v: 0.4** - Moderate brightness variation (±40%) for day/weather conditions

### **Advanced Augmentations**
```python
# Complex Augmentation Techniques
mosaic: 1.0                      # Mosaic augmentation probability
mixup: 0.0                       # MixUp augmentation probability
copy_paste: 0.0                  # Copy-paste augmentation probability
```

**Rationale for Advanced Settings:**
- **mosaic: 1.0** - Full mosaic to improve multi-object detection
- **mixup: 0.0** - Disabled to maintain object boundary clarity
- **copy_paste: 0.0** - Disabled to preserve environmental context

---

## 🎛️ Loss Function Parameters

### **Loss Component Weights**
```python
# Multi-task Loss Balancing
box: 7.5                         # Box loss weight
cls: 0.5                         # Classification loss weight
dfl: 1.5                         # Distribution Focal Loss weight
```

### **Loss Function Details**
```python
# Box Regression Loss
box_loss_type: 'CIoU'            # Complete IoU loss
iou_threshold: 0.7               # IoU threshold for positive matches

# Classification Loss
cls_loss_type: 'BCE'             # Binary Cross Entropy
label_smoothing: 0.0             # Label smoothing factor

# Distribution Focal Loss
dfl_loss_weight: 1.5             # Weight for box quality prediction
```

---

## 📊 Model Architecture Parameters

### **YOLOv11n Specifications**
```python
# Model Architecture
model_type: 'YOLOv11n'
parameters: 2_582_932            # Total trainable parameters
gflops: 6.3                      # Computational complexity
layers: 100                      # Total network layers

# Input/Output Configuration
input_shape: [3, 640, 640]       # Input tensor shape [C, H, W]
output_classes: 4                # Number of object classes
max_detections: 300              # Maximum detections per image
```

### **Network Components**
```python
# Backbone
backbone: 'C2f + SPPF'          # Feature extraction layers
neck: 'PANet'                    # Feature fusion network
head: 'Detect'                   # Detection head

# Anchor Configuration
anchor_free: True                # Anchor-free detection
stride: [8, 16, 32]             # Feature map strides
```

---

## 🔧 Hardware-Specific Parameters

### **GPU Optimization**
```python
# CUDA Settings
cuda_available: True             # CUDA acceleration
gpu_memory: '40GB'              # Available GPU memory
mixed_precision: True           # FP16 training
cuda_benchmarks: True           # Optimize CUDA kernels
```

### **Memory Management**
```python
# Memory Optimization
pin_memory: True                 # Pin memory for faster transfer
persistent_workers: True        # Keep workers alive between epochs
prefetch_factor: 2              # Number of batches to prefetch
drop_last: False                # Keep incomplete final batch
```

### **Parallel Processing**
```python
# Multi-processing
num_workers: 8                  # DataLoader worker processes
world_size: 1                   # Number of distributed processes
rank: 0                         # Process rank for distributed training
```

---

## 📈 Learning Rate Schedule Parameters

### **Cosine Annealing Configuration**
```python
# Learning Rate Scheduler
scheduler: 'cosine'              # Cosine annealing scheduler
T_max: 100                      # Maximum number of epochs
eta_min: 0.0001                 # Minimum learning rate

# Warmup Configuration
warmup_method: 'linear'         # Linear warmup
warmup_epochs: 3                # Warmup duration
warmup_decay: 0.1               # Warmup decay factor
```

### **Learning Rate Progression**
```python
# Epoch-wise LR Schedule
# Epochs 1-3:    Linear warmup (0.001 → 0.01)
# Epochs 4-100:  Cosine decay (0.01 → 0.0001)
```

---

## 🎯 Detection-Specific Parameters

### **Object Detection Configuration**
```python
# Detection Thresholds
conf_threshold: 0.001           # Confidence threshold for training
iou_threshold: 0.7              # IoU threshold for NMS
max_det: 300                    # Maximum detections per image
multi_label: False              # Single label per detection
agnostic_nms: False             # Class-agnostic NMS
```

### **Anchor and Grid Settings**
```python
# Anchor Configuration (YOLO-specific)
anchor_t: 4.0                   # Anchor multiple threshold
anchors: None                   # Auto-calculated anchors
anchor_free: True               # Anchor-free detection head
```

---

## 🔍 Validation Parameters

### **Evaluation Configuration**
```python
# Validation Settings
val_split: 0.2                  # Validation split ratio
val_iou: 0.6                    # IoU threshold for validation
val_conf: 0.001                 # Confidence threshold for validation
save_json: True                 # Save COCO-format results
plots: True                     # Generate validation plots
```

### **Metric Calculation**
```python
# Performance Metrics
compute_map: True               # Calculate mAP metrics
map_range: [0.5, 0.95]         # IoU range for mAP calculation
map_step: 0.05                  # IoU step size
per_class_metrics: True         # Per-class performance metrics
```

---

## 📁 Output and Logging Parameters

### **Model Saving Configuration**
```python
# Model Checkpoints
save_best: True                 # Save best model
save_last: True                 # Save last model
save_period: 10                 # Save every N epochs
save_dir: './runs/train'        # Output directory
name: 'yolo11n_railway'        # Experiment name
```

### **Logging and Visualization**
```python
# Progress Tracking
log_interval: 100               # Log every N batches
tensorboard: True               # TensorBoard logging
wandb: False                    # Weights & Biases logging
csv_logger: True                # CSV results logging
plot_losses: True               # Plot training losses
plot_metrics: True              # Plot validation metrics
```

---

## 🧪 Experimental Parameters

### **Ablation Study Settings**
```python
# Experiment Variations
freeze_backbone: [0, 10]        # Freeze backbone for N epochs
augment_strength: 'medium'      # Augmentation intensity
optimizer: 'SGD'                # Optimizer choice
loss_function: 'composite'      # Loss function type
```

### **Hyperparameter Search Space**
```python
# Tuning Ranges
lr_range: [0.001, 0.1]         # Learning rate search space
batch_range: [8, 32]           # Batch size options
wd_range: [1e-5, 1e-3]         # Weight decay range
momentum_range: [0.9, 0.95]    # Momentum range
```

---

## 🚀 Production Deployment Parameters

### **Inference Optimization**
```python
# Deployment Configuration
export_format: ['onnx', 'engine'] # Export formats
quantization: ['fp16', 'int8']   # Quantization options
batch_inference: 1               # Inference batch size
input_size: 640                  # Deployment input size
```

### **Edge Device Settings**
```python
# Jetson Orin Nano Configuration
target_fps: 25                  # Minimum FPS requirement
memory_limit: '8GB'             # Device memory constraint
power_mode: 'MAXN'              # Performance mode
tensorrt_precision: 'fp16'      # TensorRT optimization level
```

---

## 📊 Parameter Selection Rationale

### **Performance Optimizations**
1. **Batch Size (16)**: Optimal GPU memory utilization
2. **Learning Rate (0.01)**: Balanced convergence speed
3. **Image Size (640)**: Standard YOLO resolution
4. **Workers (8)**: CPU core utilization

### **Accuracy Optimizations**
1. **Conservative Augmentation**: Preserve railway context
2. **Transfer Learning**: Leverage COCO pre-training
3. **Loss Balancing**: Emphasize box accuracy
4. **Validation Strategy**: Robust performance evaluation

### **Deployment Considerations**
1. **Model Size**: YOLOv11n for edge compatibility
2. **Inference Speed**: Real-time processing capability
3. **Memory Efficiency**: Edge device constraints
4. **Quantization Ready**: FP16/INT8 deployment

---

## 🎯 Parameter Tuning Guidelines

### **When to Adjust Parameters**

| Scenario | Parameters to Modify | Recommended Changes |
|----------|---------------------|-------------------|
| **Low Accuracy** | lr0, weight_decay, epochs | Decrease lr0, increase epochs |
| **Overfitting** | weight_decay, augmentation | Increase weight_decay, stronger augmentation |
| **Slow Convergence** | lr0, batch_size | Increase lr0, larger batch_size |
| **Memory Issues** | batch_size, workers | Decrease batch_size, reduce workers |
| **Poor Validation** | patience, val_conf | Increase patience, adjust thresholds |

### **Critical Parameters (Do Not Modify)**
- **seed**: Required for reproducibility
- **deterministic**: Ensures consistent results
- **anchor settings**: Optimized for YOLO architecture
- **loss weights**: Balanced for multi-task learning

---

**📝 This parameter reference ensures reproducible and optimal training results for railway safety object detection models.**