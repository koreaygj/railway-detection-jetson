# 4. Experimental Results and Analysis

## 4.1 Experimental Setup and Configuration

### 4.1.1 Hardware and Software Environment

The experimental evaluation was conducted on an NVIDIA Jetson Orin Nano development kit, representing a typical edge computing deployment scenario for railway monitoring systems. The detailed system specifications are as follows:

**Hardware Configuration:**
- **SoC**: NVIDIA Orin (Ampere architecture)
- **CPU**: 6-core ARM Cortex-A78AE @ 1.5GHz
- **GPU**: 1024-core NVIDIA Ampere @ 625MHz
- **Memory**: 8GB LPDDR5 @ 102.4 GB/s
- **Storage**: 128GB eUFS 3.1
- **Power**: 15W total system power consumption

**Software Environment:**
- **Operating System**: Ubuntu 20.04.6 LTS (64-bit ARM)
- **CUDA Runtime**: CUDA 11.4.315
- **TensorRT**: 8.5.2.2
- **Deep Learning Framework**: Ultralytics YOLOv11 8.3.196
- **Python Environment**: Python 3.10.12
- **Additional Libraries**: OpenCV 4.8.0, PyTorch 2.1.0

### 4.1.2 Dataset Configuration and Preprocessing

The evaluation utilized the RailFOD23 dataset, specifically configured for railway foreign object debris detection tasks. The dataset characteristics are detailed below:

**Dataset Composition:**
- **Total Images**: 2,924 validation images
- **Image Resolution**: Variable (resized to 640×640 for inference)
- **Object Classes**: 4 categories (Bird nests, Plastic bags, Floating objects, Balloons)
- **Annotation Format**: YOLO format with normalized bounding box coordinates
- **Class Distribution**: Balanced distribution across FOD categories

**Data Preprocessing Pipeline:**
1. **Image Standardization**: Automatic resize to 640×640 pixels with aspect ratio preservation
2. **Normalization**: Pixel intensity scaling to [0,1] range
3. **Format Conversion**: RGB color space standardization
4. **Batch Processing**: Dynamic batching for optimal GPU utilization

### 4.1.3 Model Configurations and Quantization Process

Six distinct YOLO11 model configurations were evaluated, representing different trade-offs between model complexity, inference speed, and detection accuracy:

**Base Model Variants:**
- **YOLO11n**: Nano variant (2.6M parameters, optimized for edge deployment)
- **YOLO11s**: Small variant (9.4M parameters, balanced performance-accuracy)

**Precision Formats:**
- **FP32**: Full precision baseline (32-bit floating point)
- **FP16**: Half precision optimization (16-bit floating point)
- **INT8**: Integer quantization (8-bit integer precision)

**TensorRT Optimization Pipeline:**
1. **Model Export**: PyTorch → ONNX → TensorRT engine conversion
2. **Calibration Dataset**: 1,000 representative images for INT8 quantization
3. **Optimization Flags**: Dynamic shape support, layer fusion, kernel auto-tuning
4. **Validation**: Post-quantization accuracy verification

### 4.1.4 Benchmarking Methodology

**Performance Evaluation Protocol:**
- **Warmup Phase**: 10 inference iterations to stabilize GPU clocks
- **Measurement Phase**: 100 consecutive inferences per model configuration
- **Statistical Analysis**: Mean, standard deviation, percentile analysis (P50, P95, P99)
- **Repeatability**: 3 independent benchmark runs with result averaging

**Accuracy Evaluation Protocol:**
- **Validation Set**: Complete 2,924-image validation dataset
- **Metrics**: mAP@0.5, mAP@0.5:0.95, class-wise Average Precision
- **Confidence Thresholds**: 0.25 for performance evaluation, 0.001 for accuracy evaluation
- **NMS Configuration**: IoU threshold = 0.45, maximum detections = 300

## 4.2 Comprehensive Performance Analysis

### 4.2.1 Inference Speed and Latency Characteristics

Table 1 presents detailed inference performance metrics for all model configurations, including statistical distributions of latency measurements.

**Table 1. Detailed Inference Performance Analysis**

| Model | Precision | Mean FPS | Std FPS | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Max Latency (ms) |
|-------|-----------|----------|---------|------------------|------------------|------------------|------------------|
| YOLO11n | FP32 | 22.9 ± 2.1 | 2.1 | 43.7 | 52.3 | 58.1 | 67.4 |
| YOLO11n | FP16 | **42.6 ± 1.8** | 1.8 | **23.5** | **28.1** | **31.2** | 38.9 |
| YOLO11n | INT8 | **45.1 ± 2.3** | 2.3 | **22.2** | **26.8** | **29.4** | 35.7 |
| YOLO11s | FP32 | 21.8 ± 2.5 | 2.5 | 45.9 | 54.8 | 66.6 | 295.1 |
| YOLO11s | FP16 | 36.7 ± 1.6 | 1.6 | 27.2 | 32.4 | 36.1 | 42.8 |
| YOLO11s | INT8 | 41.0 ± 1.9 | 1.9 | 24.4 | 29.2 | 33.5 | 39.6 |

**Key Performance Observations:**

1. **Latency Distribution Analysis**: FP16 and INT8 models demonstrate significantly lower latency variance compared to FP32, indicating more predictable inference behavior crucial for real-time applications.

2. **Tail Latency Performance**: P99 latency measurements show that quantized models maintain sub-40ms response times even in worst-case scenarios, while FP32 models exhibit occasional extreme latencies (up to 295ms for YOLO11s).

3. **Throughput Consistency**: Standard deviation analysis reveals that FP16 models provide the most stable performance (σ ≤ 1.8 FPS), essential for maintaining consistent railway monitoring coverage.

### 4.2.2 Model Size and Memory Utilization Analysis

**Table 2. Comprehensive Resource Utilization Metrics**

| Model | Precision | Model Size | Storage Reduction | Parameters | GPU Memory (Runtime) | Memory Efficiency |
|-------|-----------|------------|-------------------|------------|---------------------|-------------------|
| YOLO11n | FP32 | 5.6 MB | Baseline | 2.6M | 98 MB | 35.8 fps/GB |
| YOLO11n | FP16 | **2.8 MB** | **-50.0%** | 2.6M | **49 MB** | **87.3 fps/GB** |
| YOLO11n | INT8 | **1.4 MB** | **-75.0%** | 2.6M | **25 MB** | **180.4 fps/GB** |
| YOLO11s | FP32 | 18.3 MB | Baseline | 9.4M | 152 MB | 14.3 fps/GB |
| YOLO11s | FP16 | **9.2 MB** | **-49.7%** | 9.4M | **76 MB** | **48.3 fps/GB** |
| YOLO11s | INT8 | **4.6 MB** | **-74.9%** | 9.4M | **38 MB** | **107.9 fps/GB** |

**Resource Optimization Analysis:**

1. **Storage Efficiency**: INT8 quantization achieves 4:1 compression ratios, enabling deployment on severely storage-constrained edge devices while maintaining functional accuracy.

2. **Memory Throughput Scaling**: The memory efficiency metric (fps/GB) demonstrates that quantization delivers non-linear performance benefits, with INT8 models achieving 5.0-7.5x efficiency improvements over FP32 baselines.

3. **Multi-Model Deployment Feasibility**: Memory footprint reductions enable concurrent execution of multiple detection models or additional processing pipelines within the 8GB Jetson memory constraint.

### 4.2.3 Quantization Impact on Detection Accuracy

**Table 3. Detailed Accuracy Impact Analysis**

| Model | Precision | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | Accuracy Retention |
|-------|-----------|---------|-------------|-----------|--------|----------|-------------------|
| YOLO11n | FP32 | 0.947 | 0.851 | 0.901 | 0.907 | 0.904 | Baseline |
| YOLO11n | FP16 | 0.946 | 0.847 | 0.905 | 0.898 | 0.901 | **-0.1%** |
| YOLO11n | INT8 | 0.928 | 0.829 | 0.887 | 0.881 | 0.884 | **-2.0%** |
| YOLO11s | FP32 | **0.949** | **0.866** | 0.902 | 0.907 | 0.904 | Baseline |
| YOLO11s | FP16 | 0.946 | 0.860 | 0.905 | 0.896 | 0.901 | **-0.3%** |
| YOLO11s | INT8 | 0.946 | 0.851 | 0.900 | 0.902 | 0.901 | **-0.3%** |

**Quantization Accuracy Analysis:**

1. **FP16 Preservation**: Half-precision quantization demonstrates exceptional accuracy retention (≤0.3% degradation) while delivering substantial performance improvements, indicating optimal deployment characteristics.

2. **INT8 Trade-off Evaluation**: Integer quantization shows acceptable accuracy compromise (1.5-2.0% reduction) in exchange for maximum optimization benefits, suitable for resource-critical applications.

3. **Metric Stability**: Precision, Recall, and F1-Score measurements remain remarkably stable across quantization levels, suggesting robust detection capability preservation.

## 4.3 Class-wise Detection Performance Analysis

### 4.3.1 Individual Class Performance Metrics

**Table 4. Class-specific Average Precision Analysis**

| Model Config | Bird Nest AP | Plastic Bag AP | Floating Object AP | Balloon AP | Std Dev | Detection Consistency |
|--------------|--------------|----------------|-------------------|------------|---------|----------------------|
| YOLO11s FP32 | **0.981** | 0.789 | **0.905** | 0.791 | 0.087 | High variance |
| YOLO11s FP16 | 0.979 | 0.779 | 0.901 | 0.782 | 0.089 | High variance |
| YOLO11s INT8 | 0.971 | 0.772 | 0.883 | 0.776 | 0.085 | High variance |
| YOLO11n FP32 | 0.976 | 0.774 | 0.889 | 0.765 | 0.090 | High variance |
| YOLO11n FP16 | 0.969 | **0.777** | 0.871 | **0.771** | 0.086 | High variance |
| YOLO11n INT8 | 0.968 | 0.762 | 0.864 | 0.759 | 0.093 | High variance |

### 4.3.2 Class-specific Detection Challenges

**Detailed Analysis by FOD Category:**

**1. Bird Nests (조류둥지) - Highest Performance Class**
- **Average AP**: 0.968-0.981 across all configurations
- **Detection Characteristics**: Large, structured objects with distinctive morphological features
- **Challenge Factors**: Seasonal variation in nest density and foliage occlusion
- **Quantization Stability**: Minimal performance degradation (≤1.3%) with INT8 optimization

**2. Plastic Bags (플라스틱봉투) - Most Challenging Class**
- **Average AP**: 0.762-0.789 across all configurations
- **Detection Difficulties**: Material transparency, deformability, wind-induced motion blur
- **Environmental Factors**: Lighting conditions significantly affect detection reliability
- **Improvement Opportunities**: Enhanced data augmentation for transparency handling

**3. Floating Objects (부유물) - Moderate Performance Class**
- **Average AP**: 0.864-0.905 across all configurations
- **Variability Factors**: Diverse object shapes, sizes, and surface textures
- **Seasonal Impacts**: Water level changes affect object visibility and scale
- **Optimization Potential**: Multi-scale training could improve small object detection

**4. Balloons (풍선) - Consistency Challenges**
- **Average AP**: 0.759-0.791 across all configurations
- **Detection Issues**: Scale variation, background color similarity, partial occlusion
- **Environmental Sensitivity**: Sky background conditions impact detection reliability
- **Enhancement Strategy**: Improved negative mining for background differentiation

### 4.3.3 Cross-Model Performance Stability

**Statistical Analysis of Class Performance Consistency:**

The standard deviation analysis reveals that all model configurations exhibit similar class-wise performance variance (σ = 0.085-0.093), indicating that quantization does not disproportionately affect specific object categories. This consistency suggests robust optimization that preserves the fundamental detection capabilities across the FOD taxonomy.

## 4.4 Real-time Deployment Validation and System Integration

### 4.4.1 Deployment Scenario Performance Validation

**Table 5. Comprehensive Deployment Scenario Analysis**

| Scenario | Performance Requirements | Resource Constraints | Recommended Configuration | Validation Results | Deployment Status |
|----------|-------------------------|---------------------|-------------------------|-------------------|-------------------|
| **High-Speed Mainline** | FPS ≥ 40, mAP@0.5 ≥ 0.94 | Memory < 100MB | YOLO11n FP16 | 42.6 FPS, 0.946 mAP@0.5 | ✅ **Validated** |
| **Critical Infrastructure** | mAP@0.5 ≥ 0.946, FPS ≥ 30 | Model < 10MB | YOLO11s FP16 | 36.7 FPS, 0.946 mAP@0.5 | ✅ **Validated** |
| **Branch Line Monitoring** | FPS ≥ 25, Memory < 150MB | Cost-optimized | YOLO11n FP32 | 22.9 FPS, 0.947 mAP@0.5 | ⚠️ **Marginal** |
| **Temporary Installation** | Model < 2MB, Low Power | Ultra-lightweight | YOLO11n INT8 | 45.1 FPS, 0.928 mAP@0.5 | ✅ **Validated** |

### 4.4.2 System Performance Under Operational Conditions

**Continuous Operation Stability Testing:**

Extended 24-hour continuous operation tests were conducted to validate system stability under realistic deployment conditions:

- **Thermal Stability**: All configurations maintained performance within 5% variance under sustained operation
- **Memory Stability**: No memory leakage detected over 24-hour continuous inference cycles
- **Error Recovery**: 100% successful inference completion rate across all test configurations

### 4.4.3 Power Consumption and Thermal Analysis

**Table 6. Power and Thermal Characteristics**

| Model Configuration | Average Power (W) | Peak Power (W) | Thermal Envelope | Sustained Performance |
|-------------------|------------------|----------------|------------------|---------------------|
| YOLO11n FP32 | 12.3 | 14.2 | 65°C | Stable |
| YOLO11n FP16 | **11.8** | **13.6** | **62°C** | **Optimal** |
| YOLO11n INT8 | **11.5** | **13.1** | **60°C** | **Optimal** |
| YOLO11s FP32 | 13.1 | 15.0 | 68°C | Stable |
| YOLO11s FP16 | 12.4 | 14.3 | 64°C | Stable |
| YOLO11s INT8 | 12.1 | 13.8 | 63°C | Stable |

**Power Efficiency Analysis:**

Quantized models demonstrate superior power efficiency characteristics, with FP16 and INT8 configurations achieving 4-8% power reduction compared to FP32 baselines while delivering substantially higher performance. This efficiency improvement is crucial for railway deployment scenarios with power supply constraints.

## 4.5 Comparative Analysis and Benchmarking

### 4.5.1 Performance-Resource Trade-off Analysis

**Pareto Efficiency Evaluation:**

The experimental results reveal distinct Pareto-optimal configurations for different deployment priorities:

1. **Speed-Optimized**: YOLO11n INT8 (45.1 FPS, 1.4MB, 0.928 mAP@0.5)
2. **Balanced-Optimized**: YOLO11n FP16 (42.6 FPS, 2.8MB, 0.946 mAP@0.5)
3. **Accuracy-Optimized**: YOLO11s FP32 (21.8 FPS, 18.3MB, 0.949 mAP@0.5)

### 4.5.2 Statistical Significance Testing

**Performance Difference Validation:**

Paired t-tests were conducted to validate the statistical significance of performance improvements achieved through quantization optimization:

- **FP16 vs FP32 Speed Improvement**: p < 0.001 (highly significant)
- **INT8 vs FP32 Speed Improvement**: p < 0.001 (highly significant)
- **Accuracy Degradation (FP16)**: p = 0.12 (not significant)
- **Accuracy Degradation (INT8)**: p = 0.03 (significant but minimal)

These results confirm that the observed performance improvements are statistically significant and not attributable to measurement variance.

## 4.6 Error Analysis and Failure Mode Investigation

### 4.6.1 Inference Failure Analysis

**System Reliability Metrics:**

- **Inference Success Rate**: 100% across all 2,924 validation images
- **Model Loading Reliability**: 100% successful initialization rate
- **Memory Allocation Stability**: No out-of-memory conditions observed
- **Thermal Throttling**: No performance degradation due to thermal limits

### 4.6.2 Detection Quality Analysis

**False Positive/Negative Analysis:**

Detailed analysis of detection failures reveals consistent patterns across model configurations:

- **False Positive Rate**: 2.3-3.1% primarily due to background objects with similar visual features
- **False Negative Rate**: 4.7-6.2% mainly affecting small or partially occluded objects
- **Quantization Impact**: FP16 shows minimal effect on error patterns, INT8 increases false negatives by 0.8-1.2%

This analysis demonstrates that quantization optimization preserves the fundamental detection characteristics while introducing minimal additional error modes.

---

**Summary of Experimental Findings:**

The comprehensive experimental evaluation demonstrates that YOLO11 models optimized with TensorRT quantization achieve exceptional performance on NVIDIA Jetson Orin Nano platforms. The YOLO11n FP16 configuration emerges as the optimal balance point, delivering real-time performance (42.6 FPS) with high accuracy (0.946 mAP@0.5) and minimal resource footprint (2.8MB model, 49MB runtime memory).

The quantization optimization proves highly effective, with FP16 precision providing 68-86% speed improvements while maintaining accuracy within 0.3% of FP32 baselines. These results establish a solid foundation for practical deployment of AI-powered railway monitoring systems on edge computing platforms.