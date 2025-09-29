# 🎨 YOLO11 Railway Detection Visualization Guide

벤치마킹과 함께 실제 객체 탐지 결과를 시각화하여 모델의 성능을 직관적으로 확인할 수 있습니다.

## 📋 목차

- [기본 사용법](#기본-사용법)
- [시각화 옵션](#시각화-옵션)
- [출력 결과](#출력-결과)
- [예시 명령어](#예시-명령어)

---

## 🚀 기본 사용법

### 1. 독립 시각화 스크립트

```bash
cd bench/

# 기본 시각화 (10개 이미지)
python3 visualize_detections.py ../data.yaml ../convert/model/yolo11s_fp16.engine

# 더 많은 이미지로 시각화
python3 visualize_detections.py ../data.yaml yolo11s.pt --max-images 20

# 낮은 신뢰도로 더 많은 객체 표시
python3 visualize_detections.py ../data.yaml yolo11s.pt --conf 0.15 --max-images 15
```

### 2. 벤치마크와 함께 시각화

```bash
# 성능 벤치마크 + 시각화 동시 실행
python3 optimized_benchmark.py ../data.yaml ../convert/model/yolo11s_fp16.engine \
    --visualize --viz-images 15 --max-images 100

# 정확도 테스트와 함께
python3 optimized_benchmark.py ../data.yaml yolo11s.pt \
    --visualize --viz-images 10 --conf 0.2 --device cpu
```

---

## ⚙️ 시각화 옵션

### 독립 스크립트 옵션 (`visualize_detections.py`)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--max-images` | 10 | 처리할 최대 이미지 수 |
| `--conf` | 0.25 | 객체 탐지 신뢰도 임계값 |
| `--iou` | 0.45 | NMS IoU 임계값 |
| `--imgsz` | 640 | 입력 이미지 크기 |
| `--device` | '0' | 사용할 디바이스 (0=GPU, cpu=CPU) |
| `--output-dir` | './visualization_results' | 결과 저장 디렉토리 |
| `--name` | auto | 실험 이름 (자동 생성) |

### 통합 벤치마크 옵션 (`optimized_benchmark.py`)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--visualize` | False | 시각화 활성화 |
| `--viz-images` | 10 | 시각화할 이미지 수 |

---

## 📁 출력 결과

시각화 실행 후 다음 구조로 결과가 저장됩니다:

```
visualization_results/
└── yolo11s_fp16_visualization_20250916_143022/
    ├── original/                    # 원본 이미지
    │   ├── image001_original.jpg
    │   ├── image002_original.jpg
    │   └── ...
    ├── detected/                    # 탐지 결과 이미지
    │   ├── image001_detected.jpg
    │   ├── image002_detected.jpg
    │   └── ...
    ├── comparison/                  # 원본-탐지 비교 이미지
    │   ├── image001_comparison.jpg
    │   ├── image002_comparison.jpg
    │   └── ...
    └── detection_stats.yaml        # 탐지 통계
```

### 🎯 탐지 통계 예시 (`detection_stats.yaml`)

```yaml
total_images: 10
total_detections: 47
class_detections:
  bird_nest: 12
  plastic_bag: 18
  floating_object: 8
  balloon: 9
avg_confidence: 0.756
avg_processing_time: 0.0273
fps: 36.7
processing_times: [0.0251, 0.0289, 0.0267, ...]
```

---

## 💡 예시 명령어

### 1. **기본 시각화**
```bash
# PyTorch 모델로 10개 이미지 시각화
python3 visualize_detections.py ../data.yaml yolo11s.pt
```

**결과**: `visualization_results/yolo11s_visualization_20250916_143022/`

### 2. **대량 시각화 (저성능 탐지 포함)**
```bash
# 30개 이미지, 낮은 신뢰도로 더 많은 객체 표시
python3 visualize_detections.py ../data.yaml yolo11s.pt \
    --max-images 30 --conf 0.15 --name "low_confidence_test"
```

**결과**: `visualization_results/low_confidence_test/`

### 3. **TensorRT 엔진 최적화 시각화**
```bash
# FP16 TensorRT 엔진으로 고속 시각화
python3 visualize_detections.py ../data.yaml ../convert/model/yolo11s_fp16.engine \
    --max-images 20 --output-dir ./tensorrt_viz
```

**결과**: `tensorrt_viz/yolo11s_fp16_visualization_20250916_143022/`

### 4. **벤치마크 + 시각화 통합**
```bash
# 성능 벤치마크와 시각화를 한번에
python3 optimized_benchmark.py ../data.yaml yolo11s.pt \
    --max-images 100 --visualize --viz-images 15 \
    --output-dir ./integrated_results --name "full_test"
```

**결과**:
```
integrated_results/
├── full_test_20250916_143022.json        # 벤치마크 결과
└── yolo11s_visualization/                 # 시각화 결과
    ├── original/
    ├── detected/
    ├── comparison/
    └── detection_stats.yaml
```

### 5. **CPU에서 시각화**
```bash
# CPU 환경에서 시각화 (Jetson 등에서 GPU 메모리 부족시)
python3 visualize_detections.py ../data.yaml yolo11s.pt \
    --device cpu --max-images 5
```

### 6. **다양한 신뢰도 비교**
```bash
# 높은 신뢰도 (확실한 탐지만)
python3 visualize_detections.py ../data.yaml yolo11s.pt \
    --conf 0.7 --name "high_confidence" --max-images 15

# 낮은 신뢰도 (모든 가능한 탐지)
python3 visualize_detections.py ../data.yaml yolo11s.pt \
    --conf 0.1 --name "all_detections" --max-images 15
```

---

## 🎨 시각화 결과 해석

### 1. **Bounding Box 색상**
- 🔵 **파란색**: bird_nest (조류 둥지)
- 🟢 **초록색**: plastic_bag (플라스틱 봉투)
- 🔴 **빨간색**: floating_object (부유물)
- 🟡 **노란색**: balloon (풍선)

### 2. **라벨 정보**
- **클래스명**: 탐지된 객체 종류
- **신뢰도**: 0.00~1.00 범위 (높을수록 확실)
- **예시**: `plastic_bag: 0.89`

### 3. **비교 이미지**
- **왼쪽**: 원본 이미지
- **오른쪽**: 탐지 결과 표시

---

## 🔍 활용 시나리오

### 1. **모델 성능 검증**
```bash
# 학습된 모델의 실제 탐지 능력 확인
python3 visualize_detections.py ../data.yaml ../result/trained_models/yolo11s_railway/weights/best.pt \
    --max-images 20 --name "training_validation"
```

### 2. **양자화 효과 비교**
```bash
# FP32 vs FP16 vs INT8 탐지 품질 비교
python3 visualize_detections.py ../data.yaml yolo11s.pt --name "fp32" --max-images 10
python3 visualize_detections.py ../data.yaml yolo11s_fp16.engine --name "fp16" --max-images 10
python3 visualize_detections.py ../data.yaml yolo11s_int8.engine --name "int8" --max-images 10
```

### 3. **임계값 튜닝**
```bash
# 다양한 신뢰도 임계값으로 최적점 찾기
for conf in 0.1 0.25 0.5 0.7; do
    python3 visualize_detections.py ../data.yaml yolo11s.pt \
        --conf $conf --name "conf_${conf}" --max-images 10
done
```

### 4. **실시간 배포 전 검증**
```bash
# 실제 배포 환경과 동일한 설정으로 테스트
python3 visualize_detections.py ../data.yaml yolo11s_fp16.engine \
    --conf 0.25 --iou 0.45 --imgsz 640 --device 0 \
    --max-images 50 --name "deployment_test"
```

---

## ⚡ 성능 팁

1. **메모리 절약**: `--max-images` 값을 줄여서 메모리 사용량 제한
2. **속도 향상**: TensorRT 엔진 사용시 더 빠른 시각화
3. **품질 확인**: `--conf` 값을 낮춰서 모든 탐지 결과 확인
4. **배치 효율**: 여러 모델 비교시 동일한 이미지셋 사용

이 시각화 도구를 통해 YOLO11 모델의 실제 탐지 성능을 직관적으로 평가하고 최적화 효과를 확인할 수 있습니다! 🎯