# Railway Detection Model Benchmark Report

## Model Information
- **Model**: best
- **Type**: .pt
- **Dataset**: data.yaml
- **Classes**: niaocao, suliaodai, piaofuwu, qiqiu
- **Device**: 0
- **Timestamp**: 20250910_212306

---

## Validation Results


| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.946** (94.6%) |
| mAP@0.5:0.95 | **0.850** (85.0%) |
| Precision | 0.915 (91.5%) |
| Recall | 0.889 (88.9%) |
| Validation Time | 79.93s |


## Inference Speed Performance

| Metric | Value |
|--------|-------|
| **Average FPS** | **21.14** |
| Min FPS | 5.16 |
| Max FPS | 26.48 |
| Average Latency | 47.31ms ± 27.73ms |
| Min Latency | 37.77ms |
| Max Latency | 193.76ms |
| Successful Predictions | 30 |
| Average Detections/Image | 3.73 |


## Resolution Benchmark

| Resolution | FPS | Latency (ms) |
|------------|-----|--------------|
| 320x320 | 34.65 | 28.86 |
| 416x416 | 33.75 | 29.63 |
| 640x640 | 31.75 | 31.50 |

---

## Summary

This benchmark report provides comprehensive performance metrics for the railway detection model.

### Recommendations:
- For real-time applications: Use 640x640 resolution for best speed/accuracy balance
- Deploy with TensorRT on Jetson for optimal edge performance
