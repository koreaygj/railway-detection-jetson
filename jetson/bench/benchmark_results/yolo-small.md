# Railway Detection Model Benchmark Report

## Model Information
- **Model**: best
- **Type**: .pt
- **Dataset**: data.yaml
- **Classes**: niaocao, suliaodai, piaofuwu, qiqiu
- **Device**: 0
- **Timestamp**: 20250910_212435

---

## Validation Results


| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.948** (94.8%) |
| mAP@0.5:0.95 | **0.865** (86.5%) |
| Precision | 0.902 (90.2%) |
| Recall | 0.905 (90.5%) |
| Validation Time | 109.51s |


## Inference Speed Performance

| Metric | Value |
|--------|-------|
| **Average FPS** | **16.31** |
| Min FPS | 4.70 |
| Max FPS | 27.26 |
| Average Latency | 61.32ms ± 31.86ms |
| Min Latency | 36.68ms |
| Max Latency | 212.55ms |
| Successful Predictions | 30 |
| Average Detections/Image | 3.70 |


## Resolution Benchmark

| Resolution | FPS | Latency (ms) |
|------------|-----|--------------|
| 320x320 | 35.13 | 28.46 |
| 416x416 | 33.33 | 30.00 |
| 640x640 | 25.92 | 38.58 |

---

## Summary

This benchmark report provides comprehensive performance metrics for the railway detection model.

### Recommendations:
- For real-time applications: Use 640x640 resolution for best speed/accuracy balance
- Deploy with TensorRT on Jetson for optimal edge performance
