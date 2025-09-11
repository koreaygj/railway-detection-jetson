#!/usr/bin/env python3
"""
Benchmark trained models on dataset (NumPy compatibility fixed)
Test accuracy, speed, and performance metrics
Author: Gyeongjin Yang
Lab: AVLab, Chungbuk National University
"""

import os
import time
import argparse
import yaml
import numpy as np
# import pandas as pd  # Removed due to numpy compatibility issues
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from tqdm import tqdm
import json

class DatasetBenchmark:
    def __init__(self, model_path, data_yaml_path, device='0'):
        """
        Initialize dataset benchmark
        
        Args:
            model_path: Path to trained model (.pt or .engine)
            data_yaml_path: Path to dataset YAML file
            device: Device to run inference on
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.model_type = Path(model_path).suffix
        
        # Load dataset config
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        self.num_classes = len(self.class_names)
        
        print(f"🚂 Dataset Benchmark Initialized")
        print(f"   Model: {Path(model_path).name}")
        print(f"   Model Type: {self.model_type}")
        print(f"   Dataset: {Path(data_yaml_path).name}")
        print(f"   Classes: {self.class_names}")
        print(f"   Device: {device}")
    
    def benchmark_validation_set(self):
        """Run validation on the validation set using YOLO's built-in validation"""
        print("\n🔍 Running validation on validation set...")
        
        start_time = time.time()
        
        # Run validation
        results = self.model.val(
            data=self.data_yaml_path,
            device=self.device,
            verbose=False,  # Reduce output
            save_json=True,
            plots=True
        )
        
        val_time = time.time() - start_time
        
        # Extract metrics
        metrics = {
            'mAP@0.5': float(results.box.map50),
            'mAP@0.5:0.95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'validation_time': val_time,
        }
        
        # Per-class metrics (if available)
        try:
            if hasattr(results.box, 'maps') and results.box.maps is not None:
                class_metrics = {}
                for i, class_name in enumerate(self.class_names):
                    if i < len(results.box.maps):
                        class_metrics[class_name] = {
                            'mAP@0.5': float(results.box.maps[i]),
                        }
                metrics['per_class'] = class_metrics
        except Exception as e:
            print(f"   ⚠️  Could not extract per-class metrics: {e}")
        
        return metrics
    
    def benchmark_inference_speed(self, test_images_dir=None, num_samples=50):
        """Benchmark inference speed on sample images"""
        print(f"\n⚡ Benchmarking inference speed ({num_samples} samples)...")
        
        # Get test images
        if test_images_dir is None:
            # Use validation images
            dataset_path = Path(self.data_config['path'])
            if not dataset_path.is_absolute():
                # Make relative to yaml file location
                yaml_dir = Path(self.data_yaml_path).parent
                dataset_path = yaml_dir / dataset_path
            
            val_path = self.data_config.get('val', 'val/images')
            
            # Handle different path formats
            if val_path.endswith('/images'):
                val_images_path = dataset_path / val_path
            elif val_path.endswith('/labels'):
                val_images_path = dataset_path / val_path.replace('/labels', '/images')
            else:
                # Assume it's a directory containing images
                val_images_path = dataset_path / val_path / 'images'
            
            # Try alternative paths if not found
            test_images_dir = val_images_path
            if not test_images_dir.exists():
                # Try without dataset_path prefix
                alternative_path = Path(self.data_yaml_path).parent / val_path
                if alternative_path.exists():
                    test_images_dir = alternative_path
                else:
                    # Try direct path from yaml location
                    test_images_dir = Path(self.data_yaml_path).parent / 'val' / 'images'
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(Path(test_images_dir).glob(ext)))
            image_files.extend(list(Path(test_images_dir).glob(ext.upper())))
        
        if not image_files:
            print(f"❌ No images found in {test_images_dir}")
            return None
        
        # Sample random images
        np.random.seed(42)
        sample_size = min(num_samples, len(image_files))
        sample_files = np.random.choice(image_files, sample_size, replace=False)
        
        print(f"   📁 Test images: {len(image_files)} total, {len(sample_files)} sampled")
        print(f"   📁 Image directory: {test_images_dir}")
        
        # Warm up
        print("   🔥 Warming up...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(5):
            try:
                self.model.predict(dummy_img, device=self.device, verbose=False)
            except:
                pass
        
        # Benchmark
        times = []
        successful_predictions = 0
        total_detections = 0
        
        print("   ⏱️  Running benchmark...")
        for img_file in tqdm(sample_files, desc="Processing"):
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                start_time = time.time()
                
                # Run inference
                results = self.model.predict(
                    img, 
                    device=self.device, 
                    verbose=False,
                    save=False
                )
                
                total_time = time.time() - start_time
                times.append(total_time)
                
                # Count detections
                if len(results) > 0 and results[0].boxes is not None:
                    total_detections += len(results[0].boxes)
                
                successful_predictions += 1
                
            except Exception as e:
                continue
        
        if not times:
            print("❌ No successful predictions!")
            return None
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        return {
            'total_mean': avg_time * 1000,  # ms
            'total_std': std_time * 1000,   # ms
            'total_min': min_time * 1000,   # ms
            'total_max': max_time * 1000,   # ms
            'fps_mean': 1.0 / avg_time,
            'fps_min': 1.0 / max_time,
            'fps_max': 1.0 / min_time,
            'successful_predictions': successful_predictions,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / successful_predictions if successful_predictions > 0 else 0
        }
    
    def benchmark_different_resolutions(self, resolutions=[320, 416, 640]):
        """Benchmark at different input resolutions"""
        print(f"\n📐 Benchmarking different resolutions: {resolutions}")
        
        resolution_results = {}
        
        for res in resolutions:
            print(f"   Testing resolution: {res}x{res}")
            
            try:
                # Test inference speed with dummy image
                dummy_img = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)
                
                # Warm up
                for _ in range(3):
                    self.model.predict(dummy_img, device=self.device, imgsz=res, verbose=False)
                
                # Time inference
                times = []
                for _ in range(10):  # Reduced iterations
                    start = time.time()
                    self.model.predict(dummy_img, device=self.device, imgsz=res, verbose=False)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                
                resolution_results[f"{res}x{res}"] = {
                    'inference_time_ms': avg_time * 1000,
                    'fps': 1.0 / avg_time
                }
                
                print(f"      ✅ {res}x{res}: {1.0/avg_time:.2f} FPS ({avg_time*1000:.2f}ms)")
                
            except Exception as e:
                print(f"      ❌ Error at resolution {res}: {str(e)}")
                resolution_results[f"{res}x{res}"] = {'error': str(e)}
        
        return resolution_results
    
    def generate_benchmark_report(self, output_dir='./benchmark_results'):
        """Generate comprehensive benchmark report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = Path(self.model_path).stem
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        print(f"\n📋 Generating comprehensive benchmark report...")
        
        # Run all benchmarks
        benchmark_results = {
            'model_info': {
                'model_path': str(self.model_path),
                'model_name': model_name,
                'model_type': self.model_type,
                'dataset': str(self.data_yaml_path),
                'classes': self.class_names,
                'num_classes': self.num_classes,
                'device': self.device,
                'timestamp': timestamp
            }
        }
        
        # 1. Validation metrics
        try:
            benchmark_results['validation'] = self.benchmark_validation_set()
        except Exception as e:
            print(f"   ❌ Validation benchmark failed: {e}")
            benchmark_results['validation'] = {'error': str(e)}
        
        # 2. Inference speed
        try:
            benchmark_results['inference_speed'] = self.benchmark_inference_speed(num_samples=30)
        except Exception as e:
            print(f"   ❌ Speed benchmark failed: {e}")
            benchmark_results['inference_speed'] = {'error': str(e)}
        
        # 3. Resolution benchmark
        try:
            benchmark_results['resolution_benchmark'] = self.benchmark_different_resolutions([320, 416, 640])
        except Exception as e:
            print(f"   ❌ Resolution benchmark failed: {e}")
            benchmark_results['resolution_benchmark'] = {'error': str(e)}
        
        # Save results
        report_file = output_dir / f'{model_name}_benchmark_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # Generate markdown report
        self.generate_markdown_report(benchmark_results, output_dir / f'{model_name}_benchmark_{timestamp}.md')
        
        print(f"✅ Benchmark complete!")
        print(f"   📄 JSON report: {report_file}")
        print(f"   📄 Markdown report: {output_dir / f'{model_name}_benchmark_{timestamp}.md'}")
        
        return benchmark_results
    
    def generate_markdown_report(self, results, output_file):
        """Generate markdown benchmark report"""
        model_name = results['model_info']['model_name']
        timestamp = results['model_info']['timestamp']
        
        md_content = f"""# Railway Detection Model Benchmark Report

## Model Information
- **Model**: {model_name}
- **Type**: {results['model_info']['model_type']}
- **Dataset**: {Path(results['model_info']['dataset']).name}
- **Classes**: {', '.join(results['model_info']['classes'])}
- **Device**: {results['model_info']['device']}
- **Timestamp**: {timestamp}

---

## Validation Results

"""
        
        if 'validation' in results and 'error' not in results['validation']:
            val = results['validation']
            md_content += f"""
| Metric | Value |
|--------|-------|
| mAP@0.5 | **{val['mAP@0.5']:.3f}** ({val['mAP@0.5']*100:.1f}%) |
| mAP@0.5:0.95 | **{val['mAP@0.5:0.95']:.3f}** ({val['mAP@0.5:0.95']*100:.1f}%) |
| Precision | {val['precision']:.3f} ({val['precision']*100:.1f}%) |
| Recall | {val['recall']:.3f} ({val['recall']*100:.1f}%) |
| Validation Time | {val['validation_time']:.2f}s |

"""
        else:
            md_content += f"❌ Validation failed: {results.get('validation', {}).get('error', 'Unknown error')}\n\n"
        
        # Inference Speed
        if 'inference_speed' in results and 'error' not in results['inference_speed']:
            speed = results['inference_speed']
            md_content += f"""
## Inference Speed Performance

| Metric | Value |
|--------|-------|
| **Average FPS** | **{speed['fps_mean']:.2f}** |
| Min FPS | {speed['fps_min']:.2f} |
| Max FPS | {speed['fps_max']:.2f} |
| Average Latency | {speed['total_mean']:.2f}ms ± {speed['total_std']:.2f}ms |
| Min Latency | {speed['total_min']:.2f}ms |
| Max Latency | {speed['total_max']:.2f}ms |
| Successful Predictions | {speed['successful_predictions']} |
| Average Detections/Image | {speed['avg_detections_per_image']:.2f} |

"""
        else:
            error_msg = "Unknown error"
            if results and 'inference_speed' in results and results['inference_speed'] and 'error' in results['inference_speed']:
                error_msg = results['inference_speed']['error']
            md_content += f"❌ Speed benchmark failed: {error_msg}\n\n"
        
        # Resolution Benchmark
        if 'resolution_benchmark' in results and 'error' not in results['resolution_benchmark']:
            md_content += """
## Resolution Benchmark

| Resolution | FPS | Latency (ms) |
|------------|-----|--------------|
"""
            for resolution, metrics in results['resolution_benchmark'].items():
                if 'error' not in metrics:
                    md_content += f"| {resolution} | {metrics['fps']:.2f} | {metrics['inference_time_ms']:.2f} |\n"
        
        md_content += """
---

## Summary

This benchmark report provides comprehensive performance metrics for the railway detection model.

### Recommendations:
- For real-time applications: Use 640x640 resolution for best speed/accuracy balance
- Deploy with TensorRT on Jetson for optimal edge performance
"""
        
        with open(output_file, 'w') as f:
            f.write(md_content)

def main():
    parser = argparse.ArgumentParser(description='Benchmark trained YOLO models on dataset')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt or .engine)')
    parser.add_argument('--data', required=True, help='Path to dataset YAML file')
    parser.add_argument('--device', default='0', help='Device to run on (0, 1, cpu)')
    parser.add_argument('--output-dir', default='./benchmark_results', help='Output directory for results')
    parser.add_argument('--speed-samples', type=int, default=30, help='Number of samples for speed benchmark')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ Dataset YAML not found: {args.data}")
        return
    
    # Initialize benchmark
    benchmark = DatasetBenchmark(args.model, args.data, args.device)
    
    # Run comprehensive benchmark
    results = benchmark.generate_benchmark_report(args.output_dir)
    
    # Print summary
    print(f"\n🏆 Benchmark Summary for {Path(args.model).name}:")
    if 'validation' in results and 'error' not in results['validation']:
        val = results['validation']
        print(f"   📊 mAP@0.5: {val['mAP@0.5']:.3f} ({val['mAP@0.5']*100:.1f}%)")
        print(f"   📊 mAP@0.5:0.95: {val['mAP@0.5:0.95']:.3f} ({val['mAP@0.5:0.95']*100:.1f}%)")
    
    if 'inference_speed' in results and 'error' not in results['inference_speed']:
        speed = results['inference_speed']
        print(f"   ⚡ Average FPS: {speed['fps_mean']:.2f}")
        print(f"   ⚡ Average Latency: {speed['total_mean']:.2f}ms")

if __name__ == "__main__":
    main()