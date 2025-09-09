#!/usr/bin/env python3
"""
YOLOv11 Model Evaluation Script for Railway Object Detection
Author: Gyeongjin Yang
Lab: AVLab, Chungbuk National University
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from collections import defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 model on Railway Detection Dataset')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, default='../../data/yolo_dataset/data.yaml',
                       help='Path to dataset yaml file')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for evaluation (0, 1, 2, etc. or cpu)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--save-dir', type=str, default='./result',
                       help='Directory to save results')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of predictions')
    
    return parser.parse_args()

class ModelEvaluator:
    def __init__(self, model_path, device='0'):
        self.model = YOLO(model_path)
        self.device = device
        self.class_names = ['niaocao', 'suliaodai', 'piaofuwu', 'qiqiu']
        self.class_names_en = ['Bird Nest', 'Plastic Bag', 'Flag', 'Balloon']
        
    def validate_model(self, data_path, img_size=640, conf=0.25, iou=0.45):
        """Run official validation on test set"""
        print("🔍 Running model validation...")
        
        results = self.model.val(
            data=data_path,
            imgsz=img_size,
            conf=conf,
            iou=iou,
            device=self.device,
            save_json=True,
            save_hybrid=False,
            plots=True
        )
        
        return results
    
    def benchmark_speed(self, img_size=640, num_iterations=100):
        """Benchmark inference speed"""
        print(f"⚡ Benchmarking inference speed ({num_iterations} iterations)...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)
        if self.device != 'cpu':
            dummy_input = dummy_input.cuda()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input, verbose=False)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input, verbose=False)
            
            if self.device != 'cpu':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{num_iterations}")
        
        times = np.array(times) * 1000  # Convert to milliseconds
        
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'fps_mean': 1000 / np.mean(times),
            'fps_median': 1000 / np.median(times)
        }
        
        return stats, times
    
    def get_model_info(self):
        """Get model information"""
        model_info = {}
        
        # Get model size
        if hasattr(self.model.model, 'model'):
            # Count parameters
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            model_info['total_params'] = total_params
            model_info['trainable_params'] = trainable_params
            model_info['model_size_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return model_info
    
    def create_confusion_matrix(self, results, save_path=None):
        """Create and save confusion matrix"""
        try:
            # Get confusion matrix from results
            cm = results.confusion_matrix.matrix
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, 
                       annot=True, 
                       fmt='d', 
                       cmap='Blues',
                       xticklabels=self.class_names_en + ['Background'],
                       yticklabels=self.class_names_en + ['Background'])
            
            plt.title('Confusion Matrix - Railway Object Detection')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Confusion matrix saved: {save_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not create confusion matrix: {str(e)}")
    
    def create_performance_plots(self, stats, times, save_dir):
        """Create performance visualization plots"""
        # Speed distribution plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(stats['mean_time'], color='red', linestyle='--', 
                   label=f'Mean: {stats["mean_time"]:.2f}ms')
        plt.axvline(stats['median_time'], color='green', linestyle='--', 
                   label=f'Median: {stats["median_time"]:.2f}ms')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        fps_values = 1000 / times
        plt.hist(fps_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(stats['fps_mean'], color='red', linestyle='--', 
                   label=f'Mean: {stats["fps_mean"]:.1f} FPS')
        plt.axvline(stats['fps_median'], color='green', linestyle='--', 
                   label=f'Median: {stats["fps_median"]:.1f} FPS')
        plt.xlabel('FPS')
        plt.ylabel('Frequency')
        plt.title('FPS Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved: {plot_path}")
    
    def save_results_summary(self, results, stats, model_info, save_path):
        """Save comprehensive results summary"""
        summary = {
            'model_info': model_info,
            'validation_results': {
                'mAP_0.5': float(results.box.map50),
                'mAP_0.5_0.95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr)) if (results.box.mp + results.box.mr) > 0 else 0.0
            },
            'performance_stats': stats,
            'class_results': {}
        }
        
        # Add per-class results if available
        if hasattr(results.box, 'maps'):
            for i, class_name in enumerate(self.class_names_en):
                if i < len(results.box.maps):
                    summary['class_results'][class_name] = {
                        'mAP_0.5_0.95': float(results.box.maps[i]),
                        'mAP_0.5': float(results.box.map50[i]) if hasattr(results.box, 'map50') and i < len(results.box.map50) else 0.0
                    }
        
        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Results summary saved: {save_path}")
        
        return summary

def print_results_table(summary):
    """Print formatted results table"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    # Model info
    model_info = summary.get('model_info', {})
    print(f"🤖 Model Information:")
    print(f"   Total Parameters: {model_info.get('total_params', 'N/A'):,}")
    print(f"   Model Size: {model_info.get('model_size_mb', 'N/A'):.2f} MB")
    
    # Validation results
    val_results = summary.get('validation_results', {})
    print(f"\n🎯 Detection Performance:")
    print(f"   mAP@0.5     : {val_results.get('mAP_0.5', 0):.4f}")
    print(f"   mAP@0.5:0.95: {val_results.get('mAP_0.5_0.95', 0):.4f}")
    print(f"   Precision   : {val_results.get('precision', 0):.4f}")
    print(f"   Recall      : {val_results.get('recall', 0):.4f}")
    print(f"   F1-Score    : {val_results.get('f1_score', 0):.4f}")
    
    # Performance stats
    perf_stats = summary.get('performance_stats', {})
    print(f"\n⚡ Speed Performance:")
    print(f"   Mean FPS    : {perf_stats.get('fps_mean', 0):.1f}")
    print(f"   Median FPS  : {perf_stats.get('fps_median', 0):.1f}")
    print(f"   Mean Time   : {perf_stats.get('mean_time', 0):.2f} ms")
    print(f"   Min Time    : {perf_stats.get('min_time', 0):.2f} ms")
    
    # Per-class results
    class_results = summary.get('class_results', {})
    if class_results:
        print(f"\n📋 Per-Class Performance:")
        for class_name, metrics in class_results.items():
            print(f"   {class_name:12}: mAP@0.5={metrics.get('mAP_0.5', 0):.4f}, "
                  f"mAP@0.5:0.95={metrics.get('mAP_0.5_0.95', 0):.4f}")
    
    print("="*60)

def main():
    args = parse_args()
    
    print(f"🚂 Railway Object Detection - Model Evaluation")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Dataset: {args.data}")
    print(f"🔧 Device: {args.device}")
    print("="*60)
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.device)
    
    # Get model information
    model_info = evaluator.get_model_info()
    
    # Run validation
    results = evaluator.validate_model(
        args.data, 
        img_size=args.img_size, 
        conf=args.conf, 
        iou=args.iou
    )
    
    # Run speed benchmark if requested
    stats = {}
    times = []
    if args.benchmark:
        stats, times = evaluator.benchmark_speed(args.img_size)
    
    # Create plots if requested
    if args.save_plots:
        # Confusion matrix
        cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        evaluator.create_confusion_matrix(results, cm_path)
        
        # Performance plots
        if args.benchmark:
            evaluator.create_performance_plots(stats, times, args.save_dir)
    
    # Save comprehensive results
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    summary = evaluator.save_results_summary(results, stats, model_info, results_path)
    
    # Print results table
    print_results_table(summary)
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()