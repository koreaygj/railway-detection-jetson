#!/usr/bin/env python3
"""
Convert YOLO models to TensorRT directly on Jetson Orin Nano
Run this script ON the Jetson device
"""

import os
import time
from pathlib import Path
from ultralytics import YOLO

def convert_models_on_jetson():
    """Convert all models to TensorRT on Jetson"""
    
    # Model paths (adjust as needed)
    models = {
        'yolo11n_railway': './models/yolo11n_railway_best.pt',
        'yolo11s_railway': './models/yolo11s_railway_best.pt', 
        'yolo11n_20250909_141751': './models/yolo11n_20250909_141751_best.pt'
    }
    
    print("🚂 Converting YOLO Models to TensorRT on Jetson Orin Nano")
    print("=" * 60)
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please check your Jetson setup.")
        return
    
    print(f"✅ CUDA available - Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Convert each model
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            continue
            
        print(f"\n🔄 Converting {model_name}...")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Convert to TensorRT FP16
            print("   📦 Exporting to TensorRT FP16...")
            start_time = time.time()
            
            engine_path = model.export(
                format='engine',
                imgsz=640,
                batch=1,
                device=0,
                half=True,        # FP16
                workspace=4,      # 4GB workspace
                verbose=True
            )
            
            export_time = time.time() - start_time
            file_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
            
            print(f"   ✅ TensorRT FP16: {file_size:.1f}MB, {export_time:.1f}s")
            print(f"   📁 Saved: {engine_path}")
            
            # Also try INT8 if you have calibration data
            # Uncomment below if you want INT8 quantization
            """
            print("   📦 Exporting to TensorRT INT8...")
            start_time = time.time()
            
            int8_engine_path = model.export(
                format='engine',
                imgsz=640,
                batch=1,
                device=0,
                int8=True,
                data='path/to/your/data.yaml',  # Calibration dataset
                workspace=4,
                verbose=True
            )
            
            export_time = time.time() - start_time
            file_size = os.path.getsize(int8_engine_path) / (1024 * 1024)  # MB
            
            print(f"   ✅ TensorRT INT8: {file_size:.1f}MB, {export_time:.1f}s")
            print(f"   📁 Saved: {int8_engine_path}")
            """
            
        except Exception as e:
            print(f"   ❌ Conversion failed: {str(e)}")
    
    print(f"\n🚀 Conversion completed!")
    print(f"📖 Run benchmark: python3 jetson_tensorrt_inference.py --engine <engine_file> --benchmark")

if __name__ == "__main__":
    convert_models_on_jetson()