#!/usr/bin/env python3
"""
Quick Start Script for Railway Object Detection
Author: Gyeongjin Yang
Lab: AVLab, Chungbuk National University

This script provides a simple way to get started with training and evaluation.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    # Package name mapping: display_name -> import_name
    package_checks = {
        'ultralytics': 'ultralytics',
        'torch': 'torch', 
        'opencv-python': 'cv2',  # opencv-python or opencv-python-headless both import as cv2
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for display_name, import_name in package_checks.items():
        try:
            __import__(import_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            missing_packages.append(display_name)
            print(f"   ❌ {display_name}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_dataset():
    """Check if dataset is available"""
    print("🔍 Checking dataset...")
    
    data_yaml = "../../data/yolo_dataset/data.yaml"
    train_images = "../../data/yolo_dataset/train/images"
    val_images = "../../data/yolo_dataset/val/images"
    
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset config not found: {data_yaml}")
        return False
    
    if not os.path.exists(train_images):
        print(f"❌ Training images not found: {train_images}")
        return False
    
    if not os.path.exists(val_images):
        print(f"❌ Validation images not found: {val_images}")
        return False
    
    # Count images
    train_count = len([f for f in os.listdir(train_images) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(val_images) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"   ✅ Training images: {train_count}")
    print(f"   ✅ Validation images: {val_count}")
    
    return True

def main():
    print("🚂 Railway Object Detection - Quick Start")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install required packages first!")
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Please ensure dataset is properly set up!")
        sys.exit(1)
    
    print("\n🎯 Choose an option:")
    print("1. Train YOLOv11n model (quick training - 50 epochs)")
    print("2. Train YOLOv11n model (full training - 100 epochs)")
    print("3. Evaluate existing model")
    print("4. Train with custom parameters")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        # Quick training
        cmd = [
            sys.executable, "train.py",
            "--model", "yolo11n.pt",
            "--epochs", "50",
            "--batch-size", "16",
            "--name", "yolo11n_quick"
        ]
        
        success = run_command(cmd, "Starting quick training (50 epochs)")
        if success:
            print("\n🎉 Quick training completed!")
            print("💡 To evaluate the model, run: python quick_start.py and choose option 3")
    
    elif choice == "2":
        # Full training
        cmd = [
            sys.executable, "train.py",
            "--model", "yolo11n.pt",
            "--epochs", "100",
            "--batch-size", "16",
            "--name", "yolo11n_full"
        ]
        
        success = run_command(cmd, "Starting full training (100 epochs)")
        if success:
            print("\n🎉 Full training completed!")
            print("💡 To evaluate the model, run: python quick_start.py and choose option 3")
    
    elif choice == "3":
        # Evaluate model
        print("\n🔍 Looking for trained models...")
        
        # Find available models
        results_dirs = ["./result", "railway-detection"]  # Check both result and old railway-detection dirs
        models = []
        
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                for exp_dir in os.listdir(results_dir):
                    exp_path = os.path.join(results_dir, exp_dir)
                    if os.path.isdir(exp_path):
                        weights_dir = os.path.join(exp_path, "weights")
                        if os.path.exists(weights_dir):
                            best_model = os.path.join(weights_dir, "best.pt")
                            if os.path.exists(best_model):
                                models.append((exp_dir, best_model))
        
        if not models:
            print("❌ No trained models found!")
            print("💡 Train a model first using options 1 or 2")
            return
        
        print(f"\n📋 Found {len(models)} trained model(s):")
        for i, (name, path) in enumerate(models, 1):
            print(f"{i}. {name} ({path})")
        
        model_choice = input(f"\nSelect model to evaluate (1-{len(models)}): ").strip()
        
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(models):
                model_name, model_path = models[model_idx]
                
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model", model_path,
                    "--save-plots",
                    "--benchmark",
                    "--save-dir", f"./result/evaluation_{model_name}"
                ]
                
                success = run_command(cmd, f"Evaluating model: {model_name}")
                if success:
                    print("\n🎉 Evaluation completed!")
            else:
                print("❌ Invalid selection!")
        except ValueError:
            print("❌ Please enter a valid number!")
    
    elif choice == "4":
        # Custom training
        print("\n⚙️ Custom Training Configuration")
        
        model_variant = input("Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt) [yolo11n.pt]: ").strip()
        if not model_variant:
            model_variant = "yolo11n.pt"
        
        epochs = input("Number of epochs [100]: ").strip()
        if not epochs:
            epochs = "100"
        
        batch_size = input("Batch size [16]: ").strip()
        if not batch_size:
            batch_size = "16"
        
        name = input("Experiment name [custom]: ").strip()
        if not name:
            name = "custom"
        
        device = input("Device (0, 1, cpu) [0]: ").strip()
        if not device:
            device = "0"
        
        cmd = [
            sys.executable, "train.py",
            "--model", model_variant,
            "--epochs", epochs,
            "--batch-size", batch_size,
            "--name", name,
            "--device", device
        ]
        
        # Ask for additional options
        if input("Enable W&B logging? (y/n) [n]: ").strip().lower() == 'y':
            cmd.append("--wandb")
        
        if input("Verbose output? (y/n) [n]: ").strip().lower() == 'y':
            cmd.append("--verbose")
        
        success = run_command(cmd, f"Starting custom training: {name}")
        if success:
            print("\n🎉 Custom training completed!")
    
    elif choice == "5":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice! Please select 1-5.")

if __name__ == "__main__":
    main()