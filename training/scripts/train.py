#!/usr/bin/env python3
"""
YOLOv11n Training Script for Railway Object Detection
Author: Gyeongjin Yang
Lab: AVLab, Chungbuk National University
"""

import os
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

def find_project_root():
    """Find the project root directory by looking for specific markers"""
    current_dir = Path(__file__).resolve().parent
    
    # Look for project markers (e.g., specific directories or files)
    markers = ['data', 'training', 'jetson', '.git']
    
    while current_dir.parent != current_dir:  # Stop at filesystem root
        if any((current_dir / marker).exists() for marker in markers):
            return current_dir
        current_dir = current_dir.parent
    
    # Fallback to two levels up from script location
    return Path(__file__).resolve().parent.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11n on Railway Object Detection Dataset')
    
    # Get project root for default paths
    project_root = find_project_root()
    
    # Model settings
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--data', type=str, default=str(project_root / 'data' / 'yolo_dataset' / 'data.yaml'),
                       help='Path to dataset yaml file')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='0',
                       help='Device to train on (0, 1, 2, etc. or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers')
    
    # Optimization settings
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')
    
    # Output settings
    parser.add_argument('--project', type=str, default=str(project_root / 'result'),
                       help='Project name for saving results')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save model every n epochs')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    
    return parser.parse_args()

def setup_wandb(args):
    """Initialize Weights & Biases logging"""
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.project,
            name=args.name or f"yolo11n_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'model': args.model,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'img_size': args.img_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'momentum': args.momentum,
            }
        )
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠️ Wandb not available. Install with: pip install wandb")

def validate_dataset(data_path):
    """Validate dataset configuration and paths"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset config file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get absolute path of dataset config file to resolve relative paths
    data_dir = os.path.dirname(os.path.abspath(data_path))
    dataset_root = data_config.get('path', './data/yolo-format')
    
    # If path is relative, make it relative to the config file location
    if not os.path.isabs(dataset_root):
        # Remove './' from the beginning if present and resolve relative to config file
        if dataset_root.startswith('./'):
            dataset_root = dataset_root[2:]
        dataset_root = os.path.join(data_dir, dataset_root)
    
    train_path = os.path.join(dataset_root, data_config.get('train', 'train/images'))
    val_path = os.path.join(dataset_root, data_config.get('val', 'val/images'))
    
    print(f"📁 Dataset paths:")
    print(f"   Config file: {data_path}")
    print(f"   Dataset root: {dataset_root}")
    print(f"   Train path: {train_path}")
    print(f"   Val path: {val_path}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training images not found: {train_path}")
    
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation images not found: {val_path}")
    
    # Count images
    train_images = len([f for f in os.listdir(train_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    val_images = len([f for f in os.listdir(val_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    print(f"✅ Dataset validation passed:")
    print(f"   Training images: {train_images}")
    print(f"   Validation images: {val_images}")
    print(f"   Classes: {data_config.get('nc', 'Unknown')}")
    print(f"   Class names: {data_config.get('names', 'Unknown')}")
    
    return data_config

def main():
    args = parse_args()
    
    # Set experiment name if not provided
    if args.name is None:
        model_name = Path(args.model).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f"{model_name}_{timestamp}"
    
    print(f"🚂 Starting Railway Object Detection Training")
    print(f"📋 Experiment: {args.name}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Dataset: {args.data}")
    print(f"🔧 Device: {args.device}")
    print("=" * 60)
    
    # Validate dataset
    data_config = validate_dataset(args.data)
    
    # Setup logging
    setup_wandb(args)
    
    # Initialize model
    if args.resume:
        print(f"🔄 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"🆕 Loading pretrained model: {args.model}")
        model = YOLO(args.model)
    
    # Check device availability
    device = args.device
    if device != 'cpu':
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, switching to CPU")
            device = 'cpu'
        elif device.isdigit() and int(device) >= torch.cuda.device_count():
            print(f"⚠️  GPU {device} not available, using GPU 0")
            device = '0'
    
    print(f"🎯 Using device: {device}")
    
    # Training configuration
    train_config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': device,
        'workers': args.workers,
        'lr0': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'project': args.project,
        'name': args.name,
        'save_period': args.save_period,
        'verbose': args.verbose,
        'seed': 42,  # For reproducibility
        'deterministic': True,
        # Augmentation settings optimized for railway detection
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    # Start training
    print("🚀 Starting training...")
    try:
        results = model.train(**train_config)
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Log final results to wandb
        if args.wandb and WANDB_AVAILABLE:
            best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                wandb.save(best_model_path)
            wandb.finish()
        
        # Print training summary
        print("\n📊 Training Summary:")
        print(f"   Best mAP@0.5: {results.box.map50:.4f}")
        print(f"   Best mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Final loss: {results.box.fitness:.4f}")
        print(f"   Model saved: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise
    finally:
        if args.wandb and WANDB_AVAILABLE and wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()