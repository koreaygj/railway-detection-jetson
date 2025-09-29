#!/usr/bin/env python3
"""
Object Detection Visualization Script for YOLO11 Railway Detection Models
Author: Gyeongjin Yang
Lab: AVLab, Chungbuk National University

Creates visualization of object detection results during benchmarking
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import time
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DetectionVisualizer:
    def __init__(self, model_path: str, data_yaml: str):
        """
        Initialize the detection visualizer

        Args:
            model_path: Path to the YOLO model
            data_yaml: Path to the dataset yaml file
        """
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)

        # Load dataset configuration
        with open(self.data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)

        # Get class names
        self.class_names = self.data_config.get('names', [])
        logger.info(f"📋 Classes: {self.class_names}")

        # Define colors for each class (BGR format)
        self.colors = [
            (255, 0, 0),    # Blue for bird_nest
            (0, 255, 0),    # Green for plastic_bag
            (0, 0, 255),    # Red for floating_object
            (255, 255, 0),  # Cyan for balloon
        ]

        # Load model
        logger.info(f"📦 Loading model: {self.model_path.name}")
        self.model = YOLO(str(self.model_path))

    def get_test_images(self, max_images: int = 10) -> List[Path]:
        """Get test images from validation dataset"""
        # Get dataset root path
        dataset_root = Path(self.data_config.get('path', ''))
        val_images_path = dataset_root / self.data_config.get('val', 'val/images')

        if not val_images_path.exists():
            logger.error(f"❌ Validation images path not found: {val_images_path}")
            return []

        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(val_images_path.glob(f'*{ext}')))
            image_files.extend(list(val_images_path.glob(f'*{ext.upper()}')))

        # Limit number of images
        selected_images = image_files[:max_images]
        logger.info(f"📊 Selected {len(selected_images)} images for visualization")

        return selected_images

    def draw_detections(self, image: np.ndarray, results, conf_threshold: float = 0.25) -> np.ndarray:
        """
        Draw detection results on image

        Args:
            image: Input image (BGR format)
            results: YOLO detection results
            conf_threshold: Confidence threshold for display

        Returns:
            Annotated image
        """
        annotated_image = image.copy()

        # Get detection results
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes

            for i, box in enumerate(boxes):
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # Skip low confidence detections
                if conf < conf_threshold:
                    continue

                # Get class name and color
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
                color = self.colors[cls % len(self.colors)]

                # Draw bounding box
                cv2.rectangle(annotated_image,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color, 2)

                # Prepare label text
                label = f"{class_name}: {conf:.2f}"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Draw label background
                cv2.rectangle(annotated_image,
                            (int(x1), int(y1) - text_height - baseline - 5),
                            (int(x1) + text_width, int(y1)),
                            color, -1)

                # Draw label text
                cv2.putText(annotated_image, label,
                           (int(x1), int(y1) - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_image

    def create_detection_visualizations(self,
                                      output_dir: str,
                                      max_images: int = 10,
                                      conf_threshold: float = 0.25,
                                      iou_threshold: float = 0.45,
                                      imgsz: int = 640,
                                      device: str = '0') -> Dict:
        """
        Create detection visualizations for benchmark images

        Args:
            output_dir: Directory to save visualization results
            max_images: Maximum number of images to process
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            imgsz: Input image size
            device: Device to use for inference

        Returns:
            Dictionary with visualization results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        original_dir = output_path / "original"
        detected_dir = output_path / "detected"
        comparison_dir = output_path / "comparison"

        for dir_path in [original_dir, detected_dir, comparison_dir]:
            dir_path.mkdir(exist_ok=True)

        # Get test images
        test_images = self.get_test_images(max_images)

        if not test_images:
            logger.error("❌ No test images found")
            return {}

        logger.info(f"🎨 Creating visualizations for {len(test_images)} images...")

        detection_stats = {
            'total_images': len(test_images),
            'total_detections': 0,
            'class_detections': {name: 0 for name in self.class_names},
            'avg_confidence': 0.0,
            'processing_times': []
        }

        total_conf = 0.0
        total_detections = 0

        for i, image_path in enumerate(test_images):
            logger.info(f"📷 Processing image {i+1}/{len(test_images)}: {image_path.name}")

            # Load original image
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                logger.warning(f"⚠️ Could not load image: {image_path}")
                continue

            # Run inference
            start_time = time.time()
            results = self.model.predict(
                source=str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False
            )[0]  # Get first result

            inference_time = time.time() - start_time
            detection_stats['processing_times'].append(inference_time)

            # Draw detections
            detected_image = self.draw_detections(original_image, results, conf_threshold)

            # Count detections and collect statistics
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes
                for box in boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    if conf >= conf_threshold:
                        total_detections += 1
                        total_conf += conf

                        if cls < len(self.class_names):
                            detection_stats['class_detections'][self.class_names[cls]] += 1

            # Save images
            image_name = image_path.stem

            # Save original
            cv2.imwrite(str(original_dir / f"{image_name}_original.jpg"), original_image)

            # Save detected
            cv2.imwrite(str(detected_dir / f"{image_name}_detected.jpg"), detected_image)

            # Create comparison (side by side)
            h, w = original_image.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = original_image
            comparison[:, w:] = detected_image

            # Add labels
            cv2.putText(comparison, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Detected", (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imwrite(str(comparison_dir / f"{image_name}_comparison.jpg"), comparison)

        # Calculate final statistics
        detection_stats['total_detections'] = total_detections
        if total_detections > 0:
            detection_stats['avg_confidence'] = total_conf / total_detections

        if detection_stats['processing_times']:
            detection_stats['avg_processing_time'] = np.mean(detection_stats['processing_times'])
            detection_stats['fps'] = 1.0 / detection_stats['avg_processing_time']

        # Save statistics
        stats_file = output_path / "detection_stats.yaml"
        with open(stats_file, 'w') as f:
            yaml.dump(detection_stats, f, default_flow_style=False)

        logger.info(f"✅ Visualization completed!")
        logger.info(f"📊 Results saved to: {output_path}")
        logger.info(f"📈 Total detections: {total_detections}")
        logger.info(f"🎯 Average confidence: {detection_stats['avg_confidence']:.3f}")
        logger.info(f"⚡ Average FPS: {detection_stats.get('fps', 0):.1f}")

        return detection_stats

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize YOLO11 Object Detection Results')

    parser.add_argument('data_yaml', type=str,
                       help='Path to dataset YAML file')
    parser.add_argument('model', type=str,
                       help='Path to YOLO model (PyTorch .pt or TensorRT .engine)')

    # Detection settings
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (0 for GPU, cpu for CPU)')

    # Output settings
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum number of images to process (default: 10)')
    parser.add_argument('--output-dir', type=str, default='./visualization_results',
                       help='Output directory for results (default: ./visualization_results)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name for output directory')

    return parser.parse_args()

def main():
    args = parse_args()

    # Set experiment name
    if args.name:
        experiment_name = args.name
    else:
        model_name = Path(args.model).stem
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{model_name}_visualization_{timestamp}"

    output_dir = Path(args.output_dir) / experiment_name

    print(f"🎨 YOLO11 Railway Detection Visualization")
    print(f"📋 Experiment: {experiment_name}")
    print(f"🤖 Model: {Path(args.model).name}")
    print(f"📊 Dataset: {Path(args.data_yaml).name}")
    print(f"🔧 Device: {args.device}")
    print("=" * 60)

    try:
        # Initialize visualizer
        visualizer = DetectionVisualizer(args.model, args.data_yaml)

        # Create visualizations
        stats = visualizer.create_detection_visualizations(
            output_dir=str(output_dir),
            max_images=args.max_images,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device
        )

        print("\n📊 Visualization Summary:")
        print(f"   Total images: {stats.get('total_images', 0)}")
        print(f"   Total detections: {stats.get('total_detections', 0)}")
        print(f"   Average confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"   Average FPS: {stats.get('fps', 0):.1f}")
        print(f"   Results saved: {output_dir}")

        # Print class-wise detections
        if stats.get('class_detections'):
            print("\n🏷️ Class-wise Detections:")
            for class_name, count in stats['class_detections'].items():
                print(f"   {class_name}: {count}")

    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        raise

if __name__ == "__main__":
    main()