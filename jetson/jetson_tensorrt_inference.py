#!/usr/bin/env python3
"""
TensorRT Inference Script for Jetson Orin Nano
Optimized for railway object detection
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path, input_size=640, conf_threshold=0.5, nms_threshold=0.4):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = ['niaocao', 'suliaodai', 'piaofuwu', 'qiqiu']
        
        # Load TensorRT engine
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        print(f"🚀 TensorRT engine loaded: {Path(engine_path).name}")
        print(f"   Input shape: {self.engine.get_binding_shape(0)}")
        print(f"   Output shape: {self.engine.get_binding_shape(1)}")
    
    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for TensorRT inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize and pad to square
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Normalize to [0, 1] and convert to CHW format
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
        
        return input_tensor.ravel(), scale, pad_h, pad_w
    
    def inference(self, input_data):
        """Run TensorRT inference"""
        # Copy input data to GPU
        np.copyto(self.inputs[0]['host'], input_data)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output back to CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']
    
    def postprocess(self, outputs, scale, pad_h, pad_w, original_shape):
        """Post-process TensorRT outputs"""
        # Reshape output (assuming YOLO format: [batch, 4+nc, num_anchors])
        outputs = outputs.reshape(1, -1, 8400)  # Adjust based on your model output
        outputs = outputs.transpose(0, 2, 1)    # [batch, num_anchors, 4+nc]
        
        detections = []
        
        for detection in outputs[0]:  # Process first (and only) batch
            # Extract box coordinates and confidences
            x, y, w, h = detection[:4]
            obj_conf = detection[4]
            class_confs = detection[5:]
            
            # Find best class
            class_id = np.argmax(class_confs)
            class_conf = class_confs[class_id]
            confidence = obj_conf * class_conf
            
            if confidence > self.conf_threshold:
                # Convert coordinates back to original image scale
                x = (x - pad_w) / scale
                y = (y - pad_h) / scale
                w = w / scale
                h = h / scale
                
                # Convert to corner coordinates
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                })
        
        # Apply NMS
        detections = self.nms(detections)
        return detections
    
    def nms(self, detections):
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping boxes
            detections = [det for det in detections if self.iou(current['bbox'], det['bbox']) < self.nms_threshold]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def predict(self, image):
        """Full prediction pipeline"""
        start_time = time.time()
        
        # Preprocess
        preprocess_start = time.time()
        input_data, scale, pad_h, pad_w = self.preprocess(image)
        preprocess_time = time.time() - preprocess_start
        
        # Inference
        inference_start = time.time()
        outputs = self.inference(input_data)
        inference_time = time.time() - inference_start
        
        # Postprocess
        postprocess_start = time.time()
        detections = self.postprocess(outputs, scale, pad_h, pad_w, image.shape[:2])
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        fps = 1.0 / total_time if total_time > 0 else 0
        
        return {
            'detections': detections,
            'timing': {
                'preprocess': preprocess_time * 1000,  # ms
                'inference': inference_time * 1000,    # ms
                'postprocess': postprocess_time * 1000, # ms
                'total': total_time * 1000,            # ms
                'fps': fps
            }
        }

def benchmark_tensorrt(engine_path, num_iterations=100):
    """Benchmark TensorRT engine performance"""
    print(f"🏃 Benchmarking TensorRT engine: {Path(engine_path).name}")
    
    inference_engine = TensorRTInference(engine_path)
    
    # Create dummy input
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up
    print("🔥 Warming up...")
    for _ in range(10):
        inference_engine.predict(dummy_image)
    
    # Benchmark
    print(f"⏱️  Running {num_iterations} iterations...")
    times = []
    inference_times = []
    
    for i in range(num_iterations):
        result = inference_engine.predict(dummy_image)
        times.append(result['timing']['total'])
        inference_times.append(result['timing']['inference'])
        
        if (i + 1) % 20 == 0:
            avg_fps = 1000.0 / np.mean(times[-20:])  # Convert ms to fps
            print(f"   Progress: {i+1}/{num_iterations} - Current FPS: {avg_fps:.1f}")
    
    # Calculate statistics
    avg_total_time = np.mean(times)
    std_total_time = np.std(times)
    avg_inference_time = np.mean(inference_times)
    avg_fps = 1000.0 / avg_total_time
    
    print(f"
📊 Benchmark Results:")
    print(f"   🚀 Average FPS: {avg_fps:.2f}")
    print(f"   ⏱️  Average total latency: {avg_total_time:.2f}ms ± {std_total_time:.2f}ms")
    print(f"   🧠 Average inference time: {avg_inference_time:.2f}ms")
    print(f"   📈 Min FPS: {1000.0/max(times):.2f}")
    print(f"   📈 Max FPS: {1000.0/min(times):.2f}")

def main():
    parser = argparse.ArgumentParser(description='TensorRT inference for railway detection')
    parser.add_argument('--engine', required=True, help='Path to TensorRT engine file')
    parser.add_argument('--source', default='0', help='Video source (camera index or video file)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark test')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save-video', help='Save output video to file')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_tensorrt(args.engine)
        return
    
    # Initialize inference engine
    inference_engine = TensorRTInference(args.engine, conf_threshold=args.conf_threshold)
    
    # Setup video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    # Video writer setup
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
    
    print("🚂 Starting railway detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    fps_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            result = inference_engine.predict(frame)
            current_fps = result['timing']['fps']
            fps_history.append(current_fps)
            
            # Keep only last 30 FPS measurements for rolling average
            if len(fps_history) > 30:
                fps_history.pop(0)
            
            avg_fps = np.mean(fps_history)
            
            # Draw detections
            for det in result['detections']:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                class_name = det['class_name']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw performance info
            perf_text = [
                f"FPS: {avg_fps:.1f}",
                f"Inference: {result['timing']['inference']:.1f}ms",
                f"Total: {result['timing']['total']:.1f}ms",
                f"Detections: {len(result['detections'])}"
            ]
            
            for i, text in enumerate(perf_text):
                y_pos = 30 + i * 25
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save frame if video writer is enabled
            if writer:
                writer.write(frame)
            
            # Display frame
            cv2.imshow('Railway Detection - TensorRT', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{frame_count:06d}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if fps_history:
            final_avg_fps = np.mean(fps_history)
            print(f"\n📊 Final Performance Summary:")
            print(f"   Average FPS: {final_avg_fps:.2f}")
            print(f"   Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()
