#!/usr/bin/env python3
"""
Jetson Nano Inference Script for Railway Object Detection
Optimized for real-time performance on edge device
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, using ONNX inference")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNXRuntime not available")

class JetsonInference:
    def __init__(self, model_path, input_size=640, conf_threshold=0.5):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.class_names = ['niaocao', 'suliaodai', 'piaofuwu', 'qiqiu']
        
        model_ext = Path(model_path).suffix.lower()
        
        if model_ext == '.engine' and TRT_AVAILABLE:
            self.engine = self._load_tensorrt_engine(model_path)
            self.inference_type = 'tensorrt'
        elif model_ext == '.onnx' and ONNX_AVAILABLE:
            self.session = ort.InferenceSession(model_path, 
                                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.inference_type = 'onnx'
        else:
            raise ValueError(f"Unsupported model format: {model_ext}")
        
        print(f"🚀 Loaded model using {self.inference_type.upper()}")
    
    def _load_tensorrt_engine(self, engine_path):
        """Load TensorRT engine"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize and pad
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Normalize
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, pad_h, pad_w
    
    def inference(self, input_tensor):
        """Run inference"""
        if self.inference_type == 'onnx':
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            return outputs[0]
        elif self.inference_type == 'tensorrt':
            # TensorRT inference implementation
            # This would require more detailed TRT setup
            pass
    
    def postprocess(self, outputs, scale, pad_h, pad_w, original_shape):
        """Post-process outputs to get bounding boxes"""
        # Implementation depends on YOLO output format
        # This is a simplified version
        detections = []
        
        for detection in outputs[0]:
            confidence = detection[4]
            if confidence > self.conf_threshold:
                # Extract box coordinates and class
                x, y, w, h = detection[:4]
                class_id = int(np.argmax(detection[5:]))
                class_conf = detection[5 + class_id]
                
                # Convert to original image coordinates
                x = (x - pad_w) / scale
                y = (y - pad_h) / scale
                w = w / scale
                h = h / scale
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(confidence * class_conf),
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        
        return detections
    
    def predict(self, image):
        """Full prediction pipeline"""
        start_time = time.time()
        
        # Preprocess
        input_tensor, scale, pad_h, pad_w = self.preprocess(image)
        preprocess_time = time.time() - start_time
        
        # Inference
        inference_start = time.time()
        outputs = self.inference(input_tensor)
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
                'preprocess': preprocess_time,
                'inference': inference_time,
                'postprocess': postprocess_time,
                'total': total_time,
                'fps': fps
            }
        }

def benchmark_model(model_path, num_iterations=100):
    """Benchmark model performance"""
    print(f"🏃 Benchmarking {model_path}...")
    
    inference_engine = JetsonInference(model_path)
    
    # Create dummy input
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(10):
        inference_engine.predict(dummy_image)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.time()
        result = inference_engine.predict(dummy_image)
        end = time.time()
        times.append(end - start)
        
        if (i + 1) % 20 == 0:
            avg_time = np.mean(times[-20:])
            avg_fps = 1.0 / avg_time
            print(f"Iteration {i+1}/{num_iterations}: {avg_fps:.1f} FPS")
    
    # Results
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_fps = 1.0 / avg_time
    
    print(f"📊 Benchmark Results:")
    print(f"   Average FPS: {avg_fps:.2f}")
    print(f"   Average latency: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"   Min FPS: {1.0/max(times):.2f}")
    print(f"   Max FPS: {1.0/min(times):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--source', default='0', help='Input source (camera index or video file)')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_model(args.model)
    else:
        # Real-time inference
        inference_engine = JetsonInference(args.model)
        
        # Setup camera or video
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source))
        else:
            cap = cv2.VideoCapture(args.source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = inference_engine.predict(frame)
            
            # Draw results
            for det in result['detections']:
                x, y, w, h = det['bbox']
                x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show FPS
            fps_text = f"FPS: {result['timing']['fps']:.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Railway Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
