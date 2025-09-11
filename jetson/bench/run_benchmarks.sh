#!/bin/bash
# Run comprehensive benchmarks on all trained models (Fixed version)
# Author: Gyeongjin Yang

echo "🚂 Running Dataset Benchmarks for Railway Detection Models"
echo "========================================================="

# Set paths - adjust these according to your directory structure
MODELS_DIR="../../result/trained_models"  # Adjust path as needed
DATA_YAML="./data/data.yaml"               # Adjust path as needed
OUTPUT_DIR="./benchmark_results"
DEVICE="0"  # Change to "cpu" if no GPU

# Alternative data paths to try
DATA_PATHS=(
    "./data/data.yaml"
    "../data/yolo_dataset/data.yaml"
    "../../data/yolo_dataset/data.yaml"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find correct data.yaml path
DATA_YAML=""
for path in "${DATA_PATHS[@]}"; do
    if [ -f "$path" ]; then
        DATA_YAML="$path"
        echo "✅ Found dataset: $DATA_YAML"
        break
    fi
done

if [ -z "$DATA_YAML" ]; then
    echo "❌ Dataset YAML not found in any of these locations:"
    for path in "${DATA_PATHS[@]}"; do
        echo "   - $path"
    done
    echo ""
    echo "Please create a data.yaml file or update the paths in this script"
    exit 1
fi

echo "📁 Dataset: $DATA_YAML"
echo "📁 Output: $OUTPUT_DIR"
echo "🔧 Device: $DEVICE"
echo ""

# Find all trained models
MODEL_FILES=($(find "$MODELS_DIR" -name "best.pt" -type f 2>/dev/null))

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "❌ No trained models found in $MODELS_DIR"
    echo "Please check the MODELS_DIR path in this script"
    exit 1
fi

echo "📦 Found ${#MODEL_FILES[@]} trained models:"
for model in "${MODEL_FILES[@]}"; do
    model_name=$(basename $(dirname $(dirname "$model")))
    echo "   - $model_name"
done
echo ""

# Run benchmarks for each model
for model_file in "${MODEL_FILES[@]}"; do
    model_name=$(basename $(dirname $(dirname "$model_file")))
    
    echo "🔄 Benchmarking: $model_name"
    echo "   Model: $model_file"
    
    # Check if model file exists
    if [ ! -f "$model_file" ]; then
        echo "   ❌ Model file not found: $model_file"
        continue
    fi
    
    python benchmark_dataset.py \
        --model "$model_file" \
        --data "$DATA_YAML" \
        --device "$DEVICE" \
        --output-dir "$OUTPUT_DIR" \
        --speed-samples 30
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Benchmark completed for $model_name"
    else
        echo "   ❌ Benchmark failed for $model_name"
    fi
    echo ""
done

# Generate comparison report (simplified to avoid f-string issues)
echo "📊 Generating comparison report..."

python3 << 'EOF'
import json
import os
from pathlib import Path
import glob

results_dir = Path('./benchmark_results')
json_files = list(results_dir.glob('*_benchmark_*.json'))

if not json_files:
    print('No benchmark results found')
    exit()

print('\n🏆 Model Comparison Summary:')
print('=' * 70)

comparison_data = []

for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        model_name = data['model_info']['model_name']
        
        # Extract key metrics
        metrics = {'model': model_name}
        
        if 'validation' in data and 'error' not in data['validation']:
            val = data['validation']
            metrics['mAP@0.5'] = val['mAP@0.5']
            metrics['mAP@0.5:0.95'] = val['mAP@0.5:0.95']
            metrics['precision'] = val['precision']
            metrics['recall'] = val['recall']
        
        if 'inference_speed' in data and 'error' not in data['inference_speed']:
            speed = data['inference_speed']
            metrics['fps'] = speed['fps_mean']
            metrics['latency_ms'] = speed['total_mean']
        
        comparison_data.append(metrics)
        
    except Exception as e:
        print('Error reading {}: {}'.format(json_file, e))

# Sort by mAP@0.5
comparison_data.sort(key=lambda x: x.get('mAP@0.5', 0), reverse=True)

# Print comparison table (avoiding f-string issues)
header = "{:<25} {:<8} {:<12} {:<8} {:<12}".format('Model', 'mAP@0.5', 'mAP@0.5:0.95', 'FPS', 'Latency(ms)')
print(header)
print('-' * 70)

for data in comparison_data:
    model = data['model'][:24]  # Truncate long names
    map50 = "{:.3f}".format(data.get('mAP@0.5', 0)) if 'mAP@0.5' in data else 'N/A'
    map50_95 = "{:.3f}".format(data.get('mAP@0.5:0.95', 0)) if 'mAP@0.5:0.95' in data else 'N/A'
    fps = "{:.1f}".format(data.get('fps', 0)) if 'fps' in data else 'N/A'
    latency = "{:.1f}".format(data.get('latency_ms', 0)) if 'latency_ms' in data else 'N/A'
    
    row = "{:<25} {:<8} {:<12} {:<8} {:<12}".format(model, map50, map50_95, fps, latency)
    print(row)

print('\n📁 Detailed reports saved in: ./benchmark_results')
print('📊 JSON files: *.json')
print('📄 Markdown reports: *.md')
EOF

echo ""
echo "✅ All benchmarks completed!"
echo "📁 Results saved in: $OUTPUT_DIR"
echo ""
echo "📋 Next steps:"
echo "1. Review benchmark results in the output directory"
echo "2. Compare model performance metrics"
echo "3. Select best model for Jetson deployment"

# Fix pandas/numpy compatibility issue
echo ""
echo "🔧 If you encountered pandas/numpy errors, try:"
echo "   pip install --upgrade numpy pandas"
echo "   or"
echo "   pip uninstall pandas && pip install pandas"