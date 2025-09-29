#!/usr/bin/env python3
"""
Comprehensive Results Viewer for YOLO11 Railway Detection Benchmarks
Author: Gyeongjin Yang
Lab: AVLab, Chungbuk National University

View and compare multiple comprehensive benchmark results in a unified interface
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class ComprehensiveResultsViewer:
    def __init__(self, results_dir: str):
        """
        Initialize comprehensive results viewer

        Args:
            results_dir: Directory containing comprehensive result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results = []
        self.df = None

    def load_results(self, pattern: str = "comprehensive_*.json") -> List[Dict]:
        """Load all comprehensive results matching pattern"""
        result_files = list(self.results_dir.glob(pattern))

        if not result_files:
            print(f"❌ No result files found matching: {pattern}")
            return []

        print(f"📁 Found {len(result_files)} result files:")

        loaded_results = []
        for file_path in sorted(result_files):
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    result['source_file'] = file_path.name
                    loaded_results.append(result)
                    print(f"   ✅ {file_path.name}")
            except Exception as e:
                print(f"   ❌ {file_path.name}: {e}")

        self.results = loaded_results
        return loaded_results

    def create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        if not self.results:
            print("❌ No results loaded. Call load_results() first.")
            return pd.DataFrame()

        data_rows = []

        for result in self.results:
            model_info = result.get('model_info', {})
            performance = result.get('performance_benchmark', {})
            accuracy = result.get('accuracy_benchmark', {})

            # Extract key metrics
            row = {
                'Model': model_info.get('model_name', 'Unknown'),
                'Model_Type': self._extract_model_type(model_info.get('model_name', '')),
                'Precision': self._extract_precision(model_info.get('model_name', '')),
                'File_Size_MB': model_info.get('file_size', 0) / (1024 * 1024),
                'Parameters_M': model_info.get('parameters', 0) / 1e6,

                # Performance metrics
                'FPS': performance.get('fps', {}).get('mean', 0),
                'FPS_Std': performance.get('fps', {}).get('std', 0),
                'Latency_ms': performance.get('latency_ms', {}).get('mean', 0),
                'Latency_Std': performance.get('latency_ms', {}).get('std', 0),
                'GPU_Memory_MB': performance.get('memory_usage', {}).get('gpu_memory_mb', 0),
                'CPU_Memory_MB': performance.get('memory_usage', {}).get('cpu_memory_mb', 0),

                # Accuracy metrics
                'mAP50': accuracy.get('metrics', {}).get('mAP50', 0),
                'mAP50-95': accuracy.get('metrics', {}).get('mAP50-95', 0),
                'Precision_Score': accuracy.get('metrics', {}).get('precision', 0),
                'Recall': accuracy.get('metrics', {}).get('recall', 0),
                'F1': accuracy.get('metrics', {}).get('f1', 0),

                # Class-wise metrics
                'bird_nest_AP': accuracy.get('class_metrics', {}).get('bird_nest', {}).get('AP', 0),
                'plastic_bag_AP': accuracy.get('class_metrics', {}).get('plastic_bag', {}).get('AP', 0),
                'floating_object_AP': accuracy.get('class_metrics', {}).get('floating_object', {}).get('AP', 0),
                'balloon_AP': accuracy.get('class_metrics', {}).get('balloon', {}).get('AP', 0),

                # Composite score
                'Composite_Score': self._calculate_composite_score(performance, accuracy),

                # Meta information
                'Source_File': result.get('source_file', ''),
                'Timestamp': result.get('timestamp', '')
            }

            data_rows.append(row)

        self.df = pd.DataFrame(data_rows)
        return self.df

    def _extract_model_type(self, model_name: str) -> str:
        """Extract model type (yolo11s, yolo11n) from model name"""
        if 'yolo11s' in model_name.lower():
            return 'YOLO11s'
        elif 'yolo11n' in model_name.lower():
            return 'YOLO11n'
        else:
            return 'Unknown'

    def _extract_precision(self, model_name: str) -> str:
        """Extract precision (FP32, FP16, INT8) from model name"""
        name_lower = model_name.lower()
        if 'int8' in name_lower:
            return 'INT8'
        elif 'fp16' in name_lower:
            return 'FP16'
        else:
            return 'FP32'

    def _calculate_composite_score(self, performance: Dict, accuracy: Dict) -> float:
        """Calculate composite score balancing accuracy and speed"""
        fps = performance.get('fps', {}).get('mean', 0)
        map50 = accuracy.get('metrics', {}).get('mAP50', 0)

        # Normalized scores (0-100)
        fps_score = min(fps / 50.0 * 100, 100)  # Max 50 FPS = 100 points
        accuracy_score = map50 * 100  # mAP is already 0-1

        # Weighted composite: 60% accuracy, 40% speed
        composite = (accuracy_score * 0.6) + (fps_score * 0.4)
        return round(composite, 1)

    def display_summary_table(self) -> None:
        """Display formatted summary table"""
        if self.df is None or self.df.empty:
            print("❌ No data available. Load results first.")
            return

        print("\n" + "="*100)
        print("🏆 COMPREHENSIVE BENCHMARK RESULTS SUMMARY")
        print("="*100)

        # Select key columns for display
        display_cols = [
            'Model_Type', 'Precision', 'FPS', 'mAP50', 'mAP50-95',
            'GPU_Memory_MB', 'Composite_Score'
        ]

        display_df = self.df[display_cols].copy()

        # Format columns
        display_df['FPS'] = display_df['FPS'].round(1)
        display_df['mAP50'] = display_df['mAP50'].round(3)
        display_df['mAP50-95'] = display_df['mAP50-95'].round(3)
        display_df['GPU_Memory_MB'] = display_df['GPU_Memory_MB'].round(0)

        # Sort by composite score
        display_df = display_df.sort_values('Composite_Score', ascending=False)

        print(display_df.to_string(index=False))

        # Add performance badges
        print("\n🏅 PERFORMANCE RATINGS:")
        for idx, row in display_df.iterrows():
            score = row['Composite_Score']
            if score >= 95:
                badge = "🔥 EXCELLENT"
            elif score >= 90:
                badge = "⭐ GREAT"
            elif score >= 85:
                badge = "✅ GOOD"
            else:
                badge = "📊 OK"

            print(f"   {row['Model_Type']} {row['Precision']}: {score:.1f} {badge}")

    def display_detailed_comparison(self) -> None:
        """Display detailed comparison of top 3 models"""
        if self.df is None or self.df.empty:
            print("❌ No data available. Load results first.")
            return

        # Get top 3 models by composite score
        top_models = self.df.nlargest(3, 'Composite_Score')

        print("\n" + "="*120)
        print("🥇 TOP 3 MODELS DETAILED COMPARISON")
        print("="*120)

        for i, (idx, model) in enumerate(top_models.iterrows(), 1):
            rank_emoji = ["🥇", "🥈", "🥉"][i-1]

            print(f"\n{rank_emoji} RANK {i}: {model['Model_Type']} {model['Precision']} (Score: {model['Composite_Score']:.1f})")
            print("-" * 80)

            # Performance metrics
            print(f"⚡ PERFORMANCE:")
            print(f"   FPS: {model['FPS']:.1f} ± {model['FPS_Std']:.1f}")
            print(f"   Latency: {model['Latency_ms']:.1f} ± {model['Latency_Std']:.1f} ms")
            print(f"   GPU Memory: {model['GPU_Memory_MB']:.0f} MB")
            print(f"   File Size: {model['File_Size_MB']:.1f} MB")

            # Accuracy metrics
            print(f"\n🎯 ACCURACY:")
            print(f"   mAP@0.5: {model['mAP50']:.3f}")
            print(f"   mAP@0.5:0.95: {model['mAP50-95']:.3f}")
            print(f"   Precision: {model['Precision_Score']:.3f}")
            print(f"   Recall: {model['Recall']:.3f}")
            print(f"   F1-Score: {model['F1']:.3f}")

            # Class-wise performance
            print(f"\n🏷️ CLASS-WISE AP:")
            print(f"   Bird Nest: {model['bird_nest_AP']:.3f}")
            print(f"   Plastic Bag: {model['plastic_bag_AP']:.3f}")
            print(f"   Floating Object: {model['floating_object_AP']:.3f}")
            print(f"   Balloon: {model['balloon_AP']:.3f}")

    def create_comparison_plots(self, output_dir: str = "./comparison_plots") -> None:
        """Create comparison plots for the results"""
        if self.df is None or self.df.empty:
            print("❌ No data available. Load results first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Performance vs Accuracy Scatter Plot
        plt.figure(figsize=(12, 8))

        # Create scatter plot with different markers for model types
        for model_type in self.df['Model_Type'].unique():
            type_data = self.df[self.df['Model_Type'] == model_type]

            for precision in type_data['Precision'].unique():
                precision_data = type_data[type_data['Precision'] == precision]

                marker = 'o' if model_type == 'YOLO11s' else '^'
                size = 150 if precision == 'FP32' else (100 if precision == 'FP16' else 70)
                alpha = 1.0 if precision == 'FP32' else (0.8 if precision == 'FP16' else 0.6)

                plt.scatter(precision_data['FPS'], precision_data['mAP50'],
                          s=size, marker=marker, alpha=alpha,
                          label=f"{model_type} {precision}")

                # Add labels for each point
                for idx, row in precision_data.iterrows():
                    plt.annotate(f"{row['Model_Type']}\n{row['Precision']}",
                               (row['FPS'], row['mAP50']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)

        plt.xlabel('FPS (Frames Per Second)', fontsize=12)
        plt.ylabel('mAP@0.5', fontsize=12)
        plt.title('YOLO11 Railway Detection: Performance vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'performance_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Model Comparison Bar Charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Sort by composite score for consistent ordering
        sorted_df = self.df.sort_values('Composite_Score', ascending=False)
        model_labels = [f"{row['Model_Type']}\n{row['Precision']}" for _, row in sorted_df.iterrows()]

        # FPS comparison
        axes[0, 0].bar(model_labels, sorted_df['FPS'], color='skyblue', alpha=0.8)
        axes[0, 0].set_title('FPS Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # mAP@0.5 comparison
        axes[0, 1].bar(model_labels, sorted_df['mAP50'], color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('mAP@0.5 Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('mAP@0.5')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # GPU Memory comparison
        axes[1, 0].bar(model_labels, sorted_df['GPU_Memory_MB'], color='salmon', alpha=0.8)
        axes[1, 0].set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Composite Score comparison
        axes[1, 1].bar(model_labels, sorted_df['Composite_Score'], color='gold', alpha=0.8)
        axes[1, 1].set_title('Composite Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Class-wise Performance Heatmap
        class_cols = ['bird_nest_AP', 'plastic_bag_AP', 'floating_object_AP', 'balloon_AP']
        class_data = sorted_df[['Model_Type', 'Precision'] + class_cols].copy()

        # Create model labels
        class_data['Model_Label'] = class_data['Model_Type'] + '_' + class_data['Precision']

        # Prepare data for heatmap
        heatmap_data = class_data.set_index('Model_Label')[class_cols].T
        heatmap_data.index = ['Bird Nest', 'Plastic Bag', 'Floating Object', 'Balloon']

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f',
                   cbar_kws={'label': 'Average Precision (AP)'})
        plt.title('Class-wise Average Precision Heatmap', fontsize=14, fontweight='bold')
        plt.ylabel('Object Classes')
        plt.xlabel('Model Configurations')
        plt.tight_layout()
        plt.savefig(output_path / 'class_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Comparison plots saved to: {output_path}")
        print(f"   📈 performance_vs_accuracy.png")
        print(f"   📊 model_comparison_charts.png")
        print(f"   🔥 class_performance_heatmap.png")

    def export_to_csv(self, output_file: str = None) -> str:
        """Export results to CSV file"""
        if self.df is None or self.df.empty:
            print("❌ No data available. Load results first.")
            return ""

        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"comprehensive_comparison_{timestamp}.csv"

        self.df.to_csv(output_file, index=False)
        print(f"📁 Results exported to: {output_file}")
        return output_file

def parse_args():
    parser = argparse.ArgumentParser(description='View and compare comprehensive benchmark results')

    parser.add_argument('results_dir', type=str,
                       help='Directory containing comprehensive result JSON files')
    parser.add_argument('--pattern', type=str, default='comprehensive_*.json',
                       help='File pattern to match (default: comprehensive_*.json)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV file')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create comparison plots')
    parser.add_argument('--plot-dir', type=str, default='./comparison_plots',
                       help='Directory to save plots (default: ./comparison_plots)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed comparison of top 3 models')

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"🔍 COMPREHENSIVE RESULTS VIEWER")
    print(f"📁 Results directory: {args.results_dir}")
    print(f"🔍 Pattern: {args.pattern}")
    print("=" * 60)

    # Initialize viewer
    viewer = ComprehensiveResultsViewer(args.results_dir)

    # Load results
    results = viewer.load_results(args.pattern)
    if not results:
        print("❌ No results found. Exiting.")
        return

    # Create DataFrame
    df = viewer.create_dataframe()
    print(f"📊 Loaded {len(df)} benchmark results")

    # Display summary table
    viewer.display_summary_table()

    # Display detailed comparison if requested
    if args.detailed:
        viewer.display_detailed_comparison()

    # Export to CSV if requested
    if args.export_csv:
        viewer.export_to_csv()

    # Create plots if requested
    if args.create_plots:
        viewer.create_comparison_plots(args.plot_dir)

if __name__ == "__main__":
    main()