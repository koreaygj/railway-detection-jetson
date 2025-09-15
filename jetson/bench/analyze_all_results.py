#!/usr/bin/env python3
"""
모든 벤치마크 결과 종합 분석 스크립트
여러 모델의 성능과 정확도를 비교하고 시각화

사용법:
    python analyze_all_results.py ./comprehensive_results/ --save-report
    python analyze_all_results.py ./comprehensive_results/ --export-csv --plot
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultAnalyzer:
    def __init__(self, results_dir: str):
        """
        결과 분석기 초기화

        Args:
            results_dir: 결과 파일들이 있는 디렉토리
        """
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.summary_df = None

        logger.info(f"🔍 결과 분석기 초기화: {self.results_dir}")



    def load_all_results(self, pattern: str = "*.json") -> int:
        """모든 결과 파일 로드"""
        result_files = list(self.results_dir.glob(pattern))

        if not result_files:
            logger.warning(f"❌ 결과 파일을 찾을 수 없습니다: {self.results_dir}/{pattern}")
            return 0

        logger.info(f"📁 {len(result_files)}개 결과 파일 발견")

        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 메타 정보 추가
                data['_file_info'] = {
                    'filename': result_file.name,
                    'filepath': str(result_file),
                    'size': result_file.stat().st_size,
                    'modified': result_file.stat().st_mtime
                }

                self.results_data.append(data)
                logger.info(f"✅ 로드: {result_file.name}")

            except Exception as e:
                logger.warning(f"⚠️ 파일 로드 실패: {result_file.name} - {e}")

        logger.info(f"📊 총 {len(self.results_data)}개 결과 로드 완료")
        return len(self.results_data)

    def extract_summary_data(self) -> pd.DataFrame:
        """결과 데이터에서 요약 정보 추출"""
        summary_data = []

        for data in self.results_data:
            try:
                summary = self._extract_single_result(data)
                if summary:
                    summary_data.append(summary)
            except Exception as e:
                logger.warning(f"결과 추출 실패: {e}")

        if not summary_data:
            logger.error("❌ 추출 가능한 결과가 없습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(summary_data)

        # 정렬
        df = df.sort_values(['model_type', 'model_name'])

        self.summary_df = df
        logger.info(f"📋 요약 테이블 생성 완료: {len(df)} 행")
        return df

    def _extract_single_result(self, data: Dict) -> Optional[Dict]:
        """단일 결과에서 정보 추출"""
        summary = {}

        # 파일 정보
        file_info = data.get('_file_info', {})
        summary['filename'] = file_info.get('filename', 'unknown')

        # 모델 정보
        if 'benchmark_info' in data:
            # comprehensive_benchmark 결과
            model_info = data['benchmark_info']
            summary['model_name'] = Path(model_info.get('model_name', 'unknown')).stem
            summary['model_path'] = model_info.get('model_path', 'unknown')
            summary['timestamp'] = model_info.get('timestamp', 'unknown')
        elif 'model_info' in data:
            # 개별 벤치마크 결과
            model_info = data['model_info']
            summary['model_name'] = Path(model_info.get('file_name', model_info.get('model_name', 'unknown'))).stem
            summary['model_path'] = model_info.get('file_path', model_info.get('model_path', 'unknown'))
            summary['timestamp'] = data.get('system_info', {}).get('timestamp', 'unknown')
        else:
            summary['model_name'] = 'unknown'
            summary['model_path'] = 'unknown'
            summary['timestamp'] = 'unknown'

        # 모델 타입 결정
        model_name = summary['model_name'].lower()
        if 'engine' in model_name or 'trt' in model_name:
            if 'fp16' in model_name:
                summary['model_type'] = 'TensorRT-FP16'
            elif 'int8' in model_name:
                summary['model_type'] = 'TensorRT-INT8'
            else:
                summary['model_type'] = 'TensorRT'
        elif '.pt' in model_name or 'pytorch' in model_name:
            if 'fp16' in model_name:
                summary['model_type'] = 'PyTorch-FP16'
            elif 'int8' in model_name:
                summary['model_type'] = 'PyTorch-INT8'
            else:
                summary['model_type'] = 'PyTorch'
        else:
            summary['model_type'] = 'Unknown'

        # 모델 크기 추정
        if 'yolo11n' in model_name:
            summary['model_size'] = 'nano'
        elif 'yolo11s' in model_name:
            summary['model_size'] = 'small'
        elif 'yolo11m' in model_name:
            summary['model_size'] = 'medium'
        elif 'yolo11l' in model_name:
            summary['model_size'] = 'large'
        else:
            summary['model_size'] = 'unknown'

        # 성능 메트릭 추출
        if 'performance' in data and 'error' not in data['performance']:
            perf = data['performance'].get('performance', data['performance'])
            summary['fps'] = perf.get('fps', 0)
            summary['avg_latency_ms'] = perf.get('avg_latency_ms', 0)
            summary['throughput'] = perf.get('throughput', 0)
            summary['success_rate'] = perf.get('success_rate', 0)
        else:
            summary['fps'] = None
            summary['avg_latency_ms'] = None
            summary['throughput'] = None
            summary['success_rate'] = None

        # 정확도 메트릭 추출
        if 'accuracy' in data and 'error' not in data['accuracy']:
            acc = data['accuracy']
            summary['map50'] = acc.get('map50', 0)
            summary['map50_95'] = acc.get('map50_95', 0)
            summary['precision'] = acc.get('precision', 0)
            summary['recall'] = acc.get('recall', 0)
            summary['f1_score'] = acc.get('f1_score', 0)
        elif 'results' in data and 'basic_validation' in data['results']:
            # accuracy_benchmark 결과
            acc = data['results']['basic_validation']
            summary['map50'] = acc.get('map50', 0)
            summary['map50_95'] = acc.get('map50_95', 0)
            summary['precision'] = acc.get('precision', 0)
            summary['recall'] = acc.get('recall', 0)
            summary['f1_score'] = acc.get('f1_score', 0)
        else:
            summary['map50'] = None
            summary['map50_95'] = None
            summary['precision'] = None
            summary['recall'] = None
            summary['f1_score'] = None

        # 종합 점수 계산
        if summary['fps'] is not None and summary['map50'] is not None:
            # 성능 점수 (FPS 기준)
            fps_score = min(100, summary['fps'] * 5)  # 20 FPS = 100점

            # 정확도 점수 (mAP 기준)
            acc_score = summary['map50'] * 100

            # 종합 점수 (성능 40% + 정확도 60%)
            summary['overall_score'] = fps_score * 0.4 + acc_score * 0.6
        else:
            summary['overall_score'] = None

        return summary

    def generate_comparison_report(self) -> str:
        """비교 리포트 생성"""
        if self.summary_df is None or self.summary_df.empty:
            return "분석할 데이터가 없습니다."

        df = self.summary_df

        # 리포트 생성
        report = []
        report.append("# 🏆 종합 벤치마크 분석 리포트")
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"분석 모델 수: {len(df)}개")
        report.append("")

        # 1. 전체 요약
        report.append("## 📊 전체 요약")
        report.append("| 순위 | 모델 | 타입 | FPS | mAP@0.5 | 종합점수 | 평가 |")
        report.append("|------|------|------|-----|---------|----------|------|")

        # 종합 점수 기준 정렬
        valid_df = df.dropna(subset=['overall_score']).sort_values('overall_score', ascending=False)

        for idx, (_, row) in enumerate(valid_df.head(10).iterrows(), 1):
            grade = self._get_grade(row['overall_score'])
            report.append(f"| {idx} | {row['model_name']} | {row['model_type']} | "
                         f"{row['fps']:.1f} | {row['map50']:.3f} | {row['overall_score']:.1f} | {grade} |")

        report.append("")

        # 2. 성능 TOP 5
        report.append("## ⚡ 성능 TOP 5 (FPS)")
        perf_df = df.dropna(subset=['fps']).sort_values('fps', ascending=False)
        for idx, (_, row) in enumerate(perf_df.head(5).iterrows(), 1):
            report.append(f"{idx}. **{row['model_name']}** ({row['model_type']}) - {row['fps']:.1f} FPS")

        report.append("")

        # 3. 정확도 TOP 5
        report.append("## 🎯 정확도 TOP 5 (mAP@0.5)")
        acc_df = df.dropna(subset=['map50']).sort_values('map50', ascending=False)
        for idx, (_, row) in enumerate(acc_df.head(5).iterrows(), 1):
            report.append(f"{idx}. **{row['model_name']}** ({row['model_type']}) - {row['map50']:.3f}")

        report.append("")

        # 4. 모델 타입별 분석
        report.append("## 🔄 모델 타입별 비교")
        type_stats = df.groupby('model_type').agg({
            'fps': ['mean', 'std', 'count'],
            'map50': ['mean', 'std'],
            'overall_score': ['mean', 'std']
        }).round(3)

        for model_type in type_stats.index:
            stats = type_stats.loc[model_type]
            report.append(f"### {model_type}")
            report.append(f"- 모델 수: {stats[('fps', 'count')]}개")
            if not pd.isna(stats[('fps', 'mean')]):
                report.append(f"- 평균 FPS: {stats[('fps', 'mean')]:.1f} ± {stats[('fps', 'std')]:.1f}")
            if not pd.isna(stats[('map50', 'mean')]):
                report.append(f"- 평균 mAP@0.5: {stats[('map50', 'mean')]:.3f} ± {stats[('map50', 'std')]:.3f}")
            if not pd.isna(stats[('overall_score', 'mean')]):
                report.append(f"- 평균 종합점수: {stats[('overall_score', 'mean')]:.1f}")
            report.append("")

        # 5. 추천사항
        report.append("## 💡 추천사항")

        # 최고 성능
        if not perf_df.empty:
            best_perf = perf_df.iloc[0]
            report.append(f"- **최고 성능**: {best_perf['model_name']} ({best_perf['fps']:.1f} FPS)")

        # 최고 정확도
        if not acc_df.empty:
            best_acc = acc_df.iloc[0]
            report.append(f"- **최고 정확도**: {best_acc['model_name']} (mAP@0.5: {best_acc['map50']:.3f})")

        # 균형 잡힌 모델
        if not valid_df.empty:
            best_overall = valid_df.iloc[0]
            report.append(f"- **균형 잡힌 모델**: {best_overall['model_name']} (종합점수: {best_overall['overall_score']:.1f})")

        return "\n".join(report)

    def _get_grade(self, score: float) -> str:
        """점수에 따른 등급 반환"""
        if score >= 90:
            return "🔥 S급"
        elif score >= 80:
            return "🚀 A급"
        elif score >= 70:
            return "✅ B급"
        elif score >= 60:
            return "📊 C급"
        else:
            return "⚠️ D급"

    def create_visualizations(self, output_dir: str = "./analysis_plots"):
        """시각화 차트 생성"""
        if self.summary_df is None or self.summary_df.empty:
            logger.warning("시각화할 데이터가 없습니다.")
            return
    
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        df = self.summary_df.dropna(subset=['fps', 'map50'])
    
        if df.empty:
            logger.warning("시각화할 유효한 데이터가 없습니다.")
            return
    
        # ✅ 데이터 전처리 함수 추가
        def parse_model_name(model_name):
            """모델명에서 베이스명과 정밀도를 분리"""
            if '_int8' in model_name or 'int8' in model_name.lower():
                return model_name.replace('_int8', '').replace('-int8', ''), 'int8'
            elif '_fp16' in model_name or 'fp16' in model_name.lower():
                return model_name.replace('_fp16', '').replace('-fp16', ''), 'fp16'
            else:
                return model_name, 'fp32'
    
        # ✅ 데이터 전처리
        df_processed = df.copy()
        df_processed[['base_model', 'precision']] = df_processed['model_name'].apply(
            lambda x: pd.Series(parse_model_name(x))
        )
        
        precision_order = ['int8', 'fp16', 'fp32']
        df_processed['precision'] = pd.Categorical(
            df_processed['precision'], 
            categories=precision_order, 
            ordered=True
        )
    
        # 1. 성능 vs 정확도 산점도 (라인 연결 추가)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['fps'], df['map50'],
                             c=df['overall_score'],
                             s=100,
                             alpha=0.7,
                             cmap='viridis')
    
        # ✅ 같은 베이스 모델끼리 라인으로 연결
        base_models = df_processed['base_model'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(base_models)))
    
        for i, base_model in enumerate(base_models):
            model_data = df_processed[df_processed['base_model'] == base_model]
            
            if len(model_data) > 1:  # 연결할 점이 2개 이상일 때만
                # 정밀도 순서대로 정렬
                model_data = model_data.sort_values('precision')
                
                # 라인으로 연결
                plt.plot(model_data['fps'], model_data['map50'], 
                        '-', color=colors[i % len(colors)], 
                        alpha=0.5, linewidth=1.5)
    
        for idx, row in df.iterrows():
            plt.annotate(row['model_name'],
                        (row['fps'], row['map50']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8)
    
        plt.colorbar(scatter, label='종합 점수')
        plt.xlabel('FPS (성능)')
        plt.ylabel('mAP@0.5 (정확도)')
        plt.title('모델 성능 vs 정확도 비교 (같은 모델의 정밀도별 연결)')
        plt.grid(True, alpha=0.3)
    
        plot_path = output_dir / 'performance_vs_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 차트 저장: {plot_path}")
    
        # 2. 정밀도별 성능 비교 차트
        fig, ax2 = plt.subplots(figsize=(14, 8))
    
        sns.pointplot(data=df_processed, 
                      x='precision', y='map50', hue='base_model',
                      ax=ax2, dodge=0.3, join=True, 
                      markers='o', linestyles='-', linewidth=2, markersize=8)
    
        ax2.set_xlabel('정밀도 타입', fontsize=12)
        ax2.set_ylabel('mAP@0.5', fontsize=12)
        ax2.set_title('모델별 정밀도에 따른 mAP@0.5 (평균 ± 95% 신뢰구간)', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
    
        plot_path = output_dir / 'model_precision_seaborn.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 Seaborn 차트 저장: {plot_path}")
    
        # 3. 모델 타입별 박스 플롯
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        # FPS 박스 플롯
        df.boxplot(column='fps', by='model_type', ax=ax1)
        ax1.set_title('모델 타입별 FPS 분포')
        ax1.set_xlabel('모델 타입')
        ax1.set_ylabel('FPS')
    
        # mAP 박스 플롯
        df.boxplot(column='map50', by='model_type', ax=ax2)
        ax2.set_title('모델 타입별 mAP@0.5 분포')
        ax2.set_xlabel('모델 타입')
        ax2.set_ylabel('mAP@0.5')
    
        plt.suptitle('')  # 자동 제목 제거
    
        plot_path = output_dir / 'model_type_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 차트 저장: {plot_path}")
    
        # 4. 종합 점수 히트맵
        pivot_df = df.pivot_table(
            values='overall_score',
            index='model_size',
            columns='model_type',
            aggfunc='mean'
        )
    
        if not pivot_df.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd')
            plt.title('모델 크기 × 타입별 종합 점수')
    
            plot_path = output_dir / 'score_heatmap.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"📈 차트 저장: {plot_path}")
    
    def export_csv(self, output_path: str = "./benchmark_summary.csv"):
        """결과를 CSV로 내보내기"""
        if self.summary_df is None:
            logger.warning("내보낼 데이터가 없습니다.")
            return

        self.summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"💾 CSV 저장: {output_path}")

    def print_summary(self):
        """요약 정보 콘솔 출력"""
        if self.summary_df is None or self.summary_df.empty:
            print("분석할 데이터가 없습니다.")
            return

        df = self.summary_df

        print("\n" + "="*80)
        print("🏆 벤치마크 결과 종합 분석")
        print("="*80)

        # TOP 3 출력
        valid_df = df.dropna(subset=['overall_score']).sort_values('overall_score', ascending=False)

        if not valid_df.empty:
            print("\n🥇 TOP 3 모델 (종합 점수)")
            for idx, (_, row) in enumerate(valid_df.head(3).iterrows(), 1):
                grade = self._get_grade(row['overall_score'])
                print(f"{idx}. {row['model_name']} ({row['model_type']})")
                print(f"   종합점수: {row['overall_score']:.1f} {grade}")
                print(f"   FPS: {row['fps']:.1f}, mAP@0.5: {row['map50']:.3f}")
                print()

        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="벤치마크 결과 종합 분석")
    parser.add_argument('results_dir', help='결과 파일 디렉토리')
    parser.add_argument('--pattern', default='*.json', help='파일 패턴 (기본값: *.json)')
    parser.add_argument('--export-csv', action='store_true', help='CSV로 내보내기')
    parser.add_argument('--save-report', action='store_true', help='마크다운 리포트 저장')
    parser.add_argument('--plot', action='store_true', help='시각화 차트 생성')
    parser.add_argument('--output-dir', default='./analysis_output', help='출력 디렉토리')

    args = parser.parse_args()

    if not Path(args.results_dir).exists():
        logger.error(f"❌ 결과 디렉토리를 찾을 수 없습니다: {args.results_dir}")
        sys.exit(1)

    try:
        # 분석기 초기화
        analyzer = ResultAnalyzer(args.results_dir)

        # 결과 로드
        loaded_count = analyzer.load_all_results(args.pattern)
        if loaded_count == 0:
            logger.error("❌ 로드된 결과가 없습니다.")
            sys.exit(1)

        # 요약 데이터 추출
        summary_df = analyzer.extract_summary_data()
        if summary_df.empty:
            logger.error("❌ 분석 가능한 데이터가 없습니다.")
            sys.exit(1)

        # 콘솔 요약 출력
        analyzer.print_summary()

        # 출력 디렉토리 생성
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV 내보내기
        if args.export_csv:
            csv_path = output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            analyzer.export_csv(csv_path)

        # 리포트 저장
        if args.save_report:
            report = analyzer.generate_comparison_report()
            report_path = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📝 리포트 저장: {report_path}")

        # 시각화
        if args.plot:
            analyzer.create_visualizations(output_dir / 'plots')

        logger.info("🎉 분석 완료!")

    except KeyboardInterrupt:
        logger.info("❌ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 분석 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()