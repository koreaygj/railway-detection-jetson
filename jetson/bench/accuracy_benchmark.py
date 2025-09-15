#!/usr/bin/env python3
"""
정확도 벤치마크 스크립트
mAP, 정밀도, 재현율 등 정확도 메트릭 측정

사용법:
    python accuracy_benchmark.py data.yaml model.engine --save-results
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union

import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AccuracyBenchmark:
    def __init__(self, data_yaml: Union[str, Path], model_path: Union[str, Path]):
        """
        정확도 벤치마크 초기화

        Args:
            data_yaml: 데이터셋 YAML 파일 경로
            model_path: 모델 파일 경로 (.engine, .pt, .onnx)
        """
        self.data_yaml = Path(data_yaml)
        self.model_path = Path(model_path)

        # 모델 정보
        self.model_type = self._determine_model_type()

        logger.info(f"🎯 정확도 벤치마크 초기화")
        logger.info(f"   데이터셋: {self.data_yaml}")
        logger.info(f"   모델: {self.model_path.name} ({self.model_type})")

    def _determine_model_type(self) -> str:
        """모델 타입 결정"""
        suffix = self.model_path.suffix.lower()
        model_types = {
            '.engine': 'TensorRT',
            '.pt': 'PyTorch',
            '.pth': 'PyTorch',
            '.onnx': 'ONNX'
        }
        return model_types.get(suffix, 'Unknown')

    def run_validation(self,
                      device: str = '0',
                      conf_threshold: float = 0.001,  # 낮은 임계값으로 모든 검출 포함
                      iou_threshold: float = 0.6,     # NMS IOU 임계값
                      max_det: int = 300,             # 최대 검출 수
                      imgsz: int = 640,
                      save_dir: Optional[str] = None,
                      plots: bool = True,
                      verbose: bool = True) -> Dict:
        """
        YOLO 검증 실행 (mAP 계산)

        Args:
            device: GPU 디바이스
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IOU 임계값
            max_det: 최대 검출 수
            imgsz: 입력 이미지 크기
            save_dir: 결과 저장 디렉토리
            plots: 시각화 차트 생성
            verbose: 상세 로그

        Returns:
            Dict: 검증 결과
        """
        logger.info(f"🔍 YOLO 검증 실행 중...")
        logger.info(f"   신뢰도 임계값: {conf_threshold}")
        logger.info(f"   NMS IOU: {iou_threshold}")
        logger.info(f"   최대 검출: {max_det}")

        try:
            # 모델 로드
            model = YOLO(str(self.model_path))

            # 검증 실행
            validation_start = time.time()

            results = model.val(
                data=str(self.data_yaml),
                device=device,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                imgsz=imgsz,
                save_json=True,
                plots=plots,
                verbose=verbose,
                project=save_dir if save_dir else None,
                name='validation_results'
            )

            validation_time = time.time() - validation_start

            # 결과 추출
            metrics = self._extract_validation_metrics(results, validation_time)

            logger.info("✅ 검증 완료")
            self._print_accuracy_summary(metrics)

            return metrics

        except Exception as e:
            logger.error(f"❌ 검증 실패: {e}")
            return None

    def _extract_validation_metrics(self, results, validation_time: float) -> Dict:
        """검증 결과에서 메트릭 추출"""

        # 메인 메트릭
        metrics = {
            'validation_time': validation_time,
            'map50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'map50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            'f1_score': 0.0,  # 계산됨
        }

        # F1 스코어 계산
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])

        # 클래스별 mAP (가능한 경우)
        try:
            if hasattr(results.box, 'maps') and results.box.maps is not None:
                class_maps = results.box.maps.tolist() if hasattr(results.box.maps, 'tolist') else results.box.maps
                metrics['class_maps'] = class_maps

                # 클래스 이름과 매핑
                if hasattr(results, 'names') and results.names:
                    class_names = results.names
                    metrics['class_metrics'] = {}
                    for i, (class_id, class_name) in enumerate(class_names.items()):
                        if i < len(class_maps):
                            metrics['class_metrics'][class_name] = {
                                'map50': float(class_maps[i]),
                                'class_id': int(class_id)
                            }
        except Exception as e:
            logger.warning(f"클래스별 메트릭 추출 실패: {e}")

        # 추가 통계
        try:
            if hasattr(results.box, 'mp') and hasattr(results.box.mp, '__len__'):
                # 클래스별 정밀도/재현율
                precisions = results.box.mp.tolist() if hasattr(results.box.mp, 'tolist') else [results.box.mp]
                recalls = results.box.mr.tolist() if hasattr(results.box.mr, 'tolist') else [results.box.mr]

                metrics['per_class_precision'] = precisions
                metrics['per_class_recall'] = recalls
        except Exception as e:
            logger.warning(f"클래스별 정밀도/재현율 추출 실패: {e}")

        return metrics

    def run_speed_vs_accuracy_test(self,
                                  conf_thresholds: List[float] = [0.001, 0.01, 0.1, 0.25, 0.5],
                                  device: str = '0',
                                  max_images: int = 100) -> Dict:
        """
        다양한 신뢰도 임계값에서 속도 vs 정확도 테스트

        Args:
            conf_thresholds: 테스트할 신뢰도 임계값들
            device: GPU 디바이스
            max_images: 테스트할 최대 이미지 수

        Returns:
            Dict: 속도-정확도 트레이드오프 결과
        """
        logger.info(f"⚖️  속도 vs 정확도 테스트 시작")
        logger.info(f"   신뢰도 임계값: {conf_thresholds}")

        results = {}

        for conf in conf_thresholds:
            logger.info(f"🔍 신뢰도 임계값 {conf} 테스트 중...")

            try:
                # 정확도 측정
                accuracy_metrics = self.run_validation(
                    device=device,
                    conf_threshold=conf,
                    plots=False,
                    verbose=False
                )

                if accuracy_metrics:
                    results[f'conf_{conf}'] = {
                        'confidence_threshold': conf,
                        'map50': accuracy_metrics['map50'],
                        'map50_95': accuracy_metrics['map50_95'],
                        'precision': accuracy_metrics['precision'],
                        'recall': accuracy_metrics['recall'],
                        'f1_score': accuracy_metrics['f1_score'],
                        'validation_time': accuracy_metrics['validation_time']
                    }

            except Exception as e:
                logger.warning(f"신뢰도 {conf}에서 테스트 실패: {e}")
                continue

        return results

    def compare_models(self,
                      other_model_paths: List[Union[str, Path]],
                      device: str = '0') -> Dict:
        """
        여러 모델 정확도 비교

        Args:
            other_model_paths: 비교할 다른 모델들
            device: GPU 디바이스

        Returns:
            Dict: 모델 비교 결과
        """
        logger.info(f"🏆 모델 정확도 비교 시작")

        comparison_results = {}

        # 현재 모델
        logger.info(f"📊 기준 모델: {self.model_path.name}")
        base_metrics = self.run_validation(device=device, plots=False, verbose=False)

        if base_metrics:
            comparison_results[self.model_path.name] = {
                'model_path': str(self.model_path),
                'model_type': self.model_type,
                'metrics': base_metrics
            }

        # 다른 모델들
        for model_path in other_model_paths:
            model_path = Path(model_path)
            logger.info(f"📊 비교 모델: {model_path.name}")

            try:
                # 임시로 모델 경로 변경
                original_model = self.model_path
                original_type = self.model_type

                self.model_path = model_path
                self.model_type = self._determine_model_type()

                metrics = self.run_validation(device=device, plots=False, verbose=False)

                if metrics:
                    comparison_results[model_path.name] = {
                        'model_path': str(model_path),
                        'model_type': self.model_type,
                        'metrics': metrics
                    }

                # 원래 모델로 복원
                self.model_path = original_model
                self.model_type = original_type

            except Exception as e:
                logger.warning(f"모델 {model_path.name} 비교 실패: {e}")
                continue

        return comparison_results

    def _print_accuracy_summary(self, metrics: Dict):
        """정확도 결과 요약 출력"""
        print("\n" + "="*60)
        print("🎯 정확도 벤치마크 결과")
        print("="*60)
        print(f"🤖 모델: {self.model_path.name}")
        print(f"📊 모델 타입: {self.model_type}")
        print("")

        print("📈 정확도 메트릭:")
        print(f"   mAP@0.5:            {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        print(f"   mAP@0.5:0.95:       {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)")
        print(f"   정밀도 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   재현율 (Recall):    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"   F1 점수:           {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print("")

        print("⏱️  검증 시간:")
        print(f"   총 검증 시간:       {metrics['validation_time']:.2f}s")
        print("")

        # 클래스별 결과 (있는 경우)
        if 'class_metrics' in metrics:
            print("📋 클래스별 mAP@0.5:")
            for class_name, class_metric in metrics['class_metrics'].items():
                print(f"   {class_name}: {class_metric['map50']:.4f} ({class_metric['map50']*100:.2f}%)")
            print("")

        # 성능 평가
        map_score = metrics['map50']
        if map_score >= 0.7:
            rating = "🔥 매우 우수"
        elif map_score >= 0.5:
            rating = "🚀 우수"
        elif map_score >= 0.3:
            rating = "✅ 양호"
        elif map_score >= 0.1:
            rating = "📊 보통"
        else:
            rating = "⚠️ 개선 필요"

        print(f"🏆 종합 평가: {rating} (mAP@0.5: {map_score:.4f})")
        print("="*60)

    def save_results(self, results: Dict, output_dir: str = './accuracy_results',
                    custom_name: Optional[str] = None):
        """결과를 JSON 파일로 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 파일명 생성
        if custom_name:
            filename = f"accuracy_{custom_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            model_name = self.model_path.stem.replace('.', '_')
            filename = f"accuracy_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = output_dir / filename

        # 메타데이터 추가
        results_with_meta = {
            'model_info': {
                'model_path': str(self.model_path),
                'model_name': self.model_path.name,
                'model_type': self.model_type,
                'file_size': self.model_path.stat().st_size
            },
            'dataset_info': {
                'data_yaml': str(self.data_yaml)
            },
            'system_info': {
                'platform': platform.system(),
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }

        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 정확도 결과 저장: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="모델 정확도 벤치마크")
    parser.add_argument('data', help='데이터셋 YAML 파일 경로')
    parser.add_argument('model', help='모델 파일 경로 (.engine, .pt, .onnx)')

    # 검증 옵션
    parser.add_argument('--device', default='0', help='GPU 디바이스 (기본값: 0)')
    parser.add_argument('--conf', type=float, default=0.001, help='신뢰도 임계값 (기본값: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6, help='NMS IOU 임계값 (기본값: 0.6)')
    parser.add_argument('--imgsz', type=int, default=640, help='입력 이미지 크기 (기본값: 640)')
    parser.add_argument('--max-det', type=int, default=300, help='최대 검출 수 (기본값: 300)')

    # 테스트 옵션
    parser.add_argument('--speed-vs-accuracy', action='store_true', help='속도 vs 정확도 테스트')
    parser.add_argument('--compare-models', nargs='+', help='비교할 다른 모델들')

    # 출력 옵션
    parser.add_argument('--output-dir', default='./accuracy_results', help='결과 저장 디렉토리')
    parser.add_argument('--name', help='결과 파일 커스텀 이름')
    parser.add_argument('--save-results', action='store_true', help='결과를 JSON으로 저장')
    parser.add_argument('--no-plots', action='store_true', help='시각화 차트 생성 안함')

    args = parser.parse_args()

    # 파일 존재 확인
    if not Path(args.data).exists():
        logger.error(f"❌ 데이터셋 YAML을 찾을 수 없습니다: {args.data}")
        sys.exit(1)

    if not Path(args.model).exists():
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)

    try:
        # 정확도 벤치마크 초기화
        benchmark = AccuracyBenchmark(args.data, args.model)

        # 기본 검증 실행
        logger.info("🎯 기본 정확도 검증 실행 중...")
        results = benchmark.run_validation(
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            plots=not args.no_plots,
            save_dir=args.output_dir if args.save_results else None
        )

        if results is None:
            logger.error("❌ 기본 검증 실패")
            sys.exit(1)

        final_results = {'basic_validation': results}

        # 속도 vs 정확도 테스트
        if args.speed_vs_accuracy:
            logger.info("⚖️  속도 vs 정확도 테스트 실행 중...")
            speed_accuracy_results = benchmark.run_speed_vs_accuracy_test(device=args.device)
            final_results['speed_vs_accuracy'] = speed_accuracy_results

        # 모델 비교
        if args.compare_models:
            logger.info("🏆 모델 비교 실행 중...")
            comparison_results = benchmark.compare_models(args.compare_models, device=args.device)
            final_results['model_comparison'] = comparison_results

        # 결과 저장
        if args.save_results:
            benchmark.save_results(final_results, args.output_dir, args.name)

        logger.info("🎉 정확도 벤치마크 완료!")

    except KeyboardInterrupt:
        logger.info("❌ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 벤치마크 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()