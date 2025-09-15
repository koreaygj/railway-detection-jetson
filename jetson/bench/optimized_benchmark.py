#!/usr/bin/env python3
"""
최적화된 단일 모델 벤치마크 스크립트
YOLO predict 방식을 사용하여 실제 배포 환경과 동일한 성능 측정

Author: Claude Code
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

class OptimizedBenchmark:
    def __init__(self, dataset_path: Union[str, Path], model_path: Union[str, Path]):
        """
        최적화된 벤치마크 초기화

        Args:
            dataset_path: 데이터셋 경로 (YAML 파일 또는 이미지 디렉토리)
            model_path: 모델 파일 경로 (.engine, .pt, .onnx)
        """
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)

        # 시스템 정보 수집
        self.system_info = self._collect_system_info()

        # 데이터셋 로드
        self.image_paths = self._load_dataset()

        # 모델 정보
        self.model_type = self._determine_model_type()
        self.model_info = self._collect_model_info()

        logger.info(f"🚀 최적화된 벤치마크 초기화 완료")
        logger.info(f"   모델: {self.model_path.name} ({self.model_type})")
        logger.info(f"   데이터셋: {len(self.image_paths)}개 이미지")
        logger.info(f"   플랫폼: {self.system_info['platform']}")

    def _collect_system_info(self) -> Dict:
        """시스템 정보 수집"""
        import psutil

        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'timestamp': datetime.now().isoformat()
        }

        # GPU 정보 (가능한 경우)
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            system_info['gpu_count'] = gpu_count

            gpu_info = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info.append({
                    'name': name,
                    'memory_total': memory.total,
                    'memory_free': memory.free
                })
            system_info['gpu_info'] = gpu_info
        except ImportError:
            system_info['gpu_info'] = "pynvml not available"
        except Exception as e:
            system_info['gpu_info'] = f"Error: {str(e)}"

        return system_info

    def _load_dataset(self) -> List[Path]:
        """데이터셋에서 이미지 경로 로드 (YOLO 호환 방식)"""
        image_paths = []

        if self.dataset_path.suffix.lower() == '.yaml':
            # YAML 데이터셋 - YOLO 호환 검증 먼저 수행
            logger.info(f"YAML 데이터셋 로드 중: {self.dataset_path}")

            # YOLO 호환 경로 해석 (accuracy_benchmark와 동일 방식)
            logger.info("YOLO 호환 경로 해석 중...")

            # YAML 파일에서 경로 정보 추출
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)

            # 데이터셋 경로 해석 (YOLO 방식)
            dataset_root = Path(data_config.get('path', '.'))
            if not dataset_root.is_absolute():
                # YAML 파일 기준으로 상대 경로 해석
                dataset_root = self.dataset_path.parent / dataset_root

            # 절대 경로로 변환
            dataset_root = dataset_root.resolve()

            # 검증 이미지 경로
            val_path = data_config.get('val', 'val/images')
            val_images_dir = dataset_root / val_path

            logger.info(f"데이터셋 루트: {dataset_root}")
            logger.info(f"검증 이미지 디렉토리: {val_images_dir}")

            # 경로 존재 확인
            if not val_images_dir.exists():
                # 다른 가능한 경로들 시도
                possible_paths = [
                    dataset_root / 'val' / 'images',
                    dataset_root / 'valid' / 'images',
                    dataset_root / 'validation' / 'images',
                    dataset_root / 'test' / 'images',
                    self.dataset_path.parent / 'val' / 'images',
                    self.dataset_path.parent / 'data' / 'val' / 'images'
                ]

                for possible_path in possible_paths:
                    if possible_path.exists():
                        logger.info(f"✅ 대체 검증 경로 발견: {possible_path}")
                        val_images_dir = possible_path
                        break
                else:
                    raise ValueError(f"검증 이미지 디렉토리를 찾을 수 없습니다: {val_images_dir}")

            # 이미지 파일 찾기
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            for ext in image_extensions:
                image_paths.extend(val_images_dir.rglob(f"*{ext}"))
                image_paths.extend(val_images_dir.rglob(f"*{ext.upper()}"))

        elif self.dataset_path.is_dir():
            # 이미지 디렉토리
            logger.info(f"이미지 디렉토리 로드 중: {self.dataset_path}")

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            for ext in image_extensions:
                image_paths.extend(self.dataset_path.rglob(f"*{ext}"))
                image_paths.extend(self.dataset_path.rglob(f"*{ext.upper()}"))
        else:
            raise ValueError(f"지원되지 않는 데이터셋 형식: {self.dataset_path}")

        if not image_paths:
            raise ValueError(f"이미지를 찾을 수 없습니다: {self.dataset_path}")

        logger.info(f"📁 총 {len(image_paths)}개 이미지 발견")
        return sorted(image_paths)

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

    def _collect_model_info(self) -> Dict:
        """모델 정보 수집"""
        model_info = {
            'file_path': str(self.model_path),
            'file_name': self.model_path.name,
            'file_size': self.model_path.stat().st_size,
            'model_type': self.model_type,
            'created_time': datetime.fromtimestamp(self.model_path.stat().st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(self.model_path.stat().st_mtime).isoformat()
        }

        return model_info

    def run_benchmark(self,
                     max_images: Optional[int] = None,
                     warmup_runs: int = 5,
                     device: str = '0',
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     imgsz: int = 640) -> Dict:
        """
        최적화된 벤치마크 실행

        Args:
            max_images: 최대 테스트 이미지 수
            warmup_runs: 워밍업 실행 횟수
            device: 실행 디바이스
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IOU 임계값
            imgsz: 입력 이미지 크기

        Returns:
            Dict: 벤치마크 결과
        """
        logger.info(f"🏃 최적화된 벤치마크 시작")
        logger.info(f"   모델: {self.model_path.name}")
        logger.info(f"   디바이스: {device}")
        logger.info(f"   이미지 크기: {imgsz}")
        logger.info(f"   워밍업: {warmup_runs}회")

        # 모델 로드
        logger.info("📦 모델 로드 중...")
        model_load_start = time.time()

        try:
            model = YOLO(str(self.model_path))
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return None

        model_load_time = time.time() - model_load_start
        logger.info(f"✅ 모델 로드 완료 ({model_load_time:.2f}s)")

        # 테스트 이미지 선택
        test_images = self.image_paths[:max_images] if max_images else self.image_paths
        logger.info(f"📊 테스트 이미지: {len(test_images)}개")

        # 워밍업
        logger.info(f"🔥 워밍업 실행 중... ({warmup_runs}회)")
        dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        warmup_start = time.time()
        for i in range(warmup_runs):
            try:
                _ = model.predict(
                    dummy_img,
                    device=device,
                    verbose=False,
                    save=False,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=imgsz
                )
                if (i + 1) % 2 == 0:
                    logger.info(f"  워밍업 진행: {i + 1}/{warmup_runs}")
            except Exception as e:
                logger.warning(f"워밍업 중 오류: {e}")

        warmup_time = time.time() - warmup_start
        logger.info(f"✅ 워밍업 완료 ({warmup_time:.2f}s)")

        # 메인 벤치마크
        logger.info("⚡ 메인 벤치마크 실행 중...")

        # 측정 변수
        inference_times = []
        preprocessing_times = []
        postprocessing_times = []
        total_times = []
        detection_counts = []
        successful_inferences = 0
        failed_inferences = 0

        benchmark_start = time.time()

        # 이미지별 추론
        for i, img_path in enumerate(tqdm(test_images, desc="Processing")):
            try:
                # 이미지 로드
                img_load_start = time.time()
                image = cv2.imread(str(img_path))
                if image is None:
                    failed_inferences += 1
                    continue
                img_load_time = time.time() - img_load_start

                # 추론 실행 (TensorRT 안전 모드)
                total_start = time.time()

                try:
                    # TensorRT 엔진인 경우 특별 처리
                    if str(self.model_path).endswith('.engine'):
                        # TensorRT 엔진에 대해 더 관대한 설정
                        results = model.predict(
                            image,
                            device=device,
                            verbose=False,
                            save=False,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=imgsz,
                            half=False,  # TensorRT는 이미 최적화됨
                            augment=False,  # 데이터 증강 비활성화
                            agnostic_nms=False,  # 클래스별 NMS
                            max_det=300  # 최대 검출 수 제한
                        )
                    else:
                        # PyTorch 모델 기본 설정
                        results = model.predict(
                            image,
                            device=device,
                            verbose=False,
                            save=False,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=imgsz
                        )

                    total_time = time.time() - total_start

                    # 결과 처리 (안전한 방식)
                    num_detections = 0
                    if results and len(results) > 0:
                        result = results[0]

                        # 여러 방식으로 검출 수 확인
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            try:
                                num_detections = len(result.boxes)
                            except:
                                num_detections = 0
                        else:
                            # 결과가 있다고 가정하고 1개로 계산
                            num_detections = 1 if results else 0

                except Exception as predict_error:
                    # predict 메소드 자체에서 오류 발생
                    total_time = time.time() - total_start
                    num_detections = 0

                    # 오류 타입에 따라 다른 처리
                    if "'images'" in str(predict_error):
                        logger.warning(f"TensorRT 출력 구조 문제로 건너뛰기: {img_path.name}")
                        failed_inferences += 1
                        continue
                    else:
                        logger.warning(f"추론 오류: {predict_error}")
                        failed_inferences += 1
                        continue

                # 시간 기록
                total_times.append(total_time)
                detection_counts.append(num_detections)
                successful_inferences += 1

                # 진행 상황 출력 (매 20개마다)
                if (i + 1) % 20 == 0:
                    recent_fps = 1.0 / np.mean(total_times[-20:])
                    logger.info(f"  진행: {i+1}/{len(test_images)} - 현재 FPS: {recent_fps:.1f}")

            except Exception as outer_e:
                # 최외곽 예외 처리 (이미지 로드 등)
                failed_inferences += 1
                logger.warning(f"이미지 {img_path.name} 외부 처리 실패: {outer_e}")
                continue

        benchmark_time = time.time() - benchmark_start

        # 결과 계산
        if not total_times:
            logger.error("❌ 성공한 추론이 없습니다.")
            return None

        # 통계 계산
        total_times_np = np.array(total_times)
        detection_counts_np = np.array(detection_counts)

        # 성능 메트릭
        performance_metrics = {
            'total_images_processed': successful_inferences,
            'failed_inferences': failed_inferences,
            'success_rate': successful_inferences / len(test_images),

            # 지연시간 통계 (ms)
            'avg_latency_ms': float(np.mean(total_times_np) * 1000),
            'std_latency_ms': float(np.std(total_times_np) * 1000),
            'min_latency_ms': float(np.min(total_times_np) * 1000),
            'max_latency_ms': float(np.max(total_times_np) * 1000),
            'p50_latency_ms': float(np.percentile(total_times_np, 50) * 1000),
            'p95_latency_ms': float(np.percentile(total_times_np, 95) * 1000),
            'p99_latency_ms': float(np.percentile(total_times_np, 99) * 1000),

            # FPS 통계
            'fps': float(1.0 / np.mean(total_times_np)),
            'min_fps': float(1.0 / np.max(total_times_np)),
            'max_fps': float(1.0 / np.min(total_times_np)),
            'throughput': float(successful_inferences / benchmark_time),

            # 검출 통계
            'avg_detections_per_image': float(np.mean(detection_counts_np)),
            'total_detections': int(np.sum(detection_counts_np)),
            'max_detections_per_image': int(np.max(detection_counts_np)),
        }

        # 전체 결과 구성
        results = {
            'model_info': self.model_info,
            'performance': performance_metrics,
            'benchmark_config': {
                'max_images': max_images,
                'warmup_runs': warmup_runs,
                'device': device,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'imgsz': imgsz,
                'model_load_time': model_load_time,
                'warmup_time': warmup_time,
                'total_benchmark_time': benchmark_time,
                'timestamp': datetime.now().isoformat()
            },
            'dataset_info': {
                'dataset_path': str(self.dataset_path),
                'total_images_available': len(self.image_paths),
                'images_tested': len(test_images)
            },
            'system_info': self.system_info
        }

        # 결과 출력
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """벤치마크 결과 요약 출력"""
        perf = results['performance']
        config = results['benchmark_config']

        print("\n" + "="*60)
        print("📊 최적화된 벤치마크 결과")
        print("="*60)
        print(f"🤖 모델: {results['model_info']['file_name']}")
        print(f"🎯 모델 타입: {results['model_info']['model_type']}")
        print(f"📏 이미지 크기: {config['imgsz']}x{config['imgsz']}")
        print(f"🖥️  디바이스: {config['device']}")
        print("")

        print("⚡ 성능 메트릭:")
        print(f"   평균 FPS:           {perf['fps']:.2f}")
        print(f"   처리량:             {perf['throughput']:.2f} images/sec")
        print(f"   평균 지연시간:       {perf['avg_latency_ms']:.2f} ± {perf['std_latency_ms']:.2f} ms")
        print(f"   P95 지연시간:        {perf['p95_latency_ms']:.2f} ms")
        print(f"   P99 지연시간:        {perf['p99_latency_ms']:.2f} ms")
        print("")

        print("📊 처리 통계:")
        print(f"   성공한 추론:         {perf['total_images_processed']}")
        print(f"   실패한 추론:         {perf['failed_inferences']}")
        print(f"   성공률:             {perf['success_rate']*100:.1f}%")
        print(f"   평균 검출 수:        {perf['avg_detections_per_image']:.2f}")
        print(f"   총 검출 수:          {perf['total_detections']}")
        print("")

        print("⏱️  시간 정보:")
        print(f"   모델 로드:          {config['model_load_time']:.2f}s")
        print(f"   워밍업:             {config['warmup_time']:.2f}s")
        print(f"   총 벤치마크:         {config['total_benchmark_time']:.2f}s")
        print("="*60)

    def save_results(self, results: Dict, output_dir: str = './benchmark_results', custom_name: Optional[str] = None):
        """결과를 JSON 파일로 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 파일명 생성
        if custom_name:
            filename = f"optimized_benchmark_{custom_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            model_name = Path(results['model_info']['file_name']).stem
            model_name_clean = model_name.replace('.', '_')
            filename = f"optimized_benchmark_{model_name_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = output_dir / filename

        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 결과 저장: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="최적화된 단일 모델 벤치마크")
    parser.add_argument('dataset', help='데이터셋 경로 (YAML 파일 또는 이미지 디렉토리)')
    parser.add_argument('model', help='모델 파일 경로 (.engine, .pt, .onnx)')
    parser.add_argument('--max-images', type=int, help='최대 테스트 이미지 수')
    parser.add_argument('--warmup', type=int, default=5, help='워밍업 실행 횟수 (기본값: 5)')
    parser.add_argument('--device', default='0', help='실행 디바이스 (기본값: 0)')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값 (기본값: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU 임계값 (기본값: 0.45)')
    parser.add_argument('--imgsz', type=int, default=640, help='입력 이미지 크기 (기본값: 640)')
    parser.add_argument('--output-dir', default='./benchmark_results', help='결과 저장 디렉토리')
    parser.add_argument('--name', help='결과 파일 커스텀 이름')
    parser.add_argument('--no-save', action='store_true', help='결과 저장 안함')

    args = parser.parse_args()

    # 파일 존재 확인
    if not Path(args.dataset).exists():
        logger.error(f"❌ 데이터셋을 찾을 수 없습니다: {args.dataset}")
        sys.exit(1)

    if not Path(args.model).exists():
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)

    try:
        # 벤치마크 초기화
        benchmark = OptimizedBenchmark(args.dataset, args.model)

        # 벤치마크 실행
        results = benchmark.run_benchmark(
            max_images=args.max_images,
            warmup_runs=args.warmup,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz
        )

        if results is None:
            logger.error("❌ 벤치마크 실패")
            sys.exit(1)

        # 결과 저장
        if not args.no_save:
            benchmark.save_results(results, args.output_dir, args.name)

        logger.info("🎉 최적화된 벤치마크 완료!")

    except KeyboardInterrupt:
        logger.info("❌ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 벤치마크 실행 중 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()