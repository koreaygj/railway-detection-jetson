#!/usr/bin/env python3
"""
YOLO 모델 TensorRT 변환 CLI 스크립트
YOLO의 export 기능을 사용하여 FP16, INT8 변환 지원

사용법:
    python yolo_export_cli.py model.pt --format engine --precision fp16
    python yolo_export_cli.py model.pt --format engine --precision int8 --calibration-data ./images/
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_yolo_model(
    model_path: str,
    format: str = "engine",
    precision: str = "fp16",
    imgsz: int = 640,
    batch: int = 1,
    device: str = "0",
    workspace: int = 4,
    calibration_data: str = None,
    simplify: bool = True,
    output_dir: str = None,
    verbose: bool = True
):
    """
    YOLO 모델을 TensorRT 엔진으로 변환

    Args:
        model_path: 입력 모델 경로 (.pt, .onnx)
        format: 출력 형식 (engine, onnx, etc.)
        precision: 정밀도 (fp32, fp16, int8)
        imgsz: 입력 이미지 크기
        batch: 배치 크기
        device: GPU 디바이스
        workspace: TensorRT 작업공간 크기 (GB)
        calibration_data: INT8 캘리브레이션 데이터 경로
        simplify: ONNX 단순화 여부
        output_dir: 출력 디렉토리
        verbose: 상세 로그 출력

    Returns:
        str: 변환된 모델 경로
    """

    # 입력 검증
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    logger.info(f"🚀 YOLO 모델 변환 시작")
    logger.info(f"   입력 모델: {model_path}")
    logger.info(f"   출력 형식: {format}")
    logger.info(f"   정밀도: {precision}")
    logger.info(f"   이미지 크기: {imgsz}")
    logger.info(f"   배치 크기: {batch}")
    logger.info(f"   디바이스: {device}")

    try:
        # YOLO 모델 로드
        logger.info("📦 YOLO 모델 로드 중...")
        model = YOLO(str(model_path))
        logger.info("✅ 모델 로드 완료")

        # 출력 디렉토리 설정
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(output_dir)

        # 변환 옵션 설정
        export_kwargs = {
            'format': format,
            'imgsz': imgsz,
            'device': device,
            'verbose': verbose,
            'simplify': simplify
        }

        # TensorRT 전용 옵션
        if format == 'engine':
            export_kwargs.update({
                'workspace': workspace,
                'batch': batch,
            })

            # 정밀도 설정
            if precision == 'fp16':
                export_kwargs['half'] = True
                logger.info("🔥 FP16 정밀도 활성화")
            elif precision == 'int8':
                export_kwargs['int8'] = True
                logger.info("🔥 INT8 정밀도 활성화")

                # INT8 캘리브레이션 데이터 설정
                if calibration_data:
                    if not Path(calibration_data).exists():
                        logger.warning(f"캘리브레이션 데이터를 찾을 수 없습니다: {calibration_data}")
                    else:
                        export_kwargs['data'] = calibration_data
                        logger.info(f"📊 캘리브레이션 데이터: {calibration_data}")
                else:
                    logger.warning("INT8 변환이지만 캘리브레이션 데이터가 지정되지 않았습니다.")
                    logger.warning("기본 캘리브레이션을 사용합니다.")

        # 모델 변환 실행
        logger.info("🔄 모델 변환 실행 중...")
        exported_model = model.export(**export_kwargs)

        # 결과 출력
        if isinstance(exported_model, str):
            output_path = Path(exported_model)
        else:
            # export 결과가 모델 객체인 경우
            output_path = Path(str(exported_model))

        logger.info("✅ 모델 변환 완료!")
        logger.info(f"   출력 파일: {output_path}")
        logger.info(f"   파일 크기: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return str(output_path)

    except Exception as e:
        logger.error(f"❌ 모델 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(
        description="YOLO 모델 TensorRT 변환 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # FP16 TensorRT 엔진 변환
  python yolo_export_cli.py yolo11n.pt --format engine --precision fp16

  # INT8 TensorRT 엔진 변환 (캘리브레이션 데이터 포함)
  python yolo_export_cli.py yolo11n.pt --format engine --precision int8 --calibration-data ./val/images/

  # 커스텀 설정
  python yolo_export_cli.py yolo11n.pt --format engine --precision fp16 --imgsz 416 --batch 4 --workspace 8

  # ONNX 변환
  python yolo_export_cli.py yolo11n.pt --format onnx --precision fp16
        """
    )

    # 필수 인자
    parser.add_argument('model', help='입력 모델 경로 (.pt, .onnx)')

    # 변환 옵션
    parser.add_argument('--format', default='engine',
                       choices=['engine', 'onnx', 'torchscript', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle'],
                       help='출력 형식 (기본값: engine)')

    parser.add_argument('--precision', default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='정밀도 (기본값: fp16)')

    # 모델 설정
    parser.add_argument('--imgsz', type=int, default=640,
                       help='입력 이미지 크기 (기본값: 640)')

    parser.add_argument('--batch', type=int, default=1,
                       help='배치 크기 (기본값: 1)')

    # 하드웨어 설정
    parser.add_argument('--device', default='0',
                       help='GPU 디바이스 (기본값: 0)')

    parser.add_argument('--workspace', type=int, default=4,
                       help='TensorRT 작업공간 크기 GB (기본값: 4)')

    # INT8 전용 옵션
    parser.add_argument('--calibration-data',
                       help='INT8 캘리브레이션용 이미지 디렉토리')

    # 기타 옵션
    parser.add_argument('--output-dir',
                       help='출력 디렉토리 (기본값: 현재 디렉토리)')

    parser.add_argument('--no-simplify', action='store_true',
                       help='ONNX 단순화 비활성화')

    parser.add_argument('--quiet', action='store_true',
                       help='상세 로그 비활성화')

    args = parser.parse_args()

    # 인자 검증
    if args.precision == 'int8' and args.format == 'engine' and not args.calibration_data:
        logger.warning("⚠️  INT8 TensorRT 변환에는 캘리브레이션 데이터를 권장합니다.")
        logger.warning("   --calibration-data 옵션을 사용하여 이미지 디렉토리를 지정하세요.")

    # 변환 실행
    try:
        result = export_yolo_model(
            model_path=args.model,
            format=args.format,
            precision=args.precision,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workspace=args.workspace,
            calibration_data=args.calibration_data,
            simplify=not args.no_simplify,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )

        if result:
            print(f"\n🎉 변환 완료!")
            print(f"📁 출력 파일: {result}")

            # 사용 예시 출력
            if args.format == 'engine':
                print(f"\n💡 사용 예시:")
                print(f"   # YOLO 추론")
                print(f"   model = YOLO('{result}')")
                print(f"   results = model.predict('image.jpg')")
                print(f"   ")
                print(f"   # 직접 TensorRT 벤치마크")
                print(f"   python direct_tensorrt_benchmark.py ./data/data.yaml {result} --max-images 50")
        else:
            print("❌ 변환 실패")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("❌ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 변환 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()