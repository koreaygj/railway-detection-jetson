#!/usr/bin/env python3
"""
영어 클래스명을 가진 YOLO 모델로부터 TensorRT 엔진 재생성
"""

import subprocess
import sys
from pathlib import Path

def export_to_tensorrt(model_path, precision='fp16'):
    """
    YOLO 모델을 TensorRT 엔진으로 변환
    """
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return False

    print(f"🔧 {model_path.name} -> TensorRT {precision.upper()} 변환 시작...")

    try:
        if precision == 'fp16':
            cmd = [
                'yolo', 'export',
                f'model={model_path}',
                'format=engine',
                'half=True',
                'device=0',
                'workspace=4',
                'verbose=False'
            ]
        elif precision == 'int8':
            cmd = [
                'yolo', 'export',
                f'model={model_path}',
                'format=engine',
                'int8=True',
                'device=0',
                'workspace=4',
                'data=data_english.yaml',  # 영어 설정 파일 사용
                'verbose=False'
            ]
        else:
            print(f"❌ 지원하지 않는 정밀도: {precision}")
            return False

        print(f"실행 명령: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {precision.upper()} 엔진 생성 완료")
            return True
        else:
            print(f"❌ 변환 실패:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """
    영어 모델들로부터 TensorRT 엔진 생성
    """
    models = [
        "convert/model/yolo11n_english.pt",
        "convert/model/yolo11s_english.pt"
    ]

    precisions = ['fp16', 'int8']

    print("🚀 영어 클래스명 TensorRT 엔진 생성 시작")
    print("=" * 50)

    for model_path in models:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"⚠️ 모델 파일을 찾을 수 없음: {model_path}")
            continue

        print(f"\n📦 처리 중: {model_path.name}")

        for precision in precisions:
            success = export_to_tensorrt(model_path, precision)
            if success:
                # 생성된 엔진 파일 확인 및 이름 변경
                generated_engine = model_path.with_suffix('.engine')
                if generated_engine.exists():
                    new_name = f"{model_path.stem}_{precision}.engine"
                    new_path = model_path.parent / new_name
                    generated_engine.rename(new_path)
                    print(f"📁 저장됨: {new_path}")

            print("-" * 30)

    print("\n✨ 모든 변환 작업 완료")

    # 생성된 파일들 확인
    print("\n📋 생성된 영어 TensorRT 엔진 파일들:")
    for engine_file in Path("convert/model").glob("*english*.engine"):
        size_mb = engine_file.stat().st_size / (1024 * 1024)
        print(f"  {engine_file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()