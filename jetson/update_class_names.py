#!/usr/bin/env python3
"""
모델 클래스 이름을 중국어에서 영어로 변경하는 스크립트
"""

import torch
import yaml
from pathlib import Path

def update_model_class_names(model_path, output_path=None):
    """
    모델 파일의 클래스 이름을 중국어에서 영어로 변경
    """
    # 기존 중국어 -> 영어 매핑
    class_mapping = {
        'niaocao': 'bird_nest',
        'suliaodai': 'plastic_bag',
        'piaofuwu': 'floating_object',
        'qiqiu': 'balloon'
    }

    print(f"로딩 중: {model_path}")

    # 모델 로드 (weights_only=False로 설정하여 YOLO 모델 로드)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 모델 메타데이터에서 클래스 이름 확인 및 수정
    if 'model' in checkpoint:
        model = checkpoint['model']

        # names 속성 찾기 및 수정
        if hasattr(model, 'names') and model.names:
            print("기존 클래스 이름:", model.names)

            # 클래스 이름 영어로 변경
            new_names = {}
            for idx, name in model.names.items():
                if name in class_mapping:
                    new_names[idx] = class_mapping[name]
                    print(f"  {name} -> {class_mapping[name]}")
                else:
                    new_names[idx] = name

            model.names = new_names
            print("새 클래스 이름:", model.names)

    # 체크포인트에서도 names 정보 수정
    if 'names' in checkpoint:
        print("체크포인트에서 기존 names:", checkpoint['names'])
        new_names_list = []
        for name in checkpoint['names']:
            if name in class_mapping:
                new_names_list.append(class_mapping[name])
            else:
                new_names_list.append(name)

        checkpoint['names'] = new_names_list
        print("체크포인트에서 새 names:", checkpoint['names'])

    # 출력 경로 설정
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_english.pt"

    # 수정된 모델 저장
    print(f"저장 중: {output_path}")
    torch.save(checkpoint, output_path)
    print("완료!")

    return output_path

def create_english_data_yaml():
    """
    영어 클래스 이름으로 된 data.yaml 파일 생성
    """
    data_config = {
        'path': './',
        'train': 'train/images',
        'val': 'val/images',
        'nc': 4,
        'names': ['bird_nest', 'plastic_bag', 'floating_object', 'balloon']
    }

    yaml_path = Path('data_english.yaml')
    print(f"영어 데이터 설정 파일 생성: {yaml_path}")

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print("data_english.yaml 파일이 생성되었습니다.")
    return yaml_path

if __name__ == "__main__":
    # 모델 파일들 찾기
    model_dir = Path("convert/model")

    if model_dir.exists():
        for model_file in model_dir.glob("*.pt"):
            print(f"\n처리 중: {model_file}")
            try:
                output_path = update_model_class_names(model_file)
                print(f"영어 버전 저장됨: {output_path}")
            except Exception as e:
                print(f"오류 발생: {e}")
    else:
        print("convert/model 디렉토리를 찾을 수 없습니다.")

    # 영어 데이터 설정 파일 생성
    create_english_data_yaml()