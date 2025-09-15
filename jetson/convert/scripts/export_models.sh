#!/bin/bash
"""
YOLO 모델 일괄 변환 스크립트
여러 모델을 다양한 정밀도로 변환
"""

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_export() {
    echo -e "${PURPLE}[EXPORT]${NC} $1"
}

# 도움말 함수
show_help() {
    cat << EOF
🚀 YOLO 모델 일괄 변환 스크립트

사용법:
    $0 [옵션]

필수 인자:
    -m, --model PATH         모델 파일 경로 (.pt)
    -o, --output DIR         출력 디렉토리

변환 옵션:
    --fp16                   FP16 TensorRT 엔진 변환
    --int8                   INT8 TensorRT 엔진 변환
    --onnx                   ONNX 형식 변환
    --all                    모든 형식 변환
    -c, --calibration DIR    INT8 캘리브레이션 데이터 디렉토리
    --imgsz SIZE            입력 이미지 크기 (기본값: 640)
    --batch SIZE            배치 크기 (기본값: 1)
    --workspace SIZE        TensorRT 작업공간 GB (기본값: 4)
    --device DEVICE         GPU 디바이스 (기본값: 0)

기타 옵션:
    -h, --help              이 도움말 표시
    --dry-run               실제 변환 없이 명령어만 출력

예시:
    # FP16 TensorRT 엔진 변환
    $0 -m yolo11n.pt -o ./engines --fp16

    # INT8 TensorRT 엔진 변환 (캘리브레이션 포함)
    $0 -m yolo11n.pt -o ./engines --int8 -c ./val/images/

    # 모든 형식 변환
    $0 -m yolo11n.pt -o ./engines --all -c ./val/images/

    # 커스텀 설정
    $0 -m yolo11n.pt -o ./engines --fp16 --imgsz 416 --batch 4
EOF
}

# 기본값 설정
MODEL=""
OUTPUT_DIR=""
EXPORT_FP16=false
EXPORT_INT8=false
EXPORT_ONNX=false
EXPORT_ALL=false
CALIBRATION_DATA=""
IMGSZ=640
BATCH=1
WORKSPACE=4
DEVICE="0"
DRY_RUN=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fp16)
            EXPORT_FP16=true
            shift
            ;;
        --int8)
            EXPORT_INT8=true
            shift
            ;;
        --onnx)
            EXPORT_ONNX=true
            shift
            ;;
        --all)
            EXPORT_ALL=true
            shift
            ;;
        -c|--calibration)
            CALIBRATION_DATA="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 유효성 검사
if [[ -z "$MODEL" ]]; then
    log_error "모델 파일을 지정해야 합니다."
    show_help
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    log_error "출력 디렉토리를 지정해야 합니다."
    show_help
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    log_error "모델 파일을 찾을 수 없습니다: $MODEL"
    exit 1
fi

# 모든 형식 변환 시 개별 플래그 설정
if [[ "$EXPORT_ALL" == true ]]; then
    EXPORT_FP16=true
    EXPORT_INT8=true
    EXPORT_ONNX=true
fi

# 최소 하나의 형식은 선택되어야 함
if [[ "$EXPORT_FP16" == false && "$EXPORT_INT8" == false && "$EXPORT_ONNX" == false ]]; then
    log_error "최소 하나의 변환 형식을 선택해야 합니다 (--fp16, --int8, --onnx, --all)"
    show_help
    exit 1
fi

# Python 스크립트 존재 확인
if [[ ! -f "yolo_export_cli.py" ]]; then
    log_error "yolo_export_cli.py 파일을 찾을 수 없습니다."
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 모델 정보 출력
MODEL_NAME=$(basename "$MODEL" .pt)
log_info "모델 변환 설정:"
echo "  모델: $MODEL"
echo "  모델명: $MODEL_NAME"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  이미지 크기: ${IMGSZ}x${IMGSZ}"
echo "  배치 크기: $BATCH"
echo "  TensorRT 작업공간: ${WORKSPACE}GB"
echo "  디바이스: $DEVICE"
[[ -n "$CALIBRATION_DATA" ]] && echo "  캘리브레이션 데이터: $CALIBRATION_DATA"
echo "  변환 형식:"
[[ "$EXPORT_FP16" == true ]] && echo "    ✓ FP16 TensorRT"
[[ "$EXPORT_INT8" == true ]] && echo "    ✓ INT8 TensorRT"
[[ "$EXPORT_ONNX" == true ]] && echo "    ✓ ONNX"
echo ""

# 변환 함수
export_model() {
    local format="$1"
    local precision="$2"
    local extra_args="$3"

    log_export "변환 시작: $format ($precision)"

    # 명령어 구성
    local cmd="python3 yolo_export_cli.py \"$MODEL\" --format $format --precision $precision"
    cmd="$cmd --imgsz $IMGSZ --batch $BATCH --device $DEVICE --output-dir \"$OUTPUT_DIR\""

    if [[ "$format" == "engine" ]]; then
        cmd="$cmd --workspace $WORKSPACE"
    fi

    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi

    log_info "실행 명령어: $cmd"

    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN 모드 - 실제 변환하지 않음"
        return 0
    fi

    # 실행
    if eval "$cmd"; then
        log_success "$format ($precision) 변환 완료"
        return 0
    else
        log_error "$format ($precision) 변환 실패"
        return 1
    fi
}

# 변환 실행
export_count=0
success_count=0

# FP16 TensorRT 변환
if [[ "$EXPORT_FP16" == true ]]; then
    export_count=$((export_count + 1))
    if export_model "engine" "fp16" ""; then
        success_count=$((success_count + 1))
    fi
    echo ""
fi

# INT8 TensorRT 변환
if [[ "$EXPORT_INT8" == true ]]; then
    export_count=$((export_count + 1))
    extra_args=""
    if [[ -n "$CALIBRATION_DATA" ]]; then
        extra_args="--calibration-data \"$CALIBRATION_DATA\""
    fi
    if export_model "engine" "int8" "$extra_args"; then
        success_count=$((success_count + 1))
    fi
    echo ""
fi

# ONNX 변환
if [[ "$EXPORT_ONNX" == true ]]; then
    export_count=$((export_count + 1))
    if export_model "onnx" "fp16" ""; then
        success_count=$((success_count + 1))
    fi
    echo ""
fi

# 결과 요약
log_info "변환 결과 요약:"
echo "  총 변환 작업: $export_count"
echo "  성공: $success_count"
echo "  실패: $((export_count - success_count))"

if [[ "$DRY_RUN" == false ]]; then
    echo ""
    log_info "출력 파일들:"
    find "$OUTPUT_DIR" -name "${MODEL_NAME}*" -type f -exec ls -lh {} \;

    echo ""
    log_info "다음 단계 제안:"
    echo "  📊 벤치마크 실행:"
    if [[ "$EXPORT_FP16" == true ]]; then
        echo "    python3 direct_tensorrt_benchmark.py ./data/data.yaml \"$OUTPUT_DIR/${MODEL_NAME}.engine\" --max-images 50"
    fi
    echo "  🔍 결과 비교:"
    echo "    python3 analyze_results.py ./benchmark_results/"
fi

if [[ $success_count -eq $export_count ]]; then
    log_success "🎉 모든 변환 작업이 성공적으로 완료되었습니다!"
    exit 0
else
    log_warning "⚠️  일부 변환 작업이 실패했습니다."
    exit 1
fi