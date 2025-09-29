import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# 한글 폰트 설정 (시스템에 맞게 조정)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

# 데이터 정의
data = {
    'Model': ['YOLO11n', 'YOLO11n_fp16', 'YOLO11n_int8', 'YOLO11s', 'YOLO11s_fp16', 'YOLO11s_int8'],
    'FPS': [22.9, 42.6, 45.1, 21.8, 36.7, 41.0],
    'mAP@0.5': [0.947, 0.946, 0.928, 0.949, 0.946, 0.946],
    'mAP@0.5:0.95': [0.851, 0.847, 0.829, 0.866, 0.860, 0.851],
    'Average_Latency': [43.7, 23.5, 22.2, 45.9, 27.2, 24.4],
    'GPU_Memory': [98, 49, 25, 152, 76, 38]
}

df = pd.DataFrame(data)

# 모델 그룹 정의
yolo11n_group = df[df['Model'].str.contains('YOLO11n')]
yolo11s_group = df[df['Model'].str.contains('YOLO11s')]

# 색상 맵 생성 (mAP@0.5:0.95 값에 따라)
def get_color_from_map_range(value, min_val=0.829, max_val=0.866):
    """mAP@0.5:0.95 값에 따른 색상 반환"""
    normalized = (value - min_val) / (max_val - min_val)
    # 보라색에서 노란색으로 그라데이션
    colors = ['#8B4A9C', '#6B5B95', '#4A6B8A', '#2E8B7F', '#1CAB74', '#4ECB69', '#A8E6A1', '#F0E68C']
    color_idx = int(normalized * (len(colors) - 1))
    color_idx = max(0, min(color_idx, len(colors) - 1))
    return colors[color_idx]

# 그래프 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 메인 그래프
# YOLO11n 계열
colors_n = [get_color_from_map_range(val) for val in yolo11n_group['mAP@0.5:0.95']]
ax1.scatter(yolo11n_group['FPS'], yolo11n_group['mAP@0.5'], 
           c=colors_n, s=100, alpha=0.8, edgecolors='black', linewidth=1.5, label='YOLO11n 계열')
ax1.plot(yolo11n_group['FPS'], yolo11n_group['mAP@0.5'], 
         color='#4A90E2', linewidth=2, alpha=0.7)

# YOLO11s 계열
colors_s = [get_color_from_map_range(val) for val in yolo11s_group['mAP@0.5:0.95']]
ax1.scatter(yolo11s_group['FPS'], yolo11s_group['mAP@0.5'], 
           c=colors_s, s=100, alpha=0.8, edgecolors='black', linewidth=1.5, label='YOLO11s 계열')
ax1.plot(yolo11s_group['FPS'], yolo11s_group['mAP@0.5'], 
         color='#E74C3C', linewidth=2, alpha=0.7)

# 각 점에 모델명 라벨 추가
for i, row in df.iterrows():
    model_name = row['Model'].replace('YOLO11', 'yolo11')
    ax1.annotate(model_name, 
                (row['FPS'], row['mAP@0.5']), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, ha='left')

ax1.set_xlabel('FPS (성능)', fontsize=12, fontweight='bold')
ax1.set_ylabel('mAP@0.5 (정확도)', fontsize=12, fontweight='bold')
ax1.set_title('모델 성능 vs 정확도 비교 (같은 모델의 정밀도별 연결)', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 축 범위 설정
ax1.set_xlim(20, 47)
ax1.set_ylim(0.925, 0.950)

# 컬러바 생성
norm = plt.Normalize(vmin=0.829, vmax=0.866)
cmap = LinearSegmentedColormap.from_list('custom', 
                                        ['#8B4A9C', '#6B5B95', '#4A6B8A', '#2E8B7F', 
                                         '#1CAB74', '#4ECB69', '#A8E6A1', '#F0E68C'])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
cbar.set_label('mAP@0.5:0.95', rotation=270, labelpad=15, fontweight='bold')

# 데이터 테이블 생성
ax2.axis('tight')
ax2.axis('off')

# 테이블 데이터 준비 (굵은 글씨 강조를 위해)
table_data = []
headers = ['Model', 'FPS', 'mAP@0.5', 'mAP@0.5:0.95', 'Avg Latency', 'GPU Memory']

# 강조할 값들 정의
highlight_values = {
    'YOLO11n_fp16': ['FPS', 'GPU_Memory'],
    'YOLO11n_int8': ['FPS', 'GPU_Memory'],
    'YOLO11s': ['mAP@0.5', 'mAP@0.5:0.95'],
    'YOLO11s_fp16': ['mAP@0.5', 'mAP@0.5:0.95'],
    'YOLO11s_int8': ['GPU_Memory']
}

for _, row in df.iterrows():
    table_row = [
        row['Model'],
        f"{row['FPS']:.1f}",
        f"{row['mAP@0.5']:.3f}",
        f"{row['mAP@0.5:0.95']:.3f}",
        f"{row['Average_Latency']:.1f}",
        f"{row['GPU_Memory']}"
    ]
    table_data.append(table_row)

# 테이블 생성
table = ax2.table(cellText=table_data,
                 colLabels=headers,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 헤더 스타일
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#f8f9fa')
    table[(0, i)].set_text_props(weight='bold')

# 강조할 셀들에 대해 굵은 글씨 적용
for row_idx, (_, row) in enumerate(df.iterrows(), 1):
    model = row['Model']
    if model in highlight_values:
        for col_name in highlight_values[model]:
            if col_name == 'FPS':
                col_idx = 1
            elif col_name == 'GPU_Memory':
                col_idx = 5
            elif col_name == 'mAP@0.5':
                col_idx = 2
            elif col_name == 'mAP@0.5:0.95':
                col_idx = 3
            
            table[(row_idx, col_idx)].set_text_props(weight='bold', color='#2563eb')

ax2.set_title('모델 성능 상세 데이터', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# 데이터프레임 출력
print("=== YOLO 모델 성능 비교 데이터 ===")
print(df.to_string(index=False))

# 성능 분석 출력
print("\n=== 성능 분석 ===")
print("1. 최고 FPS:", df.loc[df['FPS'].idxmax(), 'Model'], f"({df['FPS'].max():.1f} FPS)")
print("2. 최고 mAP@0.5:", df.loc[df['mAP@0.5'].idxmax(), 'Model'], f"({df['mAP@0.5'].max():.3f})")
print("3. 최고 mAP@0.5:0.95:", df.loc[df['mAP@0.5:0.95'].idxmax(), 'Model'], f"({df['mAP@0.5:0.95'].max():.3f})")
print("4. 최소 GPU 메모리:", df.loc[df['GPU_Memory'].idxmin(), 'Model'], f"({df['GPU_Memory'].min()} MB)")
print("5. 최소 지연시간:", df.loc[df['Average_Latency'].idxmin(), 'Model'], f"({df['Average_Latency'].min():.1f} ms)")