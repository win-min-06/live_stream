# 성능 최적화 설정 파일
# 이 파일의 값을 조정하여 성능과 정확도의 균형을 맞출 수 있습니다.

# === 기본 설정 ===
MOSAIC_LEVEL = 15  # 모자이크 강도 (숫자가 작을수록 강함)

# === 성능 최적화 설정 ===
PROCESS_EVERY_N_FRAMES = 5  # 5프레임마다 처리 (최대 성능)
FACE_DETECTION_MODEL = "hog"  # HOG 모델 (CPU 최적화)
FACE_COMPARISON_TOLERANCE = 0.7  # 관대한 비교 (빠른 처리)
CACHE_SIZE = 3  # 작은 캐시 (메모리 절약)

# === 카메라 설정 ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1  # 버퍼 크기 최소화로 지연 감소
CASCADE_FILE = 'haarcascade_frontalface_default.xml'  # Haar Cascade 파일 경로

def print_performance_info():
    """현재 성능 설정 정보를 출력합니다."""
    print("=== 현재 성능 설정 ===")
    print(f"프레임 처리 간격: {PROCESS_EVERY_N_FRAMES}프레임마다")
    print(f"얼굴 탐지 모델: {FACE_DETECTION_MODEL}")
    print(f"얼굴 비교 허용 오차: {FACE_COMPARISON_TOLERANCE}")
    print(f"캐시 크기: {CACHE_SIZE}")
    print(f"카메라 해상도: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"카메라 FPS: {CAMERA_FPS}")
    print("=====================") 