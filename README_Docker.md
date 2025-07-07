# Docker 환경에서 얼굴 인식 모자이크 프로그램 실행

## 📋 요구사항

- Docker
- Docker Compose
- 카메라 접근 권한

## 🚀 빠른 시작

### 1. 실행 스크립트 사용 (권장)
```bash
./run_docker.sh
```

### 2. 수동 실행
```bash
# 이미지 빌드
docker-compose build

# 컨테이너 실행
docker-compose up
```

## 🔧 환경별 설정

### macOS
```bash
# X11 권한 설정
xhost +localhost

# 실행
docker-compose up

# 권한 복원
xhost -localhost
```

### Linux
```bash
# X11 소켓 권한 확인
ls -la /tmp/.X11-unix

# 실행
docker-compose up
```

### Windows (WSL2)
```bash
# X11 서버 실행 필요
docker-compose up
```

## 📁 파일 구조

```
live_stream/
├── live_mosaic_high_accuracy.py  # 메인 프로그램
├── performance_config.py          # 성능 설정
├── Dockerfile                     # Docker 이미지 정의
├── docker-compose.yml            # Docker Compose 설정
├── requirements.txt              # Python 의존성
├── .dockerignore                 # Docker 빌드 제외 파일
├── run_docker.sh                 # 실행 스크립트
└── README_Docker.md              # 이 파일
```

## ⚙️ 설정 옵션

### 환경 변수
- `DISPLAY`: X11 디스플레이 설정
- `PYTHONPATH`: Python 경로 설정

### 볼륨 마운트
- `/tmp/.X11-unix`: X11 소켓 (GUI 표시용)
- `/dev/video0`: 카메라 디바이스
- `.:/app`: 현재 디렉토리 (개발용)

## 🐛 문제 해결

### 카메라 접근 오류
```bash
# 카메라 권한 확인
ls -la /dev/video*

# 권한 설정
sudo chmod 666 /dev/video0
```

### GUI 표시 오류
```bash
# X11 권한 확인
xhost

# 권한 설정
xhost +localhost
```

### 메모리 부족 오류
```bash
# Docker 메모리 증가
docker-compose up --memory=4g
```

## 📊 성능 모니터링

### 로그 확인
```bash
docker-compose logs -f
```

### 컨테이너 상태 확인
```bash
docker-compose ps
```

### 리소스 사용량 확인
```bash
docker stats
```

## 🔄 개발 모드

### 코드 변경 시 자동 반영
```bash
# 볼륨 마운트로 실시간 반영
docker-compose up --build
```

### 디버그 모드
```bash
# 상세 로그 출력
docker-compose up --verbose
```

## 🧹 정리

### 컨테이너 정지
```bash
docker-compose down
```

### 이미지 삭제
```bash
docker-compose down --rmi all
```

### 모든 Docker 리소스 정리
```bash
docker system prune -a
```

## 📝 주의사항

1. **카메라 권한**: Docker에서 카메라 접근을 위해 `privileged: true` 설정 사용
2. **GUI 표시**: X11 소켓 마운트 필요
3. **성능**: CNN 모델 사용으로 CPU/GPU 리소스 필요
4. **메모리**: 최소 2GB RAM 권장

## 🆘 지원

문제가 발생하면 다음을 확인하세요:
1. Docker 버전
2. 카메라 연결 상태
3. X11 서버 실행 상태
4. 시스템 리소스 사용량 