#!/bin/bash

# Docker 실행 스크립트
echo "=== 얼굴 인식 모자이크 프로그램 Docker 실행 ==="

# Docker 데몬 상태 확인
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker 데몬이 실행되지 않았습니다."
    echo "Docker Desktop을 실행해주세요."
    exit 1
fi

echo "✅ Docker 데몬이 실행 중입니다."

# X11 권한 설정 (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS 환경 감지"
    # xhost 명령어가 있는지 확인
    if command -v xhost > /dev/null 2>&1; then
        xhost +localhost
    else
        echo "⚠️  xhost 명령어를 찾을 수 없습니다. X11이 설치되지 않았을 수 있습니다."
        echo "GUI 없이 실행됩니다."
    fi
fi

# Docker 이미지 빌드
echo "Docker 이미지 빌드 중..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "❌ Docker 빌드 실패"
    exit 1
fi

# 컨테이너 실행
echo "컨테이너 실행 중..."
docker-compose up

# 정리
echo "프로그램 종료. 정리 중..."
docker-compose down

# X11 권한 복원 (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v xhost > /dev/null 2>&1; then
        xhost -localhost
    fi
fi

echo "완료!" 