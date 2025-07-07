# Python 3.9 기반 이미지 사용
FROM python:3.9-slim

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .

# ARM64 호환성을 위해 미리 빌드된 wheel 사용
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cmake && \
    pip install --no-cache-dir --only-binary=all dlib

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Haar Cascade 파일 다운로드
RUN wget -O haarcascade_frontalface_default.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# 권한 설정
RUN chmod +x live_mosaic_high_accuracy.py

# 환경 변수 설정
ENV DISPLAY=:99
ENV PYTHONPATH=/app

# 포트 노출 (필요시)
EXPOSE 8080

# 실행 명령
CMD ["python", "live_mosaic_high_accuracy.py"] 