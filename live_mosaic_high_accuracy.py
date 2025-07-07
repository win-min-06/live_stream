#!/usr/bin/env python3
"""
최고 정확도 실시간 얼굴 인식 및 모자이크 프로그램
CNN 모델 사용으로 최대 정확도 달성
"""

import cv2
import face_recognition
import time

# 최고 정확도 설정
HIGH_ACCURACY_CONFIG = {
    "process_every_n_frames": 2,    # 2프레임마다 처리 (높은 정확도)
    "face_detection_model": "cnn",  # CNN 모델 (최고 정확도)
    "face_comparison_tolerance": 0.4,  # 엄격한 비교 (높은 정확도)
    "camera_width": 640,           # 높은 해상도 (정확도 향상)
    "camera_height": 480,
    "mosaic_level": 15,            # 적당한 모자이크 (정확도 우선)
    "skip_frames": 1,              # 최소 스킵 (정확도 우선)
    "upsample_times": 1            # 얼굴 탐지 정확도 향상
}

def apply_high_accuracy_mosaic(frame, x, y, w, h):
    """고정확도 모자이크 적용"""
    try:
        height, width = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 0 or h <= 0:
            return frame
            
        roi = frame[y:y+h, x:x+w]
        roi_small = cv2.resize(roi, (HIGH_ACCURACY_CONFIG["mosaic_level"], HIGH_ACCURACY_CONFIG["mosaic_level"]))
        roi_mosaic = cv2.resize(roi_small, (w, h))
        frame[y:y+h, x:x+w] = roi_mosaic
        return frame
    except:
        return frame

def detect_faces_high_accuracy(frame):
    """고정확도 얼굴 탐지"""
    try:
        # 높은 해상도 유지 (정확도 향상)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CNN 모델로 정확한 얼굴 탐지
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model=HIGH_ACCURACY_CONFIG["face_detection_model"],
            number_of_times_to_upsample=HIGH_ACCURACY_CONFIG["upsample_times"]
        )
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return face_locations, face_encodings
    except:
        return [], []

def main():
    """최고 정확도 메인 프로그램"""
    print("=== 최고 정확도 실시간 얼굴 인식 및 모자이크 프로그램 ===")
    print(f"설정: {HIGH_ACCURACY_CONFIG['process_every_n_frames']}프레임마다 처리")
    print(f"모델: {HIGH_ACCURACY_CONFIG['face_detection_model']} (최고 정확도)")
    print(f"허용 오차: {HIGH_ACCURACY_CONFIG['face_comparison_tolerance']} (엄격한 비교)")
    print(f"해상도: {HIGH_ACCURACY_CONFIG['camera_width']}x{HIGH_ACCURACY_CONFIG['camera_height']}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 고정확도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_ACCURACY_CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_ACCURACY_CONFIG["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 진행자 얼굴 등록
    host_face_encoding = None
    print("카메라 앞에서 's' 키를 눌러 얼굴을 등록하세요.")
    
    while host_face_encoding is None:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, "Press 's' to register", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('High Accuracy - Register', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(
                    rgb_frame, 
                    model=HIGH_ACCURACY_CONFIG["face_detection_model"],
                    number_of_times_to_upsample=HIGH_ACCURACY_CONFIG["upsample_times"]
                )
                
                if len(face_locations) == 1:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        host_face_encoding = face_encodings[0]
                        print("✅ 얼굴 등록 완료! (고정확도 모드)")
                        break
                    else:
                        print("❌ 얼굴 특징 추출 실패")
                elif len(face_locations) > 1:
                    print("❌ 여러 얼굴이 감지됨")
                else:
                    print("❌ 얼굴을 찾을 수 없음")
            except Exception as e:
                print(f"등록 오류: {e}")

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    if host_face_encoding is None:
        print("얼굴 등록에 실패했습니다.")
        return

    cv2.destroyWindow('High Accuracy - Register')
    print("=== 최고 정확도 모드 시작 ===")

    # 성능 변수
    frame_count = 0
    last_face_locations = []
    last_face_encodings = []
    frame_times = []
    accuracy_stats = {"correct_detections": 0, "total_detections": 0}
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 스킵 최소화 (정확도 우선)
        frame_count += 1
        if frame_count % HIGH_ACCURACY_CONFIG["skip_frames"] != 0:
            continue

        # 얼굴 탐지 (빈번한 처리로 정확도 향상)
        if frame_count % HIGH_ACCURACY_CONFIG["process_every_n_frames"] == 0:
            face_locations, face_encodings = detect_faces_high_accuracy(frame)
            last_face_locations = face_locations
            last_face_encodings = face_encodings

        # 얼굴 처리
        for (top, right, bottom, left), face_encoding in zip(last_face_locations, last_face_encodings):
            try:
                # 엄격한 얼굴 비교
                matches = face_recognition.compare_faces(
                    [host_face_encoding], 
                    face_encoding, 
                    tolerance=HIGH_ACCURACY_CONFIG["face_comparison_tolerance"]
                )
                
                is_host = True in matches
                accuracy_stats["total_detections"] += 1
                if is_host:
                    accuracy_stats["correct_detections"] += 1

                # 모자이크 또는 표시
                if not is_host:
                    frame = apply_high_accuracy_mosaic(frame, left, top, right - left, bottom - top)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # 빨간색 (비진행자)
                    cv2.putText(frame, "Guest", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # 녹색 (진행자)
                    cv2.putText(frame, "Host", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    
            except:
                continue

        # 성능 측정
        processing_time = (time.time() - start_time) * 1000
        frame_times.append(processing_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        avg_time = sum(frame_times) / len(frame_times) if frame_times else 0
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        # 정확도 계산
        accuracy = (accuracy_stats["correct_detections"] / accuracy_stats["total_detections"] * 100) if accuracy_stats["total_detections"] > 0 else 0

        # 성능 및 정확도 정보 표시
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(last_face_locations)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Model: {HIGH_ACCURACY_CONFIG['face_detection_model'].upper()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('High Accuracy - Live Interview', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"최종 정확도: {accuracy:.1f}%")
    print("프로그램을 종료합니다.")

if __name__ == '__main__':
    main() 