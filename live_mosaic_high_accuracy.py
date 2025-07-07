#!/usr/bin/env python3
"""
mediapipe 기반 최고 정확도 실시간 얼굴 인식 및 모자이크 프로그램
(dlib/face_recognition 없이 모든 플랫폼에서 동작)
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import os

IS_DOCKER = os.path.exists('/.dockerenv')

CONFIG = {
    "camera_width": 640,
    "camera_height": 480,
    "mosaic_level": 15,
    "show_gui": not IS_DOCKER
}

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 특징 추출용 (진행자 등록)
def get_face_embedding(image, face_box):
    # mediapipe는 임베딩 제공X, 얼굴 ROI 픽셀값을 간단히 flatten하여 임시 임베딩으로 사용
    x, y, w, h = face_box
    face_img = cv2.resize(image[y:y+h, x:x+w], (64, 64)).flatten()
    return face_img / 255.0

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def apply_mosaic(frame, x, y, w, h):
    try:
        height, width = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        if w <= 0 or h <= 0:
            return frame
        roi = frame[y:y+h, x:x+w]
        roi_small = cv2.resize(roi, (CONFIG["mosaic_level"], CONFIG["mosaic_level"]))
        roi_mosaic = cv2.resize(roi_small, (w, h))
        frame[y:y+h, x:x+w] = roi_mosaic
        return frame
    except:
        return frame

def main():
    print("=== mediapipe 실시간 얼굴 인식 및 모자이크 프로그램 ===")
    print(f"환경: {'Docker' if IS_DOCKER else 'Local'}")
    print(f"해상도: {CONFIG['camera_width']}x{CONFIG['camera_height']}")
    print(f"GUI 표시: {CONFIG['show_gui']}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 진행자 얼굴 등록
    host_embedding = None
    print("카메라 앞에서 's' 키를 눌러 얼굴을 등록하세요.")
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        while host_embedding is None:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            face_box = None
            if results.detections:
                for det in results.detections:
                    bboxC = det.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    face_box = (x, y, w, h)
                    break
            if CONFIG["show_gui"]:
                if face_box:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "Press 's' to register", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow('Register Host', frame)
            key = cv2.waitKey(1) & 0xFF if CONFIG["show_gui"] else 0
            if key == ord('s') and face_box:
                host_embedding = get_face_embedding(frame, face_box)
                print("✅ 얼굴 등록 완료!")
                break
            elif key == ord('q'):
                cap.release()
                if CONFIG["show_gui"]:
                    cv2.destroyAllWindows()
                return
        if host_embedding is None:
            print("얼굴 등록에 실패했습니다.")
            return
        if CONFIG["show_gui"]:
            cv2.destroyWindow('Register Host')
        print("=== 인터뷰 모드 시작 ===")
        frame_times = []
        save_counter = 0
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            face_boxes = []
            embeddings = []
            if results.detections:
                for det in results.detections:
                    bboxC = det.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    face_boxes.append((x, y, w, h))
                    embeddings.append(get_face_embedding(frame, (x, y, w, h)))
            # 얼굴 비교 및 모자이크
            for i, (box, emb) in enumerate(zip(face_boxes, embeddings)):
                sim = cosine_similarity(host_embedding, emb)
                is_host = sim > 0.85
                x, y, w, h = box
                if not is_host:
                    frame = apply_mosaic(frame, x, y, w, h)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                    cv2.putText(frame, "Guest", (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, "Host", (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
            # 성능 표시
            processing_time = (time.time() - start_time) * 1000
            frame_times.append(processing_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_time = sum(frame_times) / len(frame_times) if frame_times else 0
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Faces: {len(face_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if IS_DOCKER and save_counter % 30 == 0:
                cv2.imwrite(f"output_frame_{save_counter:04d}.jpg", frame)
            save_counter += 1
            if CONFIG["show_gui"]:
                cv2.imshow('Live Interview - mediapipe', frame)
            key = cv2.waitKey(1) & 0xFF if CONFIG["show_gui"] else 0
            if key == ord('q'):
                break
        cap.release()
        if CONFIG["show_gui"]:
            cv2.destroyAllWindows()
        print("프로그램을 종료합니다.")

if __name__ == '__main__':
    main() 