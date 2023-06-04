#!/usr/bin/env python3
import cv2
import mediapipe as mp

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe Pose를 사용하여 랜드마크 추출
    image_height, image_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # 0, 12, 11, 28, 27 좌표 추출
        landmarks = results.pose_landmarks.landmark
        point0 = (int(landmarks[0].x * image_width), int(landmarks[0].y * image_height))
        point12 = (int(landmarks[12].x * image_width), int(landmarks[12].y * image_height))
        point11 = (int(landmarks[11].x * image_width), int(landmarks[11].y * image_height))
        point28 = (int(landmarks[28].x * image_width), int(landmarks[28].y * image_height))
        point27 = (int(landmarks[27].x * image_width), int(landmarks[27].y * image_height))

        # 사각형 그리기
        cv2.rectangle(frame, point0, point28, (0, 255, 0), 2)
        cv2.circle(frame, point0, 5, (0, 0, 255), -1)
        cv2.circle(frame, point12, 5, (0, 0, 255), -1)
        cv2.circle(frame, point11, 5, (0, 0, 255), -1)
        cv2.circle(frame, point28, 5, (0, 0, 255), -1)
        cv2.circle(frame, point27, 5, (0, 0, 255), -1)

    # 프레임 출력
    cv2.imshow('Frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
