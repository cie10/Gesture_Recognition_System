#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe를 위한 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# OpenCV를 사용하여 비디오를 가져옴
cap = cv2.VideoCapture(0)

# MediaPipe Pose 모델 초기화
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 이미지를 BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe에 이미지 입력
        results = pose.process(image)

        # 감지된 포즈가 있을 경우
        if results.pose_landmarks is not None:
            # 11번과 12번 좌표의 중심 좌표 계산
            landmark_11 = results.pose_landmarks.landmark[11]
            landmark_12 = results.pose_landmarks.landmark[12]
            center_x = (landmark_11.x + landmark_12.x) / 2
            center_y = (landmark_11.y + landmark_12.y) / 2
            center_z = (landmark_11.z + landmark_12.z) / 2

            # 0번 좌표와 중심 좌표의 방향 벡터 계산
            landmark_0 = results.pose_landmarks.landmark[0]
            vector_x = landmark_0.x - center_x
            vector_y = landmark_0.y - center_y
            vector_z = landmark_0.z - center_z 


            # 결과 출력
            print("중심 좌표: ({}, {}, {})".format(center_x, center_y, center_z))
            print("방향 벡터: ({}, {}, {})".format(vector_x, vector_y, vector_z))
            

        # 결과를 이미지에 그리기
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 이미지를 RGB에서 BGR로 변환하여 출력
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# 종료 시 리소스 해제
cap.release()
cv2.destroyAllWindows()
