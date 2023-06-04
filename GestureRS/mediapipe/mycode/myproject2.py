#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import math

def calculate_shoulder_hip_angle(shoulder_line, hip_line):
    dx1 = shoulder_line[1][0] - shoulder_line[0][0]
    dy1 = shoulder_line[1][1] - shoulder_line[0][1]
    dx2 = hip_line[1][0] - hip_line[0][0]
    dy2 = hip_line[1][1] - hip_line[0][1]

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    angle_degrees = math.degrees(angle1 - angle2)

    return angle_degrees

# Mediapipe를 이용하여 스켈레톤 좌표를 추출하는 함수
def extract_skeleton(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in results.pose_landmarks.landmark]
            return landmarks
        else:
            return None
# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.햐
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            
            #어깨 관절의 인덱스 가져오기 
            shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            # 엉덩이 인덱스 가져오기
            hip_index = mp_pose.PoseLandmark.LEFT_HIP.value
            shoulder_line = [skeleton[shoulder_index], skeleton[shoulder_index + 1]]
            # 엉덩이 좌표 가져오기
            hip_line = [skeleton[hip_index], skeleton[hip_index + 1]]

            angle = calculate_shoulder_hip_angle(shoulder_line, hip_line)
            
            angle=calculate_shoulder_hip_angle(shoulder_index, hip_index);

            # 각도를 비디오 화면에 출력
            angle_text = "Angle: {:.2f}".format(angle)
            cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # 보기 편하게 이미지를 좌우 반전합니다.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()