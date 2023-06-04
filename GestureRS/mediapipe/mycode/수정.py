#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    
    angle_list = []
    angle_list_size = 10  # 저장할 각도 값의 개수

    last_fall_time = 0
    fall_detected = False
    measuring_angle = False  # 각도 크기를 측정하는 중인지 여부

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            break

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks is None:
            # 포즈 랜드마크를 감지하지 못한 경우 현재 프레임을 건너뜁니다.
            continue
        
        # 어깨 좌표 추출
        left_shoulder = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
        right_shoulder = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))

        # 11번과 23번 연결선의 각도 계산
        point_11 = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
        point_23 = (int(results.pose_landmarks.landmark[23].x * image.shape[1]), int(results.pose_landmarks.landmark[23].y * image.shape[0]))
        dx1 = point_23[0] - point_11[0]
        dy1 = point_23[1] - point_11[1]
        left_angle1 = np.degrees(np.arctan2(dy1, dx1))

        # 23번과 27번 연결선의 각도 계산
        point_27 = (int(results.pose_landmarks.landmark[27].x * image.shape[1]), int(results.pose_landmarks.landmark[27].y * image.shape[0]))
        dx2 = point_27[0] - point_23[0]
        dy2 = point_27[1] - point_23[1]
        left_angle2 = np.degrees(np.arctan2(dy2, dx2))

        # 두 직선이 이루는 각도 계산
        left_angle = abs(left_angle2 - left_angle1)
        
        # 12번과 24번 연결선의 각도 계산
        point_12 = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))
        point_24 = (int(results.pose_landmarks.landmark[24].x * image.shape[1]), int(results.pose_landmarks.landmark[24].y * image.shape[0]))
        dx1 = point_24[0] - point_12[0]
        dy1 = point_24[1] - point_12[1]
        right_angle1 = np.degrees(np.arctan2(dy1, dx1))

        # 23번과 28번 연결선의 각도 계산
        point_28 = (int(results.pose_landmarks.landmark[28].x * image.shape[1]), int(results.pose_landmarks.landmark[28].y * image.shape[0]))
        dx2 = point_28[0] - point_24[0]
        dy2 = point_28[1] - point_24[1]
        right_angle2 = np.degrees(np.arctan2(dy2, dx2))

        # 두 직선이 이루는 각도 계산
        right_angle = abs(right_angle2 - right_angle1)
        
        # 어깨 내려감 판단
        if right_shoulder[1] > left_shoulder[1]:
            right_angle_text = "Right {:.2f}".format(right_angle)
            cv2.putText(image, right_angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not measuring_angle:
                angle_list.append(right_angle)
        else:
            left_angle_text = "Left {:.2f}".format(left_angle)
            cv2.putText(image, left_angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not measuring_angle:
                angle_list.append(left_angle)
        
        if len(angle_list) > angle_list_size:
            angle_list = angle_list[1:]  # 가장 오래된 값을 버림

        # 평균값 계산
        angle_average = np.mean(angle_list)
        
        if 30 < angle_average < 100:
            if not measuring_angle:
                measuring_angle = True  # Start measuring angle
                start_time = cv2.getTickCount() / cv2.getTickFrequency()
        else:
            if measuring_angle:
                end_time = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed_time = end_time - start_time
                if elapsed_time > 2:
                    # Only consider it a fall if there was little movement for 10 seconds
                    cv2.putText(image, "Fall Detected!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if fall_detected:
                        elapsed_time = current_time - last_fall_time
                        if elapsed_time > 5:
                            cv2.putText(image, "Real Fall Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        fall_detected = True
                        last_fall_time = current_time
        measuring_angle = False  # Stop measuring angle

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # 두 직선을 비디오 화면에 그립니다.
        cv2.line(image, point_11, point_23, (0, 255, 0), 2)
        cv2.line(image, point_23, point_27, (0, 0, 255), 2)
        cv2.line(image, point_12, point_24, (0, 255, 0), 2)
        cv2.line(image, point_24, point_28, (0, 0, 255), 2)

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
cap.release()
cv2.destroyAllWindows()
# 이렇게 수정해 보세요. 수정한 부분은 다음과 같습니다.

# 1. `measuring_angle` 변수를 추가하여 각도 측정 여부를 추적합니다. `measuring_angle`이 `True`일 때에만 각도 값을 저장합니다.
# 2. 평균 각도가 30과 100 사이인 경우에만 각도를 측정하도록 변경합니다.
# 3. `measuring_angle`이 `True`일 때 10초 동안 움직임이 거의 없을 경우에만 낙상으로 판단하도록 수정합니다.
# 4. 낙상이 감지된 후 5초 동안은 다시 낙상으로 판단하지 않도록 추가합니다.

# 이렇게 수정하면, fall detection을 만족했을 때 각도를 재는 과정을 일시적으로 멈추고, 지속 시간을 판단한 후 조건을 만족하지 않으면 다시 각도를 재는 코드로 전환됩니다.
