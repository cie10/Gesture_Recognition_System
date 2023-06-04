#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 지속시간 측정 위해 변수 추가 
measuring_angle = True
measuring_movement = False
movement_threshold = 10
# 움직임을 감지하기 위한 임계값 설정
movement_list = []
movement_list_size = 100  # 저장할 움직임 값의 개수

prev_left_shoulder = None
prev_right_shoulder = None
prev_left_wrist = None
prev_right_wrist = None
# 이전 프레임의 좌표 초기화

cap = cv2.VideoCapture('오른쪽 넘어짐1.mov')
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    
    angle_list = []
    angle_list_size = 15  # 저장할 각도 값의 개수

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
        
        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        #measuring_angle 값이 Ture인 경우에 각도 계산 
        if(measuring_angle):
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
                angle_list.append(right_angle)
            else:
                left_angle_text = "Left {:.2f}".format(left_angle)
                cv2.putText(image, left_angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                angle_list.append(left_angle)
            
            
            if len(angle_list) > angle_list_size:
                angle_list = angle_list[1:]  # 가장 오래된 값을 버림

            # 평균값 계산
            angle_change = np.diff(angle_list)  # 각 값의 차이값(변화량) 계산
            angle_average = np.mean(angle_change)  # 각 값의 차이값(변화량)의 평균값 계산
            cv2.putText(image, "{:.2f}".format(angle_average), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if(20 < angle_average):
            cv2.putText(image, "Fall detection!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            measuring_angle = False
            measuring_movement = True
            
        if measuring_movement:
            # 왼쪽 손목과 어깨 좌표 추출
            left_shoulder = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
            left_wrist = (int(results.pose_landmarks.landmark[15].x * image.shape[1]), int(results.pose_landmarks.landmark[15].y * image.shape[0]))

            # 오른쪽 손목과 어깨 좌표 추출
            right_shoulder = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))
            right_wrist = (int(results.pose_landmarks.landmark[16].x * image.shape[1]), int(results.pose_landmarks.landmark[16].y * image.shape[0]))

            if prev_left_shoulder is not None and prev_right_shoulder is not None:
                # 손목의 움직임 계산
                left_movement = np.sqrt(np.sum((np.array(left_wrist) - np.array(prev_left_wrist)) ** 2))
                right_movement = np.sqrt(np.sum((np.array(right_wrist) - np.array(prev_right_wrist)) ** 2))

                # 손목의 움직임을 movement_list에 추가
                movement_list.append((left_movement + right_movement) / 2)

                # movement_list의 길이가 movement_list_size보다 크면
                # 가장 오래된 값을 제거하여 길이를 유지합니다.
                if len(movement_list) > movement_list_size:
                    movement_list = movement_list[1:]

                # movement_list의 평균값 계산
                movement_average = np.mean(movement_list)

                if movement_average > movement_threshold:
                    cv2.putText(image, "Movement detected!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    measuring_angle = True
                    measuring_movement = False

            prev_left_shoulder = left_shoulder
            prev_right_shoulder = right_shoulder
            prev_left_wrist = left_wrist
            prev_right_wrist = right_wrist

        cv2.imshow('Pose Estimation', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
