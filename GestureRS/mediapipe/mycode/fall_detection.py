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
movement_threshold = 3.5
# 움직임을 감지하기 위한 임계값 설정
movement_list = []  
# 평균 편화량 저장할 리스트

prev_left_shoulder = None
prev_right_shoulder = None
prev_left_wrist = None
prev_right_wrist = None
prev_left_knee = None
prev_right_knee = None
    
# 이전 프레임의 좌표 초기화
pre_final_angle =None
# 이전 각도 초기화 
angle_change_average=0

cap = cv2.VideoCapture('dataset/sleep(1-3).mov')
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    
    angle_list = []
    angle_list_size = 10 # 저장할 각도 값의 개수

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
        
        point_11 = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
        point_23 = (int(results.pose_landmarks.landmark[23].x * image.shape[1]), int(results.pose_landmarks.landmark[23].y * image.shape[0]))
        point_25 = (int(results.pose_landmarks.landmark[25].x * image.shape[1]), int(results.pose_landmarks.landmark[25].y * image.shape[0]))
        point_12 = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))
        point_24 = (int(results.pose_landmarks.landmark[24].x * image.shape[1]), int(results.pose_landmarks.landmark[24].y * image.shape[0]))
        point_26 = (int(results.pose_landmarks.landmark[26].x * image.shape[1]), int(results.pose_landmarks.landmark[26].y * image.shape[0]))
        
        #measuring_angle 값이 Ture인 경우에 각도 계산 
        if(measuring_angle):
            # 어깨 좌표 추출
            left_shoulder = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
            right_shoulder = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))

            # 11번과 23번 연결선의 각도 계산
            dx1 = point_23[0] - point_11[0]
            dy1 = point_23[1] - point_11[1]
            left_angle1 = np.degrees(np.arctan2(dy1, dx1))

            # 23번과 25번 연결선의 각도 계산
            dx2 = point_25[0] - point_23[0]
            dy2 = point_25[1] - point_23[1]
            left_angle2 = np.degrees(np.arctan2(dy2, dx2))

            # 두 직선이 이루는 각도 계산
            left_angle = abs(left_angle2 - left_angle1)
            
            # 12번과 24번 연결선의 각도 계산
            dx1 = point_24[0] - point_12[0]
            dy1 = point_24[1] - point_12[1]
            right_angle1 = np.degrees(np.arctan2(dy1, dx1))

            # 23번과 26번 연결선의 각도 계산
            dx2 = point_26[0] - point_24[0]
            dy2 = point_26[1] - point_24[1]
            right_angle2 = np.degrees(np.arctan2(dy2, dx2))

            # 두 직선이 이루는 각도 계산
            right_angle = abs(right_angle2 - right_angle1)
            
            # 어깨 내려감 판단
            if right_shoulder[1] > left_shoulder[1]:
                right_angle_text = "Right {:.2f}".format(right_angle)
                # cv2.putText(image, right_angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                final_angle=right_angle
            else:
                left_angle_text = "Left {:.2f}".format(left_angle)
                # cv2.putText(image, left_angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                final_angle=left_angle
                
            # 확인 출력
            # cv2.putText(image, "angle {:.2f}".format(final_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            

            if (pre_final_angle is not None):
                
                angle_change = abs(final_angle - pre_final_angle)
                #변화량 계산
                angle_list.append(angle_change)
                
                if len(angle_list) > angle_list_size:
                    angle_list = angle_list[1:]  # 가장 오래된 값을 버림
                
                if(len(angle_list) >2):
                    #변화량 리스트에 저장
                    angle_change_average = np.mean(angle_list)
                    #변화량의 평균 구하기 
                    # cv2.putText(image, "angle_change_average {:.2f}".format(angle_change_average), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    angle_change_average=0
                    
            pre_final_angle = final_angle

            
                
            # angle_average = np.mean(angle_list)  # 평균값 계산
            
        if(7<angle_change_average):
            cv2.putText(image, "risk detection!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "angle_change_average {:.2f}".format(angle_change_average), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            measuring_angle=False
            measuring_movement=True
            
        if(measuring_movement):
            # 왼쪽 손목과 어깨 좌표 추출
            left_shoulder = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
            right_shoulder = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))
            left_wrist = (int(results.pose_landmarks.landmark[15].x * image.shape[1]), int(results.pose_landmarks.landmark[15].y * image.shape[0]))
            right_wrist = (int(results.pose_landmarks.landmark[16].x * image.shape[1]), int(results.pose_landmarks.landmark[16].y * image.shape[0]))
            left_knee = (int(results.pose_landmarks.landmark[25].x * image.shape[1]), int(results.pose_landmarks.landmark[25].y * image.shape[0]))
            right_knee = (int(results.pose_landmarks.landmark[26].x * image.shape[1]), int(results.pose_landmarks.landmark[26].y * image.shape[0]))
            
            if prev_left_shoulder is not None:
                #이전 프레임과 손목과 어깨의 변화량 측정
                left_shoulder_movement = np.linalg.norm(np.array(left_shoulder) - np.array(prev_left_shoulder))
                right_shoulder_movement = np.linalg.norm(np.array(right_shoulder) - np.array(prev_right_shoulder))
                left_wrist_movement = np.linalg.norm(np.array(left_wrist) - np.array(prev_left_wrist))
                right_wrist_movement = np.linalg.norm(np.array(right_wrist) - np.array(prev_right_wrist))
                left_knee_movement = np.linalg.norm(np.array(left_knee) - np.array(prev_left_knee))
                right_knee_movement = np.linalg.norm(np.array(right_knee) - np.array(prev_right_knee))
                
                # 평균 변화량 계산
                average_movement = (left_shoulder_movement+right_shoulder_movement+left_wrist_movement+right_wrist_movement+left_knee_movement+right_knee_movement) / 6
                
                #평균 변화량을 리스트에 추가 
                movement_list.append(average_movement) 
                
                
                # cv2.putText(image, "movement average {:.2f}".format(average_movement), (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 평균 변화량을 저장한 리스트의 길이가 5보다 크면 
                if len(movement_list) >= 180:
                    # 리스트의 값의 평균값을 구한다. 
                    average_threshold = np.mean(movement_list)
                    # cv2.putText(image, "final movement average {:.2f}".format(average_threshold), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    #그 때 임계값보다 작으면 
                    if average_threshold <= movement_threshold:
                        cv2.putText(image, "fall detection!", (350, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        #아니면 다시 각도를 측정
                        measuring_angle=True
                        measuring_movement=False
                    movement_list.pop(0)  
                    #이전 프레임과 손목과 어깨의 변화량 측정
                
            # 이전 프레임의 좌표 업데이트
            prev_left_shoulder = left_shoulder
            prev_right_shoulder = right_shoulder
            prev_left_wrist = left_wrist
            prev_right_wrist = right_wrist
            prev_left_knee = left_knee
            prev_right_knee = right_knee
    
           
         # 두 직선을 비디오 화면에 그립니다.
        cv2.line(image, point_11, point_23, (0, 255, 0), 2)
        cv2.line(image, point_23, point_25, (0, 0, 255), 2)
        cv2.line(image, point_12, point_24, (0, 255, 0), 2)
        cv2.line(image, point_24, point_26, (0, 0, 255), 2)

        

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    
cap.release()
cv2.destroyAllWindows()

