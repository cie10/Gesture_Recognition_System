#!/usr/bin/env python3
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # 머리와 어깨 관절의 인덱스 가져오기
            head_index = mp_pose.PoseLandmark.NOSE.value
            shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value

            # 머리와 어깨 관절의 좌표 추출
            head_coords = results.pose_landmarks.landmark[head_index].x, results.pose_landmarks.landmark[head_index].y
            shoulder_coords = results.pose_landmarks.landmark[shoulder_index].x, results.pose_landmarks.landmark[shoulder_index].y

            # 머리와 어깨 관절의 중심 계산
            center_x = (head_coords[0] + shoulder_coords[0]) / 2
            center_y = (head_coords[1] + shoulder_coords[1]) / 2

            # 중심의 수직 좌표 추출
            vertical_coordinate = results.pose_landmarks.landmark[head_index].z

            # 중심 좌표와 수직 좌표 출력
            print("중심 좌표 (x, y):", center_x, center_y)
            print("수직 좌표:", vertical_coordinate)

            # 중심점 그리기
            image_height, image_width, _ = image.shape
            center_x_pixel = int(center_x * image_width)
            center_y_pixel = int(center_y * image_height)
            cv2.circle(image, (center_x_pixel, center_y_pixel), 5, (0, 255, 0), -1)

            
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
cv2.destroyAllWindows()
