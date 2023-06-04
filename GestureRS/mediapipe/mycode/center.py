#!/usr/bin/env python3
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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
            # 12번과 11번 좌표의 중심 좌표 계산
            shoulder_left_coords = (results.pose_landmarks.landmark[11].x,
                                    results.pose_landmarks.landmark[11].y)
            shoulder_right_coords = (results.pose_landmarks.landmark[12].x,
                                     results.pose_landmarks.landmark[12].y)
            shoulder_center_coords = ((shoulder_left_coords[0] + shoulder_right_coords[0]) / 2,
                                      (shoulder_left_coords[1] + shoulder_right_coords[1]) / 2)

            # 0번 좌표와 중심 좌표의 중심 좌표 계산
            head_coords = (results.pose_landmarks.landmark[0].x,
                           results.pose_landmarks.landmark[0].y)
            center_coords = ((head_coords[0] + shoulder_center_coords[0]) / 2,
                             (head_coords[1] + shoulder_center_coords[1]) / 2)

            # 중심 좌표의 수직 좌표 계산
            vertical_coordinate = results.pose_landmarks.landmark[0].z

            # 중심 좌표와 수직 좌표 출력
            print("중심 좌표:", center_coords)
            print("수직 좌표:", vertical_coordinate)

            # 중심점 그리기
            image_height, image_width, _ = image.shape
            shoulder_center_x_pixel = int(shoulder_center_coords[0] * image_width)
            shoulder_center_y_pixel = int(shoulder_center_coords[1] * image_height)
            head_x_pixel = int(head_coords[0] * image_width)
            head_y_pixel = int(head_coords[1] * image_height)
            center_x_pixel = int(center_coords[0] * image_width)
            center_y_pixel = int(center_coords[1] * image_height)
            cv2.circle(image, (shoulder_center_x_pixel, shoulder_center_y_pixel), 5, (255, 0, 0), -1)
            cv2.circle(image, (head_x_pixel, head_y_pixel), 5, (0, 0, 255), -1)
            cv2.circle(image, (center_x_pixel, center_y_pixel), 5, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
