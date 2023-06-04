# #!/usr/bin/env python3
# import cv2
# import mediapipe as mp
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose


# # 웹캠, 영상 파일의 경우 이것을 사용하세요.:
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as pose:
#     # 초기 코의 수직 거리 설정
#     prev_distance = None
    
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("카메라를 찾을 수 없습니다.")
#             # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
#             continue

#         # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = pose.process(image)

#         # 포즈 주석을 이미지 위에 그립니다.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         mp_drawing.draw_landmarks(
#             image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

#         # 0번 landmark (코)의 위치와 발 끝 랜드마크의 수직 거리 계산
#         if results.pose_landmarks is not None:
#             landmarks = results.pose_landmarks.landmark

#             nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
#             foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

#             # 수직 거리 계산
#             distance = np.abs(nose.y - foot.y)

#             # 이전 거리와 비교하여 변화 감지
#             if prev_distance is not None:
#                 distance_threshold = 0.1  # 거리 변화 임계값 설정

#                 # 거리 변화 감지
#                 if np.abs(distance - prev_distance) > distance_threshold:
#                     print("코와 발 끝 수직 거리 변화 감지",distance)

#             # 현재 거리를 이전 거리로 업데이트
#             prev_distance = distance

#         cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
# cap.release()
