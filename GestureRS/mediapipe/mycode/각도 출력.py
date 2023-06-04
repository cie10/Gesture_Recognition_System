import cv2
import mediapipe as mp
import math

# 어깨와 엉덩이 각도 계산 함수
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

# 메인 함수
def main():
    cap = cv2.VideoCapture(0)  # 웹캠 사용을 위해 0 또는 웹캠 장치 번호를 지정

    shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    hip_index = mp_pose.PoseLandmark.LEFT_HIP.value

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        skeleton = extract_skeleton(frame)
        if skeleton is not None:
            # 어깨 관절의 좌표 가져오기
            shoulder_line = [skeleton[shoulder_index], skeleton[shoulder_index + 1]]
            # 엉덩이 좌표 가져오기
            hip_line = [skeleton[hip_index], skeleton[hip_index + 1]]

            angle = calculate_shoulder_hip_angle(shoulder_line, hip_line)
            print("어깨와 엉덩이 각도:", angle)

            # 어깨 관절과 엉덩이 좌표를 비디오 화면에 그리기
            cv2.line(frame, shoulder_line[0], shoulder_line[1], (0, 255, 0), 2)
            cv2.line(frame, hip_line[0], hip_line[1], (0, 0, 255), 2)

            # 각도를 비디오 화면에 출력
            angle_text = "각도: {:.2f}".format(angle)
            cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 
