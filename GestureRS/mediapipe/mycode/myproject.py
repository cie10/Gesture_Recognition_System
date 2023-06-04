#!/usr/bin/env python3
import cv2
import mediapipe as mp

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

# 두 점 사이의 각도를 계산하는 함수
def calculate_angle(b, c):
    angle = cv2.fastAtan2(c[1] - b[1], c[0] - b[0]) - cv2.fastAtan2(a[1] - b[1], a[0] - b[0])
    angle = angle % 360
    if angle > 180:
        angle = 360 - angle
    return angle

# 메인 함수
def main():
    cap = cv2.VideoCapture(0)  # 웹캠 사용을 위해 0 또는 웹캠 장치 번호를 지정

    angles = []  # 각도를 저장할 리스트

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        skeleton = extract_skeleton(frame)
        if skeleton is not None:
            # 11번과 12번 좌표를 연결한 직선의 각도 계산
            angle_1 = calculate_angle(skeleton[11], skeleton[12])
            # 23번과 24번 좌표를 연결한 직선의 각도 계산
            angle_2 = calculate_angle(skeleton[23], skeleton[24])

            angles.append((angle_1 + angle_2) / 2)  # 평균 각도 저장
                
            if len(angles) > 30:
                    angles = angles[1:]  # 저장된 정보가 30개를 넘어가면 가장 오래된 정보를 제거

            average_angle = sum(angles) / len(angles)  # 평균 각도 계산

            if average_angle > 30:  # 평균 각도가 30도를 초과하면 낙상으로 간주
                print("낙상 감지!")

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
