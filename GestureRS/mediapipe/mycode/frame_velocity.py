#!/usr/bin/env python3
import cv2
import time

def main():
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    
    # 이전 시간 초기화
    prev_time = 0
    
    while True:
        # 현재 시간 측정
        current_time = time.time()
        
        # 프레임 읽기
        ret, frame = cap.read()
        
        if ret:
            # 프레임 표시
            cv2.imshow('Webcam', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) == ord('q'):
                break
        
        # 경과 시간 계산
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time
        
        # 프레임 속도 출력
        print("FPS:", fps)
        
        # 이전 시간 업데이트
        prev_time = current_time
    
    # 웹캠 해제
    cap.release()
    
    # 창 닫기
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
