개별연구01_노인들을 위한 제스처 인식 시스템
# 노인을 위한 낙상 사고 감지 시스템

**1)  아이디어 배경**
   
기존 연구: 영상에서 PoseNet을 이용해 추출한 인체 골격 키포인트(Keypoints) 정보로 머리와 어깨의 키포인트들의 위치와 위치 변화 가속도를 추정함으로써 낙상 판단의 정확도를 높이기 위한 감지 방법을 연구하였다. [2]  
해당 낙상 감지 알고리즘을 이용했을 때의 문제점은 취침 동작, 넘어진 후 일어나는 동작과 낙상을 구분하지 못한다는 점이다.   
따라서, 허리 각도의 변화량을 이용하여 낙상을 감지하고, 또한 몸의 움직임의 여부를 확인하여 지속시간을 이용해 낙상의 정확도를 높이고자 하였다. 

**2) 제안하는 서비스**  

<img width="450" alt="image" src="https://github.com/cie10/Gesture_Recognition_System/assets/111051264/61f84760-3957-45eb-a048-f91939ab9d58">

**3) 알고리즘 소개**  
- 낙상 판단 알고리즘
    
   <img width="505" alt="image" src="https://github.com/cie10/Gesture_Recognition_System/assets/111051264/6b4b43d7-2bf0-4b06-b207-311fa3348a71">

- 허리 각도 변화량 측정  

  <img width="481" alt="image" src="https://github.com/cie10/Gesture_Recognition_System/assets/111051264/32eab360-65c5-4695-a1c7-73f7ef4c8647">

- 움직임 여부의 지속시간 측정

  ![image](https://github.com/cie10/Gesture_Recognition_System/assets/111051264/d68d436c-e162-4475-8171-ef66eb498819)









