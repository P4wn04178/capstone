
[캡스톤 디자인] OpenCV Python을 이용한 응용프로그램 (딥러닝을 이용한 자동 모자이크 처리 프로그램)
======================

# 1. OpenCV
## 1.1. OpenCV이란?
[**OpenCV**](https://ko.wikipedia.org/wiki/OpenCV은 실시간 컴퓨터 비전을 목적으로 한 프로그래밍 라이브러리이다. 원래는 인텔이 개발하였다. 실시간 이미지 프로세싱에 중점을 둔 라이브러리이다. 인텔 CPU에서 사용되는 경우 속도의 향상을 볼 수 있는 IPP(Intel Performance Primitives)를 지원한다. 이 라이브러리는 윈도, 리눅스 등에서 사용 가능한 크로스 플랫폼이며 오픈소스 BSD 허가서 하에서 무료로 사용할 수 있다. OpenCV는 TensorFlow , Torch / PyTorch 및 Caffe의 딥러닝 프레임워크를 지원한다.
****
# 2. Description
딥러닝을 이용한 자동 모자이크 처리 프로그램은 실시간 영상 혹은 녹화된 영상에서 원하는 특정 인물을 제외하고 다른 인물의 얼굴을 인식하어 자동으로 모자이크 처리하는 프로그램이다. openCV-python과 머신러닝 오픈소스 Tensorflow를 이용하여 얼굴을 인식하고, 모자이크 처리를 한다.

# 3. 제작 배경
최근 유튜브, 트위치, 아프리카 TV 등 인터넷 개인 방송 서비스가 유행하고 있다. 이러한 플랫폼을 통해 방송을 누구나 쉽게 시작 할 수 있게 되면서 남녀노소 불구하고 인터넷 상에 영상을 제작 및 배포를 할 수 있게 되었다. 하지만 쉬운 접근성 때문에 인터넷 상에서 벌어지는 무분별한 사진 및 동영상 공개로 초상권 침해에 대한 불만을 표하는 목소리가 커지고 있다. 이에 우리는 이 문제를 해결하고자 자동 모자이크 처리 프로그램 제작을 계획하게 되었다.

# 4. 실행 방법
```
# yolov3 on video
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi

# yolov3 on webcam 
python object_tracker.py --video 0 --output ./data/video/results.avi

# yolov3-tiny 
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-tiny.tf --tiny

# yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-custom.tf --num_classes <# CLASSES> --classes  ./data/labels/<YOUR CUSTOM .names FILE>
```

# 5. 정리


***** 

# P.S.

## ○ 참고문서

## Acknowledgments
* [Yolov3 TensorFlow Amazing Implementation](https://github.com/zzh8829/yolov3-tf2)
* [Deep SORT Repository](https://github.com/nwojke/deep_sort)
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [Yolov3_deepsort](https://github.com/theAIGuysCode/yolov3_deepsort)
