# AgeNet
Real-time age detection using deep neural networks

## Dependencies:
- NumPy 1.26.4
- OpenCV 4.10.0
- dlib 19.24.6

## How to Use:
To start face tracking and detection use command
`python -m src.face_detection.face_tracking`

To run ResNet estimation model use command
`python -m src.estimation.inference.resnet.accuracy_test -p src/estimation/weights/resnet_new`
