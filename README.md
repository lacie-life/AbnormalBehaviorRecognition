# AbnormalBehaviorRecognition
Abnormal Behavior Recognition

## Todo List

- [x] Preprocessing data
    - [x] Extract frames from video
    - [x] Remove background
    - [x] Create label
    - [x] Pose estimation label
    - [x] Create data loader
- [ ] Choose models
    - [x] Pose estimation model (resnext50 with 3 class walk, fall, fight)
    - [x] Fire detection model (yolov8, low accuracy)
    - [x] Processing strategy 

- [x] Train model
    - [x] Pose estimation model

- [x] Postprocessing data
    - [x] Calculate the start and end time of abnormal behavior

- [x] Test model
    - [x] Tuning with classes of abnormal behavior


## Acknowledgement

- [Pose estimation](https://github.com/BakingBrains/Pose_estimation/tree/main)
- [Yolov5](https://pytorch.org/hub/ultralytics_yolov5/)
- [Action recognition](https://github.com/cjf8899/Development_of_abnormal_behavior_recognition)
- [ResneXt](https://arxiv.org/abs/1611.05431)





