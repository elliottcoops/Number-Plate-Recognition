# Number Plate Recognition

## About

This repository contains a custom YOLO model trained from scratch for number plate detection. Using EasyOCR for character recognition, enabling automatic reading of vehicle number plates from videos.

![Number Plate Recognition Demo](docs/output.gif)

## ML pipeline
The number plate recognition pipeline starts with a video frame input. Each frame is passed to YOLO, which detects and tracks number plates by predicting bounding boxes in real time. The detected plate regions are then cropped and passed to EasyOCR, which reads the characters and converts them into text strings. Finally, these recognized strings are displayed on the video.

More abstractly, the process involves:

- **Frame Input:** Raw video frames as the data source.

- **Detect plates (YOLO):** Locates and follows number plates across frames.

- **Crop plates:** Crops plate from the frame. 

- **Text Recognition (EasyOCR):** Extracts characters from the detected plate.

- **Construct string:** Join extracted characters together.

![ML pipeline](docs/ml_pipeline.png)

## Training

### YOLO model training

