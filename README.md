# YOLOv8-Based Helmet Violation Detection and Tracking System

## Project Overview

The Helmet Violation Detection and Tracking System is an Artificial Intelligence and Computer Vision based project developed to automatically identify whether two-wheeler riders are wearing helmets.

The system uses a YOLOv8-based deep learning model to detect helmet compliance from images and video footage. For video processing, ByteTrack is used to maintain tracking identities across consecutive frames. OpenCV is used for image and video processing, while Python is used to integrate all components into a complete application.

The system can identify riders as:

- With Helmet
- Without Helmet

The project also provides rider statistics, visual detection results, processed video output, CSV data, and PDF reports.

## Objectives

- Detect helmet usage automatically using deep learning.
- Identify riders without helmets.
- Process both images and video footage.
- Track riders across video frames.
- Calculate rider and violation statistics.
- Display detection results using bounding boxes and labels.
- Generate structured CSV and PDF reports.
- Demonstrate the practical application of AI in road safety monitoring.

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- ByteTrack
- PyTorch
- Pandas
- ReportLab
- Tkinter

## System Workflow

Input Image / Video

↓

YOLOv8 Detection

↓

Helmet Classification

↓

With Helmet / Without Helmet

↓

ByteTrack Tracking for Video

↓

Statistics and Visualization

↓

CSV / PDF Report

## YOLOv8 Model

The project uses a pretrained YOLOv8-based helmet detection model.

The model file used during development is:

`best.pt`

The model contains the following classes:

- With Helmet
- Without Helmet

The model generates bounding boxes, class labels, and confidence scores for detected objects.

## Image Detection

The image detection module allows an input image to be processed using the YOLOv8 model.

The output displays:

- Bounding boxes
- Helmet status
- Confidence score

Detected helmeted and non-helmeted riders are visually distinguished in the output.

## Video Detection and Tracking

For video processing, OpenCV reads the input video frame by frame.

YOLOv8 performs helmet detection on each frame, while ByteTrack is used to maintain tracking IDs for detected riders.

This allows the system to estimate unique riders rather than counting the same rider repeatedly in every frame.

## Statistics

The system can maintain information such as:

- Total tracked riders
- Riders with helmets
- Riders without helmets
- Current violations
- Detection confidence
- Processing FPS

## Reporting

Detection information can be stored in CSV format using Pandas.

A PDF summary report can also be generated using ReportLab. The report can contain:

- Total Riders
- Helmet Riders
- Riders Without Helmet
- Compliance Percentage
- Violation Rate
- Number of Processed Frames

## Project Files

| File | Description |
|------|-------------|
| `helmet_tracker.py` | Main helmet detection and tracking implementation |
| `detect.py` | Detection-related implementation |
| `check_model.py` | Used to verify the loaded model and its classes |
| `track_debug.py` | Used for testing and debugging object tracking |
| `README.md` | Project documentation |

## Installation

Install the required Python packages before running the project.

```bash
pip install ultralytics opencv-python pandas reportlab
```

## Running the Project

Place the required model file in the project directory:

best.pt

Run the required Python file from the project directory.

Example:

python helmet_tracker.py

The program can then be used to process the required image or video input.

## Limitations

The accuracy of the system can be affected by:

Poor lighting
Low-resolution footage
Motion blur
Camera angle
Distance between camera and rider
Occlusion
Crowded traffic scenes
Variations in helmet appearance

The system uses a pretrained model, and therefore its performance depends on the data and conditions represented during model training.

## Future Scope

The project can be further improved by:

Fine-tuning the model using a larger dataset.
Improving detection accuracy.
Improving tracking reliability.
Supporting real-time CCTV cameras.
Adding automatic license plate recognition.
Recording timestamps and locations.
Generating automatic violation evidence.
Developing a web-based monitoring dashboard.
Deploying the system on GPU or edge-computing hardware.
Conclusion

This project demonstrates the application of YOLOv8, computer vision, and object tracking for automated helmet violation detection. The integration of detection, tracking, visualization, data processing, and reporting provides a foundation for developing more advanced intelligent traffic monitoring systems.
