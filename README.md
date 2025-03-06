# Driver Drowsiness Detection System

## Overview
This project implements a real-time Driver Drowsiness Detection System using computer vision and deep learning techniques. The system monitors the driver's facial features, particularly the eyes, to detect signs of drowsiness and alerts the driver when fatigue is detected.

## Features
- Real-time face detection
- Eye state monitoring (Open/Closed)
- Audio alerts for drowsiness warning
- User-friendly interface
- Real-time statistics and monitoring

## Requirements
- Python 3.7+
- OpenCV
- dlib
- numpy
- pygame
- imutils
- scipy

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Atharva010903/DriverDrowsinessDetection
cd DriverDrowsiness
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the shape predictor file:
- Place it in the project root directory

## Usage
1. Run the main application:
```bash
python drowsiness_detection.py
```

2. Position yourself in front of the camera
3. The system will automatically:
   - Detect your face
   - Monitor your eye state
   - Alert you if signs of drowsiness are detected

## How It Works
The system uses:
- Face detection to locate the driver's face in the video stream
- Facial landmark detection to identify key points around the eyes.
- Eye Aspect Ratio (EAR) calculation to determine eye state.
- Time-based monitoring to track prolonged eye closure.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- dlib for facial landmark detection
- OpenCV community
- All contributors to this project

## Safety Notice
This system is meant to be an aid and should not be relied upon as the sole means of preventing drowsy driving. Always ensure you are well-rested before operating a vehicle. 