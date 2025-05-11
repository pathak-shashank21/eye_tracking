# Eye Tracking with OpenSeeFace and Machine Learning

## Overview

This project provides a real-time, webcam-based eye-tracking system using OpenSeeFace for 3D facial landmark detection, combined with machine learning models for accurate gaze estimation. It's designed to track gaze points smoothly even with head movement, leveraging calibration techniques and predictive modeling.

## Features

* **Real-time gaze tracking**: Uses 3D face landmarks to accurately track gaze.
* **Blink-based Calibration**: Confirm calibration points by blinking, enabling easy interaction.
* **Smooth Gaze Visualization**: Visualize gaze points on screen with a dynamic and smooth gaze dot and trail.
* **Machine Learning Integration**: Predicts screen gaze coordinates from facial landmarks using Ridge Regression, Random Forest, or Neural Networks.
* **Robust socket-based backend**: Uses UDP sockets for efficient communication between tracking components.

## Project Structure

```
eye_tracking/
├── calibrated_eye_tracker.py    # Main gaze tracking application
├── calibrate.py                 # Calibration utility
├── facetracker.py               # Face data retrieval and processing
├── gaze_model.py                # Machine learning models for gaze estimation
├── input_reader.py              # Handles UDP socket input
├── receive_data.py              # Utility for debugging UDP socket
├── socket_test.py               # Testing utility for socket communication
├── calibration_data.csv         # Generated calibration data
├── tracker.py                   # OpenSeeFace tracking backend
└── models/
    ├── gaze_model_x.pkl
    ├── gaze_model_y.pkl
    └── scaler.pkl
```

## Installation

### Requirements

* Python 3.8+
* OpenCV
* NumPy
* SciKit-Learn
* Joblib
* ONNXRuntime
* Tkinter

### Setup

Clone the repository:

```bash
git clone https://gitlab.com/pathak.shashank01/eye_tracking.git
cd eye_tracking
```

Install dependencies:

```bash
pip install opencv-python numpy scikit-learn joblib onnxruntime tkinter
```

## Usage

### Calibration

Run calibration first to generate accurate data:

```bash
python calibrate.py
```

Follow on-screen prompts and confirm each calibration point by blinking.

### Start Eye Tracking

After calibration, launch the gaze tracker:

```bash
python calibrated_eye_tracker.py
```

Press `Esc` to exit the application.

## Machine Learning Model

The project includes a customizable gaze estimation model:

* **Ridge Regression (default)**
* **Random Forest**
* **Neural Network (MLP)**

Models are trained on calibration data automatically:

```python
# Example of creating a model
gaze_regressor = GazeRegressor(model_type='ridge', alpha=1.0)
gaze_regressor.load('calibration_data.csv')
```

## UDP Communication

* **Ports**:

  * OpenSeeFace Data: `11573`
  * Calibration Data: `11574`

### Debugging UDP

Use provided scripts to debug UDP streams:

```bash
python receive_data.py
python socket_test.py
```

## License

This project is provided as-is without a specified license. Be sure to check licensing of integrated third-party libraries like OpenSeeFace and SciKit-Learn.

## Contact

For questions, issues, or contributions, please use the GitLab repository: [GitLab Eye Tracking Project](https://gitlab.com/pathak.shashank01/eye_tracking)
