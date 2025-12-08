# Hand Gesture Movement Controller

This project implements a hand gesture movement controller using Python, OpenCV, and MediaPipe. It detects hand landmarks to calculate height, pitch, roll, and forward/backward movements, visualizing the data on a dashboard.

## Features

- **Hand Detection**: Uses MediaPipe to detect hand landmarks.
- **Height Calculation**: Calculates the height of the wrist relative to the bottom of the frame, with a stability threshold to reduce jitter.
- **Pitch & Roll**: Computes the pitch and roll angles of the hand.
- **Forward/Backward Detection**: Determines if the hand is moving forward or backward based on palm orientation.
- **Visual Dashboard**: Displays real-time data for height, pitch, and roll on the video feed.

## Prerequisites

- Python 3.x
- Webcam

## Dependencies

Install the required dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

1.  Clone the repository or download the source code.
2.  Navigate to the project directory.
3.  Run the application:

```bash
python main.py
```

4.  The application will open a window showing the camera feed with the gesture controls.
    -   **Exit**: Press `Esc` to close the application.

## Configuration

You can adjust the following constants in `main.py` to tune the application:

-   `MAX_PHYSICAL_HEIGHT_MM`: Maximum physical height corresponding to the frame height.
-   `HEIGHT_STABILITY_THRESHOLD`: Threshold for ignoring small height changes (deadzone).
