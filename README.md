
# Pose Detection System

## overview 
This application use the MediaPipe library and OpenCV to create a comprehensive real-time pose detection and analysis system. It can track body movements, analyze exercise form, count repetitions, and provide immediate feedback on posture and technique.
![Demo](/Demo.png)
- [Pose Detection System](#pose-detection-system)
  - [overview](#overview)
  - [Features](#features)
    - [Exercise Modes](#exercise-modes)
    - [Performance Tracking](#performance-tracking)
    - [Data Capture](#data-capture)
    - [Visual Interface](#visual-interface)
    - [User Controls](#user-controls)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Technical Details](#technical-details)
    - [Pose Detection](#pose-detection)
    - [Form Analysis Algorithms](#form-analysis-algorithms)
      - [Posture Analysis](#posture-analysis)
      - [Squat Analysis](#squat-analysis)
      - [Push-up Analysis](#push-up-analysis)
    - [Angle Calculation](#angle-calculation)
  - [Usage Examples](#usage-examples)
    - [Basic Usage](#basic-usage)
    - [Recording a Workout](#recording-a-workout)
    - [Form Analysis](#form-analysis)
  - [Customization](#customization)
    - [Adding New Exercise Modes](#adding-new-exercise-modes)
    - [Adjusting Sensitivity](#adjusting-sensitivity)
    - [Changing the UI](#changing-the-ui)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)



## Features

### Exercise Modes
- **Posture Analysis**: Evaluates standing posture by checking shoulder alignment, ear-shoulder-hip alignment, and body verticality
- **Squat Form Analysis**: Measures knee angles and depth to ensure proper squat technique
- **Push-up Counter**: Counts repetitions and analyzes form, including elbow angles and torso position
- **Custom Analysis**: Allows for custom angle measurements between any specified landmarks

### Performance Tracking
- Real-time accuracy percentage of correct form
- Rep counter for repetitive exercises
- Visual feedback on form quality with color-coded indicators

### Data Capture
- Automatic snapshots when good form is maintained for a set duration
- Video recording capability with timestamp organization
- Organized file storage for post-workout review

### Visual Interface
- Status bar with exercise information and performance metrics
- Skeleton overlay with joint connections
- Text with semi-transparent backgrounds for enhanced readability
- Color-coded feedback (green for correct form, red for incorrect)

### User Controls
- `Q` key: Quit the application
- `S` key: Toggle skeleton visibility
- `R` key: Start/stop recording video
- `M` key: Switch between exercise modes
- `T` key: Start a countdown timer for pose holding
- `C` key: Clear statistics and rep count

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Standard Python libraries: `time`, `datetime`, `os`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pose-detection.git
   cd pose-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. Run the application:
   ```bash
   python advanced_pose_detection.py
   ```

## Technical Details

### Pose Detection

The system uses MediaPipe's pose detection model which identifies 33 landmarks on the human body. These landmarks are then used to calculate angles and relationships between body parts for exercise analysis.
![Imgae](/pasted%20image%200.png)

### Form Analysis Algorithms

#### Posture Analysis
Evaluates:
- Shoulder alignment (horizontal)
- Ear-shoulder-hip alignment (vertical line)
- Overall body verticality
- Combined posture score based on deviations

#### Squat Analysis
Measures:
- Knee angle (between hip, knee, and ankle)
- Proper depth assessment
- Optimal squat form based on biomechanical principles

#### Push-up Analysis
Analyzes:
- Elbow angle (between shoulder, elbow, and wrist)
- Hip position to detect sagging
- Push-up phase detection (up/down) for rep counting

### Angle Calculation

The angle between three points (A, B, C) is calculated using the arctan2 function:

```python
def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle
```

## Usage Examples

### Basic Usage
1. Launch the application
2. Stand in front of your webcam
3. Use the `M` key to select your exercise mode
4. Perform your exercise and receive real-time feedback

### Recording a Workout
1. Set your desired exercise mode
2. Press `R` to start recording
3. Complete your workout with form feedback
4. Press `R` again to stop recording
5. Find your workout video in the `pose_snapshots` directory

### Form Analysis
1. Select the appropriate exercise mode
2. Perform the movement slowly
3. Watch the feedback to adjust your form
4. Hold the correct form and press `T` to start the timer
5. Maintain form until the timer completes to save a snapshot

## Customization

The application can be customized in several ways:

### Adding New Exercise Modes
Create a new analysis function following the pattern of existing ones:

```python
def get_new_exercise_feedback(landmarks):
    # Get relevant landmarks
    landmark_a = landmarks[mp_pose.PoseLandmark.LANDMARK_A]
    landmark_b = landmarks[mp_pose.PoseLandmark.LANDMARK_B]
    
    # Calculate metrics
    # ...
    
    # Return form assessment
    return is_correct, feedback_text, feedback_color
```

Then add your new exercise to the modes list:
```python
modes = ["posture", "squat", "pushup", "your_new_exercise", "custom"]
```

### Adjusting Sensitivity
Modify the thresholds in the feedback functions:

```python
# Example: Making posture detection more lenient
if posture_score < 0.20:  # Changed from 0.15
    return True, f"Excellent posture: {posture_score:.3f}", (0, 255, 0)
```

### Changing the UI
Adjust the drawing functions to modify the interface:

```python
# Example: Changing the status bar color
cv2.rectangle(frame, (0, 0), (w, 60), (70, 70, 70), -1)  # Changed from (50, 50, 50)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for their pose detection models
- [OpenCV](https://opencv.org/) for computer vision functionality

---

Created by- [Muhammad Fahd Bashir ](https://www.linkedin.com/in/mfahadbashir/)
