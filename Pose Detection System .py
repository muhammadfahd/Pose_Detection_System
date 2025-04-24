

import cv2
import mediapipe as mp
import math
import numpy as np
import time
from datetime import datetime
import os

# Create directory for saving snapshots
if not os.path.exists('pose_snapshots'):
    os.makedirs('pose_snapshots')

# Initialize model with improved settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0, 1, or 2 (higher means better accuracy but slower)
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam
capture = cv2.VideoCapture(0)

# Settings
display_skeleton = True
record_mode = False
out = None
exercise_mode = "posture"  # Default exercise mode
timer_duration = 5  # Seconds
timer_start = None
correct_pose_count = 0
total_pose_checks = 0
rep_count = 0
last_pos = None

# GUI settings
font = cv2.FONT_HERSHEY_SIMPLEX
primary_color = (0, 255, 0)  # Green
warning_color = (0, 0, 255)  # Red
info_color = (255, 255, 0)   # Yellow

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

def get_squat_feedback(landmarks):
    """Analyze squat form"""
    # Get relevant landmarks
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    
    # Calculate knee angle
    knee_angle = calculate_angle(hip, knee, ankle)
    
    if 80 <= knee_angle <= 110:
        return True, f"Good squat depth: {knee_angle:.1f}°", (0, 255, 0)
    elif knee_angle > 110:
        return False, f"Go deeper: {knee_angle:.1f}°", (0, 0, 255)
    else:
        return False, f"Too deep: {knee_angle:.1f}°", (0, 165, 255)

def get_posture_feedback(landmarks):
    """Analyze standing posture"""
    # Get relevant landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    
    # Check shoulder alignment
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    
    # Check ear-shoulder-hip alignment (should be in a vertical line)
    # Project ear and hip onto same x-coordinate as shoulder
    shoulder_ear_x_diff = abs(left_shoulder.x - left_ear.x)
    shoulder_hip_x_diff = abs(left_shoulder.x - left_hip.x)
    
    # Check if body is vertical
    hip_ankle_x_diff = abs(left_hip.x - left_ankle.x)
    
    # Combined posture score (lower is better)
    posture_score = shoulder_diff + shoulder_ear_x_diff + shoulder_hip_x_diff + hip_ankle_x_diff
    
    if posture_score < 0.15:
        return True, f"Excellent posture: {posture_score:.3f}", (0, 255, 0)
    elif posture_score < 0.25:
        return False, f"Good posture: {posture_score:.3f}", (0, 165, 255)
    else:
        return False, f"Poor posture: {posture_score:.3f}", (0, 0, 255)

def get_pushup_feedback(landmarks, frame_height):
    """Analyze pushup form"""
    # Get relevant landmarks
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    
    # Calculate elbow angle
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    
    # Check if hips are sagging (by comparing y-coordinates)
    hip_height = hip.y * frame_height
    shoulder_height = shoulder.y * frame_height
    hip_sag = hip_height - shoulder_height
    
    if elbow_angle < 90:
        phase = "down"
        if hip_sag > 30:
            return False, f"Keep hips up: {elbow_angle:.1f}°", (0, 0, 255), phase
        else:
            return True, f"Good form down: {elbow_angle:.1f}°", (0, 255, 0), phase
    else:
        phase = "up"
        if hip_sag > 30:
            return False, f"Keep hips up: {elbow_angle:.1f}°", (0, 0, 255), phase
        else:
            return True, f"Good form up: {elbow_angle:.1f}°", (0, 255, 0), phase

def draw_text_with_background(img, text, position, font, scale, text_color, thickness, bg_color):
    """Draw text with a semi-transparent background"""
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_w, text_h = text_size
    
    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, 
                 (position[0] - 5, position[1] - 5 - text_h),
                 (position[0] + text_w + 5, position[1] + 5),
                 bg_color, -1)
    
    # Add overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, position, font, scale, text_color, thickness)

while True:
    ret, frame = capture.read()
    if not ret:
        break
        
    # Flip frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Process the frame
    result = pose.process(rgb_frame)
    
    # Create status bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
    
    # Draw mode indicator
    draw_text_with_background(
        frame, 
        f"Mode: {exercise_mode.title()}", 
        (10, 30), 
        font, 0.7, 
        (255, 255, 255), 
        2, 
        (0, 100, 0)
    )
    
    # Draw recording indicator if recording
    if record_mode:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
        draw_text_with_background(
            frame, 
            "REC", 
            (w - 80, 30), 
            font, 0.7, 
            (255, 255, 255), 
            2, 
            (0, 0, 100)
        )
    
    # Draw accuracy if available
    if total_pose_checks > 0:
        accuracy = (correct_pose_count / total_pose_checks) * 100
        draw_text_with_background(
            frame, 
            f"Accuracy: {accuracy:.1f}%", 
            (w // 2 - 80, 30), 
            font, 0.7, 
            (255, 255, 255), 
            2, 
            (100, 100, 0)
        )
        
    # Draw rep counter for pushups
    if exercise_mode == "pushup":
        draw_text_with_background(
            frame, 
            f"Reps: {rep_count}", 
            (w - 150, 30), 
            font, 0.7, 
            (255, 255, 255), 
            2, 
            (100, 0, 100)
        )
    
    # Display timer if active
    if timer_start is not None:
        elapsed = time.time() - timer_start
        if elapsed < timer_duration:
            remaining = timer_duration - elapsed
            cv2.putText(
                frame,
                f"Hold for: {remaining:.1f}s", 
                (w//2 - 100, h//2), 
                font, 1.5, 
                (0, 165, 255), 
                2
            )
        else:
            # Timer completed
            cv2.putText(
                frame,
                "Great job!", 
                (w//2 - 100, h//2), 
                font, 1.5, 
                (0, 255, 0), 
                2
            )
            # Take snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_snapshots/good_pose_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            timer_start = None
    
    # Draw landmarks and analyze pose if detected
    if result.pose_landmarks:
        if display_skeleton:
            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
            )
        
        # Get landmark positions
        landmarks = result.pose_landmarks.landmark
        
        # Analyze based on selected exercise mode
        if exercise_mode == "posture":
            is_correct, feedback_text, feedback_color = get_posture_feedback(landmarks)
            
            # Start timer if pose is correct and timer not already running
            if is_correct and timer_start is None:
                timer_start = time.time()
            # Reset timer if pose becomes incorrect
            elif not is_correct:
                timer_start = None
                
        elif exercise_mode == "squat":
            is_correct, feedback_text, feedback_color = get_squat_feedback(landmarks)
            
        elif exercise_mode == "pushup":
            is_correct, feedback_text, feedback_color, phase = get_pushup_feedback(landmarks, h)
            
            # Count reps (when transitioning from down to up phase)
            if last_pos == "down" and phase == "up":
                rep_count += 1
            last_pos = phase
        
        else:
            # Default to original angle calculation
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            
            angle = math.degrees(math.atan2(right_knee.y - right_hip.y, right_knee.x - right_hip.x) - 
                               math.atan2(left_knee.y - left_hip.y, left_knee.x - left_hip.x))
            angle = abs(angle)
            
            if angle >= 45:
                is_correct = True
                feedback_color = (0, 255, 0)  # Green color for correct posture
                feedback_text = f"Correct Posture! Angle: {angle:.1f}°"
            else:
                is_correct = False
                feedback_color = (0, 0, 255)  # Red color for incorrect posture
                feedback_text = f"Incorrect Posture. Angle: {angle:.1f}°"
        
        # Draw feedback text with background
        draw_text_with_background(
            frame,
            feedback_text,
            (10, h - 30),
            font,
            1.0,
            (255, 255, 255),
            2,
            feedback_color
        )
        
        # Update pose metrics
        total_pose_checks += 1
        if is_correct:
            correct_pose_count += 1
        
        # Record video if in record mode
        if record_mode and out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(f'pose_snapshots/workout_{timestamp}.avi', fourcc, 20.0, (w, h))
            
        if record_mode and out is not None:
            out.write(frame)
    
    # Display help text
    control_text = "Controls: [Q]uit, [S]keleton, [R]ecord, [M]ode, [T]imer, [C]lear stats"
    draw_text_with_background(
        frame,
        control_text,
        (10, h - 10),
        font,
        0.5,
        (255, 255, 255),
        1,
        (100, 100, 100)
    )
    
    # Show the frame
    cv2.imshow('Advanced Pose Analysis', frame)
    
    # Process keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        display_skeleton = not display_skeleton
    elif key == ord('r'):
        record_mode = not record_mode
        if not record_mode and out is not None:
            out.release()
            out = None
    elif key == ord('m'):
        # Cycle through exercise modes
        modes = ["posture", "squat", "pushup", "custom"]
        current_index = modes.index(exercise_mode)
        exercise_mode = modes[(current_index + 1) % len(modes)]
    elif key == ord('t'):
        # Start a new timer for holding pose
        timer_start = time.time()
    elif key == ord('c'):
        # Clear stats
        correct_pose_count = 0
        total_pose_checks = 0
        rep_count = 0

# Clean up
if out is not None:
    out.release()
capture.release()
cv2.destroyAllWindows()