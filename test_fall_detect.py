import cv2
import mediapipe as mp
import numpy as np
import time
import os
import logging
from datetime import datetime
import math  # For angle calculations

# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Create snapshots directory to save images when fall is confirmed
snapshots_dir = "fall_snapshots"
os.makedirs(snapshots_dir, exist_ok=True)

# Create videos directory to save recorded videos
videos_dir = "recorded_videos"
os.makedirs(videos_dir, exist_ok=True)

# Initialize logging with updated path
log_filename = os.path.join(logs_dir, f"fall_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Fall detection system initialized.")

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open video capture
cap = cv2.VideoCapture("sample_vdo\pan1.mp4")  # or 0 for webcam

# Get video properties for recording
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:  # Fallback if FPS cannot be determined
    fps = 30

# Initialize video writer
video_filename = os.path.join(videos_dir, f"fall_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi format
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (1020, 600))  # Match the resize dimensions
logging.info(f"Video recording started: {video_filename}")

# Set FPS control
desired_fps = 30
prev_frame_time = 0
time_per_frame = 1.0 / desired_fps

# Variables for fall detection
previous_hip_y = None
pose_history = []
motion_history = []
fall_cooldown = 5
last_fall_time = 0

# Raise threshold from 0.05 to 0.1 to reduce false positives
fall_threshold = 0.1  # base threshold (scaled dynamically by frame height)
motion_threshold = 0.02

# --- New State Machine Variables ---
current_state = "unknown"  # can be 'standing', 'sitting', or 'lying'
state_confidence = 0.0     # confidence level in current state classification
# Removing has_been_standing flag to disable initial lying position logic

# --- Notification Variables ---
notification_threshold = 10  # seconds to wait before sending notification (set low for testing)
fall_event_time = None       # time when a fall event was first confirmed
notification_sent = False    # flag to ensure notification is sent only once per event
fall_snapshot_paths = []     # list to store snapshot paths
immediate_fall_threshold = 0.9  # Confidence threshold for immediate notification
fall_already_detected = False  # Flag to track if a fall has already been detected in this session
fall_detection_info = None   # Store fall detection info to display until the end

# --- Add configuration variables ---
ENABLE_DISPLAY = True        # Set to False for headless operation
DEBUG_MODE = True            # Enable additional debug information
SHOW_KEYPOINTS = True        # Set to False to hide keypoints

def draw_corner_lines_bbox(img, x_min, y_min, x_max, y_max, confidence, thickness=2):
    # Draw a simple rectangle with red color (0, 0, 255) in BGR
    color = (0, 0, 255)  # Red color for fall detection
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

def put_text_with_background(img, text, position="top-right", color=(0, 0, 160)):
    h, w = img.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    margin = 20
    if position == "top-right":
        x, y = w - text_width - margin, margin + text_height
    else:
        x, y = margin, h - margin
    cv2.rectangle(img, (x, y + baseline), (x + text_width, y - text_height), color, cv2.FILLED)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Adjust these parameters to increase fall detection sensitivity
fall_threshold = 0.05  # Lower threshold to increase sensitivity (was 0.1)
motion_threshold = 0.03  # Slightly higher to allow more movement during fall

# Add velocity history tracking
velocity_history = []
velocity_history_size = 10

# Add flag to control keypoint visualization
SHOW_KEYPOINTS = True  # Set to False to hide keypoints

def detect_fall_velocity(landmarks, frame_height):
    global previous_hip_y, velocity_history
    if previous_hip_y is None:
        previous_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        return False, 0.0
    try:
        current_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        delta = current_hip_y - previous_hip_y  # positive if moving downward
        downward_motion = (delta > 0)
        
        # Calculate velocity with increased sensitivity
        velocity_pixels = delta * frame_height / (time_per_frame * 0.5) if downward_motion else 0.0
        
        # Store velocity in history
        velocity_history.append(velocity_pixels)
        if len(velocity_history) > velocity_history_size:
            velocity_history.pop(0)
        
        # Use max velocity from recent history for more reliable detection
        max_recent_velocity = max(velocity_history) if velocity_history else velocity_pixels
        
        previous_hip_y = current_hip_y
        dynamic_threshold = fall_threshold * frame_height
        
        # Log significant motion as DEBUG instead of INFO (disabled by default)
        if downward_motion and velocity_pixels > dynamic_threshold * 0.5:
            logging.debug(f"Significant downward motion: {velocity_pixels:.2f} px/s (threshold: {dynamic_threshold:.2f})")
            
        # Return true if either current velocity or recent max velocity exceeds threshold
        is_fall = (downward_motion and (velocity_pixels > dynamic_threshold or max_recent_velocity > dynamic_threshold * 1.2))
        return is_fall, max(velocity_pixels, max_recent_velocity / 1.2)
    except ZeroDivisionError:
        return False, 0.0

def classify_pose(landmarks):
    try:
        head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        
        # Calculate multiple angles for better pose classification
        torso_angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
        
        # Add right side measurements for more robust detection
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_torso_angle = calculate_angle((r_shoulder.x, r_shoulder.y), (r_hip.x, r_hip.y), (r_knee.x, r_knee.y))
        
        # Average the angles from both sides
        avg_torso_angle = (torso_angle + r_torso_angle) / 2
        
        # Calculate horizontal alignment - key for detecting lying on furniture
        shoulder_hip_horizontal = abs(shoulder.x - hip.x)
        r_shoulder_hip_horizontal = abs(r_shoulder.x - r_hip.x)
        avg_horizontal_alignment = (shoulder_hip_horizontal + r_shoulder_hip_horizontal) / 2
        
        # Improved horizontal detection - more sensitive to detect lying on furniture
        is_horizontal = avg_horizontal_alignment > 0.12  # Lower threshold to catch more lying positions
        
        # Check for lying on back or stomach (vertical alignment of shoulders)
        shoulder_vertical_diff = abs(shoulder.y - r_shoulder.y)
        is_flat_lying = shoulder_vertical_diff < 0.05
        
        # Improved pose classification with confidence levels
        # First priority: Check for horizontal alignment (lying on side)
        if is_horizontal:
            return "lying", 0.95  # High confidence for horizontal alignment
        
        # Second priority: Check for flat lying (on back or stomach)
        if is_flat_lying and (avg_torso_angle < 140):
            return "lying", 0.9
            
        # Third priority: Check for traditional lying indicators
        if avg_torso_angle < 120:
            lying_confidence = 0.8 + (120 - avg_torso_angle) / 120
            return "lying", lying_confidence
            
        # Only classify as standing with strict criteria
        if head_y < hip_y < knee_y and not is_horizontal and not is_flat_lying:
            # Require clear vertical alignment and upright torso
            if avg_torso_angle > 160:  # More upright torso required for standing
                vertical_alignment = 1 - abs((hip_y - head_y) - (knee_y - hip_y)) / (knee_y - head_y)
                return "standing", 0.7 + vertical_alignment * 0.3
            else:
                return "sitting", 0.7  # More likely sitting if torso isn't upright
        elif abs(hip_y - knee_y) < 0.05:
            return "lying", 0.9
        else:
            return "sitting", 0.6
    except Exception as e:
        logging.error(f"Error classifying pose: {e}")
        return "unknown", 0.0

def update_state_and_detect_fall(prev_state, new_pose, new_confidence, fall_velocity_detected, velocity_score):
    fall_detected = False
    confidence = 0.0
    updated_state = prev_state
    
    # More aggressive fall detection logic
    if new_confidence > 0.5:  # Only consider poses with reasonable confidence
        # Only detect falls if there's a state transition TO lying position with high velocity
        if new_pose == "lying" and fall_velocity_detected:
            # Require higher velocity for fall detection to reduce false positives
            if velocity_score > 300 and prev_state != "lying":  # Added check for state transition
                fall_detected = True
                confidence = min(1.0, velocity_score * 1.5 * new_confidence)
                updated_state = "lying"
                logging.info(f"Fall detected! Previous state: {prev_state}, New state: {new_pose}, Velocity: {velocity_score:.2f}, Confidence: {confidence:.2f}")
            else:
                # Not enough velocity for a fall or already in lying position
                updated_state = new_pose
        else:
            # State change without fall detection
            updated_state = new_pose
            
        # Add state stability - prevent rapid state changes
        if updated_state != prev_state and prev_state != "unknown":
            # Log state transitions for debugging
            logging.debug(f"State transition: {prev_state} -> {updated_state} (conf: {new_confidence:.2f})")
    
    return updated_state, fall_detected, confidence

def detect_lack_of_movement(landmarks):
    global motion_history
    try:
        head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        motion_history.append(head_y)
        if len(motion_history) > 30:
            motion_history.pop(0)
        return (max(motion_history) - min(motion_history)) < motion_threshold
    except Exception as e:
        logging.error(f"Error detecting movement: {e}")
        return False

def send_notification(snapshot_paths, is_immediate=False):
    # Replace with your actual notification logic (email, SMS, push notification, etc.)
    if is_immediate:
        logging.info(f"URGENT NOTIFICATION: Fall detected with high confidence! Snapshots: {snapshot_paths}")
        print(f"URGENT NOTIFICATION: Fall detected with high confidence! Snapshots saved at {snapshot_paths}")
    else:
        logging.info(f"Notification sent: Person has been lying down for over the threshold duration. Snapshots: {snapshot_paths}")
        print(f"Notification: Person has been lying down. Snapshots saved at {snapshot_paths}")
    
    # Here you could add code to send email, SMS, or other notifications
    # Example (commented out as it requires additional setup):
    # send_email("Fall Alert", "A fall has been detected. Please check on the person.", snapshot_paths)
    # send_sms("Fall Alert: A fall has been detected. Please check on the person.")

def capture_snapshot(frame, prefix="fall", confidence=None):
    """Capture and save a snapshot of the current frame with timestamp and fall info"""
    # Create a copy of the frame to avoid modifying the original
    snapshot_frame = frame.copy()
    
    # Add timestamp to the image
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(snapshot_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add "Fall Detected" with confidence if provided
    if confidence is not None:
        fall_text = f"Fall Detected (Conf: {confidence:.2f})"
        cv2.putText(snapshot_frame, fall_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save the image
    snapshot_filename = f"{prefix}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
    cv2.imwrite(snapshot_path, snapshot_frame)
    logging.info(f"Snapshot captured: {snapshot_path}")
    return snapshot_path

# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - prev_frame_time
    
    # More efficient frame skipping
    if elapsed_time < time_per_frame:
        time.sleep(max(0, time_per_frame - elapsed_time))
        continue
        
    prev_frame_time = current_time

    frame = cv2.resize(frame, (1020, 600))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    fall_detected = False
    lying_detected = False
    confidence = 0.0

    if results.pose_landmarks:
        if current_time - last_fall_time < fall_cooldown:
            if not hasattr(detect_fall_velocity, 'cooldown_logged'):
                logging.info("Fall detection skipped due to cooldown period.")
                detect_fall_velocity.cooldown_logged = True
        else:
            if hasattr(detect_fall_velocity, 'cooldown_logged'):
                delattr(detect_fall_velocity, 'cooldown_logged')

            # Create a simple bounding box that surrounds the person like in the example image
            # This will create a rectangular box around the detected person
            
            # Get frame dimensions
            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark

            # Create a tight bounding box similar to YOLO detection
            # Get all visible keypoints
            visible_points = []
            for i, landmark in enumerate(landmarks):
                if landmark.visibility > 0.2:  # Only use visible landmarks
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    visible_points.append((x, y))
            
            if visible_points:
                # Find the minimum and maximum x, y coordinates
                x_points = [p[0] for p in visible_points]
                y_points = [p[1] for p in visible_points]
                
                # Calculate tight bounding box
                x_min = max(0, min(x_points))
                y_min = max(0, min(y_points))
                x_max = min(w, max(x_points))
                y_max = min(h, max(y_points))
                
                # Add minimal padding (5% of width/height)
                padding_x = int((x_max - x_min) * 0.05)
                padding_y = int((y_max - y_min) * 0.05)
                
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
                
                # Calculate box dimensions
                box_width = x_max - x_min
                box_height = y_max - y_min
            else:
                # Fallback if no landmarks with good visibility
                x_min, y_min = int(w * 0.2), int(h * 0.2)
                x_max, y_max = int(w * 0.8), int(h * 0.8)
                box_width = x_max - x_min
                box_height = y_max - y_min
            
            # 1) Check fall velocity with more sensitivity
            fall_velocity, velocity_score = detect_fall_velocity(landmarks, h)
            
            # Display velocity score in debug mode
            if DEBUG_MODE:  # Removed velocity threshold to always show velocity
                velocity_text = f"Velocity: {velocity_score:.2f}"
                put_text_with_background(frame, velocity_text, "top-left", (0, 0, 100))
            
            # 2) Classify current pose with confidence
            new_pose, pose_confidence = classify_pose(landmarks)
            
            # Now that we have the pose, we can adjust the bounding box if needed
            if new_pose == "lying" or box_width > box_height:
                # For lying pose, make the box wider to better capture the full body
                padding_x_extra = int((x_max - x_min) * 0.2)  # Additional horizontal padding
                x_min = max(0, x_min - padding_x_extra)
                x_max = min(w, x_max + padding_x_extra)
            
            # Add state tracking for fall detection - moved outside the lying condition
            prev_state = current_state  # Store previous state before updating
            
            # Add pose to history for more stable detection
            pose_history.append(new_pose)
            if len(pose_history) > 5:
                pose_history.pop(0)
            
            # 3) Update state machine with confidence
            current_state, fall_detected, confidence = update_state_and_detect_fall(
                prev_state, new_pose, pose_confidence, fall_velocity, velocity_score
            )
            state_confidence = pose_confidence  # Update global confidence
            
            # Handle fall detection and display
            if fall_detected and not fall_already_detected:
                last_fall_time = current_time
                logging.info(f"FALL DETECTED! Confidence: {confidence:.2f}, Velocity score: {velocity_score:.2f}")
                
                # Store fall detection info to display until the end
                # Use the full body bounding box we calculated earlier
                fall_detection_info = {
                    "text": f"FALL DETECTED (Conf: {confidence:.2f})",
                    "bbox": (x_min, y_min, x_max, y_max),
                    "confidence": confidence
                }
                
                if confidence > immediate_fall_threshold and velocity_score > 300:
                    urgent_snapshot_path = capture_snapshot(frame, prefix="fall", confidence=confidence)
                    send_notification([urgent_snapshot_path], is_immediate=True)
                    fall_already_detected = True
                    logging.info("First fall detected and notification sent. Ignoring subsequent falls.")

        # Draw pose landmarks if enabled (optional - you can remove if not needed)
        if SHOW_KEYPOINTS and results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

    # Always display fall detection info if it exists (moved outside the pose detection block)
    if fall_detection_info:
        put_text_with_background(frame, fall_detection_info["text"], "top-right", (0, 0, 255))
        # Removed bounding box drawing code but kept the text overlay

    # Always display the frame
    if ENABLE_DISPLAY:
        cv2.imshow("Fall Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Write the frame to video file (with all overlays)
    video_writer.write(frame)