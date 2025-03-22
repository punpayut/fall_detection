import cv2
import mediapipe as mp
import numpy as np
import time
import os
import logging
from datetime import datetime
import math

class FallDetector:
    def __init__(self, 
                 video_source="pan1.mp4", 
                 fall_threshold=0.05, 
                 motion_threshold=0.03,
                 velocity_history_size=10,
                 immediate_fall_threshold=0.9,
                 notification_threshold=10,
                 enable_display=True,
                 callback=None):
        
        # Initialize configuration parameters
        self.video_source = video_source
        self.fall_threshold = fall_threshold
        self.motion_threshold = motion_threshold
        self.velocity_history_size = velocity_history_size
        self.immediate_fall_threshold = immediate_fall_threshold
        self.notification_threshold = notification_threshold
        self.enable_display = enable_display
        self.callback = callback
        
        # Create directories if they don't exist
        self.logs_dir = "logs"
        self.snapshots_dir = "fall_snapshots"
        self.videos_dir = "recorded_videos"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        # Initialize state variables
        self.previous_hip_y = None
        self.pose_history = []
        self.motion_history = []
        self.velocity_history = []
        self.fall_cooldown = 5
        self.last_fall_time = 0
        self.current_state = "unknown"
        self.state_confidence = 0.0
        self.fall_event_time = None
        self.notification_sent = False
        self.fall_snapshot_paths = []
        self.fall_already_detected = False
        self.fall_detection_info = None
        
        # FPS control
        self.desired_fps = 30
        self.prev_frame_time = 0
        self.time_per_frame = 1.0 / self.desired_fps
        
        # Debug settings
        self.debug_mode = True
        self.show_keypoints = True
        
        logging.info("Fall detector initialized with settings:")
        logging.info(f"  Video source: {self.video_source}")
        logging.info(f"  Fall threshold: {self.fall_threshold}")
        logging.info(f"  Motion threshold: {self.motion_threshold}")
        logging.info(f"  Velocity history size: {self.velocity_history_size}")
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_filename = os.path.join(self.logs_dir, f"fall_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("Fall detection system initialized.")
    
    def draw_corner_lines_bbox(self, img, x_min, y_min, x_max, y_max, confidence, thickness=2):
        """Draw bounding box around detected person"""
        color = (0, 0, 255)  # Red color for fall detection
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    
    def put_text_with_background(self, img, text, position="top-right", color=(0, 0, 160)):
        """Add text with background to the image"""
        h, w = img.shape[:2]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        margin = 20
        if position == "top-right":
            x, y = w - text_width - margin, margin + text_height
        else:
            x, y = margin, h - margin
        cv2.rectangle(img, (x, y + baseline), (x + text_width, y - text_height), color, cv2.FILLED)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def detect_fall_velocity(self, landmarks, frame_height):
        """Detect fall velocity based on hip movement"""
        if self.previous_hip_y is None:
            self.previous_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            return False, 0.0
        try:
            current_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            delta = current_hip_y - self.previous_hip_y  # positive if moving downward
            downward_motion = (delta > 0)
            
            # Calculate velocity with increased sensitivity
            velocity_pixels = delta * frame_height / (self.time_per_frame * 0.5) if downward_motion else 0.0
            
            # Store velocity in history
            self.velocity_history.append(velocity_pixels)
            if len(self.velocity_history) > self.velocity_history_size:
                self.velocity_history.pop(0)
            
            # Use max velocity from recent history for more reliable detection
            max_recent_velocity = max(self.velocity_history) if self.velocity_history else velocity_pixels
            
            self.previous_hip_y = current_hip_y
            dynamic_threshold = self.fall_threshold * frame_height
            
            # Log significant motion as DEBUG instead of INFO (disabled by default)
            if downward_motion and velocity_pixels > dynamic_threshold * 0.5:
                logging.debug(f"Significant downward motion: {velocity_pixels:.2f} px/s (threshold: {dynamic_threshold:.2f})")
                
            # Return true if either current velocity or recent max velocity exceeds threshold
            is_fall = (downward_motion and (velocity_pixels > dynamic_threshold or max_recent_velocity > dynamic_threshold * 1.2))
            return is_fall, max(velocity_pixels, max_recent_velocity / 1.2)
        except ZeroDivisionError:
            return False, 0.0
    
    def classify_pose(self, landmarks):
        """Classify pose as standing, sitting, or lying"""
        try:
            head_y = landmarks[self.mp_pose.PoseLandmark.NOSE.value].y
            hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            knee_y = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            
            # Calculate multiple angles for better pose classification
            torso_angle = self.calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
            
            # Add right side measurements for more robust detection
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            r_torso_angle = self.calculate_angle((r_shoulder.x, r_shoulder.y), (r_hip.x, r_hip.y), (r_knee.x, r_knee.y))
            
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
    
    def update_state_and_detect_fall(self, prev_state, new_pose, new_confidence, fall_velocity_detected, velocity_score):
        """Update state machine and detect falls"""
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
    
    def detect_lack_of_movement(self, landmarks):
        """Detect if there is a lack of movement"""
        try:
            head_y = landmarks[self.mp_pose.PoseLandmark.NOSE.value].y
            self.motion_history.append(head_y)
            if len(self.motion_history) > 30:
                self.motion_history.pop(0)
            return (max(self.motion_history) - min(self.motion_history)) < self.motion_threshold
        except Exception as e:
            logging.error(f"Error detecting movement: {e}")
            return False
    
    def send_notification(self, snapshot_paths, is_immediate=False):
        """Send notification when fall is detected"""
        if is_immediate:
            logging.info(f"URGENT NOTIFICATION: Fall detected with high confidence! Snapshots: {snapshot_paths}")
            print(f"URGENT NOTIFICATION: Fall detected with high confidence! Snapshots saved at {snapshot_paths}")
        else:
            logging.info(f"Notification sent: Person has been lying down for over the threshold duration. Snapshots: {snapshot_paths}")
            print(f"Notification: Person has been lying down. Snapshots saved at {snapshot_paths}")
        
        # Call the callback function if provided
        if self.callback:
            event_data = {
                "confidence": self.fall_detection_info["confidence"] if self.fall_detection_info else 0.0,
                "snapshot_path": snapshot_paths[0] if snapshot_paths else None,
                "is_immediate": is_immediate
            }
            self.callback("fall_detected", event_data)
    
    def capture_snapshot(self, frame, prefix="fall", confidence=None):
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
        snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, snapshot_frame)
        logging.info(f"Snapshot captured: {snapshot_path}")
        return snapshot_path
        
    def run(self):
        """Main detection loop"""
        # Open video capture
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logging.error(f"Failed to open video source: {self.video_source}")
            return False
            
        # Get video properties for recording
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # Fallback if FPS cannot be determined
            fps = 30
            
        # Initialize video writer if recording is enabled
        video_writer = None
        if self.videos_dir:
            video_filename = os.path.join(self.videos_dir, f"fall_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi format
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (1020, 600))  # Match the resize dimensions
            logging.info(f"Video recording started: {video_filename}")
        
        # Main processing loop
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                elapsed_time = current_time - self.prev_frame_time
                
                # More efficient frame skipping
                if elapsed_time < self.time_per_frame:
                    time.sleep(max(0, self.time_per_frame - elapsed_time))
                    continue
                    
                self.prev_frame_time = current_time

                frame = cv2.resize(frame, (1020, 600))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                fall_detected = False
                lying_detected = False
                confidence = 0.0

                if results.pose_landmarks:
                    if current_time - self.last_fall_time < self.fall_cooldown:
                        if not hasattr(self.detect_fall_velocity, 'cooldown_logged'):
                            logging.info("Fall detection skipped due to cooldown period.")
                            self.detect_fall_velocity.cooldown_logged = True
                    else:
                        if hasattr(self.detect_fall_velocity, 'cooldown_logged'):
                            delattr(self.detect_fall_velocity, 'cooldown_logged')

                        # Create a simple bounding box that surrounds the person
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
                        fall_velocity, velocity_score = self.detect_fall_velocity(landmarks, h)
                        
                        # Display velocity score in debug mode
                        if self.debug_mode:  # Removed velocity threshold to always show velocity
                            velocity_text = f"Velocity: {velocity_score:.2f}"
                            self.put_text_with_background(frame, velocity_text, "top-left", (0, 0, 100))
                        
                        # 2) Classify current pose with confidence
                        new_pose, pose_confidence = self.classify_pose(landmarks)
                        
                        # Now that we have the pose, we can adjust the bounding box if needed
                        if new_pose == "lying" or box_width > box_height:
                            # For lying pose, make the box wider to better capture the full body
                            padding_x_extra = int((x_max - x_min) * 0.2)  # Additional horizontal padding
                            x_min = max(0, x_min - padding_x_extra)
                            x_max = min(w, x_max + padding_x_extra)
                        
                        # Add state tracking for fall detection - moved outside the lying condition
                        prev_state = self.current_state  # Store previous state before updating
                        
                        # Add pose to history for more stable detection
                        self.pose_history.append(new_pose)
                        if len(self.pose_history) > 5:
                            self.pose_history.pop(0)
                        
                        # 3) Update state machine with confidence
                        self.current_state, fall_detected, confidence = self.update_state_and_detect_fall(
                            prev_state, new_pose, pose_confidence, fall_velocity, velocity_score
                        )
                        self.state_confidence = pose_confidence  # Update global confidence
                        
                        # Handle fall detection and display
                        if fall_detected and not self.fall_already_detected:
                            self.last_fall_time = current_time
                            logging.info(f"FALL DETECTED! Confidence: {confidence:.2f}, Velocity score: {velocity_score:.2f}")
                            
                            # Store fall detection info to display until the end
                            # Use the full body bounding box we calculated earlier
                            self.fall_detection_info = {
                                "text": f"FALL DETECTED (Conf: {confidence:.2f})",
                                "bbox": (x_min, y_min, x_max, y_max),
                                "confidence": confidence
                            }
                            
                            if confidence > self.immediate_fall_threshold and velocity_score > 300:
                                urgent_snapshot_path = self.capture_snapshot(frame, prefix="fall", confidence=confidence)
                                self.send_notification([urgent_snapshot_path], is_immediate=True)
                                self.fall_already_detected = True
                                logging.info("First fall detected and notification sent. Ignoring subsequent falls.")

                    # Draw pose landmarks if enabled (optional - you can remove if not needed)
                    if self.show_keypoints and results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, 
                            results.pose_landmarks, 
                            self.mp_pose.POSE_CONNECTIONS
                        )

                # Always display fall detection info if it exists (moved outside the pose detection block)
                if self.fall_detection_info:
                    self.put_text_with_background(frame, self.fall_detection_info["text"], "top-right", (0, 0, 255))
                    # Removed bounding box drawing code but kept the text overlay

                # Always display the frame if enabled
                if self.enable_display:
                    cv2.imshow("Fall Detection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Write the frame to video file (with all overlays)
                if video_writer:
                    video_writer.write(frame)
                    
        except Exception as e:
            logging.error(f"Error in fall detection loop: {e}")
        finally:
            # Clean up resources
            cap.release()
            if video_writer:
                video_writer.release()
            if self.enable_display:
                cv2.destroyAllWindows()
            
            logging.info("Fall detection stopped")
            return True