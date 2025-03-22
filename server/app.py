import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories for storing data
os.makedirs("static/snapshots", exist_ok=True)
os.makedirs("data", exist_ok=True)

# In-memory storage for registered devices and fall events
registered_devices = []
fall_events = []

# Configuration settings with defaults
config = {
    "fall_threshold": 0.05,
    "motion_threshold": 0.03,
    "velocity_history_size": 10,
    "immediate_fall_threshold": 0.9,
    "notification_threshold": 10,
    "video_source": "pan1.mp4"  # Default video source
}

# Load configuration if exists
config_path = "data/config.json"
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            config.update(json.load(f))
        logger.info("Configuration loaded from file")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")

# Save configuration
def save_config():
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info("Configuration saved to file")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

# Fall detection thread state
fall_detector_thread = None
fall_detector_running = False

# Import fall detector module
def import_fall_detector():
    import sys
    import os
    # Add parent directory to path to import fall_detector module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fall_detector import FallDetector
    return FallDetector

# Function to start fall detection in a separate thread
def start_fall_detection():
    global fall_detector_thread, fall_detector_running
    
    if fall_detector_running:
        logger.warning("Fall detection is already running")
        return False
    
    try:
        FallDetector = import_fall_detector()
        
        # Create fall detector instance with current config
        detector = FallDetector(
            video_source=config["video_source"],
            fall_threshold=config["fall_threshold"],
            motion_threshold=config["motion_threshold"],
            velocity_history_size=config["velocity_history_size"],
            immediate_fall_threshold=config["immediate_fall_threshold"],
            notification_threshold=config["notification_threshold"],
            enable_display=False,  # Disable display for server mode
            callback=handle_fall_event
        )
        
        # Start fall detection in a separate thread
        fall_detector_running = True
        fall_detector_thread = threading.Thread(target=detector.run)
        fall_detector_thread.daemon = True
        fall_detector_thread.start()
        
        logger.info("Fall detection started")
        return True
    except Exception as e:
        logger.error(f"Error starting fall detection: {e}")
        fall_detector_running = False
        return False

# Function to stop fall detection
def stop_fall_detection():
    global fall_detector_running
    fall_detector_running = False
    logger.info("Fall detection stopped")
    return True

# Callback function for fall detection events
def handle_fall_event(event_type, data):
    if event_type == "fall_detected":
        # Create fall event record
        event = {
            "id": len(fall_events) + 1,
            "timestamp": datetime.now().isoformat(),
            "confidence": data.get("confidence", 0),
            "snapshot_path": data.get("snapshot_path", ""),
            "acknowledged": False
        }
        
        # Add to events list
        fall_events.append(event)
        
        # Save snapshot to static directory if it exists
        if event["snapshot_path"] and os.path.exists(event["snapshot_path"]):
            filename = os.path.basename(event["snapshot_path"])
            destination = os.path.join("static/snapshots", filename)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Copy file
            import shutil
            shutil.copy2(event["snapshot_path"], destination)
            
            # Update path in event
            event["snapshot_url"] = f"/api/snapshots/{filename}"
        
        # Emit event via WebSocket
        socketio.emit("fall_detected", event)
        
        # Send push notifications to registered devices
        send_push_notifications(event)
        
        logger.info(f"Fall event processed: {event}")

# Function to send push notifications
def send_push_notifications(event):
    # In a real implementation, this would use Firebase Cloud Messaging
    # or another push notification service to send to registered devices
    logger.info(f"Would send push notification for event {event['id']} to {len(registered_devices)} devices")
    
    # For now, just log the notification
    for device in registered_devices:
        logger.info(f"Sending notification to device: {device['token']}")

# API Routes
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "running" if fall_detector_running else "stopped",
        "config": config,
        "registered_devices": len(registered_devices),
        "fall_events": len(fall_events)
    })

@app.route("/api/start", methods=["POST"])
def api_start_detection():
    success = start_fall_detection()
    return jsonify({"success": success, "status": "running" if success else "error"})

@app.route("/api/stop", methods=["POST"])
def api_stop_detection():
    success = stop_fall_detection()
    return jsonify({"success": success, "status": "stopped" if success else "error"})

@app.route("/api/register-device", methods=["POST"])
def register_device():
    data = request.json
    if not data or "token" not in data:
        return jsonify({"error": "Device token is required"}), 400
    
    # Check if device is already registered
    for device in registered_devices:
        if device["token"] == data["token"]:
            return jsonify({"message": "Device already registered"}), 200
    
    # Register new device
    device_info = {
        "token": data["token"],
        "platform": data.get("platform", "unknown"),
        "registered_at": datetime.now().isoformat()
    }
    registered_devices.append(device_info)
    
    logger.info(f"New device registered: {device_info['platform']}")
    return jsonify({"success": True, "message": "Device registered successfully"})

@app.route("/api/events", methods=["GET"])
def get_events():
    # Optional filtering by acknowledged status
    acknowledged = request.args.get("acknowledged")
    if acknowledged is not None:
        acknowledged = acknowledged.lower() == "true"
        filtered_events = [e for e in fall_events if e["acknowledged"] == acknowledged]
        return jsonify(filtered_events)
    
    return jsonify(fall_events)

@app.route("/api/events/<int:event_id>/acknowledge", methods=["POST"])
def acknowledge_event(event_id):
    for event in fall_events:
        if event["id"] == event_id:
            event["acknowledged"] = True
            return jsonify({"success": True})
    
    return jsonify({"error": "Event not found"}), 404

@app.route("/api/snapshots/<filename>", methods=["GET"])
def get_snapshot(filename):
    return send_from_directory("static/snapshots", filename)

@app.route("/api/settings", methods=["GET", "POST"])
def handle_settings():
    if request.method == "GET":
        return jsonify(config)
    
    # Update settings
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Update config with new values
    for key, value in data.items():
        if key in config:
            config[key] = value
    
    # Save updated config
    save_config()
    
    # Restart fall detection if running
    if fall_detector_running:
        stop_fall_detection()
        start_fall_detection()
    
    return jsonify({"success": True, "config": config})

# WebSocket events
@socketio.on("connect")
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

# Main entry point
if __name__ == "__main__":
    # Start fall detection on server startup
    start_fall_detection()
    
    # Start Flask server
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)