import cv2
import numpy as np
from ultralytics import YOLO
import os
import redis
import json
from datetime import datetime

# ------------------- CONFIGURABLE VARIABLES -------------------
INPUT_VIDEO_PATH = "assets/video7.mp4"
SAVE_FOLDER = "annotated_images"
YOLO_MODEL_PATH = "models/yolo11s.pt" # Using a standard model name
VEHICLE_CONFIDENCE = 0.5

# Redis Message Queue Settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "violation_queue"

# Classes for vehicle detection (COCO model class IDs)
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Traffic light color thresholds in HSV space
RED_LOWER = np.array([0, 120, 70])
RED_UPPER = np.array([10, 255, 255])
YELLOW_LOWER = np.array([15, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])
GREEN_LOWER = np.array([40, 100, 100])
GREEN_UPPER = np.array([80, 255, 255])
# Minimum number of pixels to detect a color
PIXEL_THRESHOLD = 10
VIOLATION_LIGHT = "GREEN"

# Frame resize limits to maintain performance
MAX_WIDTH = 1700
MAX_HEIGHT = 956
# --------------------------------------------------------------

# Global variables for ROI selection
traffic_light_roi = []
road_roi = []
current_roi = "traffic_light"

def select_roi_callback(event, x, y, flags, param):
    """
    Mouse callback function to capture points for ROIs.
    Appends the (x, y) coordinates to the appropriate list based on the current selection mode.
    """
    global traffic_light_roi, road_roi, current_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_roi == "traffic_light":
            traffic_light_roi.append((x, y))
        else:
            road_roi.append((x, y))

def get_signal_status(frame, roi):
    """
    Determines the traffic light color within its ROI by analyzing pixel colors in the HSV space.
    
    Args:
        frame: The current video frame.
        roi: The list of points defining the traffic light's region of interest.
        
    Returns:
        A string representing the detected color ("RED", "YELLOW", "GREEN", or "UNDEFINED").
    """
    if not roi:
        return "UNDEFINED"

    # Create a mask for the ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi, dtype=np.int32)], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convert to HSV for better color detection
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Check for each color
    red_mask = cv2.inRange(hsv_frame, RED_LOWER, RED_UPPER)
    yellow_mask = cv2.inRange(hsv_frame, YELLOW_LOWER, YELLOW_UPPER)
    green_mask = cv2.inRange(hsv_frame, GREEN_LOWER, GREEN_UPPER)

    if cv2.countNonZero(red_mask) > PIXEL_THRESHOLD:
        return "RED"
    elif cv2.countNonZero(yellow_mask) > PIXEL_THRESHOLD:
        return "YELLOW"
    elif cv2.countNonZero(green_mask) > PIXEL_THRESHOLD:
        return "GREEN"
    else:
        return "UNDEFINED"

def main():
    """
    Main function to run the red light violation detection system.
    """
    global traffic_light_roi, road_roi, current_roi

    # Ensure the folder for saving violation images exists
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # --- Establish connection to Redis ---
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Error connecting to Redis: {e}")
        print("Please ensure the Redis server is running and accessible.")
        return

    # Load the YOLO model
    model = YOLO(YOLO_MODEL_PATH)

    # Open the video file
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at '{INPUT_VIDEO_PATH}'.")
        return

    # Read the first frame for ROI selection
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't read the first frame from the video.")
        return

    # --- Resize frame if it's too large ---
    original_height, original_width = first_frame.shape[:2]
    should_resize = original_width > MAX_WIDTH or original_height > MAX_HEIGHT
    if should_resize:
        aspect_ratio = original_width / original_height
        if original_width / MAX_WIDTH > original_height / MAX_HEIGHT:
            width = MAX_WIDTH
            height = int(width / aspect_ratio)
        else:
            height = MAX_HEIGHT
            width = int(height * aspect_ratio)
        first_frame = cv2.resize(first_frame, (width, height))
    else:
        width, height = original_width, original_height

    # --- ROI selection UI loop ---
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi_callback)
    print("🟢 Select traffic light ROI by clicking points. Press 'n' when done.")
    while True:
        display_frame = first_frame.copy()
        if current_roi == "traffic_light":
            cv2.putText(display_frame, "Select TRAFFIC LIGHT ROI - Press 'n' for next step", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if traffic_light_roi:
                cv2.polylines(display_frame, [np.array(traffic_light_roi)], True, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Select ROAD ROI - Press 's' to start processing", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if traffic_light_roi: # Keep showing the first ROI
                cv2.polylines(display_frame, [np.array(traffic_light_roi)], True, (0, 255, 255), 2)
            if road_roi:
                cv2.polylines(display_frame, [np.array(road_roi)], True, (0, 0, 255), 2)

        cv2.imshow("Select ROI", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n') and traffic_light_roi:
            print("🟡 Traffic light ROI selected. Now select ROAD ROI.")
            current_roi = "road"
        elif key == ord('s') and road_roi:
            if not traffic_light_roi:
                print("❌ Error: Traffic light ROI must be selected first.")
                continue
            break
    cv2.destroyWindow("Select ROI")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video to the beginning

    # --- NEW: Set to store IDs of vehicles that have already violated ---
    violated_vehicle_ids = set()
    prev_signal_status = None

    # --- Main processing loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if should_resize:
            frame = cv2.resize(frame, (width, height))

        # Get current traffic signal status
        signal_status = get_signal_status(frame, traffic_light_roi)

        # --- NEW: Clear violation records when light is no longer red ---
        if signal_status != VIOLATION_LIGHT and prev_signal_status == VIOLATION_LIGHT:
            violated_vehicle_ids.clear()
            print("🚦 Signal changed from RED. Clearing violation set for the new cycle.")
        prev_signal_status = signal_status

        # Use YOLOv8 tracking
        results = model.track(frame, persist=True, verbose=False, conf=VEHICLE_CONFIDENCE, classes=VEHICLE_CLASS_IDS)
        
        road_roi_color = (0, 255, 0) # Green by default
        if signal_status == VIOLATION_LIGHT:
            road_roi_color = (0, 0, 255) # Red when light is red

        # Check if tracking IDs are present in the results
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                box_color = (0, 255, 0) # Default green
                label = model.names[int(cls)]

                # Check for violation: vehicle in road ROI during a red light
                if signal_status == VIOLATION_LIGHT and cv2.pointPolygonTest(np.array(road_roi), center_point, False) >= 0:
                    box_color = (0, 0, 255) # Red for violation
                    label = f"VIOLATION: {model.names[int(cls)]}"

                    # --- NEW: Check if this vehicle ID has already been flagged ---
                    if track_id not in violated_vehicle_ids:
                        # This is a new violation, process it
                        violated_vehicle_ids.add(track_id)
                        print(f"🚨 NEW VIOLATION DETECTED! Vehicle ID: {track_id}")

                        # --- START: Violation detected, send to queue (Copied Logic) ---
                        
                        # 1. Prepare violation data
                        timestamp = datetime.now().isoformat()
                        image_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}.jpg"
                        save_path = os.path.join(SAVE_FOLDER, image_name)

                        # 2. Save snapshot image with bounding box
                        snapshot = frame.copy()
                        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(snapshot, "WRONG WAY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imwrite(save_path, snapshot)
                        
                        # 3. Create JSON message
                        violation_message = {
                            "image_name": image_name,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "class": "wrong_direction",
                            "timestamp": timestamp
                        }
                        message_string = json.dumps(violation_message)
                        
                        # 4. Push message to Redis queue
                        try:
                            redis_client.lpush(QUEUE_NAME, message_string)
                            print(f"📦 Sent wrong_direction violation for ID {track_id} to queue.")
                        except redis.exceptions.RedisError as e:
                            print(f"❌ Could not send to Redis: {e}")
                        # --- END: Copied Logic ---


                # Draw bounding box and label on the display frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                # Add track ID to the label for clarity
                cv2.putText(frame, f"ID {track_id} - {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # Draw ROIs and signal status on the display frame
        cv2.polylines(frame, [np.array(traffic_light_roi)], True, (0, 255, 255), 2)
        cv2.polylines(frame, [np.array(road_roi)], True, road_roi_color, 2)
        cv2.putText(frame, f"Signal: {signal_status}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Violations this cycle: {len(violated_vehicle_ids)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("Red Light Violation System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 'q' pressed. Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing complete. Violation images saved in '{SAVE_FOLDER}'.")

if __name__ == "__main__":
    main()