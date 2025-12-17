# File: detect_wrong_direction.py
import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import redis
import json
from datetime import datetime

# ------------------------------
# SettingsE:\vmukti\anpr_pipeline\anpr_pipeline
# ------------------------------
VIDEO_PATH = "assets/video3.mp4"
YOLO_MODEL_PATH = "models/yolo11s.pt"
ROI_CONFIG_FILE = "roi_data/wrong_direction_rois.json" # Path to the saved ROI data
SAVE_FOLDER = "annotated_images"
ANGLE_THRESHOLD_WRONG = 140

# Redis Message Queue Settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "violation_queue"

# Create save folder if it doesn't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Establish connection to Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client.ping()
    print("✅ Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"❌ Error connecting to Redis: {e}. Please ensure Redis server is running.")
    exit()

# ------------------------------
# Vector angle helper
# ------------------------------
def calculate_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return math.degrees(math.acos(dot_product))

# --- NEW: Load ROIs and Directions from file ---
defined_rois = []
defined_directions = []
try:
    with open(ROI_CONFIG_FILE, 'r') as f:
        roi_data = json.load(f)
        for data in roi_data:
            # Convert loaded lists back to numpy arrays
            roi_points = np.array(data["roi_points"], dtype=np.int32)
            direction_vector = np.array(data["direction_vector"])
            defined_rois.append(roi_points)
            defined_directions.append(direction_vector)
    if not defined_rois:
        print(f"⚠️ ROI file '{ROI_CONFIG_FILE}' is empty. No areas to monitor.")
        exit()
    print(f"✅ Successfully loaded {len(defined_rois)} ROIs and directions from '{ROI_CONFIG_FILE}'.")
except FileNotFoundError:
    print(f"❌ Error: ROI file not found at '{ROI_CONFIG_FILE}'.")
    print("Please run 'draw_rois_and_directions.py' first to create it.")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: Could not parse '{ROI_CONFIG_FILE}'. Make sure it's a valid JSON.")
    exit()


# ------------------------------
# Video and YOLO Setup
# ------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties for scaling
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
scale = min(1920 / W, 1080 / H, 1.0)
new_size = (int(W * scale), int(H * scale)) if scale < 1 else (W, H)

model = YOLO(YOLO_MODEL_PATH)
track_history = {}
wrong_direction_ids = set()
frame_count = 0
FRAME_INTERVAL = max(1, int(fps / 5))

# ------------------------------
# Main Detection Loop
# ------------------------------
print("\n--- 🚫➡️ Starting Wrong Direction Detection ---")
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    if scale < 1:
        frame = cv2.resize(frame, new_size)

    # Draw the loaded ROIs and directions on each frame for visualization
    for i, roi in enumerate(defined_rois):
        cv2.polylines(frame, [roi], True, (0, 255, 0), 2)
        roi_center = np.mean(roi, axis=0).astype(int)
        end_point = roi_center + defined_directions[i] // 4
        cv2.arrowedLine(frame, tuple(roi_center), tuple(end_point), (255, 0, 0), 3)

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device='cuda')[0]

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        clss = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if cls not in [2, 3, 5, 7]: continue # car, motorcycle, bus, truck

            l, t, r, b = box
            cx, cy = (l + r) // 2, (t + b) // 2
            
            # Check if the center of the vehicle is inside any ROI
            relevant_main_dir_vector = None
            for i, roi in enumerate(defined_rois):
                if cv2.pointPolygonTest(roi, (int(cx), int(cy)), False) >= 0:
                    relevant_main_dir_vector = defined_directions[i]
                    break

            if relevant_main_dir_vector is not None:
                if track_id not in track_history: track_history[track_id] = []
                track_history[track_id].append((frame_count, (cx, cy), relevant_main_dir_vector))

                if len(track_history[track_id]) >= 2:
                    first_frame_idx, first_pos, _ = track_history[track_id][0]
                    current_frame_idx, current_pos, current_relevant_main_dir_vector = track_history[track_id][-1]

                    if current_frame_idx - first_frame_idx >= FRAME_INTERVAL:
                        if track_id not in wrong_direction_ids:
                            current_vector = np.array(current_pos) - np.array(first_pos)

                            if np.linalg.norm(current_vector) > 10: # Check for minimum movement
                                angle = calculate_angle(current_relevant_main_dir_vector, current_vector)
                                if ANGLE_THRESHOLD_WRONG <= angle <= 180:
                                    wrong_direction_ids.add(track_id)
                                    print(f"🚨 VIOLATION: Vehicle ID {track_id} going the wrong way.")

                                    # --- Violation detected, send to queue ---
                                    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-000")
                                    image_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}.jpg"
                                    save_path = os.path.join(SAVE_FOLDER, image_name)
                                    
                                    snapshot = frame.copy()
                                    cv2.rectangle(snapshot, (l, t), (r, b), (0, 0, 255), 2)
                                    cv2.putText(snapshot, "WRONG WAY", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    cv2.imwrite(save_path, snapshot)
                                    
                                    violation_message = {
                                        "image_name": image_name,
                                        "bbox": [int(l), int(t), int(r), int(b)],
                                        "class": "wrong_direction",
                                        "timestamp": timestamp
                                    }
                                    try:
                                        redis_client.lpush(QUEUE_NAME, json.dumps(violation_message))
                                        print(f"📦 Sent violation for ID {track_id} to Redis queue.")
                                    except redis.exceptions.RedisError as e:
                                        print(f"❌ Could not send to Redis: {e}")
                        
                        # Reset history for this vehicle to calculate next movement
                        track_history[track_id] = [(current_frame_idx, current_pos, current_relevant_main_dir_vector)]
            
            # Draw bounding box on the frame
            box_color = (0, 0, 255) if track_id in wrong_direction_ids else (255, 150, 0)
            label = f"ID:{track_id} WRONG" if track_id in wrong_direction_ids else f"ID:{track_id}"
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # cv2.imshow("Wrong Direction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 'q' pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Processing complete.")

