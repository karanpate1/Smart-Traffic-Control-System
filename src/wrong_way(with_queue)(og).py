import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import redis  # NEW: Import redis library
import json   # NEW: Import json library
from datetime import datetime # NEW: Import datetime for timestamps

# ------------------------------
# Settings
# ------------------------------
VIDEO_PATH = "assets/video3.mp4"
YOLO_MODEL_PATH = "models/yolo11s.pt" # Using yolov8s.pt as a common model
SAVE_FOLDER = "annotated_images" # NEW: Changed to the common folder for all violations
ANGLE_THRESHOLD_WRONG = 140

# NEW: Redis Message Queue Settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "violation_queue"

# Create save folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# NEW: Establish connection to Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client.ping() # Check if the connection is successful
    print("✅ Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"❌ Error connecting to Redis: {e}")
    print("Please ensure Redis server is running.")
    exit()

# ------------------------------
# Globals for Mouse Callbacks (No changes in this section)
# ------------------------------
defined_rois = []
defined_directions = []
current_roi_points = []
current_direction_points = []
drawing_roi = False
drawing_direction = False
current_roi_index = -1

# ------------------------------
# Mouse callback for ROI polygon and direction vector (No changes in this section)
# ------------------------------
def draw_interactive_elements(event, x, y, flags, param):
    global current_roi_points, current_direction_points, drawing_roi, drawing_direction, defined_rois, defined_directions, current_roi_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_roi:
            current_roi_points.append((x, y))
        elif drawing_direction and len(current_direction_points) < 2:
            current_direction_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if drawing_roi:
            if len(current_roi_points) > 2:
                defined_rois.append(np.array(current_roi_points, dtype=np.int32))
                current_roi_points = []
                drawing_roi = False
                drawing_direction = True
                current_roi_index = len(defined_rois) - 1
                print(f"ROI {current_roi_index + 1} defined. Now define its direction.")
            else:
                print("ROI needs at least 3 points.")
        elif drawing_direction and len(current_direction_points) == 2:
            main_dir_vector = np.array(current_direction_points[1]) - np.array(current_direction_points[0])
            defined_directions.append(main_dir_vector)
            current_direction_points = []
            drawing_direction = False
            current_roi_index = -1
            drawing_roi = True
            print(f"Direction for ROI {len(defined_directions)} defined. Ready for next ROI or press 's'.")
        elif not drawing_roi and not drawing_direction:
            drawing_roi = True
            print("Starting new ROI. Left-click to add points, right-click to complete ROI.")

# ------------------------------
# Vector angle helper (No changes in this section)
# ------------------------------
def calculate_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle = math.degrees(math.acos(dot_product))
    return angle

# ------------------------------
# Video and ROI Setup (No changes in this section)
# ------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()
H, W, _ = first_frame.shape
scale = min(1920 / W, 1080 / H, 1.0)
if scale < 1:
    new_size = (int(W * scale), int(H * scale))
    first_frame = cv2.resize(first_frame, new_size)
else:
    new_size = (W, H)

# --- ROI and Direction Definition UI Loop (No changes) ---
print("--- Setup Phase ---")
# (Instructions print statements are omitted for brevity but are unchanged)
cv2.namedWindow("Define ROIs and Directions")
cv2.setMouseCallback("Define ROIs and Directions", draw_interactive_elements)
drawing_roi = True
while True:
    temp_frame = first_frame.copy()
    # (Drawing logic for ROIs and directions is unchanged)
    for i, roi in enumerate(defined_rois):
        cv2.polylines(temp_frame, [roi], True, (0, 255, 0), 2)
    if drawing_roi and len(current_roi_points) > 0:
        cv2.polylines(temp_frame, [np.array(current_roi_points)], False, (0, 255, 255), 2)
    for i, direction_vec in enumerate(defined_directions):
        if len(defined_rois) > i:
            roi_center = np.mean(defined_rois[i], axis=0).astype(int)
            end_point = roi_center + direction_vec // 5
            cv2.arrowedLine(temp_frame, tuple(roi_center), tuple(end_point), (255, 0, 0), 3)
    if drawing_direction and len(current_direction_points) == 2:
        cv2.arrowedLine(temp_frame, current_direction_points[0], current_direction_points[1], (255, 0, 255), 3)
    cv2.imshow("Define ROIs and Directions", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if len(defined_rois) > 0 and len(defined_rois) == len(defined_directions):
            print("Setup complete. Starting inference.")
            break
        else:
            print("Please define at least one ROI and its corresponding direction.")
cv2.destroyWindow("Define ROIs and Directions")
# (Exit logic if no ROIs are defined is unchanged)
# ...

# ------------------------------
# YOLO with ByteTrack Setup
# ------------------------------
model = YOLO(YOLO_MODEL_PATH)
track_history = {}
wrong_direction_ids = set()
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_INTERVAL = max(1, int(fps / 5))

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if scale < 1:
        frame = cv2.resize(frame, new_size)

    # (Drawing ROIs and Directions on frame is unchanged)
    # ...

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        clss = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if cls in [2, 3, 5, 7]: # car, motorcycle, bus, truck
                l, t, r, b = box
                cx, cy = (l + r) // 2, (t + b) // 2
                relevant_main_dir_vector = None
                for i, roi in enumerate(defined_rois):
                    if cv2.pointPolygonTest(roi, (int(cx), int(cy)), False) >= 0:
                        relevant_main_dir_vector = defined_directions[i]
                        break

                if relevant_main_dir_vector is not None:
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((frame_count, (cx, cy), relevant_main_dir_vector))

                    if len(track_history[track_id]) >= 2:
                        first_frame_idx, first_pos, _ = track_history[track_id][0]
                        current_frame_idx, current_pos, current_relevant_main_dir_vector = track_history[track_id][-1]

                        if current_frame_idx - first_frame_idx >= FRAME_INTERVAL:
                            if track_id not in wrong_direction_ids:
                                current_vector = np.array(current_pos) - np.array(first_pos)

                                if np.linalg.norm(current_vector) > 10:
                                    angle = calculate_angle(current_relevant_main_dir_vector, current_vector)
                                    if ANGLE_THRESHOLD_WRONG <= angle <= 180:
                                        wrong_direction_ids.add(track_id)

                                        # --- NEW: Violation detected, send to queue ---
                                        
                                        # 1. Prepare violation data
                                        timestamp = datetime.now().isoformat()
                                        image_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}.jpg"
                                        save_path = os.path.join(SAVE_FOLDER, image_name)

                                        # 2. Save snapshot image with bounding box
                                        snapshot = frame.copy()
                                        cv2.rectangle(snapshot, (l, t), (r, b), (0, 0, 255), 2)
                                        cv2.putText(snapshot, "WRONG WAY", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                        cv2.imwrite(save_path, snapshot)
                                        
                                        # 3. Create JSON message
                                        violation_message = {
                                            "image_name": image_name,
                                            "bbox": [int(l), int(t), int(r), int(b)],
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
                                        # --- End of NEW section ---

                            track_history[track_id] = [(current_frame_idx, current_pos, current_relevant_main_dir_vector)]

                # (Drawing boxes on frame is unchanged)
                box_color = (0, 0, 255) if track_id in wrong_direction_ids else (255, 150, 0)
                label = f"ID:{track_id} WRONG" if track_id in wrong_direction_ids else f"ID:{track_id}"
                cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
                cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imshow("Wrong Direction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()