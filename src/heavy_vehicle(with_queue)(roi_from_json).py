import cv2
import numpy as np
from ultralytics import YOLO
import os
import redis
import json
from datetime import datetime

# ------------------- CONFIGURABLE VARIABLES -------------------
# --- Paths ---
INPUT_VIDEO_PATH = "assets/truck1.mp4"               #  UPDATE this path
ROI_CONFIG_FILE = "roi_data/heavy_vehicle_roi.json"                  #  UPDATE this path if needed
SAVE_FOLDER = "annotated_images"
YOLO_MODEL_PATH = "models/yolo11s.pt"                 # YOLO model path

# --- Detection Settings ---
TRUCK_CLASS_ID = 7                            # COCO dataset class ID for 'truck'
DETECTION_CONFIDENCE = 0.5                    # Minimum detection confidence

# --- Redis Message Queue Settings ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "violation_queue"

# --- Resize Settings ---
MAX_WIDTH = 1700
MAX_HEIGHT = 956
# --------------------------------------------------------------

def resize_if_needed(frame):
    """Resize only if frame exceeds max dimensions."""
    height, width = frame.shape[:2]
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        return cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame

def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # --- Redis connection ---
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Error connecting to Redis: {e}")
        return

    # --- Load ROI ---
    try:
        with open(ROI_CONFIG_FILE, 'r') as f:
            roi_data = json.load(f)
            roi_polygon = roi_data.get("traffic_light_roi", [])
        if not roi_polygon:
            print(f"❌ ROI polygon not found in '{ROI_CONFIG_FILE}'.")
            return
        roi_polygon_np = np.array(roi_polygon, dtype=np.int32)
        print(f"✅ Loaded ROI from '{ROI_CONFIG_FILE}'.")
    except FileNotFoundError:
        print(f"❌ ROI file not found at '{ROI_CONFIG_FILE}'.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error parsing '{ROI_CONFIG_FILE}'. Invalid JSON.")
        return

    # --- Load YOLO model ---
    model = YOLO(YOLO_MODEL_PATH)

    # --- Video capture ---

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video at '{INPUT_VIDEO_PATH}'.")
        return

    violated_truck_ids = set()

    print("\n--- 🚚 Starting Heavy Vehicle Detection ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Resize if needed
        frame = resize_if_needed(frame)

        # Run detection
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=DETECTION_CONFIDENCE,
            classes=[TRUCK_CLASS_ID],
            device='cuda'
        )

        # Draw ROI polygon
        cv2.polylines(frame, [roi_polygon_np], isClosed=True, color=(255, 255, 0), thickness=2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

                is_inside_roi = cv2.pointPolygonTest(roi_polygon_np, center_point, False) >= 0

                if is_inside_roi:
                    box_color = (0, 0, 255)
                    if track_id not in violated_truck_ids:
                        violated_truck_ids.add(track_id)
                        print(f"🚨 VIOLATION DETECTED! Truck ID: {track_id}")

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        image_name = f"truck_violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}.jpg"
                        save_path = os.path.join(SAVE_FOLDER, image_name)

                        snapshot = frame.copy()
                        cv2.rectangle(snapshot, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(snapshot, "VIOLATION", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                        cv2.imwrite(save_path, snapshot)

                        violation_message = {
                            "image_name": image_name,
                            "bbox": [x1, y1, x2, y2],
                            "class": "heavy_vehicle_violation",
                            "timestamp": timestamp
                        }
                        try:
                            redis_client.lpush(QUEUE_NAME, json.dumps(violation_message))
                            print(f"📦 Sent violation data for Truck ID {track_id} to Redis queue '{QUEUE_NAME}'.")
                        except redis.exceptions.RedisError as e:
                            print(f"❌ Redis error: {e}")
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"Truck ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # cv2.imshow("Truck Violation Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("🛑 'q' pressed. Exiting...")
        #     break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Processing complete. Annotated images saved in '{SAVE_FOLDER}'.")

if __name__ == "__main__":
    main()
