import redis
import json
import os
import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ------------------------------
# Settings
# ------------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "violation_queue"

# Folder paths
ANNOTATED_IMAGES_FOLDER = "annotated_images"
FINAL_IMAGES_FOLDER = "final_images" # NEW: Folder for images with plate text drawn on them

# YOLO model for number plate detection
PLATE_MODEL_PATH = "models/best(plate).pt"

# The final JSON log file
FINAL_LOG_FILE = "violations_log.json"

# NEW: Create final images folder if it doesn't exist
if not os.path.exists(FINAL_IMAGES_FOLDER):
    os.makedirs(FINAL_IMAGES_FOLDER)

# ------------------------------
# Initialization
# ------------------------------
print("🚀 Initializing processor script...")
# (Initialization for models and Redis remains the same)
try:
    plate_model = YOLO(PLATE_MODEL_PATH)
    print("✅ YOLO number plate model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit()
try:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu", # Set to "gpu" if your environment is fixed
        lang='en'
    )
    print("✅ PaddleOCR initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing PaddleOCR: {e}")
    exit()
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client.ping()
    print("✅ Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"❌ Error connecting to Redis: {e}")
    exit()

# ------------------------------
# Helper function to update the JSON log file (remains the same)
# ------------------------------
def update_log_file(log_entry):
    try:
        if os.path.exists(FINAL_LOG_FILE) and os.path.getsize(FINAL_LOG_FILE) > 0:
            with open(FINAL_LOG_FILE, 'r') as f:
                logs = json.load(f)
        else:
            logs = {}
    except (json.JSONDecodeError, FileNotFoundError):
        logs = {}
    image_name = log_entry.pop("image_name_for_grouping")
    if image_name not in logs:
        logs[image_name] = []
    logs[image_name].append(log_entry)
    with open(FINAL_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

# ------------------------------
# Main Processing Loop
# ------------------------------
print(f"\n🎧 Waiting for violation messages from queue '{QUEUE_NAME}'...")
while True:
    try:
        # Block and wait for a message from the queue
        message = redis_client.brpop(QUEUE_NAME, timeout=0)
        data_string = message[1].decode('utf-8')
        violation_data = json.loads(data_string)
        
        print(f"\n📩 Received new violation: {violation_data['class']} in {violation_data['image_name']}")

        # --- 1. Load the annotated image ---
        image_name = violation_data["image_name"]
        image_path = os.path.join(ANNOTATED_IMAGES_FOLDER, image_name)
        if not os.path.exists(image_path):
            print(f"   -> ⚠️  Image not found: {image_path}. Skipping.")
            continue
        frame = cv2.imread(image_path)
        
        # --- 2. Crop the vehicle ---
        l, t, r, b = violation_data["bbox"]
        vehicle_crop = frame[t:b, l:r]
        if vehicle_crop.size == 0:
            print("   -> ⚠️  Vehicle crop failed (zero size). Skipping.")
            continue

        # --- 3. Detect number plate & perform OCR ---
        plate_results = plate_model(vehicle_crop, verbose=False, device ='cuda')
        recognized_text = "N/A"
        plate_confidence = 0.0

        if len(plate_results[0].boxes) > 0:
            best_plate_box = plate_results[0].boxes[0]
            pl, pt, pr, pb = best_plate_box.xyxy[0].cpu().numpy().astype(int)
            plate_crop = vehicle_crop[pt:pb, pl:pr]
            
            if plate_crop.size > 0:
                print("   -> 🔎 Plate detected, performing OCR...")
                ocr_result = ocr.predict(plate_crop)
                
                if ocr_result and ocr_result[0]:
                    image_result = ocr_result[0]
                    texts = image_result.get('rec_texts', [])
                    scores = image_result.get('rec_scores', [])

                    recognized_text = "".join(texts)
                    if scores:
                        confidence_score = sum(scores) / len(scores)
                        plate_confidence = confidence_score
                        print(f"   -> ✅ OCR Result: '{recognized_text}' (Confidence: {plate_confidence:.2f})")
                else:
                        print("   -> ⚠️  OCR could not read text from the plate.")
            else:
                 print("   -> ⚠️  Plate crop failed (zero size).")
        else:
            print("   -> ⚠️  No number plate detected on the vehicle.")
        
        # --- NEW: 4. Draw plate text on the image and save it ---
        final_image_path = os.path.join(FINAL_IMAGES_FOLDER, image_name)
        
        # Define text properties
        text_to_draw = f"Plate: {recognized_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255) # White
        bg_color = (0, 0, 255) # Red background for visibility
        
        # Get text size to draw a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text_to_draw, font, font_scale, font_thickness)
        
        # Position for the text and background
        text_x = l
        text_y = t - 10
        bg_rect_start = (text_x, text_y - text_height - baseline)
        bg_rect_end = (text_x + text_width, text_y + baseline)
        
        # Draw the background rectangle and the text
        cv2.rectangle(frame, bg_rect_start, bg_rect_end, bg_color, -1) # -1 for filled rectangle
        cv2.putText(frame, text_to_draw, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Save the final image
        cv2.imwrite(final_image_path, frame)
        print(f"   -> 🖼️  Final image saved to '{final_image_path}'")
        
        # --- 5. Update the final JSON log ---
        final_log_entry = {
            "image_name_for_grouping": image_name,
            "bbox": violation_data["bbox"],
            "class": violation_data["class"],
            "timestamp": violation_data["timestamp"],
            "plate_text": recognized_text,
            "plate_confidence": round(plate_confidence, 2),
            "final_image_path": final_image_path # NEW: Add the path to the final image
        }
        
        update_log_file(final_log_entry)
        print(f"   -> 💾 Log updated for {image_name}")

    except json.JSONDecodeError:
        print("Error: Could not decode message from Redis.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        time.sleep(5)