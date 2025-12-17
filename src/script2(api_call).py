import redis
import json
import os
import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
import requests
import threading
import datetime
import uuid
# Import ContentSettings explicitly from the Azure SDK
from azure.storage.blob import BlobServiceClient, ContentSettings

# ------------------------------
# Settings
# ------------------------------
# Redis and Local Folder Settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_NAME = "violation_queue"
ANNOTATED_IMAGES_FOLDER = "annotated_images"
FINAL_IMAGES_FOLDER = "final_images"
FINAL_LOG_FILE = "violations_log.json"

# Model Paths
PLATE_MODEL_PATH = "models/best(plate).pt"

# --- NEW: Infrastructure & API Configuration ---
# Azure Blob Storage details (from reference script)
AZURE_BLOB_CONNECTION_STRING = "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-09-29T18:30:00Z&st=2025-08-01T05:08:24Z&spr=https,http&sig=TU%2BYXGW4h4j9WfoEx2k%2Bxqbizbfu2juvwujLsjX9Gjg%3D"
AZURE_CONTAINER_NAME = 'nvrdatashinobi' # The container for storing violation images

# API details (from reference script, adjust as needed)
API_ENDPOINT = "https://arcisai.vmukti.com:8082/analyticsimage/analytics"
CAMERA_ID = "tushar-mobile"  # A specific ID for your traffic camera
ANALYTICS_ID = "4"  # A unique ID for traffic violation analytics
IMAGE_COUNT_FOR_API = "1" # Typically "1" for a single violation image

# --- Global variable for the Azure client ---
blob_service_client = None

# Create final images folder if it doesn't exist
if not os.path.exists(FINAL_IMAGES_FOLDER):
    os.makedirs(FINAL_IMAGES_FOLDER)

# ------------------------------
# NEW: Azure and API Functions
# ------------------------------
def initialize_azure_blob_storage():
    """
    Initializes the BlobServiceClient using the connection string and verifies container existence.
    """
    global blob_service_client
    if not AZURE_BLOB_CONNECTION_STRING or "..." in AZURE_BLOB_CONNECTION_STRING:
        print("❌ Azure Blob Storage connection string is a placeholder. API alerts will fail.")
        return False
    if not API_ENDPOINT or "YOUR_API_ENDPOINT_URL" in API_ENDPOINT:
        print("❌ API Endpoint is not configured. API alerts will not be sent.")
        return False
        
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
        print("✅ Azure Blob Service client initialized.")
        
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        try:
            container_client.get_container_properties()
            print(f"   -> Blob container '{AZURE_CONTAINER_NAME}' already exists.")
        except Exception:
            print(f"   -> Blob container '{AZURE_CONTAINER_NAME}' not found. Attempting to create...")
            try:
                blob_service_client.create_container(AZURE_CONTAINER_NAME)
                print(f"   -> Container '{AZURE_CONTAINER_NAME}' created successfully.")
            except Exception as e_create:
                print(f"   -> ❌ Failed to create container '{AZURE_CONTAINER_NAME}': {e_create}")
                blob_service_client = None
                return False
        return True
    except ImportError:
        print("❌ Azure SDK not found. Please install it: pip install azure-storage-blob requests")
    except Exception as e:
        print(f"❌ Error initializing Azure Blob Service client: {e}")
    blob_service_client = None
    return False

def send_alert_to_api(frame_to_send, violation_details):
    """
    Uploads an annotated violation image to Azure Blob Storage and sends a notification to the API.
    
    Args:
        frame_to_send (np.ndarray): The image frame (already annotated) to be uploaded.
        violation_details (dict): A dictionary containing metadata about the violation.
    """
    global blob_service_client
    if not blob_service_client:
        print("   -> ⚠️ Blob service client not initialized. Cannot send image for alert.")
        return False
    if frame_to_send is None or frame_to_send.size == 0:
        print("   -> ⚠️ Cannot send empty frame for alert.")
        return False

    try:
        is_success, img_encoded = cv2.imencode(".jpg", frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not is_success:
            print("   -> ❌ Error encoding image to JPEG for alert.")
            return False
        img_bytes = img_encoded.tobytes()
        
        blob_name_prefix = violation_details.get("class", "unknown_violation").replace(" ", "_")
        blob_name = f"{blob_name_prefix}_{CAMERA_ID}_{ANALYTICS_ID}_{uuid.uuid4().hex}.jpg"
        
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
        blob_content_settings = ContentSettings(content_type='image/jpeg')

        blob_client.upload_blob(img_bytes, blob_type="BlockBlob", content_settings=blob_content_settings, overwrite=True)
        blob_url = blob_client.url
        print(f"   -> ✅ Alert image uploaded to Azure Blob Storage.")

        # send_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-000")

        payload = {
            "cameradid": str(CAMERA_ID),
            "sendtime": violation_details.get("timestamp"),
            # "sendtime": send_time,
            "imgurl": str(blob_url),
            "an_id": str(ANALYTICS_ID),
            "ImgCount": str(IMAGE_COUNT_FOR_API),
            "person_name": None,
            "numberplateid": violation_details.get("plate_text", "N/A"),
            "eventType": violation_details.get("class", "unknown")
        }
        headers = {'Content-Type': 'application/json'}
        
        def api_call_thread_func(api_endpoint_url, json_payload, req_headers, b_name):
            try:
                response = requests.post(api_endpoint_url, json=json_payload, headers=req_headers, timeout=15)
                response.raise_for_status()
                print(f"   -> ✅ API call successful ({response.status_code}) for {b_name}")
            except requests.exceptions.Timeout:
                print(f"   -> ❌ API call timed out for {b_name}")
            except requests.exceptions.RequestException as e:
                print(f"   -> ❌ API call failed for {b_name}: {e}")
            except Exception as e:
                print(f"   -> ❌ Unexpected error during API call for {b_name}: {e}")

        thread = threading.Thread(target=api_call_thread_func, args=(API_ENDPOINT, payload, headers, blob_name))
        thread.start()
        return True
        
    except Exception as e:
        print(f"   -> ❌ An error occurred during the alert sending process: {e}")
        return False

# ------------------------------
# Initialization
# ------------------------------
print("🚀 Initializing processor script...")
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

# --- NEW: Initialize Azure integration ---
print("🚀 Initializing Azure integration...")
storage_initialized = initialize_azure_blob_storage()
if not storage_initialized:
    print("⚠️ Warning: Azure/API is not configured. Cloud alerts will be disabled.")
# ------------------------------

# Helper function to update the JSON log file
def update_log_file(log_entry):
    try:
        logs = {}
        if os.path.exists(FINAL_LOG_FILE) and os.path.getsize(FINAL_LOG_FILE) > 0:
            with open(FINAL_LOG_FILE, 'r') as f:
                logs = json.load(f)
        
        image_name = log_entry.pop("image_name_for_grouping")
        if image_name not in logs:
            logs[image_name] = []
        logs[image_name].append(log_entry)
        
        with open(FINAL_LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"   -> ❌ Error updating local log file: {e}")

# ------------------------------
# Main Processing Loop
# ------------------------------
print(f"\n🎧 Waiting for violation messages from queue '{QUEUE_NAME}'...")
while True:
    try:
        message = redis_client.brpop(QUEUE_NAME, timeout=0)
        data_string = message[1].decode('utf-8')
        violation_data = json.loads(data_string)
        
        print(f"\n📩 Received new violation: {violation_data['class']} in {violation_data['image_name']}")

        # 1. Load the annotated image
        image_name = violation_data["image_name"]
        image_path = os.path.join(ANNOTATED_IMAGES_FOLDER, image_name)
        if not os.path.exists(image_path):
            print(f"   -> ⚠️ Image not found: {image_path}. Skipping.")
            continue
        frame = cv2.imread(image_path)
        
        # 2. Crop the vehicle
        l, t, r, b = violation_data["bbox"]
        vehicle_crop = frame[t:b, l:r]
        if vehicle_crop.size == 0:
            print("   -> ⚠️ Vehicle crop failed (zero size). Skipping.")
            continue

        # 3. Detect number plate & perform OCR
        plate_results = plate_model(vehicle_crop, verbose=False)
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
        
        # 4. Draw plate text on the main image for context
        final_image_path = os.path.join(FINAL_IMAGES_FOLDER, image_name)
        text_to_draw = f"Plate: {recognized_text}"
        # cv2.putText(frame, text_to_draw, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ##--------------------------------------------------------------------##
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

        ##_---------------------------------------------------------------------------------------------------##
        
        # 5. Save the final image locally
        cv2.imwrite(final_image_path, frame)
        print(f"   -> 🖼️  Final image saved locally to '{final_image_path}'")
        
        # 6. Prepare data for logging and API call
        final_log_entry = {
            "image_name_for_grouping": image_name,
            "bbox": violation_data["bbox"],
            "class": violation_data["class"],
            "timestamp": violation_data["timestamp"],
            "plate_text": recognized_text,
            "plate_confidence": round(plate_confidence, 2),
            "final_image_path": final_image_path
        }
        
        # 7. Send alert to API and Blob Storage
        if storage_initialized:
            send_alert_to_api(frame, final_log_entry)
        
        # 8. Update the final JSON log
        update_log_file(final_log_entry)
        print(f"   -> 💾 Local log updated for {image_name}")

    except json.JSONDecodeError:
        print("Error: Could not decode message from Redis.")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        time.sleep(5)