# File: draw_roi.py
import cv2
import numpy as np
import json
import os

# ------------------- CONFIGURABLE VARIABLES -------------------
INPUT_VIDEO_PATH = "assets/video1.mp4"
ROI_CONFIG_FILE = "roi_data/red_light_rois.json"
MAX_WIDTH = 1700
MAX_HEIGHT = 956
# --------------------------------------------------------------

# Global variables for ROI selection
traffic_light_roi = []
restricted_area_roi = []
current_roi = "traffic_light"

def select_roi_callback(event, x, y, flags, param):
    """
    Mouse callback function to capture points for ROIs.
    Appends the (x, y) coordinates to the appropriate list based on the current selection mode.
    """
    global traffic_light_roi, restricted_area_roi, current_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_roi == "traffic_light":
            traffic_light_roi.append([x, y])
        else:
            restricted_area_roi.append([x, y])
        print(f"🔹 Point added: ({x}, {y})")

def main():
    """
    Main function to draw and save ROIs to a JSON file.
    """
    global traffic_light_roi, restricted_area_roi, current_roi

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at '{INPUT_VIDEO_PATH}'.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't read the first frame from the video.")
        cap.release()
        return
    cap.release() # We only need the first frame

    # --- Resize frame if it's too large ---
    original_height, original_width = first_frame.shape[:2]
    if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
        aspect_ratio = original_width / original_height
        if original_width / MAX_WIDTH > original_height / MAX_HEIGHT:
            width = MAX_WIDTH
            height = int(width / aspect_ratio)
        else:
            height = MAX_HEIGHT
            width = int(height * aspect_ratio)
        first_frame = cv2.resize(first_frame, (width, height))

    # --- ROI selection UI loop ---
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi_callback)
    print("🟢 Select traffic light ROI by clicking points. Press 'n' to switch to restricted area.")

    while True:
        display_frame = first_frame.copy()

        # Display instructions
        if current_roi == "traffic_light":
            cv2.putText(display_frame, "Select TRAFFIC LIGHT ROI - Press 'n' for next step", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Select RESTRICTED AREA ROI - Press 's' to save and exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw existing ROIs
        if traffic_light_roi:
            cv2.polylines(display_frame, [np.array(traffic_light_roi)], True, (0, 255, 255), 2)
        if restricted_area_roi:
            cv2.polylines(display_frame, [np.array(restricted_area_roi)], True, (0, 0, 255), 2)

        cv2.imshow("Select ROI", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n') and traffic_light_roi:
            print("🟡 Traffic light ROI selected. Now select RESTRICTED AREA ROI.")
            current_roi = "restricted_area"

        elif key == ord('s') and restricted_area_roi:
            if not traffic_light_roi:
                print("❌ Error: Traffic light ROI must be selected first.")
                continue

            # Save ROIs to JSON file
            roi_data = {
                "traffic_light_roi": traffic_light_roi,
                "restricted_area_roi": restricted_area_roi
            }
            with open(ROI_CONFIG_FILE, 'w') as f:
                json.dump(roi_data, f, indent=4)

            print(f"✅ ROIs saved successfully to '{ROI_CONFIG_FILE}'.")
            break

        elif key == ord('q'):
            print("🛑 Exiting without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()