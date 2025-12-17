# File: draw_rois_and_directions.py
import cv2
import numpy as np
import json
import os

# ------------------------------
# Settings
# ------------------------------
VIDEO_PATH = "assets/video3.mp4"
ROI_CONFIG_FILE = "roi_data/wrong_direction_rois.json"

# ------------------------------
# Globals for Mouse Callbacks
# ------------------------------
defined_rois = []
defined_directions = []
current_roi_points = []
current_direction_points = []
drawing_roi = False
drawing_direction = False
current_roi_index = -1

# ------------------------------
# Mouse callback for ROI polygon and direction vector
# ------------------------------
def draw_interactive_elements(event, x, y, flags, param):
    global current_roi_points, current_direction_points, drawing_roi, drawing_direction, defined_rois, defined_directions, current_roi_index

    # Start drawing a new ROI
    if event == cv2.EVENT_RBUTTONDOWN and not drawing_roi and not drawing_direction:
        drawing_roi = True
        print("🟢 Starting new ROI. Left-click to add points, right-click to complete ROI.")
        return

    # Add points to the current ROI
    if event == cv2.EVENT_LBUTTONDOWN and drawing_roi:
        current_roi_points.append((x, y))
        print(f"  > Point added: {(x, y)}")

    # Finish ROI and start drawing its direction
    elif event == cv2.EVENT_RBUTTONDOWN and drawing_roi:
        if len(current_roi_points) > 2:
            defined_rois.append(np.array(current_roi_points, dtype=np.int32))
            current_roi_points = []
            drawing_roi = False
            drawing_direction = True
            current_roi_index = len(defined_rois) - 1
            print(f"✅ ROI {current_roi_index + 1} defined. Now define its direction vector (2 left-clicks).")
        else:
            print("⚠️ ROI needs at least 3 points.")

    # Add points for the direction vector
    elif event == cv2.EVENT_LBUTTONDOWN and drawing_direction and len(current_direction_points) < 2:
        current_direction_points.append((x, y))
        if len(current_direction_points) == 2:
             print("  > Direction vector defined. Right-click to confirm.")


    # Finish the direction vector
    elif event == cv2.EVENT_RBUTTONDOWN and drawing_direction and len(current_direction_points) == 2:
        main_dir_vector = np.array(current_direction_points[1]) - np.array(current_direction_points[0])
        defined_directions.append(main_dir_vector)
        current_direction_points = []
        drawing_direction = False
        current_roi_index = -1
        print(f"✅ Direction for ROI {len(defined_directions)} defined. Right-click for next ROI or press 's' to save.")


def main():
    """ Main function to run the interactive ROI and direction setup. """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_PATH}")
        return
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read the first frame.")
        cap.release()
        return
    cap.release()

    H, W, _ = first_frame.shape
    scale = min(1920 / W, 1080 / H, 1.0)
    if scale < 1:
        new_size = (int(W * scale), int(H * scale))
        first_frame = cv2.resize(first_frame, new_size)

    # --- ROI and Direction Definition UI Loop ---
    print("\n--- 🚀 ROI & Direction Setup ---")
    print("1. Right-click to start a new ROI.")
    print("2. Left-click to add points to the polygon.")
    print("3. Right-click again to finish the polygon.")
    print("4. Left-click twice to draw the allowed direction vector.")
    print("5. Right-click to confirm the direction.")
    print("6. Repeat for all ROIs, then press 's' to save and exit.")

    cv2.namedWindow("Define ROIs and Directions")
    cv2.setMouseCallback("Define ROIs and Directions", draw_interactive_elements)

    while True:
        temp_frame = first_frame.copy()

        # Draw completed ROIs and directions
        for i, roi in enumerate(defined_rois):
            cv2.polylines(temp_frame, [roi], True, (0, 255, 0), 2)
            cv2.putText(temp_frame, f"ROI {i+1}", tuple(roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if i < len(defined_directions):
                roi_center = np.mean(defined_rois[i], axis=0).astype(int)
                end_point = roi_center + defined_directions[i] // 4 # Scaled for visibility
                cv2.arrowedLine(temp_frame, tuple(roi_center), tuple(end_point), (255, 0, 0), 3)

        # Draw current (in-progress) ROI
        if drawing_roi and len(current_roi_points) > 0:
            cv2.polylines(temp_frame, [np.array(current_roi_points)], False, (0, 255, 255), 2)

        # Draw current (in-progress) direction
        if drawing_direction and len(current_direction_points) == 2:
            cv2.arrowedLine(temp_frame, current_direction_points[0], current_direction_points[1], (255, 0, 255), 3)

        cv2.imshow("Define ROIs and Directions", temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(defined_rois) > 0 and len(defined_rois) == len(defined_directions):
                # --- Save data to JSON ---
                output_data = []
                for roi, direction in zip(defined_rois, defined_directions):
                    output_data.append({
                        "roi_points": roi.tolist(), # Convert numpy array to list for JSON
                        "direction_vector": direction.tolist()
                    })

                with open(ROI_CONFIG_FILE, 'w') as f:
                    json.dump(output_data, f, indent=4)

                print(f"\n✅ Setup complete. Configuration for {len(output_data)} ROIs saved to '{ROI_CONFIG_FILE}'.")
                break
            else:
                print("⚠️ Please define at least one ROI and its corresponding direction, or ensure all ROIs have a direction.")
        elif key == ord('q'):
            print("🛑 Exiting without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()