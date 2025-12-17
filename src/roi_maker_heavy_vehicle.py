import cv2
import json

# ======= CONFIG =======
video_path = 'assets/truck1.mp4'  # Replace with your actual path
json_path = 'roi_data/heavy_vehicle_roi.json'
MAX_WIDTH = 1700
MAX_HEIGHT = 956
# =======================

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])  # Save as list, not tuple

def resize_if_needed(frame):
    height, width = frame.shape[:2]
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        return cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT))
    return frame

def main():
    global points

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read the first frame.")
        return

    frame = resize_if_needed(frame)
    clone = frame.copy()

    cv2.namedWindow("Draw ROI (press 's' to save, 'q' to quit)")
    cv2.setMouseCallback("Draw ROI (press 's' to save, 'q' to quit)", click_event)

    while True:
        display_frame = clone.copy()

        if points:
            for i in range(len(points)):
                cv2.circle(display_frame, tuple(points[i]), 4, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display_frame, tuple(points[i - 1]), tuple(points[i]), (255, 0, 0), 2)
            if len(points) > 2:
                cv2.line(display_frame, tuple(points[-1]), tuple(points[0]), (0, 0, 255), 1)

        cv2.imshow("Draw ROI (press 's' to save, 'q' to quit)", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            with open(json_path, 'w') as f:
                json.dump({"traffic_light_roi": points}, f, indent=4)
            print(f"✅ ROI saved to {json_path}")
            break

        elif key == ord('q'):
            print("❌ Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
