import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO
import torch
import time
import os

# Configuration constants
GAME_WINDOW_NAME = "Cuphead"
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

def get_game_window():
    """
    Find the game window by title and return its region as a dictionary.
    """
    try:
        win = gw.getWindowsWithTitle(GAME_WINDOW_NAME)[0]
        return {
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height
        }
    except IndexError:
        raise Exception("Game window not found!")

def detection_loop():
    """
    Continuously capture the game window, run YOLO detection on the image,
    display the detections, and take a screenshot when the 'h' key is pressed.
    """
    # Create a screen capture instance
    sct = mss.mss()
    try:
        region = get_game_window()
    except Exception as e:
        print(e)
        return

    # Load the YOLO model on the appropriate device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("runs/detect/train/weights/best.pt").to(device)

    # Create (or reuse) an OpenCV window for displaying detection results.
    cv2.namedWindow("Cuphead Detection", cv2.WINDOW_NORMAL)

    count = 0
    total_inference_time = 0.0

    while True:
        # Capture a frame of the game window
        frame = np.array(sct.grab(region), dtype=np.uint8)[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run inference on the captured frame
        start_time = time.time()
        results = model(frame, verbose=False)[0]
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        count += 1

        # Process results and draw bounding boxes and labels for detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"{model.names[cls_id]} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        # Display the detection result in the window
        cv2.imshow("Cuphead Detection", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Average inference time:", total_inference_time / count)
            break
        elif key == ord("h"):
            # Save a screenshot of the current frame when 'h' is pressed
            screenshot_filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(screenshot_filename, frame)
            print(f"Screenshot saved as {screenshot_filename}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run only the detection loop (train_rl() removed)
    detection_loop()