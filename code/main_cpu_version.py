import cv2
import numpy as np
import mss
import pygetwindow as gw
from inference import get_model
import supervision as sv

# Configuration
GAME_WINDOW_NAME = "Cuphead"
MODEL_ID = "yolov8n-640"
CONFIDENCE_THRESHOLD = 0.5

def get_game_window():
    """Get the game window coordinates"""
    try:
        win = gw.getWindowsWithTitle(GAME_WINDOW_NAME)[0]
        return {
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height
        }
    except IndexError:
        raise Exception("Game window not found - make sure the game is running!")

def main():
    # Initialize screen capture
    sct = mss.mss()
    region = get_game_window()
    
    # Initialize YOLOv8 model (remove device parameter)
    model = get_model(model_id=MODEL_ID)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Create the window once outside the loop
    cv2.namedWindow("Cuphead Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Cuphead Detection", cv2.WND_PROP_TOPMOST, 1)

    while True:
        # Capture screen
        img = np.array(sct.grab(region))[:, :, :3]
        
        # Run inference
        results = model.infer(img)[0]
        detections = sv.Detections.from_inference(results)
        
        # Filter by confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(img.copy(), detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections)
        
        # Display output
        cv2.imshow("Cuphead Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        q
        # Ensure window updates properly
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()