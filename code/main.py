import cv2
import numpy as np
import mss
import pygetwindow as gw
from inference import get_model
import supervision as sv

# Install required libraries if not already installed
# pip install inference supervision pygetwindow mss opencv-python numpy
# you may need to enable long paths in windows if certain paths are too long

# Configuration
GAME_WINDOW_NAME = "Cuphead"
MODEL_ID = "yolov8n-640"  # Choose from yolov8n/s/m/l/x
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
	
	# Initialize YOLOv8 model
	model = get_model(model_id=MODEL_ID, device="cuda")
	
	# Initialize annotators
	box_annotator = sv.BoxAnnotator(thickness=2, text_scale=0.5)
	label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
	
	while True:
		# Capture screen
		img = np.array(sct.grab(region))[:, :, :3]  # Remove alpha channel
		
		# Run inference
		results = model.infer(img)[0]
		detections = sv.Detections.from_inference(results)
		
		# Filter by confidence
		detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
		
		# Annotate frame
		annotated_frame = box_annotator.annotate(
			scene=img.copy(), 
			detections=detections
		)
		annotated_frame = label_annotator.annotate(
			scene=annotated_frame, 
			detections=detections
		)
		
		# Display output (convert RGB to BGR for OpenCV)
		cv2.imshow("Cuphead Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
		
		# Exit key
		if cv2.waitKey(1) & 0xFF == ord("q"):
			cv2.destroyAllWindows()
			break

if __name__ == "__main__":
	main()