import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO
import torch

# Configuration
GAME_WINDOW_NAME = "Cuphead"
MODEL_PATH = "yolov8n.pt"  # Local model file
CONFIDENCE_THRESHOLD = 0.5

def get_game_window():
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

def main():
	sct = mss.mss()
	region = get_game_window()
	
	# Initialize YOLOv8 with GPU support
	model = YOLO(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
	
	# Ensure OpenCV reuses the window instead of opening new ones
	cv2.namedWindow("Cuphead Detection", cv2.WINDOW_NORMAL)

	while True:
		# Capture and format screen
		img = np.array(sct.grab(region), dtype=np.uint8)[:, :, :3]  # Ensure uint8
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR for OpenCV

		# Run inference on GPU
		results = model(img, verbose=False)[0]
		
		# Process results
		for result in results:
			boxes = result.boxes.xyxy.cpu().numpy()
			confidences = result.boxes.conf.cpu().numpy()
			class_ids = result.boxes.cls.cpu().numpy().astype(int)
			
			for box, conf, cls_id in zip(boxes, confidences, class_ids):
				if conf > CONFIDENCE_THRESHOLD:
					x1, y1, x2, y2 = map(int, box)
					cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.putText(img, 
							f"{model.names[cls_id]} {conf:.2f}",
							(x1, y1 - 10), 
							cv2.FONT_HERSHEY_SIMPLEX, 
							0.5, (0, 255, 0), 2)
		
		# Show the updated frame
		cv2.imshow("Cuphead Detection", img)
		
		# Check for exit key
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	# Cleanup after exiting the loop
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
