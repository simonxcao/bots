import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO
import torch
import time
import threading
import os
import argparse

from agent import DQNAgent
from environment import CupheadEnv

# Configuration constants
GAME_WINDOW_NAME = "Cuphead"
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_agent.pth")


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
	and display the detections in a separate window.
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

		# Allow the user to press 'q' to exit the detection loop.
		if cv2.waitKey(1) & 0xFF == ord("q"):
			print("Average inference time:", total_inference_time / count)
			break

	cv2.destroyAllWindows()

def train_rl():
	"""
	Train the RL agent to play the game
	"""
	# Create checkpoint directory
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)

	# Initialize environment and agent
	parser = argparse.ArgumentParser()
	parser.add_argument("--load", type=str, help="Path to agent checkpoint")
	args = parser.parse_args()

	env = CupheadEnv()
	state_size = len(env.get_state())  # Get state size from environment
	action_size = len(env.action_map)
	
	# Initialize agent with optional loading
	agent = DQNAgent(state_size, action_size, load_path=args.load or CHECKPOINT_PATH)
	
	episodes = 1300
	update_target_freq = 10
	save_checkpoint_freq = 50  # Save every 50 episodes

	log_file = "training_log.txt"
	
	with open(log_file, "a") as f:
		for episode in range(episodes):
			state = env.reset()
			total_reward = 0
			done = False
			
			while not done:
				action = agent.act(state)
				env.execute_action(action)
				next_state = env.get_state()
				reward = env.get_reward()
				done = env.done
				
				agent.remember(state, action, reward, next_state, done)
				agent.replay()
				
				total_reward += reward
				state = next_state
				
				# Decay epsilon after each step
				if agent.epsilon > agent.epsilon_min:
					agent.epsilon *= agent.epsilon_decay
			
			# Update the target network periodically
			if episode % update_target_freq == 0:
				agent.update_target_model()
				
			# Save the model checkpoint
			if episode % save_checkpoint_freq == 0:
				agent.save(f"checkpoint_{episode}.pth")
				print(f"Checkpoint saved at episode {episode}")
			
			# Log episode results
			f.write(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}\n")
			f.flush()


if __name__ == "__main__":
	# Start the detection loop in a separate daemon thread so that it
	# runs concurrently with the RL training loop.
	detection_thread = threading.Thread(target=detection_loop, daemon=True)
	detection_thread.start()
	
	# Run the RL training loop in the main thread.
	train_rl()
