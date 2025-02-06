import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import os

# Configuration
GAME_WINDOW_NAME = "Cuphead"
SAVE_FOLDER = "datasets/cuphead/images"
CAPTURE_INTERVAL = 1  # Seconds between screenshots

# Ensure save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

def get_game_window():
	"""Find the Cuphead game window"""
	try:
		win = gw.getWindowsWithTitle(GAME_WINDOW_NAME)[0]
		return {
			"left": win.left,
			"top": win.top,
			"width": win.width,
			"height": win.height
		}
	except IndexError:
		raise Exception("Game window not found! Make sure Cuphead is running.")

def capture_screenshots():
	"""Capture and save screenshots from Cuphead"""
	sct = mss.mss()
	region = get_game_window()
	
	count = len(os.listdir(SAVE_FOLDER))  # Start from last saved file

	print(f"📸 Capturing screenshots... Press CTRL+C to stop.")
	
	try:
		while True:
			# Capture screen
			img = np.array(sct.grab(region))[:, :, :3]
			
			# Convert to BGR for OpenCV
			img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

			# Save the screenshot
			filename = os.path.join(SAVE_FOLDER, f"cuphead_{count:04d}.jpg")
			cv2.imwrite(filename, img)
			print(f"✅ Saved: {filename}")

			count += 1
			time.sleep(CAPTURE_INTERVAL)  # Wait before next screenshot

	except KeyboardInterrupt:
		print("🛑 Screenshot capture stopped.")

if __name__ == "__main__":
	capture_screenshots()