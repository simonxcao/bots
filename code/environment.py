import time
import numpy as np
import pydirectinput as pdi
from ultralytics import YOLO
import cv2
import mss
import pygetwindow as gw
import torch

class CupheadEnv:
	def __init__(self, model_path="runs/detect/train/weights/best.pt", 
				game_window_name="Cuphead", action_delay=0.1):
		# Game control parameters
		self.done = False
		self.action_delay = action_delay
		self.action_map = {
			0: ('a', 'left'),    # Move left
			1: ('d', 'right'),   # Move right
			2: ('l', 'jump'),    # Jump
			3: ('e', 'nothing'), 	 # agent presses a key that does nothing
		}
		
		# Screen capture setup
		self.sct = mss.mss()
		self.game_region = self._get_game_window(game_window_name)
		
		# Detection model setup
		self.detection_model = YOLO(model_path)
		self.previous_state = None
		self.current_state = None
		
		# Reward tracking
		self.hps = ['hp4', 'hp3', 'hp2', 'hp1']
		self.current_health = 4
		self.last_health = 4
		self.last_score = 0
		self.survival_time = 0

	def _get_game_window(self, window_name):
		"""Helper to find game window coordinates"""
		try:
			win = gw.getWindowsWithTitle(window_name)[0]
			return {
				"left": win.left,
				"top": win.top,
				"width": win.width,
				"height": win.height
			}
		except IndexError:
			raise Exception("Game window not found!")

	def _capture_frame(self):
		"""Capture current game screen"""
		img = np.array(self.sct.grab(self.game_region), dtype=np.uint8)[:, :, :3]
		return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

	def _process_detections(self, results):
		"""Convert YOLO detections to state vector"""
		state = {
			'player': None,
			'enemies': [],
		}
		
		# Process detections
		for result in results:
			for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
				class_name = result.names[int(cls_id)]
				x_center = (box[0] + box[2]) / 2
				y_center = (box[1] + box[3]) / 2
				
				if class_name == 'player':
					state['player'] = (x_center, y_center)
				elif class_name in self.hps:
					new_health = int(class_name[2])
					if new_health != self.current_health:
						self.current_health = new_health
						if self.current_health == 0:
							self.done = True
				elif class_name == 'player_progress':
					self.done = True
				else:
					state['enemies'].append((x_center, y_center))
		return self._vectorize_state(state)

	def _vectorize_state(self, state_dict):
		"""Convert state dictionary to numerical vector"""
		vector = []
		
		# Player position (normalized)
		if state_dict['player']:
			vector.extend([
				state_dict['player'][0] / self.game_region['width'],
				state_dict['player'][1] / self.game_region['height']
			])
		else:
			vector.extend([0, 0])
		
		# Nearest enemy relative position
		if state_dict['enemies']:
			nearest_enemy = min(state_dict['enemies'], 
							key=lambda e: abs(e[0] - vector[0]))
			vector.extend([
				(nearest_enemy[0] - vector[0]) / self.game_region['width'],
				(nearest_enemy[1] - vector[1]) / self.game_region['height']
			])
		else:
			vector.extend([0, 0])
		vector = [elem.cpu().item() if isinstance(elem, torch.Tensor) else elem for elem in vector]
		return np.array(vector, dtype=np.float32)

	def get_state(self):
		"""Get current game state"""
		frame = self._capture_frame()
		results = self.detection_model(frame, verbose=False)
		self.previous_state = self.current_state
		self.current_state = self._process_detections(results)
		return self.current_state

	def execute_action(self, action):
		"""Execute game action with proper timing"""
		key, action_type = self.action_map[action]
		
		# tbh I don't think this makes a difference to hold down for 0.1s vs just tapping. Should fix if we redo agent
		if action_type in ['left', 'right']: 	# move left or right for full action delay period (0.1s)
			pdi.keyDown(key)
			time.sleep(self.action_delay)
			pdi.keyUp(key)
		else:									# perform small jump/parry (since it isn't held down) or do nothing until next action
			pdi.press(key)
			time.sleep(self.action_delay) 

	def get_reward(self):
		"""Calculate reward based on state changes"""
		reward = 0
		
		# Survival reward
		reward += 0.05
		
		# Health change penalty/reward
		current_health = self.current_health
		if current_health < self.last_health:
			reward -= (self.last_health - current_health) * 20
		self.last_health = current_health
		return reward

	def reset(self):
		"""Reset environment"""
		# Release shooting keys before retry
		pdi.keyUp('w')
		pdi.keyUp('j')
		
		self.done = False
		time.sleep(5) # wait for five seconds for player_progress to reach its completion distance
		pdi.press('z')  # Retry key

		# Hold aim and shoot keys
		pdi.keyDown('w')
		pdi.keyDown('j')

		self.current_health = 4
		self.last_health = 4
		self.previous_state = None
		self.current_state = None
		# self.survival_time = 0
		return self.get_state()