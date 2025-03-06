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
				game_window_name="Cuphead", action_delay=0.01):
		# Game control parameters
		self.done = False
		self.win_detected = False
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
		
		# Phase tracking
		self.second_phase_reached = False
		self.carrot_detected = False

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
		
		# Initialize needed attributes if not present
		if not hasattr(self, 'carrot_detected'):
			self.carrot_detected = False
			
		if not hasattr(self, 'second_phase_reached'):
			# print("second_phase_reached")
			self.second_phase_reached = False
			
		if not hasattr(self, 'win_detected'):
			self.win_detected = False
		
		# Process detections
		for result in results:
			for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
				class_name = result.names[int(cls_id)]
				x_center = (box[0] + box[2]) / 2
				y_center = (box[1] + box[3]) / 2
				
				# Detect phase transitions
				if class_name == 'onion_boss':
					self.second_phase_reached = True
				
				# Detect carrot for third phase
				if class_name == 'carrot':
					self.carrot_detected = True
				
				# Check for win condition
				if class_name == 'win_results':
					self.win_detected = True
					self.done = True
					return self._vectorize_state(state)
					
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
		
		# Store the last action for reward calculation
		self.last_action = action
		
		# Track movement direction and consecutive moves for third phase rewards
		if not hasattr(self, 'movement_direction'):
			self.movement_direction = None
			self.consecutive_moves = 0
			self.movement_direction_changed = False
		
		# Update movement tracking
		if action == 0:  # Left
			if self.movement_direction == 'left':
				self.consecutive_moves += 1
			else:
				if self.movement_direction == 'right' and self.consecutive_moves >= 5:
					self.movement_direction_changed = True
				self.movement_direction = 'left'
				self.consecutive_moves = 1
		elif action == 1:  # Right
			if self.movement_direction == 'right':
				self.consecutive_moves += 1
			else:
				if self.movement_direction == 'left' and self.consecutive_moves >= 5:
					self.movement_direction_changed = True
				self.movement_direction = 'right'
				self.consecutive_moves = 1
		elif action not in [0, 1]:  # Jump - reset consecutive moves counter
			self.consecutive_moves = 0
		
		# Original action execution logic
		if action_type in ['left', 'right']:
			pdi.keyDown(key)
			time.sleep(self.action_delay)
			pdi.keyUp(key)
		else:
			pdi.press(key)
			time.sleep(self.action_delay)

	def get_reward(self):
		"""Calculate reward based on state changes"""
		reward = 0
		
		# Survival reward
		reward += 0.08
		
		# Health change penalty/reward
		current_health = self.current_health
		if current_health < self.last_health:
			reward -= (self.last_health - current_health) * 10
		self.last_health = current_health
		
		# Position-based rewards and penalties
		if self.current_state is not None and len(self.current_state) >= 2:
			player_x_normalized = self.current_state[0]
			
			# Penalty for being too close to left edge
			if player_x_normalized < 0.05:
				# print("hugging wall penalty")
				reward -= 0.02
			# Penalty for being too close to right edge
			if player_x_normalized > 0.95:
				reward -= 0.01
		
		# Apply jumping penalty after second phase is reached
		if hasattr(self, 'second_phase_reached') and self.second_phase_reached:
			# print("2nd phase check")
			# Penalize jumping after second phase (action 2 is jump)
			if hasattr(self, 'last_action') and self.last_action == 2:
				# print("jump penalty")
				reward -= 1
				
		# Third phase logic - carrot detection and movement incentives
		if hasattr(self, 'carrot_detected') and self.carrot_detected:
			# We're in the third phase (carrot was detected)
			# Reward horizontal movement patterns
			if hasattr(self, 'movement_direction') and hasattr(self, 'consecutive_moves'):
				# print("consecutive movement", self.consecutive_moves)
				# Reward for maintaining consecutive moves in the same direction
				if self.consecutive_moves >= 3 and self.consecutive_moves <= 15:
					reward += 0.01 * self.consecutive_moves
				# Extra reward for switching direction after a good sequence
				if self.consecutive_moves >= 5 and self.movement_direction_changed:
					reward += 0.2
					self.movement_direction_changed = False
			
			# Reward for staying in the middle 60% of the screen
			if 0.2 <= player_x_normalized <= 0.8:
				print("middle position reward")
				reward += 0.1

			# larger hugging wall penalty for third phase
			if player_x_normalized < 0.05:
				# print("hugging wall penalty")
				reward -= 0.1
				
		return reward

	def reset(self):
		"""Reset environment"""
		# Release shooting keys before retry
		pdi.keyUp('w')
		pdi.keyUp('j')
		
		self.done = False
		# Reset phase detection
		self.second_phase_reached = False
		self.carrot_detected = False
		self.movement_direction = None
		self.consecutive_moves = 0
		self.movement_direction_changed = False
		self.last_action = None
		
		time.sleep(5) # wait for five seconds for player_progress to reach its completion distance
		pdi.press('z')  # Retry key

		# Hold aim and shoot keys
		pdi.keyDown('w')
		pdi.keyDown('j')

		self.current_health = 4
		self.last_health = 4
		self.previous_state = None
		self.current_state = None
		return self.get_state()
	
	def reset_on_win(self):
		"""Reset environment after winning a level"""
		# Release shooting keys first
		pdi.keyUp('w')
		pdi.keyUp('j')
		
		# Wait 5 seconds
		time.sleep(5)
		
		# Press F8
		pdi.press('f8')
		
		# Press ESC
		pdi.press('esc')
		
		# Wait 4 seconds
		time.sleep(4)
		
		# Press 'a' four times
		for _ in range(4):
			pdi.press('a')
		
		# Press 'z'
		pdi.press('z')
		
		# Wait 2 seconds
		time.sleep(2)
		
		# Press 'z' again
		pdi.press('z')
		
		# Reset environment variables
		self.done = False
		self.win_detected = False
		self.second_phase_reached = False
		self.carrot_detected = False
		self.movement_direction = None
		self.consecutive_moves = 0
		self.movement_direction_changed = False
		self.last_action = None
		self.current_health = 4
		self.last_health = 4
		self.previous_state = None
		self.current_state = None
		
		# Hold aim and shoot keys again
		pdi.keyDown('w')
		pdi.keyDown('j')
		
		return self.get_state()