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
			3: ('z', 'retry'), 	 # Retry agent may choose this as well while alive which is a nice way to choose no action
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
		self.current_phase = 1
		self.last_phase = 1
		self.phase_transition_time = time.time()
		
		# Position tracking
		self.optimal_x_pos = self.game_region['width'] * 0.6  # 60% of screen width
		self.last_projectile_distance = float('inf')

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
			'projectiles': [],
			'boss': None
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
				elif class_name == 'boss':
					state['boss'] = (x_center, y_center)
					# Update phase based on boss position/behavior
					if y_center < self.game_region['height'] * 0.3:
						self.current_phase = 3
					elif y_center < self.game_region['height'] * 0.6:
						self.current_phase = 2
					else:
						self.current_phase = 1
				elif 'projectile' in class_name:
					state['projectiles'].append((x_center, y_center))
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
		
		# Base survival reward (reduced to emphasize other behaviors)
		reward += 0.02
		
		# Phase progression rewards
		if self.current_phase > self.last_phase:
			phase_time = time.time() - self.phase_transition_time
			phase_reward = {
				2: 50,
				3: 100
			}.get(self.current_phase, 0)
			
			# Bonus for quick phase transitions (under 60 seconds)
			if phase_time < 60:
				phase_reward *= 1.5
			
			# Bonus for perfect phase transitions (no damage taken)
			if self.current_health == self.last_health:
				phase_reward *= 1.25
				
			reward += phase_reward
			self.last_phase = self.current_phase
			self.phase_transition_time = time.time()
		
		# Health change penalties (increased for later phases)
		if self.current_health < self.last_health:
			health_penalty = 20 * (1 + (self.current_phase - 1) * 0.5)
			reward -= (self.last_health - self.current_health) * health_penalty
		self.last_health = self.current_health
		
		# Positioning rewards
		if self.current_state is not None and len(self.current_state) >= 2:
			player_x = self.current_state[0] * self.game_region['width']
			
			# Reward for maintaining optimal attack position
			distance_from_optimal = abs(player_x - self.optimal_x_pos)
			position_reward = 0.1 * (1 - distance_from_optimal / self.game_region['width'])
			reward += position_reward
			
			# Penalize being at screen edges when not dodging
			if player_x < self.game_region['width'] * 0.1 or player_x > self.game_region['width'] * 0.9:
				reward -= 0.1
		
		# Projectile avoidance rewards
		if self.current_state is not None and len(self.current_state) >= 4:
			nearest_projectile_x = self.current_state[2] * self.game_region['width']
			nearest_projectile_y = self.current_state[3] * self.game_region['height']
			
			if nearest_projectile_x != 0 or nearest_projectile_y != 0:  # If projectile exists
				player_x = self.current_state[0] * self.game_region['width']
				player_y = self.current_state[1] * self.game_region['height']
				
				current_projectile_distance = ((player_x - nearest_projectile_x) ** 2 + 
											(player_y - nearest_projectile_y) ** 2) ** 0.5
				
				# Reward for maintaining safe distance from projectiles
				if current_projectile_distance > self.last_projectile_distance:
					reward += 0.1 * (self.current_phase ** 0.5)  # Higher reward in later phases
				
				self.last_projectile_distance = current_projectile_distance
		
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
		self.current_phase = 1
		self.last_phase = 1
		self.phase_transition_time = time.time()
		self.last_projectile_distance = float('inf')
		return self.get_state()