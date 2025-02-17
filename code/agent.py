import torch
import numpy as np
import random
import os
from collections import deque

class DQNAgent:
	def __init__(self, state_size, action_size, load_path=None):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=10000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0   # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.model = self._build_model()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		self.target_model = self._build_model()
		self.update_target_model()

		if load_path and os.path.exists(load_path):
			self.load(load_path)

	def _build_model(self):
		model = torch.nn.Sequential(
			torch.nn.Linear(self.state_size, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, self.action_size)
		)
		return model

	def update_target_model(self):
		self.target_model.load_state_dict(self.model.state_dict())

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		state = torch.FloatTensor(state).unsqueeze(0)
		with torch.no_grad():
			act_values = self.model(state)
		return torch.argmax(act_values[0]).item()

	def replay(self, batch_size=32):
		if len(self.memory) < batch_size:
			return
		
		minibatch = random.sample(self.memory, batch_size)
		states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
		actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
		rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
		next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
		dones = torch.BoolTensor(np.array([t[4] for t in minibatch]))

		current_q = self.model(states).gather(1, actions.unsqueeze(1))
		next_q = self.target_model(next_states).max(1)[0].detach()
		target = rewards + (1 - dones.float()) * self.gamma * next_q

		loss = torch.nn.functional.mse_loss(current_q.squeeze(), target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		

	def save(self, name):
		"""Save full agent state including epsilon and memory"""
		torch.save({
			'model_state_dict': self.model.state_dict(),
			'target_model_state_dict': self.target_model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'epsilon': self.epsilon,
			'memory': list(self.memory),
		}, name)

	def load(self, name):
		"""Load full agent state"""
		if os.path.exists(name):
			checkpoint = torch.load(name)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.epsilon = checkpoint.get('epsilon', 1.0)
			self.memory = deque(checkpoint.get('memory', []), maxlen=10000)
			print(f"Loaded agent from {name}")