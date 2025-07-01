import random
import numpy as np
from collections import deque
from .memory import MemoryUnit

class AffectiveAgent:
    def __init__(self, memory_size=1000):
        self.memory = deque(maxlen=memory_size)
        self.time = 0
        self.last_obs = None
        self.last_action = None
        self.repetition_count = 0

    def remember(self, obs, reward, done, action):
        x = self._emotion_label(reward)
        epsilon = abs(reward)
        mu = -1 if done or reward == 0 else reward

        if self.last_action == action:
            self.repetition_count += 1
        else:
            self.repetition_count = 0

        if self.repetition_count >= 3:
            x = "frustration"
            mu = -1
            epsilon = 1.0

        zeta = MemoryUnit(x, self.time, obs, epsilon, mu, action)
        self.memory.append(zeta)
        self.last_obs = obs
        self.last_action = action
        self.time += 1

    def act(self, obs):
        memory = self._retrieve(obs)
        if memory:
            return 1 - memory.action if memory.mu < 0 else memory.action
        return random.choice([0, 1])

    def _retrieve(self, obs):
        for z in reversed(self.memory):
            dist = np.linalg.norm(np.array(z.phi) - np.array(obs))
            if dist < 0.5 and z.epsilon > 0.8:
                return z
        return None

    def _emotion_label(self, reward):
        if reward > 0:
            return "curiosity"
        elif reward < 0:
            return "frustration"
        return "neutral"

