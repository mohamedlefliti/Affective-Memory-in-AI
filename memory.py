class MemoryUnit:
    def __init__(self, x, t, phi, epsilon, mu, action):
        self.x = x              # emotion label
        self.t = t              # internal time
        self.phi = phi          # perceptual state
        self.epsilon = epsilon  # intensity
        self.mu = mu            # structural impact
        self.action = action    # action taken
