from src.rl.q_agent import QAgent
from src.rl.gym.environments import SingleAgentSpeedEnv

env = SingleAgentSpeedEnv(width=5, height=5)

model = QAgent(env, dynamic=True)
model.learn(nb_episodes=100000, verbose=True)
model.save("../../../res/rl/models/q_test", timestamp=False)
