from src.rl.q_agent import QAgent
from src.rl.gym.environments import SingleAgentSpeedEnv

env = SingleAgentSpeedEnv()
model = QAgent(env)
model.load("../../../res/rl/models/q_test")
model.test()
