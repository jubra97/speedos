from src.rl.q_agent import QAgent
from src.rl.gym.environments import SingleAgentSpeedEnv

env = SingleAgentSpeedEnv()

model = QAgent(env)
model.learn(nb_episodes=100000, verbose=False)
model.save("../../../res/rl/models/q_test", timestamp=False)
