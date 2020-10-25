from src.rl.q_agent import QAgent
from src.rl.gym.environments import SingleAgentSpeedEnv
from src.rl.gym.observers import GlobalObserver

env = SingleAgentSpeedEnv(width=5, height=5, observer=GlobalObserver(nb_agents=1, width=5, height=5))

model = QAgent(env, dynamic=True)
model.learn(nb_episodes=100000, verbose=True)
model.save("../../../res/rl/models/q_test", timestamp=False)
