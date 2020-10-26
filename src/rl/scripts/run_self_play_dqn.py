from src.rl.self_play_dqn import SelfPlayDQN
from src.rl.gym.environments import SelfPlaySpeedEnv
from src.rl.gym.observers import GlobalObserver

env = SelfPlaySpeedEnv(width=7, height=7, observer=GlobalObserver(nb_agents=3, width=7, height=7))
env.model.verbose = True
model = SelfPlayDQN.load("../../../res/rl/models/self_play_dqn")

obs = env.reset()
done = False
while env.model.running:
    action, _states = model.predict(obs)
    env.step(action)
    env.render()
