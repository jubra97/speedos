from stable_baselines import DQN
from src.rl.gym.environments import SingleAgentSpeedEnv

env = SingleAgentSpeedEnv()
model = DQN.load("../../../res/rl/models/dqn_test")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
