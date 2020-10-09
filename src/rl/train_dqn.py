from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from src.rl.gym.environments import SingleAgentSpeedEnv

env = SingleAgentSpeedEnv()

model = DQN(MlpPolicy, env, verbose=1, learning_starts=100, target_network_update_freq=50, exploration_fraction=0.5)
model.learn(total_timesteps=10000)
model.save("../../res/rl/models/dqn_test")
