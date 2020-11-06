from stable_baselines import DQN
from src.rl.gym.environments import HeuristicsSpeedEnv
from src.rl.gym.observers import GlobalMixedImageObserver, GlobalImageObserver
from src.model.agents import RandomAgent, AgentDummy, OneStepSurvivalAgent
from src.rl.policies import nature_cnn_mlp_mix


policy_kwargs = dict(cnn_extractor=nature_cnn_mlp_mix)
env = HeuristicsSpeedEnv(width=16, height=16, observer=GlobalImageObserver(nb_agents=2, width=64, height=64),
                         agent_classes=[AgentDummy, OneStepSurvivalAgent])
env.model.verbose = True
model = DQN.load("../../../res/rl/models/DQN_vs_OneStepSurvivalAgent", policy_kwargs=policy_kwargs)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
