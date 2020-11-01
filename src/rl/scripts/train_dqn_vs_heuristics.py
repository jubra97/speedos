from stable_baselines import DQN
from src.rl.gym.environments import HeuristicsSpeedEnv
from src.rl.gym.observers import GlobalObserver, GlobalMixedImageObserver, GlobalImageObserver
from src.model.agents import RandomAgent, AgentDummy, OneStepSurvivalAgent
from src.rl.policies import nature_cnn_mlp_mix


# parameters
model_name = "DQN_vs_OneStepSurvivalAgent"
nb_training_steps = 100_000
#hidden_layers = [64, 64, 32]
erm_size = 100_000

# init environment and agent
policy_kwargs = dict(cnn_extractor=nature_cnn_mlp_mix)
env = HeuristicsSpeedEnv(width=16, height=16, observer=GlobalMixedImageObserver(nb_agents=2, width=64, height=64),
                         agent_classes=[AgentDummy, OneStepSurvivalAgent])
model = DQN("CnnPolicy", env, verbose=1, buffer_size=erm_size, target_network_update_freq=50, exploration_fraction=0.3,
            policy_kwargs=policy_kwargs, tensorboard_log="../../../res/rl/tensorboard/")

# train and save
model.learn(total_timesteps=nb_training_steps)
model.save("../../../res/rl/models/" + model_name)
