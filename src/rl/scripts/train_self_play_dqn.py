from stable_baselines.deepq.policies import FeedForwardPolicy
from src.rl.gym.environments import SelfPlaySpeedEnv
from src.rl.self_play_dqn import SelfPlayDQN
from src.rl.gym.observers import GlobalObserver


# parameters
model_name = "self_play_dqn"
nb_training_steps = 300_000
hidden_layers = [128, 64, 32, 16]
erm_size = 100_000

# init environment and agent
env = SelfPlaySpeedEnv(width=7, height=7, observer=GlobalObserver(nb_agents=3, width=7, height=7))
policy_kwargs = dict(net_arch=hidden_layers)
model = SelfPlayDQN("MlpPolicy", env, verbose=1, buffer_size=erm_size, policy_kwargs=policy_kwargs)

# Train and save the agent
model.learn(total_timesteps=nb_training_steps, tb_log_name=model_name, log_interval=5000, callback=None)
file_name = '../../../res/rl/models/' + model_name
model.save(file_name)
