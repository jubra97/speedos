from stable_baselines.deepq.policies import FeedForwardPolicy
from src.rl.gym.environments import SelfPlaySpeedEnv
from src.rl.self_play_dqn import SelfPlayDQN
from src.rl.gym.observers import GlobalObserver


# parameters
model_name = "self_play_dqn"
nb_training_steps = 100_000
hidden_layers = [64, 64]
erm_size = 100_000

env = SelfPlaySpeedEnv(width=5, height=5, observer=GlobalObserver(nb_agents=2, width=5, height=5))


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=hidden_layers,
                                              layer_norm=False,
                                              feature_extraction="mlp")


model = SelfPlayDQN(CustomDQNPolicy, env, verbose=1, buffer_size=erm_size)

# Train and save the agent
model.learn(total_timesteps=nb_training_steps, tb_log_name=model_name, log_interval=5000, callback=None)
file_name = '../../../res/rl/models/' + model_name
model.save(file_name)
