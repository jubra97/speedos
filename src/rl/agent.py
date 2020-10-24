from abc import ABC, abstractmethod
import numpy as np
import random
import tensorflow as tf
import datetime


class RLAgent(ABC):
    """
    Abstract that all agents implement.
    """

    def __init__(self, env):
        """
        Initializes the agent.
        :param env: The environment the agent lives in.
        """
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.tb_writer = None
        self.avg_episode_rewards = []

    @abstractmethod
    def learn(self, nb_steps, callback=None, visualize=False, tb_log_name=None, nb_max_episode_steps=None):
        """
        Trains the agent.
        :param nb_steps: Number of training steps.
        :param callback: A lambda/function that is executed every training step.
        :param visualize: Whether or not the environment should be visualized.
        :param tb_log_name: A path to save a tensorboard log of the training progress to.
        :param nb_max_episode_steps: Maximum number of steps before an episode will be automatically ended.
        :return: A list of the average reward per step for each episode.
        """
        pass

    def test(self, nb_episodes=1, callback=None, visualize=True, verbose=True, tb_log_name=None,
             nb_max_episode_steps=None):
        """
        Tests/Evaluates a trained agent.
        :param nb_episodes: Number of episodes to perform.
        :param callback: A lambda/function that is executed every step.
        :param visualize: Whether or not the environment should be visualized.
        :param verbose: Whether or not detailed information should be printed during execution.
        :param tb_log_name: A path to save a tensorboard log to.
        :param nb_max_episode_steps: Maximum number of steps before an episode will be automatically ended.
        :return: A list of the average reward per step for each episode.
        """
        self._init_log(tb_log_name)
        self.env._max_episode_steps = nb_max_episode_steps
        for episode in range(nb_episodes):
            state = self.env.reset()
            step = 0
            ret = 0
            done = False
            if step == 900:
                print("JA")
            while not done and (nb_max_episode_steps is None or step < nb_max_episode_steps):
                (action, reward, next_state, done) = self.act(state)

                state = next_state
                step += 1
                ret += reward

                if visualize:
                    self.env.render()
                if callback is not None:
                    callback(self)

            self._log(episode, ret, verbose=verbose)

        return self.avg_episode_rewards

    @abstractmethod
    def act(self, state, deterministic=False):
        """
        Gets the agent's action for the given state.
        :param state: The environment state.
        :param deterministic: Whether or not the action selection should be deterministic.
        :return: The selected action.
        """
        pass

    @abstractmethod
    def reset_states(self):
        """
        Resets all internal states.
        :return: None
        """
        pass

    @abstractmethod
    def save(self, filename, timestamp):
        """
        Saves the agent's model.
        :param filename: The path of the file that the model should be saved to.
        :param timestamp: Whether or not a timestamp should be added to the filename.
        :return: None
        """
        pass

    @abstractmethod
    def load(self, filename):
        """
        Loads a saved model from the given file.
        :param filename: The path of the file that the model should be loaded from.
        :return: None
        """
        pass

    def set_random_seed(self, seed):
        """
        Sets a seed for all internally used random number generators.
        :param seed: The integer seed.
        :return: None
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and gym
        np.random.seed(seed)
        random.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
            # Seed the action space
            # useful when selecting random actions
            self.env.action_space.seed(seed)
        self.action_space.seed(seed)

    def _init_log(self, log_dir):
        """
        Initializes logging and the tensorboard log.
        :param log_dir: Directory path where the tensorboard log should be created.
        :return: None
        """
        self.avg_episode_rewards = []
        if log_dir is not None:
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.tb_writer = tf.summary.create_file_writer(log_dir + '_' + current_time)

    def _log(self, episode, ret, verbose):
        """
        Logs the average reward per step for an episode.
        :param episode: The episode number.
        :param ret: The return (accumulated reward) of the episode.
        :param verbose: Whether or not the results should be printed.
        :return: None
        """
        self.avg_episode_rewards.append(ret)
        if self.tb_writer is not None:
            with self.tb_writer.as_default():
                tf.summary.scalar('return', ret, step=episode)
        if verbose:
            print('Episode {} finished. Return: {}'.format(episode, ret))
