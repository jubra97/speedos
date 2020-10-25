from stable_baselines import DQN
from stable_baselines import logger
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer

import tensorflow as tf
import numpy as np


class SelfPlayDQN(DQN):
    """
    An Extension of Stable Baselines DQN implementation for self play in multi agent environments.
    """

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="MADQN",
              reset_num_timesteps=True, replay_wrapper=None):
        """
        This function is heavily based on the original learning algorithm of a DQN and has just a few adjustments for
        multiple agents and parameter sharing.

        :param total_timesteps: The amount of total training steps for the DQN
        (The amount of executed environment steps are total_timesteps/num_agents)
        :param callback: A function that is called every training step
        :param seed: A random seed
        :param log_interval: The interval of training steps in which progress will be logged
        :param tb_log_name: The name of the tensorboard log
        :param reset_num_timesteps: Whether the amount of time steps should be set back to 0 at the beginning
        (only important for several runs in a row)
        :param replay_wrapper: A replay wrapper that should be used for experience replay memory
        :return: self
        """

        environment_steps = int(total_timesteps / self.env.nb_agents)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = environment_steps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * environment_steps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []
            observations = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))

            kwargs = {}
            if not self.param_noise:
                update_eps = self.exploration.value(self.num_timesteps)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = \
                    -np.log(1. - self.exploration.value(self.num_timesteps) +
                            self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            for env_step in range(environment_steps):
                acts = []
                new_observations = []
                with self.sess.as_default():
                    for i in range(self.env.nb_agents):
                        acts.append(self.act(np.array(observations[i])[None], update_eps=update_eps, **kwargs)[0])
                step_returns = self.env.step(acts)

                for i in range(self.env.nb_agents):
                    self.num_timesteps += 1

                    obs = observations[i]
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    # Take action and update exploration to the newest value
                    kwargs = {}
                    if not self.param_noise:
                        update_eps = self.exploration.value(env_step)
                        update_param_noise_threshold = 0.
                    else:
                        update_eps = 0.
                        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                        # for detailed explanation.
                        update_param_noise_threshold = \
                            -np.log(1. - self.exploration.value(env_step) +
                                    self.exploration.value(env_step) / float(self.env.action_space.n))
                        kwargs['reset'] = reset
                        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                        kwargs['update_param_noise_scale'] = True

                    action = acts[i]
                    reset = False
                    new_obs, rew, done, info = step_returns[i]
                    new_observations.append(new_obs)
                    # don't learn from a step that is marked with skip
                    if info["skip"]:
                        continue

                    # Store transition in the replay buffer.
                    self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                    obs = new_obs

                    if writer is not None:
                        ep_rew = np.array([rew]).reshape((1, -1))
                        ep_done = np.array([done]).reshape((1, -1))
                        tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                            self.num_timesteps)

                    episode_rewards[-1] += rew
                    if done:
                        maybe_is_success = info.get('is_success')
                        if maybe_is_success is not None:
                            episode_successes.append(float(maybe_is_success))
                        if not isinstance(self.env, VecEnv):
                            obs = self.env.reset()[0]
                        episode_rewards.append(0.0)
                        reset = True

                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample and self.num_timesteps > self.learning_starts \
                            and self.num_timesteps % self.train_freq == 0:
                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        if self.prioritized_replay:
                            experience = self.replay_buffer.sample(self.batch_size,
                                                                   beta=self.beta_schedule.value(env_step))
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                            weights, batch_idxes = np.ones_like(rewards), None

                        if writer is not None and i == 0:
                            # run loss backprop with summary, but once every 100 steps save the metadata
                            # (memory, compute time, ...)
                            if (1 + env_step) % 100 == 0:
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                      dones, weights, sess=self.sess,
                                                                      options=run_options,
                                                                      run_metadata=run_metadata)
                                writer.add_run_metadata(run_metadata, 'step%d' % env_step)
                            else:
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                      dones, weights, sess=self.sess)
                            writer.add_summary(summary, env_step)
                        else:
                            _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones,
                                                            weights,
                                                            sess=self.sess)

                        if self.prioritized_replay:
                            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    if can_sample and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                    if len(episode_rewards[-101:-1]) == 0:
                        mean_100ep_reward = -np.inf
                    else:
                        mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                    num_episodes = len(episode_rewards)
                    if self.verbose >= 1 and log_interval is not None and self.num_timesteps % log_interval == 0:
                        logger.record_tabular("steps", self.num_timesteps)
                        logger.record_tabular("episodes", num_episodes)
                        if len(episode_successes) > 0:
                            logger.logkv("success rate", np.mean(episode_successes[-100:]))
                        logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                        logger.record_tabular("% time spent exploring",
                                              int(100 * self.exploration.value(env_step)))
                        logger.dump_tabular()

                observations = new_observations

        return self
