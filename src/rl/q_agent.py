from src.rl.agent import RLAgent
from src.utils import arg_maxes
from functools import reduce
import numpy as np
import pandas as pd
import datetime


class QAgent(RLAgent):

    def __init__(self, env, dynamic=False):
        super().__init__(env)
        self.q_table = QTable(self.observation_space, self.action_space, dynamic)

    def learn(self, nb_episodes, alpha_start=0.1, alpha_min=0.1, gamma=0.99, epsilon_start=1, epsilon_min=0.02,
              exploration_fraction=0.6, nb_max_episode_steps=None, callback=None, visualize=False, tb_log_name=None,
              verbose=False):
        self._init_log(tb_log_name)
        self.env._max_episode_steps = nb_max_episode_steps
        epsilon = epsilon_start
        alpha = alpha_start
        for episode in range(nb_episodes):
            state = self.env.reset()
            step = 0
            ret = 0
            done = False
            while not done and (nb_max_episode_steps is None or step < nb_max_episode_steps):
                # Act
                (action, reward, next_state, done) = self.act(state, epsilon)

                # Update q table
                self.q_table.update((state, action, reward, next_state, done), alpha, gamma)

                state = next_state
                step += 1
                ret += reward

                if visualize:
                    self.env.render()
                if callback is not None:
                    callback(self)

            # Update Epsilon
            new_eps = epsilon_start - ((epsilon_start - epsilon_min) * episode / (nb_episodes * exploration_fraction))
            epsilon = max(new_eps, epsilon_min)
            alpha = alpha_start - ((alpha_start - alpha_min) * episode / nb_episodes)

            self._log(episode, ret, verbose)

        return self.avg_episode_rewards

    def act(self, state, epsilon=0):
        if np.random.random() < epsilon:
            # Random action
            action = self.action_space.sample()
        else:
            # Greedy action (pick action with highest q value for the given state)
            maxes = arg_maxes(self.q_table.get_qs(state))
            action = np.random.choice(maxes)
        # Execute chosen action on environment
        next_state, reward, done, _ = self.env.step(action)

        return action, reward, next_state, done

    def reset_states(self):
        pass

    def save(self, filename, timestamp=True):
        self.q_table.save(filename + (datetime.datetime.now().strftime('%Y%m%d-%H%M%S') if timestamp else ''))

    def load(self, filename):
        self.q_table.load(filename)


class QTable:
    """
    Only works for box observation space with one-dimensional shape and discrete action space
    """

    def __init__(self, observation_space, action_space, dynamic=False):
        self.observation_space = observation_space
        self.action_space = action_space
        self.dynamic = dynamic
        self.q = self._init_q()

    def update(self, transition, alpha, gamma):
        state, action, reward, next_state, done = transition
        old_q = self.get_q(state, action)
        if done:
            new_q = old_q + alpha * (reward - old_q)
        else:
            new_q = old_q + alpha * (reward + gamma * max(self.get_qs(next_state)) - old_q)
        self.set_q(state, action, new_q)

    def get_qs(self, state):
        encoded_state = self.encode_observation(state)
        if self.dynamic:
            self._check_state(encoded_state)
            return self.q.loc[encoded_state]
        else:
            return self.q[encoded_state]

    def get_q(self, state, action):
        encoded_state = self.encode_observation(state)
        if self.dynamic:
            self._check_state(encoded_state)
            return self.q.loc[encoded_state][action]
        else:
            return self.q[encoded_state, action]

    def set_q(self, state, action, q):
        encoded_state = self.encode_observation(state)
        if self.dynamic:
            self.q.loc[encoded_state][action] = q
        else:
            self.q[encoded_state][action] = q

    def save(self, filename):
        if self.dynamic:
            self.q.to_pickle(filename)
        else:
            np.save(filename, self.q)

    def load(self, filename):
        if self.dynamic:
            self.q = pd.read_pickle(filename)
        else:
            self.q = np.load(filename + '.npy', allow_pickle=True)

    def encode_observation(self, obs):
        if self.dynamic:
            return ''.join(map(str, obs))
        else:
            base = 1
            encoded_obs = 0
            dimensionality = (self.observation_space.high - self.observation_space.low + 1).flatten()
            obs = np.asarray(obs).flatten()
            lows = self.observation_space.low.flatten()
            for idx, obs_dim in enumerate(obs):
                encoded_obs += base * (obs_dim - lows[idx])
                base *= dimensionality[idx]
            return encoded_obs

    def _init_q(self):
        if self.dynamic:
            return pd.DataFrame(columns=range(self.action_space.n), dtype='float')
        else:
            obs_space_size = self.encode_observation(self.observation_space.high) + 1
            return np.zeros((obs_space_size, self.action_space.n))

    def _check_state(self, state):
        encoded_state = self.encode_observation(state)
        if encoded_state not in self.q.index:
            self.q = self.q.append(pd.Series([0] * self.action_space.n, index=self.q.columns, name=encoded_state))
