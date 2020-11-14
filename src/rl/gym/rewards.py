from abc import ABC, abstractmethod


class Reward(ABC):

    @abstractmethod
    def payout(self, state, action, next_state, done, agent_id):
        return NotImplementedError

    @property
    @abstractmethod
    def reward_range(self):
        return NotImplementedError


class LongSurvivalReward(Reward):

    def payout(self, state, action, next_state, done, agent_id):
        if next_state["players"][str(agent_id)]["active"]:
            return 1
        else:
            return -1

    @property
    def reward_range(self):
        return -1, 1


class WinLossReward(Reward):

    def payout(self, state, action, next_state, done, agent_id):
        if done:
            if next_state["players"][str(agent_id)]["active"]:
                # won
                return 1
            elif len(list(filter(lambda player: player["active"], list(next_state["players"].values())))) == 0:
                # tie (everyone else died last round or earlier as well)
                return 0
            else:
                # lost
                return -1
        else:
            # neither won nor lost yet
            return 0

    @property
    def reward_range(self):
        return -1, 1
