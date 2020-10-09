from abc import ABC, abstractmethod


class Reward(ABC):

    @abstractmethod
    def payout(self, state, action, next_state):
        return NotImplementedError

    @property
    @abstractmethod
    def reward_range(self):
        return NotImplementedError


class LongSurvivalReward(Reward):

    def payout(self, state, action, next_state):
        agent_id = next_state["you"]
        if next_state["players"][str(agent_id)]["active"]:
            return 1
        else:
            return 0

    @property
    def reward_range(self):
        return 0, 1
