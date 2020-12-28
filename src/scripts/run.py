from src.agents import NStepSurvivalAgent, BaseMultiMiniMaxAgent
from src.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(10, 10, 2, agent_classes=[BaseMultiMiniMaxAgent for i in range(2)])
    print(model.schedule.steps, model.active_speed_agents[0].game_step)
    while model.running:
        model.step()
        print(model.schedule.steps, model.active_speed_agents[0].game_step)
