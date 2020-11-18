import numpy as np
import pandas as pd
import webbrowser
from datetime import datetime

from src.model.model import SpeedModel


class Evaluator:

    def __init__(self, model_params, parameter_settings_info=None):
        self.model_params = model_params
        self.parameter_settings_info = parameter_settings_info

        self.model = SpeedModel(**model_params)
        self.win_table = None
        self.elimination_step_table = None
        self.elimination_action_table = None
        self.agent_independent_table = None

    def evaluate(self, repetitions, seeds=None, show=True, save=False):
        self._init_tables(repetitions)
        for rep in range(repetitions):
            self.model = SpeedModel(**self.model_params)
            if seeds is not None:
                self.model.reset_randomizer(seeds[rep])
            self.model.run_model()
            self._update_tables(rep)
        self._process_results(repetitions, show, save)

    def _process_results(self, repetitions, show, save):
        if not show and not save:
            return

        index = [f"Agent {i + 1} ({str(type(self.model.get_agent_by_id(i + 1)).__name__)})"
                 for i in range(self.model.nb_agents)]

        self.win_table *= 100 / repetitions  # convert to percentages
        win_df = pd.DataFrame(self.win_table, index=index, columns=["Wins [%]", "Ties [%]", "Losses [%]"])

        elimination_data = {"ES Mean": [], "ES Std": []}
        for agent_data in self.elimination_step_table:
            data = agent_data[np.nonzero(agent_data)]
            elimination_data["ES Mean"].append(np.mean(data))
            elimination_data["ES Std"].append(np.std(data))
        elimination_step_df = pd.DataFrame(elimination_data, index=index)

        elimination_action_df = pd.DataFrame(self.elimination_action_table, index=index,
                                             columns=["EA Left", "EA Right", "EA Slow Down",
                                                      "EA Speed Up", "EA Change Nothing"])

        agent_table = win_df.join(elimination_step_df).join(elimination_action_df)

        self.agent_independent_table[1] *= 100 / (self.model.width * self.model.height)  # convert to percentages
        agent_independent_df = pd.DataFrame({
            "Game Duration Mean": np.mean(self.agent_independent_table[0]),
            "Game Duration Std": np.std(self.agent_independent_table[0]),
            "Empty Cells Mean [%]": np.mean(self.agent_independent_table[1]),
            "Empty Cells Std [%]": np.std(self.agent_independent_table[1])
        }, index=["Data"])

        parameter_settings = {
            "Width": self.model.width,
            "Height": self.model.height,
            "Repetitions": repetitions
        }
        if self.parameter_settings_info:
            parameter_settings = {**parameter_settings, **self.parameter_settings_info} # join the dicts
        parameter_settings_df = pd.DataFrame(parameter_settings, index=["Data"])

        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        if save:
            writer = pd.ExcelWriter(f"../../res/evaluation/eval_{timestamp}.xlsx", engine='xlsxwriter')

            agent_table.to_excel(writer, sheet_name='Agents')
            agent_independent_df.to_excel(writer, sheet_name='Global')
            parameter_settings_df.to_excel(writer, sheet_name='Parameter Settings')

            writer.save()

        if show:
            agent_table.to_html('temp_agents.html')
            webbrowser.open_new_tab('temp_agents.html')

            agent_independent_df.to_html('temp_global.html')
            webbrowser.open_new_tab('temp_global.html')

            parameter_settings_df.to_html('temp_settings.html')
            webbrowser.open_new_tab('temp_settings.html')

    def _init_tables(self, repetitions):
        self.win_table = np.zeros((self.model.nb_agents, 3))
        self.elimination_step_table = np.zeros((self.model.nb_agents, repetitions))
        self.elimination_action_table = np.zeros((self.model.nb_agents, 5), dtype=np.int)
        self.agent_independent_table = np.empty((2, repetitions))

    def _update_tables(self, repetition):
        for agent in self.model.speed_agents:
            a_idx = agent.unique_id - 1
            if agent.active:
                # win
                self.win_table[a_idx, 0] += 1
            elif len(self.model.active_speed_agents) == 0 and agent.elimination_step == self.model.schedule.steps:
                # tie
                self.win_table[a_idx, 1] += 1
                self.elimination_step_table[a_idx, repetition] = agent.elimination_step
                self.elimination_action_table[a_idx, agent.action.value] += 1
            else:
                # loss
                self.win_table[a_idx, 2] += 1
                self.elimination_step_table[a_idx, repetition] = agent.elimination_step
                self.elimination_action_table[a_idx, agent.action.value] += 1

        self.agent_independent_table[0, repetition] = self.model.schedule.steps
        self.agent_independent_table[1, repetition] = np.count_nonzero(self.model.cells == 0)
