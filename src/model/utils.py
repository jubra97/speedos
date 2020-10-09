from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __str__(self):
        return self.name.lower()


class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    SLOW_DOWN = 2
    SPEED_UP = 3
    CHANGE_NOTHING = 4

    def __str__(self):
        return self.name.lower()


def agent_to_json(agent):
    x, y = agent.pos
    return {
        "x": x,
        "y": y,
        "direction": str(agent.direction),
        "speed": agent.speed,
        "active": agent.active
    }


def model_to_json(model):
    players = dict()
    for agent in model.speed_agents:
        players[str(agent.unique_id)] = agent_to_json(agent)

    return {
        "width": model.width,
        "height": model.height,
        "cells": model.cells,
        "players": players,
        "running": model.running
    }


def get_state(model, agent):
    state = model_to_json(model)
    state["you"] = agent.unique_id
    return state
