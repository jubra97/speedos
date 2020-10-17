import json
import os

cells = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
         ]

players = {"1": {
    "x": 10,
    "y": 4,
    "direction": "down",
    "speed": 1,
    "active": True
},
    "2": {
        "x": 7,
        "y": 5,
        "direction": "up",
        "speed": 1,
        "active": True
    },
    "3": {
        "x": 4,
        "y": 11,
        "direction": "up",
        "speed": 1,
        "active": True
    }
}

values = {
    "width": 15,
    "height": 12,
    "cells": cells,
    "players": players
}
print(values)
with open(os.path.abspath("..") + "/res/originalGames/test_game.json", "w") as f:
    json.dump(values, f)
