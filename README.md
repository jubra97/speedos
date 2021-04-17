# Speedos 

**<p align="center">Welcome to the world of Spe_ed, where Team Speedos rules</p>**

This repository contains our approach to the [InformatiCup 2021](https://github.com/informatiCup/InformatiCup2021) competition. Here we will give you a brief introduction to the underlying algorithm as well as the project structure and architecture. Furthermore, we will guide you through the installation process and provide a code example to get you started.

## Algorithm

We are using a multi-player extension of the well known [Minimax-Algorithm](https://en.wikipedia.org/wiki/Minimax) - called [Muli-Minimax](https://link.springer.com/chapter/10.1007/978-3-030-35288-2_4). The evaluation function for non-final game states is mainly based on the so called [Voronoi-Heuristic](https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html). In a nutshell, the Voronoi-Heuristic calculates how many cells each player could potentially reach ahead of every other player. The returned value is then calculated as the difference betweeen the amount of cells that the maximizing and minimizing players can reach first. This encourages our player to control as mouch space as possible and to corral opponents. In order to deal with the variable time limit for action responses we implemented [Depth-First Iterative Deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search). Furthermore, we enhanced our approach with extensions, such as [Alpha-Beta-Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) or Wall-Hugging.

## Project Structure & Architecture

The project is split into the following packages: core, evaluation and scripts. The *core* package is the functional core of our project - it contains the
Spe_ed model and all of the different player algorithms (agents); the *evaluation* package contains everything that can be used to evaluate agents; the *scripts* package contains application oriented scripts, such as a script for the online execution of our agents. The figure below shows the dependency hirarchy between the upper mentioned packages. The basic idea behind the structure is that a user can simply extend or use the scripts and evaluation tools without having to worry about the concrete core implementation.

<p align="center"><img src="https://user-images.githubusercontent.com/72612967/115119802-8afc9d80-9faa-11eb-83df-6a6872fc4228.png" /></p>

The *model* can be viewed as a black-box replica of the game Spe_ed. It provides the exact same interfaces as the original so that algorithms can use both game instances without any adjustments. We use the agent-based modelling framework [Mesa](https://mesa.readthedocs.io/en/master/) t... A simplified The model is mainly used by Muli-Minimax-agents to simulate future actions.


## Project Architecture

Das grundlegende Modell ist im Mesa-Framework geschrieben. Ein generischer Agent wird dank Modularität und Vererbung zu Agenten erweitert, die verschiedene Lösungen implementieren. Das Modell kann sehr einfach gestartet und konfiguriert werden...


## Getting Started

### Installing Requirements

Execute the following comands in the project directory:
```shell
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### How to use Docker

Clone the repository and build a docker image:
```shell
git clone https://github.com/jubra97/speedos.git
cd speedos
docker build -t speedosagent .
```

Alternatively: Use our pre-built docker image
```shell
docker pull ghcr.io/luk-ha/liveagent:latest
docker tag ghcr.io/luk-ha/liveagent:latest speedosagent
```

Start the docker container:
```shell
docker run -e URL="wss://msoll.de/spe_ed" -e KEY="IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO" -e TIME_URL="https://msoll.de/spe_ed_time" speedosagent
```

### Start Coding

The following code snippet shows how easy it is to create and run a fully functional game with different agents:
```python
model = SpeedModel(60, 60, 2, agent_classes=[RandomAgent, SlidingWindowVoronoiMultiMiniMaxAgent], verbose=True)
model.run_model()
```
You can also have a look at the [scripts folder](https://github.com/jubra97/speedos/tree/main/src/scripts) to see how we used the project core to deploy, test and evaluate our software.

## Contact and Contribution

We welcome everyone to contribute to our project and will gladly receive and answer any suggestions or questions that you might have. The best way to contact us is via [e-mail](mailto:maximilian.demmler@student.uni-augsburg.de).
