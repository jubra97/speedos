# Speedos 

This repository contains our approach to the [InformatiCup 2021](https://github.com/informatiCup/InformatiCup2021) competition. This Readme will give you a brief introduction to the underlying algorithm as well as the project structure and basic architecture. Furthermore, it guides you through the installation process and gives a quick code example to get you started.

## Algorithm

We are using a multi-player extension of the well known [Minimax-Algorithm](https://en.wikipedia.org/wiki/Minimax) - called [Muli-Minimax](https://link.springer.com/chapter/10.1007/978-3-030-35288-2_4). The evaluation function for non-final game states is mainly based on the so called [Voronoi-Heuristic](https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html). In a nutshell, the Voronoi-Heuristic calculates how many cells each player could potentially reach ahead of every other player. The returned value is then calculated as the difference betweeen the amount of cells that the maximizing and minimizing players can reach first. This encourages our player to control as mouch space as possible and to corral opponents. In order to deal with the variable time limit for action responses we implemented [Depth-First Iterative Deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search). Furthermore, we enhanced our approach with extensions, such as [Alpha-Beta-Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) or Wall-Hugging.

## Project Structure


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
