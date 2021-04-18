**<p align="center">Welcome to the world of Spe_ed, where Team Speedos rules</p>**

<p align="center"><img src="https://user-images.githubusercontent.com/72612967/115153514-a634df00-a076-11eb-8afa-2624515ebb33.jpg" /></p>

## Algorithm

We are using a multi-player extension of the well known [Minimax-Algorithm](https://en.wikipedia.org/wiki/Minimax) - called [Multi-Minimax](https://link.springer.com/chapter/10.1007/978-3-030-35288-2_4). The evaluation function for non-final game states is mainly based on the so called [Voronoi-Heuristic](https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html). In a nutshell, the Voronoi-Heuristic calculates how many cells each player could potentially reach ahead of every other player. The returned value is then calculated as the difference betweeen the amount of cells that the maximizing and minimizing players can reach first. This encourages our player to control as much space as possible and to corral opponents. In order to deal with the variable time limit for action responses we implemented [Depth-First Iterative Deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search). Furthermore, we enhanced our approach with extensions, such as [Alpha-Beta-Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) or Wall-Hugging.

## Project Structure & Architecture

The project is split into the following packages: *core*, *evaluation* and *scripts*. The *core* package is the functional core of our project - it contains a model of Spe_ed and all of the different player algorithms (agents); the *evaluation* package contains everything that can be used to evaluate agents; the *scripts* package contains application oriented scripts, such as a script for the online execution of our agents. The figure below shows the dependency hirarchy between the upper mentioned packages. The basic idea behind the structure is that a user can simply extend or use the scripts and evaluation tools without having to worry about the concrete *core*-implementation.

<p align="center"><img src="https://user-images.githubusercontent.com/72612967/115119802-8afc9d80-9faa-11eb-83df-6a6872fc4228.png" /></p>

The *model* can be viewed as a black-box replica of the game Spe_ed. It provides the exact same interfaces as the original so that algorithms can use both game instances without any adjustments. The model is created within the agent-based modelling framework [Mesa](https://mesa.readthedocs.io/en/master/) which provides additional tools for visualization and data science. A simplified class diagram of the model architecture can be seen in the figure below. Most importantly it contains the *SpeedModel* and an abstract *SpeedAgent*. The *SpeedModel* implements game rules and is used to control the execution of a game instance (e.g. create, run or step forward). An instance of the *SpeedModel* class is mainly used by Muli-Minimax-agents to simulate future actions and scenarios. The *SpeedAgent* class provides the abstract method *act(state)*. Every functional agent is a subclass of *SpeedAgent* that implements this function. *act* receives a game state and returns an action. A detailed description of the game state format and possible actions can be found in the [InformatiCup repository](https://github.com/informatiCup/InformatiCup2021). In case you want to implement an agent yourself, you could also take a look at [agents](https://github.com/jubra97/speedos/tree/main/src/core/agents.py) we already implemented.

<p align="center"><img src="https://user-images.githubusercontent.com/72612967/115124512-f3f00f80-9fc2-11eb-947a-0dd8c7e343ea.png" /></p>

The [res folder](https://github.com/jubra97/speedos/tree/main/res) contains evaluation results and recorded games from the original game. These records are mainly used to test the model and make sure that it functions exactly like the original game. All other core software parts are also tested with unit tests. Tests are placed in a subfolder (called *tests*) within the respectively tested source folder - as its standard for python unit tests.


## Getting Started

### Installing Requirements

Execute the following comands in the project directory:
```shell
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### How to use Docker

We use docker to deploy our software. If you want to deploy a specific version you can bould a docker image with the following commands (if you already have a local repository you can skip the *git clone* command):
```shell
git clone https://github.com/jubra97/speedos.git
cd speedos
docker build -t speedosagent .
```

If you just want to use the latest version you can simply use the pre-built docker image that we provide:
```shell
docker pull docker.pkg.github.com/jubra97/speedos/speedos-agent:latest
docker tag docker.pkg.github.com/jubra97/speedos/speedos-agent:latest speedosagent
```

To start the docker container execute the following commands:
```shell
docker run -e URL="wss://msoll.de/spe_ed" -e KEY="IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO" -e TIME_URL="https://msoll.de/spe_ed_time" speedosagent
```

### Start Coding

The following code snippet shows how easy it is to create and run a fully functional game with different agents:
```python
model = SpeedModel(60, 60, 2, agent_classes=[RandomAgent, SlidingWindowVoronoiMultiMiniMaxAgent], verbose=True)
model.run_model()
```
You can also have a look at the [scripts folder](https://github.com/jubra97/speedos/tree/main/src/scripts) to see how we used the projects core to deploy, test and evaluate our software.

## Contact, Contribution & Further Use

We welcome everyone to contribute to our project and will gladly receive and answer any suggestions or questions that you might have. We encourage you to create GitHub-Issues in case of bug encounters or feature suggestions. In other cases the best way to contact us is via [e-mail](mailto:maximilian.demmler@student.uni-augsburg.de).

Our code is free to use for everybody under the conditions stated in our [license](https://github.com/jubra97/speedos/blob/main/LICENSE). However, we would like to kindly ask you to acknowledge our work if you wish to use it for research, educational or commercial purposes.
