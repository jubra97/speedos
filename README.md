# Speedos 
Unsere Lösung zum [InformatiCup 2021](https://github.com/informatiCup/InformatiCup2021).

## Background

Wir verwenden eine Erweiterung des Minimax-Algorithmus aus der Spieltheorie. Dieser wird angepasst zum Multi-Minimax für mehrere Gegner und mittels Voronoi werden Spielsituationen evaluiert.

## Software

Das grundlegende Modell ist im Mesa-Framework geschrieben. Ein generischer Agent wird dank Modularität und Vererbung zu Agenten erweitert, die verschiedene Lösungen implementieren. Das Modell kann sehr einfach gestartet und konfiguriert werden: 

```python
model = SpeedModel(60, 60, 2, agent_classes=[RandomAgent, SlidingWindowVoronoiMultiMiniMaxAgent], verbose=True)
model.run_model()
```

## Handbuch Docker 

### Repository klonen und Docker Image bauen
```shell
git clone https://github.com/jubra97/speedos.git
cd speedos
docker build -t speedosagent .
```

### Alternativ: Pre-built Image laden
```shell
docker pull ghcr.io/luk-ha/liveagent:latest
docker tag ghcr.io/luk-ha/liveagent:latest speedosagent
```

### Docker Container starten
```shell
docker run -e URL="wss://msoll.de/spe_ed" -e KEY="<Your API key>" -e TIME_URL="https://msoll.de/spe_ed_time" speedosagent
```


