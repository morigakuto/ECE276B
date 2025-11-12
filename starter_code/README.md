# ECE276B PR2 Spring 2025

## Overview
In this assignment, you will implement and compare the performance of search-based and sampling-based motion planning algorithms on several 3-D environments.

### 1. main.py
This file contains examples of how to load and display the environments and how to call a motion planner and plot the planned path. Feel free to modify this file to fit your specific goals for the project. In particular, you should certainly replace Line 104 with a call to a function which checks whether the planned path intersects the boundary or any of the blocks in the environment.

### 2. Planner.py
This file contains an implementation of a baseline planner. The baseline planner gets stuck in complex environments and is not very careful with collision checking. Modify this file in any way necessary for your own implementation.

### 3. astar.py
This file contains a class defining a node for the A* algorithm as well as an incomplete implementation of A*. Feel free to continue implementing the A* algorithm here or start over with your own approach.

### 4. maps
This folder contains 7 test environments described via a rectangular outer boundary and a list of rectangular obstacles. The start and goal points for each environment are specified in main.py.

### 5. Batch evaluation utility
The helper script `run_all_planners.py` (callable as `python -m starter_code.run_all_planners`) runs every planner in `Planner.py` on the benchmark maps, records runtime and path-quality statistics, and saves 3-D visualizations. By default it creates a timestamped folder under `starter_code/reports/` containing:

- `summary.json` and `summary.csv` with aggregate metrics.
- One subdirectory per planner with per-map images (`.png`), raw waypoint exports (`_path.csv`), and detailed metrics (`_metrics.json`).

You can limit the run to a subset of planners or maps:

```
python -m starter_code.run_all_planners --planners MyAStarPlanner MyRRTStarPlanner --maps maze window
```

To change the output location, pass `--output-root <dir>`. Long runs can be launched inside a terminal multiplexer (e.g. `tmux` or `screen`) so you can detach while results are generated; install one via your package manager if you do not already have it.


