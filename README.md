# A deep reinforcement learning and a dynamic graph neural network-based scheduling agent to control a multi-task robot

## Project presentation
1. This repository in linked to a scientific paper under review, the pre-print is available at: [Robotics and Computer-Integrated Manufacturing (RCIM)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)

2. A video presentation of the project is available at: [Presentation video on YouTube (FRENCH)](https://www.youtube.com/watch?v=wMU39mVTmOg)

3. This repository is under a [MIT License](https://github.com/Hedi-Boukamcha/GNRS/blob/main/LICENSE)

## Introduction to the problem solved
> *"We propose a comprehensive approach to schedule the operations and control the detailed movements of a real-world multi-task robotic cell operating in a modern manufacturing environment. The problem addressed in this paper exhibits similarities with several well-known scheduling problems, especially the Hoist Scheduling Problem (HSP) and the dynamic version of the Flexible Job Shop Scheduling Problem (FJSSP). However, existing formulations in the literature are not sufficient to handle the scheduling problem at hand. To address this gap, this paper proposes a mathematical formulation and scheduling agent based on a dynamic Graph Neural Network (GNN), trained with an adapted ϵ-greedy deep Q-learning algorithm. In addition to the pure deep reinforcement learning policy derived from Q-values, the agent relies on a custom decision simulator to generate feasible dates and movements, respecting all system constraints and operational logic. The complete agent incorporates a dedicated solving strategy based on a Q-values guided beam search and a local improvement operator. For large problems, the agent requires less than a minute to find high-quality solutions. Yet, the solving stage is a search process resulting in 10 distinct solutions: the actual GNN-based agent only needs 0.04 to 0.08 seconds to construct a single solution for the largest instances. The memory usage is negligible, even during training. By contrast, the mathematical model, optimized via a constraint programming solver, used the maximum allowed computation time and memory (24 hours and 185 GB RAM) to find its best solution. For small instances where the mathematical model achieves optimal solutions, the agent reached a median deviation of 2.97%. For large-sized problems, for which the mathematical model only finds feasible solutions, the agent outperformed the mathematical model for most instances and achieved a median deviation of −9.84%. Our agent also outperformed, both in terms of solution quality and computing time, several heuristic approaches, including a Nested Tabu Search"*
> — [Boukamcha et al. (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)

### **Fig. 1: STUDIED ROBOT**, extracted from [Boukamcha et al. (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)
![Robotic cell](/docs/1.png)
![Context under study](/docs/2.png)

## Folder architecture
```shell
🗂️ project_root/
├── 📁 data/
│   ├── 📁 gantts/                  # Output folder for generated Gantt charts
│   ├── 📁 instances/
│   │   ├── 📁 test/                # Test datasets (subfolders: s, m, l, xl)
│   │   └── 📁 train/               # Training datasets (subfolders: s, m, l, xl)
│   └── 📁 training/                # Saved RL weights (.pth) and training logs
├── 📁 gantt_builder/
│   ├── 📄 cp_gantt.py              # Visualization for CP solutions
│   └── 📄 gnn_gantt.py             # Visualization for GNN/Heuristic solutions
├── 📁 heuristic/
│   ├── 📄 local_search.py          # Local Search (LS) logic
│   └── 📄 tabu_search.py           # Tabu Search (TS) logic
├── 📁 models/
│   ├── 📁 gnn/
│   │   ├── 📄 basic.py             # Basic GNN architecture
│   │   └── 📄 custom.py            # Custom GNN architecture (QNet, Embeddings)
│   ├── 📄 agent.py                 # DQN Agent implementation
│   ├── 📄 environment.py           # RL Environment wrapper
│   ├── 📄 instance.py              # Data structures (Job, Operation, Instance)
│   ├── 📄 memory.py                # Replay Memory (Experience Replay)
│   └── 📄 state.py                 # State representation (Robot, Machines, Stations)
├── 📁 simulators/
│   └── 📄 gnn_simulator.py         # Step-by-step discrete event simulator
├── 📁 utils/
│   └── 📑 common.py                # Helper functions (device handling, conversions)
├── ⚙️ conf.py                      # Global configuration (Constants, Hyperparameters)
├── ▶️ cp_solver.py                 # Exact Solver (Google OR-Tools)
├── ▶️ gnn_solver.py                # Main entry point for GNN training and testing
├── ▶️ heuristic_solver.py          # Main entry point for LS and Tabu Search
├── ▶️ instance_generator.py        # Script to generate synthetic datasets
├── 📜 LICENSE                      # MIT License
└── 📄 README.md
```

## Test the code
1. `python3 -m venv gnrs_env`
2. `source gnrs_env/bin/activate`
3. `pip3 install --upgrade pip`
4. Several possible modes:
    * For the DRL and heuristic agents `pip3 install -r requirements/drl.txt`
    * For the OR solver: `pip3 install -r requirements/or.txt`
5. Several possible modes:
    * Training stage: `python gnn_solver.py --mode=train --interactive=true --load=false --custom=true --path=.`
    * Test one problem (DRL): `python gnn_solver.py --mode=test_one --size=s --id=1 --improve=true --interactive=false --load=true --custom=true --beam=true --path=.`
    * Solve all instances (DRL): `python gnn_solver.py --mode=test_all --improve=true --interactive=false --load=true --custom=true --beam=true --path=.`
    
    * Test one problem (Heuristic, with or without Tabu search): `python heuristic_solver.py --mode=test_one --size=s --id=1 --tabu=true --path=./`
    * Solve all instances (Heuristic, with or without Tabu search):  `python heuristic_solver.py --mode=test_all --tabu=true --path=.`

    * Solve one problem (OR): `python cp_solver.py --type=test --size=s --id=1 --path=./`

## Proposed approach (Figures extracted from [Boukamcha et al. (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825))
![Approach overview](/docs/3.png)
![Graph representation](/docs/4.png)
![GNN architecture](/docs/5.png)
![Simulator](/docs/6.png)
![Training process](/docs/7.png)
![Local search](/docs/8.png)
![Beam decoding](/docs/9.png)

## Refer to this repository in scientific documents
BOUKAMCHA, Hedi et al. (2026). A deep reinforcement learning and a dynamic graph neural network-based scheduling agent to control a multi-task robot. *GitHub repository: https://github.com/Hedi-Boukamcha/GNRS*.

```bibtex
    @misc{GNRS26,
      authors = {BOUKAMCHA, Hedi and NEUMANN, Anas and REKIK, Monia, and HAJJI, Adnene, CARON GUILLEMETTE, Gabriel, and FARAH, Mohamed},
      title = {A deep reinforcement learning and a dynamic graph neural network-based scheduling agent to control a multi-task robot},
      year = {2026},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Hedi-Boukamcha/GNRS}},
      commit = {main}
    }
```