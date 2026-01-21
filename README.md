# A reinforcement learning and a dynamic graph neural network-based scheduling agent to control a multi-task robot

## Project presentation
1. This repository in linked to a scientific paper under review, the pre-print is available at: [SSRN pre-print for Robotics and Computer-Integrated Manufacturing (RCIM)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)

2. A video presentation of the project is available at: [Presentation video on YouTube (FRENCH)](https://www.youtube.com/watch?v=wMU39mVTmOg)

3. This repository is under a [MIT License](https://github.com/Hedi-Boukamcha/GNRS/blob/main/LICENSE)

## Introduction to the problem solved
> *"In the last fifteen years, manufacturing systems have undergone a profound transformation driven by Industry 4.0 technologies. The integration of smart resources and robots, sensors, and digital platforms offers companies the opportunity to make more informed assessments, adapt decisions in real-time, and ultimately achieve better performance. Yet, this integration has also introduced new challenges in the management and optimization of industrial processes, particularly in operation scheduling. The robotic resource studied in the paper is a complex welding cell comprising several collaborating components: three loading stations, a robotic arm responsible for picking up and transporting parts, two different welding machines, and a positioner designed to hold a part during processing. The main objective is to minimize both the makespan of all parts and their tardiness. To this end, this paper proposes a mathematical formulation for an optimization model as well as a scheduling agent based on a dynamic Graph Neural Network (GNN), trained with an adapted ϵ-greedy deep Q-learning algorithm. In addition to the GNNbased policy, our agent utilizes a custom decision simulator to generate dates and movements, respecting all system constraints and logic. For large instances, our agent requires between 0.9 and 2.35 seconds to complete. Yet, the solving stage is a search process that includes a local improvement heuristic: the actual GNN-based agent only needs 0.07 to 0.14 seconds to construct a single solution. The memory usage is negligible, even during training. In contrast, the mathematical model, optimized via a constraint programming solver, used the maximum allowed computation time and memory (24 hours and 185 GB RAM) to find its best solution. For small instances where the mathematical model achieves optimal solutions, our agent reached a median deviation of 6.64%. For large-sized problems (for which the mathematical model only finds feasible solutions), the agent outperformed the mathematical model for 64% of the instances and achieved a median deviation of -2.82%."*
> — [Boukamcha et al. (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)

### **Fig. 1: STUDIED ROBOT**, extracted from [Boukamcha et al. (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)
![robot presentation](/docs/robot.png)

## Test the code
1. `python3 -m venv gnrs_env`
2. `source gnrs_env/bin/activate`
3. `pip3 install --upgrade pip`
4. Several possible modes:
    * For the DRL and heuristic agents `pip3 install -r requirements/drl.txt`
    * For the OR solver: `pip3 install -r requirements/or.txt`
5. Several possible modes:
    * Training stage: `python gnn_solver.py --mode=train --interactive=true --load=false --custom=true --path=.`
    * Test one problem (DRL): `python gnn_solver.py --mode=test_one --size=s --id=1 --improve=true --interactive=false --load=true --custom=true --path=.`
    * Solve all instances (DRL): `python gnn_solver.py --mode=test_all --improve=true --interactive=false --load=true --custom=true --path=.`
    
    * Test one problem (Heuristic, with or without Tabu search): `python heuristic_solver.py --mode=test_one --size=s --id=1 --tabu=true --path=./`
    * Solve all instances (Heuristic, with or without Tabu search):  `python heuristic_solver.py --mode=test_all --tabu=true --path=.`

    * Solve one problem (OR): `python cp_solver.py --type=test --size=s --id=1 --path=./`

## Proposed approach

### **Fig. 2: AI AGENT OVERVIEW**, extracted from [Boukamcha et al. (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)
![approach overview](/docs/overview.png)

### **Fig. 3: GNN MODEL ARCHITECTURE**, extracted from [Boukamcha et al. (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5797825)
![GNN architecture](/docs/gnn2.png)

## Refer to this repository in scientific documents
BOUKAMCHA, Hedi et al. (2025). A reinforcement learning and a dynamic graph neural network-based scheduling agent to control a multi-task robot. *GitHub repository: https://github.com/Hedi-Boukamcha/GNRS*.

```bibtex
    @misc{GNRS25,
      authors = {BOUKAMCHA, Hedi and NEUMANN, Anas and REKIK, Monia, and HAJJI, Adnene and FARAH, Mohamed},
      title = {A reinforcement learning and a dynamic graph neural network-based scheduling agent to control a multi-task robot},
      year = {2025},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Hedi-Boukamcha/GNRS}},
      commit = {main}
    }
```