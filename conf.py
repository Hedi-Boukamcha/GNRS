M: int = 3
L: int = 2

NB_STATIONS: int = 3
BIG_STATION: int = 1

PROCEDE_1: int = 0
PROCEDE_2: int = 1

PROCEDE_1_SEQ_MODE_A: int = 0
PROCEDE_1_PARALLEL_MODE_B: int = 1
PROCEDE_2_MODE_C: int = 2

STATION_1: int = 0
STATION_2: int = 1
STATION_3: int = 2

FIRST_OP: int = 0

POS_STATION: int   = 0
POS_PROCESS_1: int = 1
POS_PROCESS_2: int = 2
LOCATION_NAMES: list[str] = ["stations", "process 1", "process 2"]

LOAD: int    = 0
MOVE: int    = 1
HOLD: int    = 2
POS:  int    = 3
EXECUTE: int = 4
UNLOAD: int  = 5
EVENT_NAMES: list[str] = ["load", "move", "hold", "pos", "execute", "unload"]

NOT_YET: int = 0
IN_SYSTEM: int = 1
IN_EXECUTION: int = 2
DONE: int = 3

COLOR = {
    "job":   "tab:blue",
    "proc_1":    "tab:orange",
    "proc_2":    "tab:orange",
    "station": "tab:green",
    "robot":   "tab:red",
}

INSTANCES_SIZES: list[str] = [("s", 3, 5), ("m", 7, 10), ("l", 15, 20), ("xl", 30, 50)]

# DL model configuration
DROPOUT: float        = 0.1
ATTENTION_HEADS: int  = 4
JOB_DIM: int          = 16
NODE_DIM: int         = 8
GRAPH_DIM: int        = 32

# Nb raw features
JOB_FEATURES: int     = 13
STATION_FEATURES: int = 2
MACHINE_FEATURES: int = 3
ROBOT_FEATURES: int   = 4

# Training configuration
BATCH_SIZE          = 256
CAPACITY            = 100_000
OPTIMIZATION_RATE   = 5
GAMMA               = 0.99  # discount factor
TAU                 = 0.003 # update rate of the target network
LR                  = 1e-4  # learning rate of AdamW 
EPS_START           = 0.99  # starting value of epsilon
EPS_END             = 0.005 # final value of epsilon
EPS_DECAY_RATE      = 0.33  # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)
NB_EPISODES         = 6000