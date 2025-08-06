# Vocabulary
M: int = 3
L: int = 2

NB_STATIONS: int = 3
BIG_STATION: int = 1

MACHINE_1: int = 0
MACHINE_2: int = 1

MACHINE_1_SEQ_MODE_A: int = 0
MACHINE_1_PARALLEL_MODE_B: int = 1
MACHINE_2_MODE_C: int = 2

STATION_1: int = 0
STATION_2: int = 1
STATION_3: int = 2

FIRST_OP: int = 0 

POS_STATION: int   = 0
POS_MACHINE_1: int = 1
POS_MACHINE_2: int = 2
LOCATION_NAMES: list[str] = ["stations", "machine 1", "machine 2"]

LOAD: int    = 0
MOVE: int    = 1
HOLD: int    = 2
POS:  int    = 3
EXECUTE: int = 4
UNLOAD: int  = 5
AWAIT: int   = 6
EVENT_NAMES: list[str] = ["load", "move", "hold", "pos", "execute", "unload", "await"]

NOT_YET: int = 0
IN_SYSTEM: int = 1
IN_EXECUTION: int = 2
DONE: int = 3

INSTANCES_SIZES: list[str] = [("s", 3, 5), ("m", 7, 10), ("l", 15, 20), ("xl", 30, 50)]
NB_TRAIN: int              = 150

# Solving stage configuration
RETRIES: int = 10

# DL model configuration
DROPOUT: float        = 0.1
ATTENTION_HEADS: int  = 4
JOB_DIM: int          = 16
NODE_DIM: int         = 8
GRAPH_DIM: int        = 32
PB_SIZE_DIM: int      = 6

# Nb raw features
JOB_FEATURES: int     = 13
STATION_FEATURES: int = 2
MACHINE_FEATURES: int = 3
ROBOT_FEATURES: int   = 4
PB_SIZE_FEATURES: int = 10

# Transition weights
T_WEIGTHS: dict = {
    's': 3.0,
    'm': 2.0,
    'l': 1.15,
    'xl': 0.7,
}

# Training configuration
BATCH_SIZE          = 256     # batch size for training
CAPACITY            = 300_000 # replay memory capacity
SAVING_RATE         = 500     # nb episodes before saving the model
SWITCH_RATE         = 30      # nb episodes before switching from an instance to another
GAMMA               = 1.0     # discount factor (none in our case)
TAU                 = 0.003   # update rate of the target network
LR                  = 1e-3    # starting learning rate of AdamW 
MIN_LR              = 1.25e-4 # min learning rate of AdamW 
EPS_START           = 0.99    # starting value of epsilon
EPS_END             = 0.005   # final value of epsilon
EPS_DECAY_RATE      = 18_000  # controls the rate of exponential decay of epsilon
NB_EPISODES         = 60_000  # X (changes) episodes per instances on average
COMPLEXITY_RATE     = 6000    # curriculum learning rate: nb episodes before adding larger instances to the training set
MAX_GRAD_NORM       = 30.0    # max norm for gradient clipping 
LR_PATIENCE         = 800     # patience for the learning rate scheduler (in number of episodes)
LR_REDUCE_RATE      = 3000    # threshold for the learning rate scheduler
REWARD_SCALE        = 1.      # scale factor for the reward
BETA                = 35      # beta parameter for the Huber loss function
TRADE_OFF           = 0.85    # trade-off between the current-value-based reward and the lower-bound-based reward
VALIDATE_RATE       = 200     # nb episodes before validating the model
WARMUP_EPISODES     = 24_000  # nb episodes before starting to adapt (reduce) LR

# Gantt configuration
JOB_COLORS        = ['#8dd3c7', '#80b1d3', '#fb8072', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
GNN_GANTT_LEVELS  = ["Station 1", "Station 2", "Station 3", "Robot", "Machine 1", "Machine 2"]
CP_GANTT_LEVELS   = ["Station 1", "Station 2", "Station 3", "Machine 1", "Machine 2"]
STATIONS          = {"Station 1", "Station 2", "Station 3"}
EVENT_COLORS      = { EXECUTE: "#8dd3c7", LOAD:    "#80b1d3", UNLOAD:  "#fb8072", MOVE:    "#fdb462", HOLD:    "#b3de69", POS:     "#fccde5"}
MIN_REAL_DURATION = 1e-6
BOLD_EVENTS       = {EXECUTE, HOLD, LOAD, UNLOAD, MOVE, POS, AWAIT}