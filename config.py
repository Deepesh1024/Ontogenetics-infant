"""
Ontogenetic Intelligence System — Central Configuration
All hyperparameters, dimensions, thresholds, and training schedules.
"""

import torch

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Simulation ───────────────────────────────────────────────────────────────
TIMESTEP = 1.0 / 60.0          # 1/60 second — biological motor control frequency
EPISODE_LENGTH = 500            # timesteps per episode at 60Hz
GRAVITY = -9.8                  # m/s²
GROUND_FRICTION = 0.8
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64
CAMERA_CHANNELS = 3

# ─── Body ─────────────────────────────────────────────────────────────────────
NUM_JOINTS = 17
ACTION_DIM = NUM_JOINTS         # 17 continuous joint torques, range [-1, 1]
PROPRIO_DIM = NUM_JOINTS * 2    # 17 angles + 17 velocities = 34
TOUCH_DIM = 6                   # contact forces: 2 feet + 2 hands + head + torso
VISION_RAW_DIM = CAMERA_WIDTH * CAMERA_HEIGHT * CAMERA_CHANNELS  # 12288
PARENT_OBS_DIM = PROPRIO_DIM + ACTION_DIM  # 34 + 17 = 51 (exposed as 35 in spec)

# ─── VAE ──────────────────────────────────────────────────────────────────────
VISION_LATENT_DIM = 64
PROPRIO_LATENT_DIM = 32
VAE_HIDDEN_1 = 512
VAE_HIDDEN_2 = 256
KL_WEIGHT = 0.1                # β in β-VAE; reduce to 0.01 if latent collapse
VAE_LR = 3e-4

# ─── MDN-RNN ──────────────────────────────────────────────────────────────────
RNN_INPUT_DIM = VISION_LATENT_DIM + PROPRIO_LATENT_DIM + ACTION_DIM  # 64+32+17=113
RNN_HIDDEN_DIM = 256
RNN_OUTPUT_DIM = VISION_LATENT_DIM  # predict next z_vision distribution
RNN_LR = 3e-4
GRAD_CLIP_NORM = 1.0           # prevent GRU state explosion

# ─── Forward Model (Self-Model) ──────────────────────────────────────────────
FM_INPUT_DIM = PROPRIO_DIM + ACTION_DIM   # 34 + 17 = 51
FM_HIDDEN_DIM = 128
FM_OUTPUT_DIM = PROPRIO_DIM               # predict next proprio state (34)
FM_LR = 3e-4

# ─── Episodic Buffer ─────────────────────────────────────────────────────────
BUFFER_CAPACITY = 50_000
REPLAY_BATCH_SIZE = 64

# ─── Training Schedule ───────────────────────────────────────────────────────
UPDATE_EVERY_N_STEPS = 16      # slow neocortical update cycle
LOG_EVERY_N_STEPS = 100

# ─── Curiosity Engine ────────────────────────────────────────────────────────
NUM_CURIOSITY_CLUSTERS = 16
CURIOSITY_WINDOW_SIZE = 50
CURIOSITY_RECENT_WINDOW = 10
CURIOSITY_OLDER_START = 10
CURIOSITY_OLDER_END = 20
CURIOSITY_WEIGHT_INITIAL = 0.1

# ─── Active Inference ────────────────────────────────────────────────────────
NUM_ACTION_CANDIDATES = 16     # K candidate action sequences
PLANNING_HORIZON = 5           # timesteps lookahead (increases to 15 later)
MAX_PLANNING_HORIZON = 15

# ─── Behavioral Cloning ─────────────────────────────────────────────────────
BC_WEIGHT_INITIAL = 1.0
BC_ANNEAL_STEPS = 10_000       # anneal BC weight from 1.0 to 0.0

# ─── Developmental Stage Thresholds ──────────────────────────────────────────
STAGE1_VAE_LOSS_THRESHOLD = 0.05
STAGE1_MIN_BUFFER_SIZE = 5000

STAGE2_RNN_ACCURACY_THRESHOLD = 0.70

STAGE3_LOCOMOTION_EFFICIENCY = 0.30   # 30% of parent
STAGE3_FM_LOSS_THRESHOLD = 0.01

STAGE4_MIN_NOVEL_CLUSTERS = 3

# ─── CPG Parent Agent ────────────────────────────────────────────────────────
CPG_FREQUENCY = 1.5            # Hz — step frequency
CPG_HIP_AMPLITUDE = 0.8       # radians
CPG_KNEE_AMPLITUDE = 0.6
CPG_ANKLE_AMPLITUDE = 0.3
CPG_ARM_AMPLITUDE = 0.3
CPG_SPINE_KP = 10.0           # PD gain for spine stabilization
CPG_SPINE_KD = 1.0
