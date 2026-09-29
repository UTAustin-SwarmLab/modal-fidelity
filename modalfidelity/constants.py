"""Constants shared by the router, the baselines and the evaluation (paper Sec. 2 and 3)."""
from __future__ import annotations

#: the four per-window actions; NONE must stay 0 so an argmax of 0 means "acquire nothing"
ACTIONS = ("none", "audio", "image", "both")
NONE, AUDIO, IMAGE, BOTH = 0, 1, 2, 3

#: cost of each action in detector calls (paper: c^m = 1, so a window read in both streams costs 2)
UNIT_COST = (0.0, 1.0, 1.0, 2.0)

#: the remaining budget b_t and horizon T - t enter the router as absolute counts divided by
#: this fixed constant (numerics only; it never turns them into ratios)
INPUT_SCALE = 100.0

#: preview feature width and LSTM width of the router (2.37 M trainable parameters in total)
FEATURE_DIM = 2048
LSTM_HIDDEN = 256

#: per-window geometry of AV-Deepfake1M windows: 1 s windows (25 frames) at a 0.24 s stride
WINDOW_FRAMES, WINDOW_STRIDE, FPS = 25, 6, 25

#: fixed detector operating points tau^m (Youden-J on the held-out benchmark)
THRESHOLD_AUDIO = 4.514        # W2V2-AASIST
THRESHOLD_IMAGE = 0.738        # GenD (CLIP ViT-L)

#: budgets reported in the paper, as a fraction rho of the video's windows
RHOS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)

#: false-alarm penalty p of the reward used by the released router
FALSE_ALARM = 0.25
