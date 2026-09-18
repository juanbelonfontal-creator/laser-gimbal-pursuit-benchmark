import cv2
import numpy as np
import serial
import time
import csv
import math
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# On an old dual-core Intel Mac, tiny tensors are often FASTER on 1-2 threads
# than on 4, because thread sync costs more than the math. Tune if needed.
torch.set_num_threads(2)

# ==========================================
# 1. AI FOUNDATION (Q-TABLE & PYTORCH DQN)
# ==========================================
ACTION_MAP = [
    (0, -1), (0, 1), (-1, 0), (1, 0), 
    (-1, -1), (1, -1), (-1, 1), (1, 1)
]

# --- TABULAR Q-LEARNING ---
Q_TABLE = np.zeros((15, 15, 2, 8)) 

def get_state(tx, ty, hx, hy, t_dir):
    dx = tx - hx
    dy = ty - hy
    bin_dx = int(np.clip((dx + 500) / 1000 * 14, 0, 14))
    bin_dy = int(np.clip((dy + 500) / 1000 * 14, 0, 14))
    dir_idx = 1 if t_dir > 0 else 0
    return (bin_dx, bin_dy, dir_idx)

# --- DEEP Q-NETWORK (DQN) ---
class DQNBrain(nn.Module):
    def __init__(self, input_dim=3, output_dim=8):
        super(DQNBrain, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

policy_net = DQNBrain()
target_net = DQNBrain()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Matched to the CNN's configuration, which reached 100% greedy twice.
# Previously BATCH_SIZE=64 with a backprop on EVERY step, so each stored
# transition was replayed ~64 times before ageing out of the buffer, against
# a target net refreshed only every 1000 steps -- the standard recipe for
# value overestimation. The symptom was the DQN's exploring catch rate
# (67.7%) beating its own greedy rate (47.0%): the policy was degrading late
# in training. Batch 32 every 2 steps gives ~16 replays, like the CNN.
BATCH_SIZE = 32
DQN_OPTIMIZE_EVERY = 2
GAMMA = 0.95        
LR = 1e-3           
TARGET_UPDATE = 1000 
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss() 

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(capacity=10000)

def optimize_model():
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch_state = torch.stack([t[0] for t in transitions])
    batch_action = torch.tensor([t[1] for t in transitions], dtype=torch.int64).unsqueeze(1)
    batch_reward = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    batch_next_state = torch.stack([t[3] for t in transitions])
    batch_done = torch.tensor([t[4] for t in transitions], dtype=torch.float32)

    current_q_values = policy_net(batch_state).gather(1, batch_action).squeeze(1)
    with torch.no_grad():
        max_next_q_values = target_net(batch_next_state).max(1)[0]
        expected_q_values = batch_reward + (GAMMA * max_next_q_values * (1 - batch_done))

    loss = loss_fn(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_state_tensor(tx, ty, hx, hy, t_dir):
    dx = (tx - hx) / 500.0 
    dy = (ty - hy) / 500.0
    dir_val = 1.0 if t_dir > 0 else -1.0
    return torch.tensor([dx, dy, dir_val], dtype=torch.float32)

# ==========================================
# 1b. CONVOLUTIONAL DEEP Q-NETWORK (CNN)  <-- NEW 4th ALGORITHM
# ==========================================
# The DQN above is handed pre-computed coordinates (dx, dy, dir).
# The CNN is handed a PICTURE of the arena instead: a small GRID_SIZE x
# GRID_SIZE image with the target painted in channel 0, the hunter in
# channel 1, and the sweep direction as a constant plane in channel 2.
# It has to learn the geometry from pixels, exactly like it would from a
# real downsampled camera feed. Same reward, same action space, same
# simulator - only the state representation and the encoder differ.

GRID_SIZE = 24            # resolution of the rendered state image
CNN_CHANNELS = 3          # 0 = target, 1 = hunter, 2 = direction plane

CNN_EPISODES = 700        # lower than DQN: conv layers are much slower on CPU
CNN_BATCH_SIZE = 32
CNN_LR = 5e-4
CNN_GAMMA = 0.95
CNN_TARGET_UPDATE = 500
CNN_OPTIMIZE_EVERY = 2    # backprop every N environment steps, not every step
CNN_EPS_DECAY = 0.993

class CNNBrain(nn.Module):
    def __init__(self, in_ch=CNN_CHANNELS, output_dim=8):
        super(CNNBrain, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1)  # 24 -> 12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)     # 12 -> 6
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)     # 6  -> 3
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.dim() == 3:              # single un-batched state -> add batch dim
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

cnn_policy_net = CNNBrain()
cnn_target_net = CNNBrain()
cnn_target_net.load_state_dict(cnn_policy_net.state_dict())
cnn_target_net.eval()

cnn_optimizer = optim.Adam(cnn_policy_net.parameters(), lr=CNN_LR)
cnn_memory = ReplayMemory(capacity=8000)

def get_board_bounds(board):
    """Axis-aligned bounding box of the 4 orange dots."""
    xs = [board['tl'][0], board['tr'][0], board['bl'][0], board['br'][0]]
    ys = [board['tl'][1], board['tr'][1], board['bl'][1], board['br'][1]]
    return (min(xs), max(xs), min(ys), max(ys))

def render_state_grid(tx, ty, hx, hy, t_dir, bounds):
    """Turn the world into a small multi-channel image for the CNN."""
    x_min, x_max, y_min, y_max = bounds
    span_x = max(1.0, float(x_max - x_min))
    span_y = max(1.0, float(y_max - y_min))
    grid = np.zeros((CNN_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def to_cell(px, py):
        gx = int(np.clip((px - x_min) / span_x * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
        gy = int(np.clip((py - y_min) / span_y * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
        return gx, gy

    tgx, tgy = to_cell(tx, ty)
    hgx, hgy = to_cell(hx, hy)
    # paint 3x3 blobs so a single pixel does not disappear through the strides
    grid[0, max(0, tgy - 1):tgy + 2, max(0, tgx - 1):tgx + 2] = 1.0
    grid[1, max(0, hgy - 1):hgy + 2, max(0, hgx - 1):hgx + 2] = 1.0
    grid[2, :, :] = 1.0 if t_dir > 0 else -1.0
    return torch.from_numpy(grid)

def optimize_cnn_model():
    if len(cnn_memory) < CNN_BATCH_SIZE: return
    transitions = cnn_memory.sample(CNN_BATCH_SIZE)
    batch_state = torch.stack([t[0] for t in transitions])
    batch_action = torch.tensor([t[1] for t in transitions], dtype=torch.int64).unsqueeze(1)
    batch_reward = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    batch_next_state = torch.stack([t[3] for t in transitions])
    batch_done = torch.tensor([t[4] for t in transitions], dtype=torch.float32)

    current_q_values = cnn_policy_net(batch_state).gather(1, batch_action).squeeze(1)
    with torch.no_grad():
        max_next_q_values = cnn_target_net(batch_next_state).max(1)[0]
        expected_q_values = batch_reward + (CNN_GAMMA * max_next_q_values * (1 - batch_done))

    loss = loss_fn(current_q_values, expected_q_values)
    cnn_optimizer.zero_grad()
    loss.backward()
    cnn_optimizer.step()

def draw_cnn_view(frame, grid_tensor, size=96):
    """Picture-in-picture of what the CNN actually sees. Great for the video."""
    g = grid_tensor.numpy()
    vis = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    vis[:, :, 1] = np.clip(g[0] * 255, 0, 255).astype(np.uint8)   # target -> green
    vis[:, :, 0] = np.clip(g[1] * 255, 0, 255).astype(np.uint8)   # hunter -> blue
    vis = cv2.resize(vis, (size, size), interpolation=cv2.INTER_NEAREST)
    h, w = frame.shape[:2]
    x0, y0 = w - size - 10, 70
    if x0 > 0 and (y0 + size) < h:
        frame[y0:y0 + size, x0:x0 + size] = vis
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + size, y0 + size), (255, 255, 255), 1)
        cv2.putText(frame, "CNN INPUT", (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

# ==========================================
# 1c. GREEDY POLICY EVALUATION
# ==========================================
# The catch counters printed during training are measured WITH exploration
# still switched on, and each algorithm uses a different epsilon schedule and
# episode count. That makes those numbers useless for comparing algorithms.
#
# This runs each finished policy with epsilon = 0, no learning, no replay
# buffer writes, over the same number of fresh episodes. One honest number
# per algorithm, all measured the same way.

EVAL_EPISODES = 200

def evaluate_policy(choose_action, n_episodes=EVAL_EPISODES):
    """Greedy rollouts in the training simulator. Returns catch rate in %.
    Reads the simulator parameters (sim_tx_min/max, sim_ty, actual_h_speed,
    center_x, home_y) as globals, so it must be called after Phase 1."""
    hits = 0
    for _ in range(n_episodes):
        t_dir = random.choice([-1, 1])
        sim_tx = sim_tx_min if t_dir == 1 else sim_tx_max
        sim_hx = center_x + random.randint(-20, 20)
        sim_hy = home_y + random.randint(-20, 20)
        step_count = 0
        while True:
            step_count += 1
            action = choose_action(sim_tx, sim_ty, sim_hx, sim_hy, t_dir)
            move_x, move_y = ACTION_MAP[action]
            sim_hx += move_x * actual_h_speed
            sim_hy += move_y * actual_h_speed
            sim_tx += V_SPEED * t_dir

            sim_hy = min(sim_hy, home_y + SIM_FLOOR_PX)
            hunter_lost = (sim_hx < bb_x_min or sim_hx > bb_x_max or sim_hy < bb_y_min)

            if math.hypot(sim_tx - sim_hx, sim_ty - sim_hy) < 30:
                hits += 1
                break
            if (sim_tx >= sim_tx_max or sim_tx <= sim_tx_min) or step_count > 150 or hunter_lost:
                break
    return 100.0 * hits / n_episodes

def act_qtable(tx, ty, hx, hy, d):
    s = get_state(tx, ty, hx, hy, d)
    return int(np.argmax(Q_TABLE[s[0], s[1], s[2]]))

def act_dqn(tx, ty, hx, hy, d):
    with torch.no_grad():
        return int(policy_net(get_state_tensor(tx, ty, hx, hy, d)).argmax().item())

def act_cnn(tx, ty, hx, hy, d):
    with torch.no_grad():
        return int(cnn_policy_net(render_state_grid(tx, ty, hx, hy, d, board_bounds)).argmax().item())

def evaluate_pcontrol(Kp, Ki, n_episodes=EVAL_EPISODES):
    """Same simulator, but for the continuous classical controllers.
    Commanded degrees are converted to pixels through the measured mapping
    (MAX_HUNTER_SPEED degrees <-> actual_h_speed pixels). Pass Ki=0 for pure P.
    Lets you tune PI_Ki in seconds instead of by 45-minute hardware runs.
    CAVEAT: the simulator has no board edges for the hunter, so it cannot
    punish an overshoot that would lose laser detection in reality."""
    hits = 0
    lim = (MAX_HUNTER_SPEED / Ki) if Ki > 0 else 0.0
    for _ in range(n_episodes):
        t_dir = random.choice([-1, 1])
        sim_tx = sim_tx_min if t_dir == 1 else sim_tx_max
        sim_hx = center_x + random.randint(-20, 20)
        sim_hy = home_y + random.randint(-20, 20)
        ix = iy = 0.0
        step_count = 0
        while True:
            step_count += 1
            ex = sim_tx - sim_hx
            ey = sim_ty - sim_hy
            if Ki > 0:
                ix = max(-lim, min(lim, ix + ex))
                iy = max(-lim, min(lim, iy + ey))
            pan = max(-MAX_HUNTER_SPEED, min(MAX_HUNTER_SPEED, ex * Kp + ix * Ki))
            tilt = max(-MAX_HUNTER_SPEED, min(MAX_HUNTER_SPEED, ey * Kp + iy * Ki))
            # degrees -> pixels via the measured servo speed
            sim_hx += (pan / MAX_HUNTER_SPEED) * actual_h_speed
            sim_hy += (tilt / MAX_HUNTER_SPEED) * actual_h_speed
            sim_tx += V_SPEED * t_dir

            sim_hy = min(sim_hy, home_y + SIM_FLOOR_PX)
            hunter_lost = (sim_hx < bb_x_min or sim_hx > bb_x_max or sim_hy < bb_y_min)

            if math.hypot(sim_tx - sim_hx, sim_ty - sim_hy) < 30:
                hits += 1
                break
            if (sim_tx >= sim_tx_max or sim_tx <= sim_tx_min) or step_count > 150 or hunter_lost:
                break
    return 100.0 * hits / n_episodes

def log_eval(algorithm, train_rate, greedy_rate, n_seeds=1, spread=""):
    print(f"[{algorithm}] sim catch rate (training, exploring): {train_rate:.1f}%")
    print(f"[{algorithm}] sim catch rate (GREEDY, {EVAL_EPISODES} eps): {greedy_rate:.1f}%")
    if n_seeds > 1:
        print(f"[{algorithm}] {n_seeds} seeds, greedy spread: {spread}  (deployed = median)")
    with open(eval_csv_filename, mode='a', newline='') as f:
        csv.writer(f).writerow([algorithm, round(train_rate, 1), round(greedy_rate, 1),
                                n_seeds, spread])

# ==========================================
# 2. HARDWARE CONTROL MODULE
# ==========================================
SERIAL_PORT = '/dev/cu.usbmodem14101' 
# 9600 cost 14.6 ms per command -- 23% of a frame at 17 fps -- for no reason.
# 115200 cuts that to 1.2 ms. THE ARDUINO SKETCH MUST BE RE-UPLOADED with a
# matching Serial.begin(115200) or the link will produce garbage.
BAUD_RATE = 115200

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print("Connected to Arduino!")
except:
    print("ERROR: Could not connect to Arduino. Check your port!")
    exit()

def send_angles(h_pan, h_tilt, t_pan, t_tilt):
    safe_hp = max(0, min(180, int(h_pan)))
    safe_ht = max(0, min(180, int(h_tilt)))
    safe_tp = max(0, min(180, int(t_pan)))
    safe_tt = max(0, min(180, int(t_tilt)))
    command = f"{safe_hp},{safe_ht},{safe_tp},{safe_tt}\n"
    arduino.write(command.encode('utf-8'))

# ==========================================
# 3. COMPUTER VISION MODULE
# ==========================================
def find_board_boundaries(hsv_frame, display_frame):
    lower_orange = np.array([6, 229, 219])
    upper_orange = np.array([58, 254, 255])
    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    orange_mask = cv2.dilate(orange_mask, None, iterations=2)
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + (w // 2), y + (h // 2)
            corners.append((cx, cy))
            cv2.circle(display_frame, (cx, cy), 5, (0, 165, 255), -1)
            
    if len(corners) == 4:
        corners = sorted(corners, key=lambda pt: pt[1])
        top_corners = sorted(corners[:2], key=lambda pt: pt[0])
        bottom_corners = sorted(corners[2:], key=lambda pt: pt[0])
        tl, tr = top_corners[0], top_corners[1]
        bl, br = bottom_corners[0], bottom_corners[1]
        
        board_pts = np.array([tl, tr, br, bl], np.int32)
        cv2.polylines(display_frame, [board_pts], True, (255, 255, 255), 2)
        return {'tl': tl, 'tr': tr, 'bl': bl, 'br': br}, len(corners)
    return None, len(corners)

# ---------- ROI TRACKING ----------
# The laser moves at most ~21 px per frame, so searching the whole 1280x720
# frame (920k pixels) to find one ~10 px dot is wasteful. Once the dot has been
# seen, only a window around its last position is converted to HSV and searched
# (~32k pixels, about 3% of the work). Any failure falls back to a full-frame
# reacquisition, so a dropout costs one slow frame rather than losing the dot.
ROI_HALF = 90                 # half-width of the search window, px
track_x, track_y = 0, 0
track_ok = False
roi_hits, roi_misses = 0, 0   # diagnostics, reported at shutdown


def _scan_red(frame_bgr, display_frame, corners, x0, y0, x1, y1):
    """Threshold for the red dot inside one rectangle. Returns frame coords."""
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False, 0, 0

    sub = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 150, 150]), np.array([5, 255, 255])
    lower_red2, upper_red2 = np.array([170, 150, 150]), np.array([180, 255, 255])
    mask = (cv2.inRange(hsv, lower_red1, upper_red1)
            + cv2.inRange(hsv, lower_red2, upper_red2))

    # blank the orange corner markers, but only those near this window
    for (cx, cy) in corners:
        px, py = cx - x0, cy - y0
        if -30 <= px <= (x1 - x0) + 30 and -30 <= py <= (y1 - y0) + 30:
            cv2.circle(mask, (px, py), 30, 0, -1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            bx, by, bw, bh = cv2.boundingRect(c)
            hx, hy = x0 + bx + bw // 2, y0 + by + bh // 2
            cv2.circle(display_frame, (hx, hy), 10, (255, 0, 0), 2)
            return True, hx, hy
    return False, 0, 0


def find_hunter(frame_bgr, display_frame, board):
    """ROI-first laser detection with full-frame fallback. The frame has already
    been masked to the board, so nothing outside it can register."""
    global track_x, track_y, track_ok, roi_hits, roi_misses
    corners = [board['tl'], board['tr'], board['bl'], board['br']]

    if track_ok:
        ok, hx, hy = _scan_red(frame_bgr, display_frame, corners,
                               track_x - ROI_HALF, track_y - ROI_HALF,
                               track_x + ROI_HALF, track_y + ROI_HALF)
        if ok:
            roi_hits += 1
            track_x, track_y = hx, hy
            cv2.rectangle(display_frame,
                          (track_x - ROI_HALF, track_y - ROI_HALF),
                          (track_x + ROI_HALF, track_y + ROI_HALF),
                          (60, 60, 60), 1)
            return True, hx, hy

    roi_misses += 1                      # reacquire over the whole frame
    h, w = frame_bgr.shape[:2]
    ok, hx, hy = _scan_red(frame_bgr, display_frame, corners, 0, 0, w, h)
    track_ok = ok
    if ok:
        track_x, track_y = hx, hy
    return ok, hx, hy


def find_hunter_laser(hsv_frame, display_frame, board):
    lower_red1, upper_red1 = np.array([0, 150, 150]), np.array([5, 255, 255])
    lower_red2, upper_red2 = np.array([170, 150, 150]), np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red1, upper_red1) + cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    for point in [board['tl'], board['tr'], board['bl'], board['br']]:
        cv2.circle(red_mask, point, 30, 0, -1)
        
    board_mask = np.zeros_like(red_mask)
    board_poly = np.array([board['tl'], board['tr'], board['br'], board['bl']], np.int32)
    cv2.fillPoly(board_mask, [board_poly], 255)
    valid_zone_mask = cv2.bitwise_and(red_mask, board_mask)
    
    hx, hy = 0, 0
    h_found = False
    
    h_contours, _ = cv2.findContours(valid_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if h_contours:
        largest_h = max(h_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_h) > 10:
            x, y, w, h = cv2.boundingRect(largest_h)
            hx, hy = x + (w//2), y + (h//2)
            h_found = True
            cv2.circle(display_frame, (hx, hy), 10, (255, 0, 0), 2)

    return h_found, hx, hy

def calculate_hunter_step(goal_x, goal_y, hx, hy, Kp, max_speed):
    error_x = goal_x - hx
    error_y = goal_y - hy
    pan_jump = max(min(-error_x * Kp, max_speed), -max_speed)
    tilt_jump = max(min(-error_y * Kp, max_speed), -max_speed)
    return pan_jump, tilt_jump

# ---------- PI CONTROLLER (5th algorithm) ----------
# CLASSIC is pure proportional, which leaves a finite steady-state error when
# tracking a constantly-moving target -- the 40-50 px lag in the results. An
# integral term drives that ramp error to zero.
#
# ANTI-WINDUP MATTERS HERE. With Kp=0.02 and a ~400 px error the proportional
# term alone asks for 8 deg/frame against a 2.0 deg cap, so the controller is
# saturated for most of the chase. An unguarded integrator would accumulate
# thousands of px-frames during that time and fling the gimbal off the board
# when the error finally shrinks -- and off the board means h_found goes False
# and control stops entirely. The integral is therefore clamped so that the I
# term on its own can never exceed the speed cap.

PI_Kp = 0.02        # identical to CLASSIC, so the I term is the ONLY difference
# Chosen from a simulator sweep, not guessed. There is a sharp threshold at
# Ki ~= 0.0018: below it the catch rate is 0%, at 0.002+ it is 100%. Overshoot
# above the target line actually DROPS as Ki rises (a stronger integrator
# closes earlier in the sweep): ~67 px at Ki=0.002, ~61 px at 0.003, ~18 px at
# 0.01. There is roughly 110 px of board above the target line before the laser
# leaves the detection polygon, so 0.003 sits clear of the cliff with margin on
# both sides. Re-check with evaluate_pcontrol() if the board or camera moves.
PI_Ki = 0.003

# ---------- SIMULATOR BOUNDARY (sim-to-real fix) ----------
# The simulator used to let the hunter drift anywhere, so overshooting the
# board cost nothing. On the real rig, leaving the board puts the laser
# outside the detection polygon: h_found goes False, the whole control block
# is skipped, the gimbal freezes and the sweep is lost. That is why the DQN
# scored 100% in sim and 1/3 on hardware.
#   - top / left / right: leaving = detection lost = terminate as an escape
#   - bottom: the real "edge firewall" CLAMPS tilt at home - 2 deg, so the
#     simulator clamps rather than terminates there (2 deg ~ 20 px)
SIM_FLOOR_PX = 20.0

# ---------- RL -> PI TERMINAL HANDOFF (hybrid arms) ----------
# The learned policies have 8 discrete actions at a FIXED full speed, so near
# the target they can only bang back and forth -- their accuracy floor is about
# half a step (~10 px). PI is continuous and can taper. Each learned arm
# therefore runs twice: 3 sweeps pure, then 3 sweeps with PI taking over inside
# HANDOFF_PX. Simulation says closest approach improves from ~5 px mean
# (worst ~10) to ~2 px (worst ~2); the handoff radius itself barely matters.
# Must stay well above the 30 px catch radius or the handoff never happens.
HANDOFF_PX = 100.0

# ---------- RUN TAG ----------
# Every output is prefixed with this, so a new run never overwrites the old one.
RUN_TAG = "v2_run1"

# ---------- STAGED GUIDANCE (scenarios A and B) ----------
# Parameters chosen by search_phases.py in a simulator calibrated against this
# rig (lag 2 frames, detection noise 1.0 px, which reproduces the measured
# CLASSIC/discrete/PI closest approaches to within 0.6 px).
#
# The phases, outermost first:
#   Q-TABLE   -- learned policy closes the gap
#   BANG-BANG -- full speed, but heading computed continuously instead of being
#                snapped to 8 directions; aims LEAD frames ahead to cancel lag
#   PI        -- integral term kills the residual tracking error (scenario A only)
#   HOLD      -- "stay on top": matches the target's OWN velocity as feedforward
#                and uses feedback only to trim. An error-driven controller has
#                to let error build before it reacts, which is what makes the
#                laser flicker on and off the target.
#
# Simulated dwell within 15 px: discrete alone 2.4 frames, Q-TABLE+PI 12.4,
# scenario B 13.2, scenario A 18.5.
A_BB_PX, A_PI_PX, A_HOLD_PX, A_KH = 100.0, 60.0, 22.0, 0.30
B_BB_PX, B_HOLD_PX, B_KH = 160.0, 22.0, 0.15
LEAD_FRAMES = 2.0          # retune if the loop rate moves far from ~16 fps
ON_TOP_PX = 15.0           # "on target" radius used for the dwell metric

# ---------- VIDEO PLAYBACK SPEED ----------
# Playback speed = VIDEO_FPS / (actual loop rate). The loop measures ~18 fps on
# this machine (1280x720 HSV + contours on a dual-core i5), so the old
# hard-coded 30 made every recording play 1.66x too fast. 9 gives half speed.
# The true loop rate is printed at shutdown -- set this to (that / 2).
VIDEO_FPS = 9.0

# Seconds each "PHASE COMPLETE" title card blocks for. 0 = no wall-clock wait;
# a few frames of the card are still written so the video keeps its markers.
PAUSE_SECONDS = 0.0
in_terminal = False
in_stage = None
# Arm order for this run: CLASSIC (also the system-ID phase) -> Q-TABLE ->
# Q-TABLE+PI -> scenario A -> scenario B -> DQN. CNN and the CNN/DQN +PI arms
# are skipped: this run is about comparing A and B against the two previous
# top performers, and dropping the CNN saves its whole training phase.
# Scenario B only (the PI stage of scenario A hurt on hardware: dwell 3.4
# frames vs B's 7.2, and PI sits one frame of lag away from instability on this
# rig). This run sweeps the HOLD feedback gain kh, which was picked by a
# simulator search that got the A-vs-B ranking backwards -- so it is tuned on
# hardware instead.
#
# DRIFT CONTROL: kh=0.15 runs FIRST and again LAST. The rig changes between
# sessions (Q-TABLE closest approach moved 9.4 -> 15.0 px across two runs with
# identical code), so four settings run back-to-back would confound kh with
# drift. If the two 0.15 arms disagree, the sweep is not trustworthy.
# (r_bb, kh) per arm. kh=0.40 won the last sweep at r_bb=160; both knobs had
# their optimum at the EDGE of what was tested, so this run brackets them.
#   kh 0.50 / 0.60 -- the simulator predicts a COLLAPSE past ~0.5 from lag
#                     instability. A falsifiable prediction: if 0.60 holds up,
#                     the lag model is wrong about something.
#   rbb220         -- simulator says 16.8 frames of dwell vs 11.1 at 160.
#   bbonly         -- bang-bang runs the whole approach, the learned policy
#                     never acts. Answers "does Q-TABLE contribute at all?"
#   kh0.40r        -- repeat of the first arm, drift control.
# FOUR arms instead of eight, 30 sweeps each instead of 10. At 10 sweeps the
# dwell sd was as large as its mean, so nothing separated (every p > 0.27) and
# kh 0.50 / r_bb 220 showed no effect worth chasing. Sweeps are cheap -- only
# training costs real time -- so the budget is better spent on n.
#   kh0.40   -- current best
#   kh0.60   -- tests the simulator's predicted lag-instability collapse
#   bbonly   -- does the Q-TABLE phase contribute? Two runs have DISAGREED on
#               this (worse at p=0.043, then best on min distance), so it needs
#               the extra n more than anything else here.
#   kh0.40r  -- drift control
ARM_CFG = {"B-kh0.40":  (160.0, 0.40),
           "B-kh0.60":  (160.0, 0.60),
           "B-bbonly":  (9999.0, 0.40),
           "B-kh0.40r": (160.0, 0.40)}

ARM_NEXT = {"Q-TABLE":  "B-kh0.40",
            "B-kh0.40": "B-kh0.60",
            "B-kh0.60": "B-bbonly",
            "B-bbonly": "B-kh0.40r"}
ARM_IMG = {"CLASSIC":   f"{RUN_TAG}_1_Classic.jpg",
           "Q-TABLE":   f"{RUN_TAG}_2_QTable.jpg",
           "B-kh0.40":  f"{RUN_TAG}_3_kh040.jpg",
           "B-kh0.60":  f"{RUN_TAG}_4_kh060.jpg",
           "B-bbonly":  f"{RUN_TAG}_5_bbonly.jpg",
           "B-kh0.40r": f"{RUN_TAG}_6_kh040_repeat.jpg"}
bb_x_min = bb_x_max = bb_y_min = bb_y_max = 0

# ---------- MULTI-SEED TRAINING ----------
# The CNN scored 100%, 100%, then 45.5% greedy across three runs with
# identical hyperparameters, because nothing was seeded. A single run is not
# evidence. Each learned method now trains N_SEEDS times; the MEDIAN policy is
# deployed to hardware (not the best -- that would be cherry-picking) and the
# full spread is recorded. Cost: roughly N_SEEDS x the training time.
N_SEEDS = 3
qt_seed_idx = dqn_seed_idx = cnn_seed_idx = 0
qt_rates, dqn_rates, cnn_rates = [], [], []
qt_trains, dqn_trains, cnn_trains = [], [], []
qt_tables, dqn_weights, cnn_weights = [], [], []

def pick_median(rates):
    """Index of the median-performing seed."""
    return sorted(range(len(rates)), key=lambda i: rates[i])[len(rates) // 2]

pi_int_x, pi_int_y = 0.0, 0.0

def bang_bang_step(goal_x, goal_y, hx, hy, t_dir, max_speed, lead=LEAD_FRAMES):
    """Full-speed propulsion, continuously steered. Unlike the 8-action learned
    policies this can point ANYWHERE, and it aims `lead` frames ahead of the
    target so the command that finally lands is pointed where the target will
    be rather than where it was."""
    ex = (goal_x + V_SPEED * t_dir * lead) - hx
    ey = goal_y - hy
    n = math.hypot(ex, ey)
    if n < 1e-6:
        return 0.0, 0.0
    return -(ex / n) * max_speed, -(ey / n) * max_speed


def hold_step(goal_x, goal_y, hx, hy, t_dir, kh, max_speed, px_per_step):
    """Stay on top of the target. Feedforward = the target's own velocity, so
    there is no steady-state error to chase; feedback only trims the residual."""
    vx = V_SPEED * t_dir + kh * (goal_x - hx)      # px per frame
    vy = kh * (goal_y - hy)
    # px/frame -> degrees/frame using the measured servo scale
    scale = max_speed / max(1e-6, px_per_step)
    pan, tilt = -vx * scale, -vy * scale
    n = math.hypot(pan, tilt)
    if n > max_speed:
        pan, tilt = pan * max_speed / n, tilt * max_speed / n
    return pan, tilt


def reset_pi_integral():
    global pi_int_x, pi_int_y
    pi_int_x, pi_int_y = 0.0, 0.0

def calculate_hunter_step_pi(goal_x, goal_y, hx, hy, Kp, Ki, max_speed):
    global pi_int_x, pi_int_y
    error_x = goal_x - hx
    error_y = goal_y - hy

    pi_int_x += error_x
    pi_int_y += error_y

    lim = (max_speed / Ki) if Ki > 0 else 0.0      # anti-windup clamp
    pi_int_x = max(-lim, min(lim, pi_int_x))
    pi_int_y = max(-lim, min(lim, pi_int_y))

    pan_jump = -(error_x * Kp + pi_int_x * Ki)
    tilt_jump = -(error_y * Kp + pi_int_y * Ki)

    pan_jump = max(min(pan_jump, max_speed), -max_speed)
    tilt_jump = max(min(tilt_jump, max_speed), -max_speed)
    return pan_jump, tilt_jump

# HELPER: Safe UI Pause
def pause_screen(bg_frame, text1, text2, seconds=None):
    secs = PAUSE_SECONDS if seconds is None else seconds
    if secs <= 0:
        # No wall-clock wait, but still stamp a few frames into the video so the
        # phase boundary stays visible when reviewing the recording.
        disp = bg_frame.copy()
        cv2.putText(disp, text1, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(disp, text2, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for _ in range(6):
            if is_recording: video_out.write(disp)
        cv2.imshow("Robotics Benchmark", disp)
        cv2.waitKey(1)
        print(f"--- {text1} ---")
        return
    start_t = time.time()
    while time.time() - start_t < secs:
        disp = bg_frame.copy()
        cv2.putText(disp, text1, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(disp, text2, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Robotics Benchmark", disp)
        if is_recording: video_out.write(disp)
        cv2.waitKey(33)

# ==========================================
# 4. MASTER EXPERIMENT CONFIGURATION
# ==========================================
MAX_HUNTER_SPEED = 2.0 
Kp = 0.02
# THE TARGET'S SPEED WAS TIED TO THE FRAME RATE. V_SPEED is pixels per FRAME,
# so when ROI tracking + 115200 baud took the loop from ~17 to ~29 fps, the
# target silently got 75% faster (136 -> 238 px/s) and every arm was measured
# on a harder problem. Classic's miss distance went 39 -> 46 px for that reason
# alone, and the kh/r_bb comparisons became untestable.
#
# Difficulty is now specified in px/SECOND and V_SPEED is derived from the frame
# rate measured during homing, just before the first sweep. Future speed
# optimisations will improve control instead of raising the bar.
TARGET_PX_PER_SEC = 136.0      # matches the original 8 px/frame at 17 fps
V_SPEED = 8.0                  # placeholder; recomputed at the end of homing

target_pan, target_tilt = 90.0, 125.0
hunter_pan, hunter_tilt = 90.0, 170.0
hunter_home_pan, hunter_home_tilt = 90.0, 170.0 

v_tx, v_ty = 0, 0
v_dir = 1 
target_escaped = False
escape_time = 0

current_mode = "CLASSIC" 
sweep_count = 0
# 3 sweeps could not resolve the differences that matter now: every arm except
# CLASSIC catches, so the question is the DISTRIBUTION of closest approach, and
# 3 samples cannot separate a 6.3 px mean from an 8.3 px one. Sweeps cost no
# training time -- 10 per arm adds only ~4-5 minutes across all 8 arms.
# Must be EVEN so each direction (L-to-R and R-to-L) gets the same count:
# sweeps alternate direction, and the board is trapezoidal in frame, so an odd
# count would bias the mean toward whichever direction ran one extra time.
max_sweeps = 30              # default for the arms under test
# CLASSIC and Q-TABLE are context, not the experiment -- CLASSIC only needs
# enough sweeps to measure the servo speed. Keeping them short buys ~4 minutes
# for the arms that actually need the n. Counts must stay EVEN so each
# direction gets the same number (sweeps alternate L-to-R / R-to-L).
SWEEPS_OF = {"CLASSIC": 10, "Q-TABLE": 10}

hunter_pixel_speeds = [] 

trail_canvas = None
draw_hx, draw_hy, draw_tx, draw_ty = 0, 0, 0, 0
last_hx, last_hy = 0, 0 
last_event_text = "SWEEPING..."

board_data = None
board_bounds = None          # NEW: bounding box used by the CNN renderer
results_log = {}             # arm -> [(min_dist, time_to_catch or None, hit)]
total_sweep_time = 0.0       # wall seconds spent in live sweeps
last_good_board = None       # latched calibration, so the polygon stops flickering
last_good_time = 0.0
board_bgmask = None          # built once when the board is locked
home_t0, home_frames = None, 0   # used to size V_SPEED from the real frame rate
current_state = 0 
is_recording = False 

csv_filename = f"{RUN_TAG}_Results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # "Final Error (px)" keeps its old meaning (0 on a catch) so this file stays
    # comparable with earlier runs. "Min Dist (px)" is the new accuracy metric:
    # the closest the hunter actually got, which is what separates the discrete
    # learned policies from the continuous PI terminal stage.
    writer.writerow(["Algorithm", "Sweep", "Direction", "Final Error (px)",
                     "Duration (s)", "Result", "Min Dist (px)", "Time to Catch (s)",
                     "Dwell (frames)", "Dwell (s)", "Post-hit Err (px)"])

# Kept separate from the physical results above on purpose: these are
# simulator numbers, not measurements of the real gimbal.
eval_csv_filename = f"{RUN_TAG}_SimEval.csv"
with open(eval_csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Algorithm", "Train Catch Rate (%, exploring)",
                     "Greedy Catch Rate (%)", "N Seeds", "Greedy Spread (min-max)"])

# Built-in FaceTime camera. NOTE: this index is assigned in device-enumeration
# order, so connecting/disconnecting an iPhone (Continuity Camera) can move it.
# If the script exits immediately, re-run the enumeration snippet to re-check.
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"ERROR: could not open camera index {CAMERA_INDEX}.")
    arduino.close()
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_out = cv2.VideoWriter(f'Master_{RUN_TAG}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (frame_width, frame_height))

loop_t0 = time.time()
frames_seen = 0
loop_fps = 16.3          # updated live once enough frames have been seen
sweep_frames = 0         # frames spent in live sweeps only
sweep_seconds = 0.0      # wall time of those sweeps

print("STATE 0: Adjust Camera, press SPACEBAR.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: camera index {CAMERA_INDEX} returned no frame. Wrong index, "
              "or the device was disconnected mid-run.")
        break
    
    if current_state >= 2 and board_bgmask is not None:
        frame = cv2.bitwise_and(frame, board_bgmask)   # mask built once at lock
        
    # Snapshot BEFORE any HUD/markers are drawn, so saved figures stay clean.
    # NOTE: only the IMAGE SAVES use this; the displayed/recorded frame keeps
    # its overlays.
    clean_frame = frame.copy()

    # Full-frame HSV is only needed by the corner detector during calibration.
    # From state 2 on, find_hunter() converts just its ROI.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if current_state <= 1 else None
    if trail_canvas is None: trail_canvas = np.zeros_like(frame)
    
    if current_state == 0:
        cv2.putText(frame, "ADJUST CAMERA. PRESS SPACEBAR.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if cv2.waitKey(1) == 32: 
            is_recording = True 
            current_state = 1
            print("STATE 1: calibrating - waiting for 4 orange corners, then SPACEBAR")
            time.sleep(0.5) 
            
    elif current_state == 1:
        # The detector only returns a board when EXACTLY 4 orange blobs are
        # visible. Detection wobbles between 3/4/5 blobs, so the polygon used
        # to flicker and the spacebar only worked during a good frame. Now the
        # last good detection is LATCHED: the polygon stays on screen and the
        # spacebar accepts the latched board whenever you press it.
        found, n_corners = find_board_boundaries(hsv, frame)
        if found is not None:
            last_good_board = found
            last_good_time = time.time()

        if last_good_board is not None:
            poly = np.array([last_good_board['tl'], last_good_board['tr'],
                             last_good_board['br'], last_good_board['bl']], np.int32)
            cv2.polylines(frame, [poly], True, (255, 255, 255), 2)
            for k in ('tl', 'tr', 'bl', 'br'):
                cv2.circle(frame, last_good_board[k], 7, (255, 255, 255), 2)
            age = time.time() - last_good_time
            cv2.putText(frame, f"4 CORNERS LATCHED ({age:.1f}s ago) - PRESS SPACEBAR",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"SEEKING 4 ORANGE CORNERS - seeing {n_corners}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, f"blobs detected now: {n_corners}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if cv2.waitKey(1) == 32 and last_good_board is not None:
            board_data = last_good_board
            board_bounds = get_board_bounds(board_data)
            board_bgmask = np.zeros_like(frame)
            cv2.fillPoly(board_bgmask,
                         [np.array([board_data['tl'], board_data['tr'],
                                    board_data['br'], board_data['bl']], np.int32)],
                         (255, 255, 255))
            print(f"STATE 2: board locked {board_data} -> homing hunter")
            current_state = 2
            time.sleep(0.5) 
            
    elif current_state > 1 and board_data is not None:
        h_found, hx, hy = find_hunter(frame, frame, board_data)
        
        center_x = int((board_data['bl'][0] + board_data['br'][0]) / 2)
        home_y = int((board_data['bl'][1] + board_data['br'][1]) / 2) - 40
        
        # --- STATE 2: VISUAL HOMING ---
        if current_state == 2:
            # Homing STARTS with the laser on the ceiling, off the board and
            # therefore invisible, so every early frame costs a full-frame
            # reacquisition and runs ~30% slow. Timing only starts once the
            # laser has actually been found, which is the regime a sweep runs in.
            if h_found:
                if home_t0 is None:
                    home_t0, home_frames = time.time(), 0
                home_frames += 1
            cv2.putText(frame, f"HOMING HUNTER FOR {current_mode}...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            # If these numbers CHANGE but the gimbal does not move, the problem
            # is hardware (wiring/power/pins). If they are frozen, it is software.
            cv2.putText(frame, f"cmd pan={hunter_pan:6.1f}  tilt={hunter_tilt:6.1f}   laser found: {h_found}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(frame, (center_x, home_y), 8, (0, 255, 255), -1)

            if h_found:
                dist_to_home = np.sqrt((center_x - hx)**2 + (home_y - hy)**2)
                if dist_to_home < 25: 
                    # Homing uses the same code path as a sweep (ROI detection,
                    # no full-frame HSV), so its frame rate is representative.
                    _hf = time.time() - home_t0 if home_t0 else 0.0
                    if home_frames >= 20 and _hf > 0.5:
                        _fps_home = home_frames / _hf
                        V_SPEED = max(2.0, min(12.0, TARGET_PX_PER_SEC / _fps_home))
                        print(f"Measured {_fps_home:.1f} fps during homing -> "
                              f"V_SPEED = {V_SPEED:.2f} px/frame "
                              f"({TARGET_PX_PER_SEC:.0f} px/s)")
                    else:
                        print(f"Homing too short to measure fps; keeping "
                              f"V_SPEED = {V_SPEED:.2f}")
                    hunter_home_pan = hunter_pan
                    hunter_home_tilt = hunter_tilt
                    
                    v_dir = 1
                    v_tx = board_data['tl'][0] + 30
                    v_ty = int(board_data['tl'][1] + (board_data['bl'][1] - board_data['tl'][1]) * 0.25)
                    cv2.waitKey(2000)
                    sweep_start_time = time.time()
                    sweep_min_dist = 1e9
                    sweep_hit = False
                    sweep_hit_time = 0.0
                    sweep_dwell = 0
                    sweep_dwell_s = 0.0
                    track_err_sum = 0.0
                    track_err_n = 0
                    last_frame_t = 0.0
                    current_state = 3
                else:
                    pan_jump, tilt_jump = calculate_hunter_step(center_x, home_y, hx, hy, Kp, MAX_HUNTER_SPEED)
                    hunter_pan += pan_jump
                    hunter_tilt += tilt_jump
            else: 
                hunter_tilt -= 0.5 
                
            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)

        # --- STATE 3: THE ACTIVE CHASE (4-WAY TESTING) ---
        elif current_state == 3:
            
            if not target_escaped:
                hud = f"{current_mode} SWEEP {sweep_count + 1}"
                if sweep_hit: hud += f"  [HIT @ {sweep_hit_time:.2f}s]"
                cv2.putText(frame, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                v_tx += V_SPEED * v_dir        # float now: V_SPEED is not integral
                v_tx_i = int(round(v_tx))
                cv2.circle(frame, (v_tx_i, v_ty), 15, (0, 255, 0), -1)
                
                if h_found:
                    if current_mode == "CLASSIC" and last_hx != 0 and last_hy != 0:
                        speed_this_frame = math.hypot(hx - last_hx, hy - last_hy)
                        if speed_this_frame > 1.0: 
                            hunter_pixel_speeds.append(speed_this_frame)
                    
                    last_hx, last_hy = hx, hy 
                    
                    # Hybrid arms ("<MODE>+PI") hand control to PI inside
                    # HANDOFF_PX; outside it the learned policy drives.
                    base_mode = current_mode[:-3] if current_mode.endswith("+PI") else current_mode
                    terminal_pi = (current_mode.endswith("+PI")
                                   and math.hypot(v_tx - hx, v_ty - hy) < HANDOFF_PX)
                    if terminal_pi and not in_terminal:
                        reset_pi_integral()      # fresh integral on each entry
                        in_terminal = True
                    elif not terminal_pi:
                        in_terminal = False

                    if current_mode in ARM_CFG:
                        d_now = math.hypot(v_tx - hx, v_ty - hy)
                        r_bb, kh = ARM_CFG[current_mode]
                        r_pi, r_hold = -1.0, B_HOLD_PX

                        if d_now > r_bb:
                            stage = "QT"
                        elif d_now > max(r_pi, r_hold):
                            stage = "BB"
                        elif d_now > r_hold:
                            stage = "PI"
                        else:
                            stage = "HOLD"

                        if stage != in_stage:
                            if stage == "PI": reset_pi_integral()
                            in_stage = stage

                        if stage == "QT":
                            st_ = get_state(v_tx, v_ty, hx, hy, v_dir)
                            mx, my = ACTION_MAP[int(np.argmax(Q_TABLE[st_[0], st_[1], st_[2]]))]
                            pan_jump = -mx * MAX_HUNTER_SPEED
                            tilt_jump = -my * MAX_HUNTER_SPEED
                            col = (255, 255, 0)
                        elif stage == "BB":
                            pan_jump, tilt_jump = bang_bang_step(
                                v_tx, v_ty, hx, hy, v_dir, MAX_HUNTER_SPEED)
                            col = (0, 165, 255)
                        elif stage == "PI":
                            pan_jump, tilt_jump = calculate_hunter_step_pi(
                                v_tx, v_ty, hx, hy, PI_Kp, PI_Ki, MAX_HUNTER_SPEED)
                            col = (255, 0, 255)
                        else:
                            pan_jump, tilt_jump = hold_step(
                                v_tx, v_ty, hx, hy, v_dir, kh,
                                MAX_HUNTER_SPEED, actual_h_speed)
                            col = (0, 255, 0)

                        hunter_pan += pan_jump
                        hunter_tilt += tilt_jump
                        cv2.line(frame, (hx, hy), (v_tx_i, v_ty), col, 2)
                        cv2.putText(frame, f"stage: {stage}", (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                    elif terminal_pi:
                        cv2.line(frame, (hx, hy), (v_tx_i, v_ty), (255, 0, 255), 2)
                        pan_jump, tilt_jump = calculate_hunter_step_pi(
                            v_tx, v_ty, hx, hy, PI_Kp, PI_Ki, MAX_HUNTER_SPEED)
                        hunter_pan += pan_jump
                        hunter_tilt += tilt_jump

                    elif base_mode == "CLASSIC":
                        cv2.line(frame, (hx, hy), (v_tx_i, v_ty), (0, 255, 0), 2)
                        pan_jump, tilt_jump = calculate_hunter_step(v_tx, v_ty, hx, hy, Kp, MAX_HUNTER_SPEED)
                        hunter_pan += pan_jump
                        hunter_tilt += tilt_jump
                        
                    elif base_mode == "Q-TABLE":
                        state = get_state(v_tx, v_ty, hx, hy, v_dir)
                        best_action = np.argmax(Q_TABLE[state[0], state[1], state[2]])
                        move_x, move_y = ACTION_MAP[best_action]
                        hunter_pan += (-move_x * MAX_HUNTER_SPEED)
                        hunter_tilt += (-move_y * MAX_HUNTER_SPEED)
                        
                    elif base_mode == "DQN":
                        state_tensor = get_state_tensor(v_tx, v_ty, hx, hy, v_dir)
                        with torch.no_grad():
                            q_values = policy_net(state_tensor)
                            best_action = q_values.argmax().item()
                        move_x, move_y = ACTION_MAP[best_action]
                        hunter_pan += (-move_x * MAX_HUNTER_SPEED)
                        hunter_tilt += (-move_y * MAX_HUNTER_SPEED)

                    elif base_mode == "PI":                        # 5th ALGORITHM
                        cv2.line(frame, (hx, hy), (v_tx_i, v_ty), (255, 0, 255), 2)
                        pan_jump, tilt_jump = calculate_hunter_step_pi(
                            v_tx, v_ty, hx, hy, PI_Kp, PI_Ki, MAX_HUNTER_SPEED)
                        hunter_pan += pan_jump
                        hunter_tilt += tilt_jump

                    elif base_mode == "CNN":                       # NEW
                        grid_tensor = render_state_grid(v_tx, v_ty, hx, hy, v_dir, board_bounds)
                        with torch.no_grad():
                            best_action = cnn_policy_net(grid_tensor).argmax().item()
                        move_x, move_y = ACTION_MAP[best_action]
                        hunter_pan += (-move_x * MAX_HUNTER_SPEED)
                        hunter_tilt += (-move_y * MAX_HUNTER_SPEED)
                        draw_cnn_view(frame, grid_tensor)
                        
                    # THE EDGE FIREWALL
                    hunter_tilt = max(hunter_tilt, hunter_home_tilt - 2.0)
                    
                    if draw_hx != 0 and draw_hy != 0:
                        cv2.line(trail_canvas, (draw_hx, draw_hy), (hx, hy), (255, 255, 0), 2) 
                    if draw_tx != 0 and draw_ty != 0:
                        cv2.line(trail_canvas, (draw_tx, draw_ty), (v_tx_i, v_ty), (0, 0, 255), 2) 
                    draw_hx, draw_hy, draw_tx, draw_ty = hx, hy, v_tx_i, v_ty
                
                check_hx = hx if h_found else last_hx
                check_hy = hy if h_found else last_hy
                
                distance = math.hypot(v_tx - check_hx, v_ty - check_hy)
                _now = time.time()
                _dt = _now - last_frame_t if last_frame_t else 0.0
                last_frame_t = _now
                if h_found:
                    sweep_min_dist = min(sweep_min_dist, distance)
                    if distance < ON_TOP_PX:
                        sweep_dwell += 1
                        sweep_dwell_s += _dt      # real elapsed time, not
                                                  # frames / an averaged fps
                    # POST-HIT TRACKING ERROR: a continuous measure of "stays on
                    # target". Dwell counts threshold crossings and is nearly
                    # binary per sweep (locked on or not), so its sd came out as
                    # large as its mean. Mean distance after the first hit
                    # measures the same thing with far less variance.
                    if sweep_hit:
                        track_err_sum += distance
                        track_err_n += 1
                duration = time.time() - sweep_start_time
                sweep_seconds = max(sweep_seconds, 0.0)
                
                # A hit no longer ENDS the sweep. Every sweep runs the full
                # traverse, so Min Dist is the TRUE closest approach instead of
                # "the first reading under 30 px" -- which was censoring the
                # metric and hiding what the PI terminal stage actually does.
                # Time to Catch preserves the old time-to-intercept number.
                if distance < 30 and not sweep_hit:
                    sweep_hit = True
                    sweep_hit_time = time.time() - sweep_start_time

                is_escape = (v_dir == 1 and v_tx > board_data['tr'][0] - 30) or (v_dir == -1 and v_tx < board_data['tl'][0] + 30)
                
                if is_escape:
                    final_error = 0 if sweep_hit else int(distance)
                    last_event_text = "CATCH!" if sweep_hit else "ESCAPED!"
                    dir_str = "L to R" if v_dir == 1 else "R to L"
                    print(f"[{current_mode}] Sweep {sweep_count + 1} {last_event_text} Error: {final_error}px, Time: {duration:.2f}s")
                    
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([current_mode, sweep_count + 1, dir_str, final_error,
                                         round(duration, 2), last_event_text.rstrip("!"),
                                         int(sweep_min_dist) if sweep_min_dist < 1e8 else "",
                                         round(sweep_hit_time, 2) if sweep_hit else "",
                                         sweep_dwell, round(sweep_dwell_s, 3),
                                         round(track_err_sum / track_err_n, 1)
                                         if track_err_n else ""])
                        
                    total_sweep_time += duration
                    results_log.setdefault(current_mode, []).append(
                        (sweep_min_dist, sweep_hit_time if sweep_hit else None,
                         sweep_hit, sweep_dwell,
                         track_err_sum / track_err_n if track_err_n else None))

                    target_escaped = True
                    escape_time = time.time()
                    sweep_count += 1
                    draw_hx, draw_hy, draw_tx, draw_ty = 0, 0, 0, 0 
                    
                    # PHASE 4-WAY ROUTING
                    if sweep_count >= SWEEPS_OF.get(current_mode, max_sweeps):
                        if current_mode == "CLASSIC":
                            current_state = 4 
                        elif current_mode in ARM_NEXT:
                            cv2.imwrite(ARM_IMG[current_mode], cv2.add(clean_frame, trail_canvas))
                            current_mode = ARM_NEXT[current_mode]
                            in_stage = None
                            sweep_count = 0
                            target_escaped = False
                            reset_pi_integral()
                            in_terminal = False
                            v_dir = 1
                            v_tx = board_data['tl'][0] + 30
                            hunter_pan, hunter_tilt = hunter_home_pan, hunter_home_tilt
                            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)
                            time.sleep(1.5)
                            sweep_start_time = time.time()
                            sweep_min_dist = 1e9
                            sweep_hit = False
                            sweep_hit_time = 0.0
                            sweep_dwell = 0
                            sweep_dwell_s = 0.0
                            track_err_sum = 0.0
                            track_err_n = 0
                            last_frame_t = 0.0
                            trail_canvas = np.zeros_like(frame)
                        elif current_mode == "B-kh0.40r":
                            cv2.imwrite(ARM_IMG["B-kh0.40r"], cv2.add(clean_frame, trail_canvas))
                            current_state = 8          # finale -- no DQN phase
                                                       # in this run, so its
                                                       # training is skipped
                    
            else:
                text_color = (0, 255, 0) if last_event_text == "CATCH!" else (0, 0, 255)
                cv2.putText(frame, f"{last_event_text} HUNTER RESETTING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.circle(frame, (center_x, home_y), 8, (0, 255, 255), -1) 
                
                h_is_home = False
                if h_found:
                    dist_to_home = np.sqrt((center_x - hx)**2 + (home_y - hy)**2)
                    if dist_to_home < 25: h_is_home = True
                    else:
                        pan_jump, tilt_jump = calculate_hunter_step(center_x, home_y, hx, hy, Kp, MAX_HUNTER_SPEED)
                        hunter_pan += pan_jump
                        hunter_tilt += tilt_jump
                else:
                    # BLIND RECOVERY
                    hunter_pan += (hunter_home_pan - hunter_pan) * 0.05
                    hunter_tilt += (hunter_home_tilt - hunter_tilt) * 0.05
                
                if h_is_home and (time.time() - escape_time > 1.5):
                    target_escaped = False
                    reset_pi_integral()          # stale integral must not carry between sweeps
                    v_dir *= -1 
                    v_tx = (board_data['tl'][0] + 30) if v_dir == 1 else (board_data['tr'][0] - 30)
                    sweep_start_time = time.time()
                    sweep_min_dist = 1e9
                    sweep_hit = False
                    sweep_hit_time = 0.0
                    sweep_dwell = 0
                    sweep_dwell_s = 0.0
                    track_err_sum = 0.0
                    track_err_n = 0
                    last_frame_t = 0.0
                    trail_canvas = np.zeros_like(frame)

            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)

        # --- STATE 4: Q-TABLE TRAINING ---
        elif current_state == 4:
            combined = cv2.add(clean_frame, trail_canvas)
            pass  # arm images are saved by the routing block
            
            if qt_seed_idx == 0: pause_screen(np.zeros_like(frame), "CLASSIC BENCHMARK COMPLETE!", f"Next: Q-Table x{N_SEEDS} seeds...")
            
            actual_h_speed = float(np.mean(hunter_pixel_speeds)) if len(hunter_pixel_speeds) > 0 else 8.0
            print(f"\n--- SYSTEM IDENTIFICATION COMPLETE ---")
            print(f"Measured Physical Speed: {actual_h_speed:.2f} px/frame")
            
            sim_tx_min = board_data['tl'][0] + 30
            sim_tx_max = board_data['tr'][0] - 30
            sim_ty = int(board_data['tl'][1] + (board_data['bl'][1] - board_data['tl'][1]) * 0.25)
            
            bb_x_min, bb_x_max, bb_y_min, bb_y_max = board_bounds

            # Same seed values are used for every algorithm, so all of them face
            # identical episode sequences -- a paired comparison, not independent
            # draws.
            random.seed(1000 + qt_seed_idx)
            np.random.seed(1000 + qt_seed_idx)

            Q_TABLE = np.zeros((15, 15, 2, 8))
            epsilon = 1.0
            catches = 0                      # DIAGNOSTIC
            
            bg_mask = np.zeros_like(frame)
            cv2.putText(bg_mask, "ANALYZING PHYSICAL PHYSICS...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(bg_mask, f"Measured Servo Speed: {actual_h_speed:.2f} px/frame", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for episode in range(15000):
                if episode % 3000 == 0:
                    display_mask = bg_mask.copy()
                    cv2.putText(display_mask, f"TRAINING Q-TABLE: {episode}/15000", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    if is_recording: video_out.write(display_mask)
                    cv2.imshow("Robotics Benchmark", display_mask)
                    cv2.waitKey(1) 
                
                t_dir = random.choice([-1, 1])
                sim_tx = sim_tx_min if t_dir == 1 else sim_tx_max
                
                # SIM2REAL FIX: Domain Randomization
                sim_hx = center_x + random.randint(-20, 20)
                sim_hy = home_y + random.randint(-20, 20)
                
                state = get_state(sim_tx, sim_ty, sim_hx, sim_hy, t_dir)
                done = False
                step_count = 0
                
                while not done:
                    step_count += 1
                    if random.uniform(0, 1) < epsilon: action = random.randint(0, 7)
                    else: action = np.argmax(Q_TABLE[state[0], state[1], state[2]])
                        
                    move_x, move_y = ACTION_MAP[action]
                    sim_hx += move_x * actual_h_speed
                    sim_hy += move_y * actual_h_speed
                    sim_tx += V_SPEED * t_dir
                    
                    distance = math.hypot(sim_tx - sim_hx, sim_ty - sim_hy)
                    is_catch = distance < 30
                    # SIM BOUNDARY: clamp the floor (mirrors the real edge
                    # firewall), terminate if the hunter leaves the board at
                    # top/left/right (real rig would lose laser detection).
                    sim_hy = min(sim_hy, home_y + SIM_FLOOR_PX)
                    hunter_lost = (sim_hx < bb_x_min or sim_hx > bb_x_max
                                   or sim_hy < bb_y_min)
                    is_escape = ((sim_tx >= sim_tx_max or sim_tx <= sim_tx_min)
                                 or step_count > 150 or hunter_lost)
                    
                    reward = -1 
                    if is_catch:
                        reward, done = 1000, True
                    elif is_escape:
                        reward, done = -1000, True
                        
                    if is_catch: catches += 1        # DIAGNOSTIC

                    new_state = get_state(sim_tx, sim_ty, sim_hx, sim_hy, t_dir)
                    best_future = 0.0 if done else np.max(Q_TABLE[new_state[0], new_state[1], new_state[2]])
                    
                    Q_TABLE[state[0], state[1], state[2], action] += 0.1 * (
                        reward + 0.95 * best_future - Q_TABLE[state[0], state[1], state[2], action]
                    )
                    state = new_state
                    
                if epsilon > 0.05: epsilon *= 0.999

            qt_trains.append(100.0 * catches / 15000)
            qt_rates.append(evaluate_policy(act_qtable))
            qt_tables.append(Q_TABLE.copy())
            print(f"[Q-TABLE] seed {qt_seed_idx + 1}/{N_SEEDS} greedy: {qt_rates[-1]:.1f}%")
            qt_seed_idx += 1
            if qt_seed_idx < N_SEEDS:
                continue                     # re-enter state 4 for the next seed

            _m = pick_median(qt_rates)
            Q_TABLE = qt_tables[_m]           # deploy the MEDIAN seed, not the best
            log_eval("Q-TABLE", sum(qt_trains) / len(qt_trains), qt_rates[_m],
                     N_SEEDS, f"{min(qt_rates):.1f}-{max(qt_rates):.1f}")
            pause_screen(np.zeros_like(frame), "Q-TABLE READY!", "Starting Phase 2 Sweeps (6 seconds)...")
            
            current_mode = "Q-TABLE"
            sweep_count = 0
            target_escaped = False
            v_dir = 1 
            v_tx = board_data['tl'][0] + 30
            
            hunter_pan, hunter_tilt = hunter_home_pan, hunter_home_tilt
            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)
            time.sleep(1.5) 
                
            sweep_start_time = time.time()
            sweep_min_dist = 1e9
            sweep_hit = False
            sweep_hit_time = 0.0
            sweep_dwell = 0
            sweep_dwell_s = 0.0
            track_err_sum = 0.0
            track_err_n = 0
            last_frame_t = 0.0
            trail_canvas = np.zeros_like(frame) 
            current_state = 3 

        # --- STATE 5: DQN TRAINING ---
        elif current_state == 5:
            combined = cv2.add(clean_frame, trail_canvas)
            pass  # arm images are saved by the routing block
            
            if dqn_seed_idx == 0: pause_screen(np.zeros_like(frame), "Q-TABLE BENCHMARK COMPLETE!", f"Next: PyTorch DQN x{N_SEEDS} seeds...")
            
            random.seed(1000 + dqn_seed_idx)
            np.random.seed(1000 + dqn_seed_idx)
            torch.manual_seed(1000 + dqn_seed_idx)

            policy_net = DQNBrain()           # fresh init for this seed
            target_net = DQNBrain()
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            optimizer = optim.Adam(policy_net.parameters(), lr=LR)
            memory = ReplayMemory(capacity=10000)

            epsilon = 1.0
            total_steps = 0
            catches = 0                      # DIAGNOSTIC
            
            bg_mask = np.zeros_like(frame)
            cv2.putText(bg_mask, "TRAINING DEEP NEURAL NETWORK...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            for episode in range(2500):
                if episode % 500 == 0:
                    display_mask = bg_mask.copy()
                    cv2.putText(display_mask, f"TRAINING NEURAL NET: {episode}/2500", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    if is_recording: video_out.write(display_mask)
                    cv2.imshow("Robotics Benchmark", display_mask)
                    cv2.waitKey(1) 
                
                t_dir = random.choice([-1, 1])
                sim_tx = sim_tx_min if t_dir == 1 else sim_tx_max
                
                # SIM2REAL FIX: Domain Randomization
                sim_hx = center_x + random.randint(-20, 20)
                sim_hy = home_y + random.randint(-20, 20)
                
                state_tensor = get_state_tensor(sim_tx, sim_ty, sim_hx, sim_hy, t_dir)
                done = False
                step_count = 0
                
                while not done:
                    step_count += 1
                    total_steps += 1
                    
                    if random.uniform(0, 1) < epsilon: action = random.randint(0, 7)
                    else:
                        with torch.no_grad():
                            action = policy_net(state_tensor).argmax().item()
                            
                    move_x, move_y = ACTION_MAP[action]
                    sim_hx += move_x * actual_h_speed
                    sim_hy += move_y * actual_h_speed
                    sim_tx += V_SPEED * t_dir
                    
                    distance = math.hypot(sim_tx - sim_hx, sim_ty - sim_hy)
                    is_catch = distance < 30
                    # SIM BOUNDARY: clamp the floor (mirrors the real edge
                    # firewall), terminate if the hunter leaves the board at
                    # top/left/right (real rig would lose laser detection).
                    sim_hy = min(sim_hy, home_y + SIM_FLOOR_PX)
                    hunter_lost = (sim_hx < bb_x_min or sim_hx > bb_x_max
                                   or sim_hy < bb_y_min)
                    is_escape = ((sim_tx >= sim_tx_max or sim_tx <= sim_tx_min)
                                 or step_count > 150 or hunter_lost)
                    
                    # RESCALED REWARDS. Tabular Q-learning is invariant to
                    # reward scale; a neural net is not. With +/-1000 targets
                    # and Adam at lr=1e-3, the network cannot grow its outputs
                    # to that magnitude within the training budget, and Huber
                    # loss sits in its flat region the whole time.
                    reward = -0.01
                    if is_catch:
                        reward, done = 1.0, True
                    elif is_escape:
                        reward, done = -1.0, True

                    if is_catch: catches += 1        # DIAGNOSTIC

                    next_state_tensor = get_state_tensor(sim_tx, sim_ty, sim_hx, sim_hy, t_dir)
                    memory.push(state_tensor, action, reward, next_state_tensor, done)
                    state_tensor = next_state_tensor
                    
                    if total_steps % DQN_OPTIMIZE_EVERY == 0:
                        optimize_model()
                    if total_steps % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                        
                # Was 0.999, which left epsilon at 0.082 after 2500 episodes --
                # the DQN never got a sustained near-greedy phase. 0.998 hits
                # the 0.05 floor around episode 1500, leaving the last ~40% of
                # training exploiting, matching the CNN's proportion.
                if epsilon > 0.05: epsilon *= 0.998

            dqn_trains.append(100.0 * catches / 2500)
            dqn_rates.append(evaluate_policy(act_dqn))
            dqn_weights.append({k: v.clone() for k, v in policy_net.state_dict().items()})
            print(f"[DQN] seed {dqn_seed_idx + 1}/{N_SEEDS} greedy: {dqn_rates[-1]:.1f}%")
            dqn_seed_idx += 1
            if dqn_seed_idx < N_SEEDS:
                continue                     # re-enter state 5 for the next seed

            _m = pick_median(dqn_rates)
            policy_net.load_state_dict(dqn_weights[_m])   # deploy the MEDIAN seed
            log_eval("DQN", sum(dqn_trains) / len(dqn_trains), dqn_rates[_m],
                     N_SEEDS, f"{min(dqn_rates):.1f}-{max(dqn_rates):.1f}")
            pause_screen(np.zeros_like(frame), "DQN READY!", "Starting Phase 3 Sweeps (6 seconds)...")
            
            current_mode = "DQN"
            sweep_count = 0
            target_escaped = False
            v_dir = 1 
            v_tx = board_data['tl'][0] + 30
            
            hunter_pan, hunter_tilt = hunter_home_pan, hunter_home_tilt
            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)
            time.sleep(1.5) 
                
            sweep_start_time = time.time()
            sweep_min_dist = 1e9
            sweep_hit = False
            sweep_hit_time = 0.0
            sweep_dwell = 0
            sweep_dwell_s = 0.0
            track_err_sum = 0.0
            track_err_n = 0
            last_frame_t = 0.0
            trail_canvas = np.zeros_like(frame) 
            current_state = 3 

        # --- STATE 6: CNN TRAINING (NEW 4th ALGORITHM) ---
        elif current_state == 6:
            combined = cv2.add(clean_frame, trail_canvas)
            if cnn_seed_idx == 0: cv2.imwrite("3b_DQN_PI_Trajectory.jpg", combined)
            
            if cnn_seed_idx == 0: pause_screen(np.zeros_like(frame), "DQN BENCHMARK COMPLETE!", f"Next: Convolutional Net x{N_SEEDS} seeds...")
            
            random.seed(1000 + cnn_seed_idx)
            np.random.seed(1000 + cnn_seed_idx)
            torch.manual_seed(1000 + cnn_seed_idx)

            cnn_policy_net = CNNBrain()       # fresh init for this seed
            cnn_target_net = CNNBrain()
            cnn_target_net.load_state_dict(cnn_policy_net.state_dict())
            cnn_target_net.eval()
            cnn_optimizer = optim.Adam(cnn_policy_net.parameters(), lr=CNN_LR)
            cnn_memory = ReplayMemory(capacity=8000)

            epsilon = 1.0
            cnn_total_steps = 0
            cnn_t_start = time.time()
            catches = 0                      # DIAGNOSTIC
            
            bg_mask = np.zeros_like(frame)
            cv2.putText(bg_mask, "TRAINING CONVOLUTIONAL NET...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(bg_mask, f"State = {GRID_SIZE}x{GRID_SIZE} rendered image", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for episode in range(CNN_EPISODES):
                if episode % 25 == 0:
                    elapsed = time.time() - cnn_t_start
                    display_mask = bg_mask.copy()
                    cv2.putText(display_mask, f"TRAINING CNN: {episode}/{CNN_EPISODES}", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    cv2.putText(display_mask, f"Elapsed: {elapsed:.0f}s   Epsilon: {epsilon:.2f}", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if is_recording: video_out.write(display_mask)
                    cv2.imshow("Robotics Benchmark", display_mask)
                    cv2.waitKey(1) 
                
                t_dir = random.choice([-1, 1])
                sim_tx = sim_tx_min if t_dir == 1 else sim_tx_max
                
                # SIM2REAL FIX: Domain Randomization
                sim_hx = center_x + random.randint(-20, 20)
                sim_hy = home_y + random.randint(-20, 20)
                
                state_grid = render_state_grid(sim_tx, sim_ty, sim_hx, sim_hy, t_dir, board_bounds)
                done = False
                step_count = 0
                
                while not done:
                    step_count += 1
                    cnn_total_steps += 1
                    
                    if random.uniform(0, 1) < epsilon: action = random.randint(0, 7)
                    else:
                        with torch.no_grad():
                            action = cnn_policy_net(state_grid).argmax().item()
                            
                    move_x, move_y = ACTION_MAP[action]
                    sim_hx += move_x * actual_h_speed
                    sim_hy += move_y * actual_h_speed
                    sim_tx += V_SPEED * t_dir
                    
                    distance = math.hypot(sim_tx - sim_hx, sim_ty - sim_hy)
                    is_catch = distance < 30
                    # SIM BOUNDARY: clamp the floor (mirrors the real edge
                    # firewall), terminate if the hunter leaves the board at
                    # top/left/right (real rig would lose laser detection).
                    sim_hy = min(sim_hy, home_y + SIM_FLOOR_PX)
                    hunter_lost = (sim_hx < bb_x_min or sim_hx > bb_x_max
                                   or sim_hy < bb_y_min)
                    is_escape = ((sim_tx >= sim_tx_max or sim_tx <= sim_tx_min)
                                 or step_count > 150 or hunter_lost)
                    
                    # RESCALED REWARDS -- same reasoning as the DQN phase.
                    reward = -0.01
                    if is_catch:
                        reward, done = 1.0, True
                    elif is_escape:
                        reward, done = -1.0, True

                    if is_catch: catches += 1        # DIAGNOSTIC

                    next_state_grid = render_state_grid(sim_tx, sim_ty, sim_hx, sim_hy, t_dir, board_bounds)
                    cnn_memory.push(state_grid, action, reward, next_state_grid, done)
                    state_grid = next_state_grid
                    
                    if cnn_total_steps % CNN_OPTIMIZE_EVERY == 0:
                        optimize_cnn_model()
                    if cnn_total_steps % CNN_TARGET_UPDATE == 0:
                        cnn_target_net.load_state_dict(cnn_policy_net.state_dict())
                        
                if epsilon > 0.05: epsilon *= CNN_EPS_DECAY

            print(f"CNN training done in {time.time() - cnn_t_start:.0f}s ({cnn_total_steps} steps)")
            cnn_trains.append(100.0 * catches / CNN_EPISODES)
            cnn_rates.append(evaluate_policy(act_cnn))
            cnn_weights.append({k: v.clone() for k, v in cnn_policy_net.state_dict().items()})
            print(f"[CNN] seed {cnn_seed_idx + 1}/{N_SEEDS} greedy: {cnn_rates[-1]:.1f}%")
            cnn_seed_idx += 1
            if cnn_seed_idx < N_SEEDS:
                continue                     # re-enter state 6 for the next seed

            _m = pick_median(cnn_rates)
            cnn_policy_net.load_state_dict(cnn_weights[_m])   # deploy the MEDIAN seed
            log_eval("CNN", sum(cnn_trains) / len(cnn_trains), cnn_rates[_m],
                     N_SEEDS, f"{min(cnn_rates):.1f}-{max(cnn_rates):.1f}")
            pause_screen(np.zeros_like(frame), "CNN READY!", "Starting Phase 4 Sweeps (6 seconds)...")
            
            current_mode = "CNN"
            sweep_count = 0
            target_escaped = False
            v_dir = 1 
            v_tx = board_data['tl'][0] + 30
            
            hunter_pan, hunter_tilt = hunter_home_pan, hunter_home_tilt
            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)
            time.sleep(1.5) 
                
            sweep_start_time = time.time()
            sweep_min_dist = 1e9
            sweep_hit = False
            sweep_hit_time = 0.0
            sweep_dwell = 0
            sweep_dwell_s = 0.0
            track_err_sum = 0.0
            track_err_n = 0
            last_frame_t = 0.0
            trail_canvas = np.zeros_like(frame) 
            current_state = 3 

        # --- STATE 7: HANDOFF TO PI CONTROLLER (5th ALGORITHM) ---
        elif current_state == 7:
            combined = cv2.add(clean_frame, trail_canvas)
            cv2.imwrite("4b_CNN_PI_Trajectory.jpg", combined)

            pause_screen(np.zeros_like(frame), "CNN BENCHMARK COMPLETE!", "Next: PI Controller (no training needed)...")

            # No training phase -- PI is analytic. Just report how it does in
            # the same simulator, alongside pure P for reference.
            log_eval("CLASSIC (P)", 0.0, evaluate_pcontrol(Kp, 0.0))
            log_eval("PI", 0.0, evaluate_pcontrol(PI_Kp, PI_Ki))

            current_mode = "PI"
            sweep_count = 0
            target_escaped = False
            reset_pi_integral()
            v_dir = 1
            v_tx = board_data['tl'][0] + 30

            hunter_pan, hunter_tilt = hunter_home_pan, hunter_home_tilt
            send_angles(hunter_pan, hunter_tilt, target_pan, target_tilt)
            time.sleep(1.5)

            sweep_start_time = time.time()
            sweep_min_dist = 1e9
            sweep_hit = False
            sweep_hit_time = 0.0
            sweep_dwell = 0
            sweep_dwell_s = 0.0
            track_err_sum = 0.0
            track_err_n = 0
            last_frame_t = 0.0
            trail_canvas = np.zeros_like(frame)
            current_state = 3

        # --- STATE 8: THE FINALE ---
        elif current_state == 8:
            combined = cv2.add(clean_frame, trail_canvas)
            cv2.imwrite("5_PI_Trajectory.jpg", combined)
            
            pause_screen(np.zeros_like(frame), "5-WAY BENCHMARK COMPLETE!", "Saving files and shutting down...")
            
            # Aggregate summary -- with 10 sweeps per arm these means are worth
            # reading directly instead of re-deriving them from the CSV.
            print("\n" + "=" * 74)
            print(f"{'Arm':14}{'n':>4}{'catch%':>9}{'minDist':>10}{'sd':>7}"
                  f"{'worst':>7}{'t2catch':>10}{'dwell(f)':>9}{'postHitErr':>10}")
            print("=" * 74)
            for arm, rows in results_log.items():
                n = len(rows)
                caught = sum(1 for r in rows if r[2])
                md = [r[0] for r in rows]
                mean = sum(md) / n
                sd = (sum((x - mean) ** 2 for x in md) / n) ** 0.5
                tt = [r[1] for r in rows if r[1] is not None]
                tstr = f"{sum(tt)/len(tt):.2f}" if tt else "-"
                dw = sum(r[3] for r in rows) / n
                te = [r[4] for r in rows if r[4] is not None]
                print(f"{arm:14}{n:4d}{100.0*caught/n:8.0f}%{mean:10.1f}{sd:7.1f}"
                      f"{max(md):7.0f}{tstr:>10}{dw:9.1f}"
                      f"{(sum(te)/len(te) if te else float('nan')):10.1f}")
            print("=" * 74)

            send_angles(90, 170, 90, 125)
            print("Experiment finished. Video, CSV, and 5 Images saved.")
            break

    if trail_canvas is not None:
        frame = cv2.add(frame, trail_canvas)
        
    frames_seen += 1
    if current_state == 3 and not target_escaped:
        sweep_frames += 1      # ACTIVE sweep frames only -- the reset/re-home
                               # between sweeps is state 3 too, and counting it
                               # against active-sweep seconds inflated the rate
    if frames_seen % 60 == 0:
        _el = time.time() - loop_t0
        if _el > 1.0: loop_fps = frames_seen / _el

    if is_recording:
        video_out.write(frame)
        
    cv2.imshow("Robotics Benchmark", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Safely quitting and cleaning up...")
        break

# --- SAFELY CLEAN UP ---
_elapsed = time.time() - loop_t0
if frames_seen > 0 and _elapsed > 0:
    _fps = frames_seen / _elapsed
    print(f"\nOverall loop rate: {_fps:.1f} fps over {frames_seen} frames "
          f"(diluted -- training phases block the loop while the clock runs).")
    if sweep_frames > 0 and total_sweep_time > 0:
        _sfps = sweep_frames / total_sweep_time
        print(f"SWEEP-PHASE loop rate: {_sfps:.1f} fps over {sweep_frames} frames "
              f"-- this is the number that matters.")
        print(f"For half-speed playback set VIDEO_FPS = {_sfps/2:.1f}.")

    _tot = roi_hits + roi_misses
    if _tot:
        print(f"ROI tracking: {roi_hits}/{_tot} frames found in the small window "
              f"({100.0*roi_hits/_tot:.1f}%). A low rate means ROI_HALF={ROI_HALF} "
              f"is too small or the dot drops out often.")

send_angles(90, 170, 90, 125) 
cap.release()
if is_recording: video_out.release()
cv2.destroyAllWindows()

for _ in range(10):
    cv2.waitKey(1)
    
arduino.close()
print("Cleanup complete. Goodbye!")
