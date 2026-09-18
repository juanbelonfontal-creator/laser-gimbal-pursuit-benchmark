"""
calibrate_sim.py -- fit the simulator to the measured hardware results.

The original simulator was frictionless: a command took effect instantly and the
hunter's position was known exactly. That is why it predicted PI would reach
~2 px closest approach when the rig actually delivers 14.1 px.

Two physical effects are added here:

  LAG    (L frames) -- a servo does not reach a commanded angle within one
                       frame, and the serial write + camera capture + detection
                       pipeline adds more delay. Modelled as a delay queue on
                       the commanded motion.

  NOISE  (sigma px) -- the laser centroid from the HSV blob detector jitters
                       frame to frame. Controllers see a noisy position.

(L, sigma) are fitted so the simulator reproduces the closest-approach means
actually measured over 10 sweeps per arm on the real gimbal.
"""

import math
import random
import statistics as st

# ---------------------------------------------------------------- geometry
# Board corners read off the run's trajectory images; escape span and target
# line are derived exactly as the benchmark derives them.
BOARD_TOP, BOARD_BOT = 40, 537
BOARD_L, BOARD_R = 533, 996
sim_ty = int(BOARD_TOP + (BOARD_BOT - BOARD_TOP) * 0.25)
sim_tx_min, sim_tx_max = BOARD_L + 45, BOARD_R - 98
center_x, home_y = 764, BOARD_BOT - 40

SPEED = 20.6        # measured servo speed, px/frame (printed each run)
V_SPEED = 8         # virtual target speed, px/frame
MAX = 2.0           # MAX_HUNTER_SPEED, degrees/frame
FLOOR = 20.0        # SIM_FLOOR_PX, the real edge-firewall clamp

CATCH_PX = 30.0     # benchmark catch radius

ACTION_MAP = [(0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (1, -1), (-1, 1), (1, 1)]

# measured on hardware, 10 sweeps per arm, mean closest approach in px
REAL = {"CLASSIC": 38.1, "Q-TABLE": 9.4, "DQN": 7.2, "PI": 14.1}


# ---------------------------------------------------------------- controllers
def ctl_discrete(tx, ty, hx, hy, st_):
    """Stand-in for a converged Q-TABLE/DQN policy: 8 directions, full speed."""
    best, bi = 1e9, 0
    for i, (mx, my) in enumerate(ACTION_MAP):
        d = math.hypot(tx - (hx + mx * SPEED), ty - (hy + my * SPEED))
        if d < best:
            best, bi = d, i
    mx, my = ACTION_MAP[bi]
    return mx * SPEED, my * SPEED


def ctl_p(tx, ty, hx, hy, st_, Kp=0.02):
    """Pure proportional, degrees clipped to MAX, then converted to px."""
    pan = max(-MAX, min(MAX, (tx - hx) * Kp))
    tilt = max(-MAX, min(MAX, (ty - hy) * Kp))
    return (pan / MAX) * SPEED, (tilt / MAX) * SPEED


def ctl_pi(tx, ty, hx, hy, st_, Kp=0.02, Ki=0.003):
    lim = MAX / Ki
    st_['ix'] = max(-lim, min(lim, st_.get('ix', 0.0) + (tx - hx)))
    st_['iy'] = max(-lim, min(lim, st_.get('iy', 0.0) + (ty - hy)))
    pan = max(-MAX, min(MAX, (tx - hx) * Kp + st_['ix'] * Ki))
    tilt = max(-MAX, min(MAX, (ty - hy) * Kp + st_['iy'] * Ki))
    return (pan / MAX) * SPEED, (tilt / MAX) * SPEED


# ---------------------------------------------------------------- simulator
def rollout(controller, L, sigma, seed_ep, hold_from=None):
    """One sweep. Returns (min_dist, frames_to_hit or None, dwell_frames)."""
    rng = random.Random(seed_ep)
    d = rng.choice([-1, 1])
    tx = sim_tx_min if d == 1 else sim_tx_max
    hx = center_x + rng.randint(-20, 20)
    hy = home_y + rng.randint(-20, 20)

    queue = [(0.0, 0.0)] * max(0, L)      # commanded motion in flight
    st_ = {}
    md, t_hit, dwell, step = 1e9, None, 0, 0

    while True:
        step += 1
        # controller sees a NOISY estimate of where the laser is
        ohx = hx + rng.gauss(0, sigma)
        ohy = hy + rng.gauss(0, sigma)

        mvx, mvy = controller(tx, sim_ty, ohx, ohy, st_)

        queue.append((mvx, mvy))          # command enters the pipeline
        amvx, amvy = queue.pop(0)         # a delayed command takes effect
        hx += amvx
        hy += amvy

        tx += V_SPEED * d
        hy = min(hy, home_y + FLOOR)

        dist = math.hypot(tx - hx, sim_ty - hy)
        md = min(md, dist)
        if t_hit is None and dist < CATCH_PX:
            t_hit = step
        if dist < 15.0:
            dwell += 1

        if tx >= sim_tx_max or tx <= sim_tx_min or step > 150:
            return md, t_hit, dwell


def arm_mean(controller, L, sigma, n=300, base=0):
    return st.mean(rollout(controller, L, sigma, base + i)[0] for i in range(n))


# ---------------------------------------------------------------- calibration
def fit():
    best = None
    print(f"{'L':>3}{'sigma':>7}  {'CLASSIC':>9}{'discrete':>10}{'PI':>8}   {'error':>8}")
    print("-" * 52)
    for L in range(0, 6):
        for sigma in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]:
            c = arm_mean(ctl_p, L, sigma)
            g = arm_mean(ctl_discrete, L, sigma)
            p = arm_mean(ctl_pi, L, sigma)
            # discrete is compared against the mean of the two coordinate-state
            # learned arms actually measured (Q-TABLE 9.4, DQN 7.2)
            err = ((c - REAL["CLASSIC"]) ** 2
                   + (g - (REAL["Q-TABLE"] + REAL["DQN"]) / 2) ** 2
                   + (p - REAL["PI"]) ** 2) ** 0.5
            print(f"{L:3d}{sigma:7.1f}  {c:9.1f}{g:10.1f}{p:8.1f}   {err:8.2f}")
            if best is None or err < best[0]:
                best = (err, L, sigma, c, g, p)
    return best


if __name__ == "__main__":
    print("Measured on hardware (10 sweeps/arm, mean closest approach px):")
    print(f"  CLASSIC {REAL['CLASSIC']}   Q-TABLE {REAL['Q-TABLE']}   "
          f"DQN {REAL['DQN']}   PI {REAL['PI']}\n")
    err, L, sigma, c, g, p = fit()
    print("\n" + "=" * 52)
    print(f"BEST FIT: lag L = {L} frames, detection noise sigma = {sigma} px")
    print(f"  CLASSIC  sim {c:5.1f}  vs real {REAL['CLASSIC']:5.1f}")
    print(f"  discrete sim {g:5.1f}  vs real {(REAL['Q-TABLE']+REAL['DQN'])/2:5.1f}")
    print(f"  PI       sim {p:5.1f}  vs real {REAL['PI']:5.1f}")
    print("=" * 52)
