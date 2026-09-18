"""
search_phases.py -- find the best section sizes for two staged guidance laws,
using the simulator calibrated in calibrate_sim.py (lag L=2 frames, detection
noise sigma=1.0 px, which reproduces the measured hardware results to <1 px).

SCENARIO A:  Q-TABLE -> bang-bang(continuous steering) -> PI -> HOLD
SCENARIO B:  Q-TABLE -> bang-bang(continuous steering) -> HOLD

HOLD is the "stay on top of the target" mode. The key idea is FEEDFORWARD: the
target's velocity is known exactly (it is virtual, V_SPEED * direction), so the
hunter matches that velocity outright and uses feedback only to trim the
residual. A purely error-driven controller has to let error BUILD before it
reacts, which is what makes the laser flicker on and off the target.

Objective, in order: 100% hit rate, then maximum DWELL (frames within
ON_TOP_PX of the target), then minimum time to first hit.
"""

import math
import random
import statistics as st
from itertools import product

BOARD_TOP, BOARD_BOT = 40, 537
BOARD_L, BOARD_R = 533, 996
sim_ty = int(BOARD_TOP + (BOARD_BOT - BOARD_TOP) * 0.25)
sim_tx_min, sim_tx_max = BOARD_L + 45, BOARD_R - 98
center_x, home_y = 764, BOARD_BOT - 40

SPEED, V_SPEED, MAX, FLOOR = 20.6, 8, 2.0, 20.0
CATCH_PX, ON_TOP_PX = 30.0, 15.0
FPS = 16.3                      # derived: span/V_SPEED frames over measured 2.45 s

LAG, NOISE = 2, 1.0             # calibrated against hardware

ACTION_MAP = [(0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (1, -1), (-1, 1), (1, 1)]


def c_discrete(tx, ty, hx, hy, s):
    best, bi = 1e9, 0
    for i, (mx, my) in enumerate(ACTION_MAP):
        d = math.hypot(tx - (hx + mx * SPEED), ty - (hy + my * SPEED))
        if d < best:
            best, bi = d, i
    mx, my = ACTION_MAP[bi]
    return mx * SPEED, my * SPEED


def c_bb(tx, ty, hx, hy, s):
    """Bang-bang propulsion, continuous steering: always full speed, heading
    computed continuously. 'lead' aims ahead of the target to compensate lag."""
    ax = tx + V_SPEED * s['dir'] * s['lead']
    ex, ey = ax - hx, ty - hy
    n = math.hypot(ex, ey)
    if n < 1e-6:
        return 0.0, 0.0
    return SPEED * ex / n, SPEED * ey / n


def c_pi(tx, ty, hx, hy, s):
    Kp, Ki = 0.02, 0.003
    lim = MAX / Ki
    s['ix'] = max(-lim, min(lim, s.get('ix', 0.0) + (tx - hx)))
    s['iy'] = max(-lim, min(lim, s.get('iy', 0.0) + (ty - hy)))
    pan = max(-MAX, min(MAX, (tx - hx) * Kp + s['ix'] * Ki))
    tilt = max(-MAX, min(MAX, (ty - hy) * Kp + s['iy'] * Ki))
    return (pan / MAX) * SPEED, (tilt / MAX) * SPEED


def c_hold(tx, ty, hx, hy, s):
    """Stay on top: match the target's velocity (feedforward) + trim the error."""
    kh = s['kh']
    vx = V_SPEED * s['dir'] + kh * (tx - hx)
    vy = kh * (ty - hy)
    n = math.hypot(vx, vy)
    if n > SPEED:
        vx, vy = vx * SPEED / n, vy * SPEED / n
    return vx, vy


def rollout(r_bb, r_pi, r_hold, kh, lead, seed_ep, use_pi=True):
    rng = random.Random(seed_ep)
    d = rng.choice([-1, 1])
    tx = sim_tx_min if d == 1 else sim_tx_max
    hx = center_x + rng.randint(-20, 20)
    hy = home_y + rng.randint(-20, 20)

    q = [(0.0, 0.0)] * LAG
    s = {'dir': d, 'kh': kh, 'lead': lead}
    md, t_hit, dwell, step, phase = 1e9, None, 0, 0, None

    while True:
        step += 1
        ohx, ohy = hx + rng.gauss(0, NOISE), hy + rng.gauss(0, NOISE)
        dist_o = math.hypot(tx - ohx, sim_ty - ohy)

        if dist_o > r_bb:
            ph, ctl = 'D', c_discrete
        elif use_pi and dist_o > r_pi:
            ph, ctl = 'B', c_bb
        elif use_pi and dist_o > r_hold:
            ph, ctl = 'P', c_pi
        elif (not use_pi) and dist_o > r_hold:
            ph, ctl = 'B', c_bb
        else:
            ph, ctl = 'H', c_hold

        if ph != phase:                 # fresh integral on entering PI
            if ph == 'P':
                s['ix'] = s['iy'] = 0.0
            phase = ph

        mvx, mvy = ctl(tx, sim_ty, ohx, ohy, s)
        q.append((mvx, mvy))
        amvx, amvy = q.pop(0)
        hx += amvx
        hy += amvy

        tx += V_SPEED * d
        hy = min(hy, home_y + FLOOR)

        dist = math.hypot(tx - hx, sim_ty - hy)
        md = min(md, dist)
        if t_hit is None and dist < CATCH_PX:
            t_hit = step
        if dist < ON_TOP_PX:
            dwell += 1

        if tx >= sim_tx_max or tx <= sim_tx_min or step > 150:
            return md, t_hit, dwell


def score(r_bb, r_pi, r_hold, kh, lead, n=150, use_pi=True, base=0):
    hits = dwell = 0
    tt, mds = [], []
    for i in range(n):
        md, th, dw = rollout(r_bb, r_pi, r_hold, kh, lead, base + i, use_pi)
        mds.append(md)
        dwell += dw
        if th is not None:
            hits += 1
            tt.append(th)
    return (100.0 * hits / n, dwell / n, st.mean(tt) / FPS if tt else 9.9,
            st.mean(mds))


def report(title, rows, k):
    print(f"\n{title}")
    print(f"{'params':34}{'hit%':>6}{'dwell(f)':>10}{'t_hit(s)':>10}{'minDist':>9}")
    print("-" * 69)
    for p, (h, dw, t, md) in rows[:k]:
        print(f"{p:34}{h:6.0f}{dw:10.1f}{t:10.2f}{md:9.1f}")


if __name__ == "__main__":
    print(f"Calibrated simulator: lag={LAG} frames, noise={NOISE} px, "
          f"{FPS:.1f} fps")
    print(f"Objective: 100% hit, then max dwell within {ON_TOP_PX:.0f} px, "
          f"then min time to hit")

    # ---------------- baselines, same simulator ----------------
    print("\nBASELINES (current hardware top performers):")
    b1 = score(0, 0, 0, 0, 0, use_pi=True)          # r_bb=0 -> discrete only
    print(f"  {'Q-TABLE / DQN (discrete only)':34}{b1[0]:6.0f}{b1[1]:10.1f}"
          f"{b1[2]:10.2f}{b1[3]:9.1f}")
    b2 = score(0, 100, 0, 0, 0, use_pi=True)        # discrete then PI at 100 px
    print(f"  {'Q-TABLE+PI (handoff at 100 px)':34}{b2[0]:6.0f}{b2[1]:10.1f}"
          f"{b2[2]:10.2f}{b2[3]:9.1f}")

    # ---------------- scenario A ----------------
    resA = []
    for r_bb, r_pi, r_hold, kh, lead in product(
            [160, 130, 100, 80], [60, 45, 30], [22, 16, 10],
            [0.15, 0.30, 0.50], [0, 1, 2]):
        if not (r_bb > r_pi > r_hold):
            continue
        resA.append((f"bb>{r_bb} pi>{r_pi} hold<={r_hold} kh={kh} lead={lead}",
                     score(r_bb, r_pi, r_hold, kh, lead, use_pi=True)))
    resA.sort(key=lambda r: (-r[1][0], -r[1][1], r[1][2]))
    report("SCENARIO A -- discrete -> bang-bang -> PI -> hold", resA, 8)

    # ---------------- scenario B ----------------
    resB = []
    for r_bb, r_hold, kh, lead in product(
            [160, 130, 100, 80], [22, 16, 10], [0.15, 0.30, 0.50], [0, 1, 2]):
        if not (r_bb > r_hold):
            continue
        resB.append((f"bb>{r_bb} hold<={r_hold} kh={kh} lead={lead}",
                     score(r_bb, 0, r_hold, kh, lead, use_pi=False)))
    resB.sort(key=lambda r: (-r[1][0], -r[1][1], r[1][2]))
    report("SCENARIO B -- discrete -> bang-bang -> hold (no PI)", resB, 8)

    # ---------------- confirm winners on fresh seeds ----------------
    print("\nCONFIRMATION on 600 unseen episodes:")
    for name, rows, use_pi in (("A", resA, True), ("B", resB, False)):
        p = rows[0][0]
        v = {kv.split('=')[0] if '=' in kv else kv.split('>')[0]:
             kv.split('=')[-1].split('>')[-1] for kv in p.split()}
        rb = float(v['bb'])
        rp = float(v.get('pi', 0))
        rh = float(v['hold<'].lstrip('=')) if 'hold<' in v else float(v['hold<='])
        kh, ld = float(v['kh']), int(v['lead'])
        r = score(rb, rp, rh, kh, ld, n=600, use_pi=use_pi, base=50000)
        print(f"  {name}: {p}")
        print(f"     hit {r[0]:.0f}%   dwell {r[1]:.1f} frames "
              f"({r[1]/FPS:.2f}s)   t_hit {r[2]:.2f}s   minDist {r[3]:.1f}px")
