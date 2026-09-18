# Laser Gimbal Pursuit Benchmark — Handoff

Paste this at the start of a new chat, along with `benchmark_v2.py` and
`gimbal_receiver.ino`.

---

## 1. What the rig is

- **Mac**: MacBook Pro 13" 2017, dual-core i5, 8 GB, Ventura. Intel, so PyTorch
  is capped at 2.2.2 (last Intel build).
- **Arduino Uno** over USB. Two servos on a pan/tilt gimbal carrying a cheap red
  laser pointer. **Pan = D3, tilt = D5.** Baud **115200** (both ends must match).
- **Camera**: built-in FaceTime, 1280×720. Index is unstable — iPhone Continuity
  Camera shifts it between sessions. Enumerate before every run.
- **Board**: matte black, four orange dots mark the corners. The script masks
  everything outside that quad; nothing beyond it is visible to the code.
- Laser parks at tilt 170 (ceiling) at startup; home is bottom-centre of board.

### Working environment (do not upgrade blindly)
`numpy 1.26.4`, `opencv-python 4.10.0.84`, `torch 2.2.2`.
NumPy ≥2 breaks torch. OpenCV 4.12 needs NumPy ≥2 and **silently returns no
frames** with NumPy 1.x — it fails as a dead camera, not an error.

---

## 2. What the experiment does

A virtual target (green dot, drawn in software) sweeps horizontally across the
top quarter of the board. The laser ("hunter") starts at bottom-centre and
chases it. Each algorithm runs N sweeps, alternating direction.

The script measures the servo's real pixel speed during the CLASSIC phase, then
builds a matched simulator from the detected board geometry, trains policies in
it, and deploys them to the same hardware. Sim-to-real by construction.

### Metrics (CSV columns)
| Column | Meaning |
|---|---|
| Final Error (px) | distance at sweep end; **0 on any catch** — saturated, legacy only |
| Result | CATCH if ever within 30 px |
| Min Dist (px) | closest approach over the whole sweep — accuracy |
| Time to Catch (s) | first crossing inside 30 px — interception speed |
| Dwell (frames/s) | time spent within 15 px — "stays on target" |
| Post-hit Err (px) | mean distance after first hit — same idea, ~3× less noisy |

Sweeps run the **full traverse** — a catch no longer ends them. That was
necessary; while catches ended sweeps, both accuracy metrics were censored.

---

## 3. Final result

**Winner: Scenario B**, a three-stage staged guidance law.

| Distance to target | Stage | Controller |
|---|---|---|
| > 160 px | Q-TABLE | learned policy, 8 discrete directions, full speed |
| 160 → 22 px | BANG-BANG | full speed, *continuous* heading, aims 2 frames ahead |
| < 22 px | HOLD | matches target velocity (feedforward) + `kh` × error |

`kh = 0.40` (0.60 is statistically identical — this knob is flat).

Final numbers (n=30 per arm, `v2_run1`):

| Arm | Hit | Min dist | Dwell | Post-hit err |
|---|---|---|---|---|
| B (kh 0.40) | 100% | 4.6 | 8.1 f | 15.5 px |
| B (kh 0.60) | 100% | 5.1 | 7.5 f | 16.8 px |
| B, no Q-Table | 100% | 7.9 | 3.9 f | 30.8 px |
| Q-Table alone | 80% | 17.1 | 1.2 f | 40.5 px |
| Classic P | 0% | 43.2 | 0 | — |

---

## 4. Findings worth keeping (each cost a hardware run)

1. **Reward scaling broke the neural methods.** ±1000 rewards with Adam at
   lr=1e-3: the network cannot grow its outputs to that magnitude in the
   training budget, and Huber loss sits flat the whole time. Rescaling to ±1
   took the CNN from 5% to 100%. Tabular Q-learning is invariant to this, which
   is why it "won" at first.
2. **Replay ratio caused value overestimation.** DQN backpropped every step at
   batch 64 (~64 replays/transition) vs the CNN's 16. Symptom: the *exploring*
   catch rate beat the *greedy* rate — the policy degraded late in training.
   Batch 32 every 2 steps fixed it, 47% → 100%.
3. **The simulator let the hunter leave the board.** Overshoot was free in sim;
   in reality leaving the board loses laser detection and freezes control. DQN
   scored 100% in sim and 1/3 on hardware until the boundary was added.
4. **Feedforward is what makes the laser stay on target.** Pure error-driven
   control must let error build before reacting → the dot flickers on and off.
   Matching the target's known velocity raised dwell 4–9×. This was the user's
   idea and it is the core result.
5. **Classic P's 38–43 px lag is structural**, not tuning: steady-state error of
   a proportional controller tracking a ramp. Adding an integrator (PI) fixed it
   exactly as theory predicts — 0/3 → 3/3.
6. **PI sits one frame of lag from instability on this rig.** It works at L=2 and
   collapses at L=3. Adding a PI stage to the staged controller (Scenario A)
   *halved* dwell vs leaving it out (Scenario B).
7. **The Q-Table phase genuinely contributes.** Removing it costs 15 px of
   post-hit error at p ≤ 0.0001 (n=30). Two earlier n=10 runs disagreed on this;
   only n=30 settled it.

### Where the simulator can and cannot be trusted
Calibrated at **lag = 2 frames, detection noise = 1.0 px**, it reproduced
Classic/discrete/PI closest approach to within 0.6 px. It still:
- predicted PI would reach 2 px (hardware: 14 px) — no lag/noise in fine control
- ranked Scenario A above B (hardware: reversed)
- predicted `kh = 0.60` would collapse (hardware: identical to 0.40)

**Use it to generate hypotheses and bracket parameters. Do not use it to rank
architectures.**

---

## 5. Methodology that proved necessary

- **n matters more than arm count.** At n=10 dwell's sd ≈ its mean and nothing
  separated (all p > 0.27). At n=30 effects appeared at p ≤ 0.0001.
- **Drift control**: repeat the first arm last. The rig genuinely changes within
  and between sessions (Q-Table min dist moved 9.4 → 15.0 px across two runs
  with identical code).
- **Permutation tests**, not eyeballing. Several "obvious" effects were noise.
- **Keep sweep counts even** — direction alternates, so odd counts bias the mean.
- **Optimising the loop changes the task.** `V_SPEED` is px/*frame*, so speeding
  the loop from 17→29 fps made the target 75% faster and invalidated a whole
  run. Difficulty is now `TARGET_PX_PER_SEC` with `V_SPEED` derived from the
  measured frame rate.

---

## 6. Known open defects

1. **Position-dependent gain.** `actual_h_speed` is one scalar, but a pan/tilt
   aimed at a plane has px/degree scaling as 1/cos²θ. The board spans ~42° of
   pan, so the edges differ from the centre by ~15%, and the camera's
   trapezoidal view compounds it. This corrupts the HOLD feedforward directly
   and is the most likely reason post-hit error sits at ~15 px. **Biggest
   remaining improvement.** Fix: log (x, y, px moved) during CLASSIC, fit a gain
   map, use the local value in `hold_step`.
2. Sweep duration is ~1.5–1.9 s against a designed 2.35 s — the difficulty match
   is close but not exact.
3. `LEAD_FRAMES = 2` was tuned at 16 fps and has not been re-derived since the
   loop reached ~24 fps.

---

## 7. Performance notes

- ROI tracking (search a 180×180 window around the last known dot) + 115200 baud
  took the loop from ~17 to ~24–29 fps. ROI hit rate ~96%.
- Latency budget at 17 fps was ~118 ms total: serial 14.6 ms (now 1.2), servo
  refresh ~10 ms (Servo library is 50 Hz), camera frame ~59 ms, plus processing
  and servo mechanical response.
- Remaining hardware levers, in order: faster camera (frame period is the
  largest single term), digital servos at 200–333 Hz, camera mounted
  perpendicular to the board to remove the trapezoid.

---

## 8. Files

- `benchmark_v2.py` — the current script (all fixes, 4-arm design, n=30)
- `gimbal_receiver.ino` — Arduino firmware, pan D3 / tilt D5, 115200
- `servo_test.py` — hardware-only check, no camera; isolates wiring from vision
- `calibrate_sim.py` — fits lag/noise to measured hardware results
- `search_phases.py`, `pi_tune.py`, `hybrid_test.py` — offline parameter search.
  **All have board geometry hardcoded from screenshots** — update the constants
  before trusting any output.

---

## 9. Roadmap — what comes next

The benchmark above is **finished**. Everything below is the follow-on work.

### Part 1 — Gimbal tracks an RC car (START HERE)

Replace the virtual target with a real vehicle moving at varying speed.

**What carries over:** the gimbal, board, camera, Arduino, laser detection,
ROI tracking, and the entire staged controller (Q-TABLE > 160 px, bang-bang
160→22 px, HOLD < 22 px, kh = 0.40).

**What is genuinely new — the whole point of Part 1:** the target's velocity
becomes *unknown*. HOLD's feedforward currently uses `V_SPEED * t_dir`, exact
because the target is software. With a real car it must be **estimated** from
vision. Raw frame differencing will be too noisy; use an alpha-beta filter
(or Kalman). The car also moves in 2D, so feedforward is needed on both axes,
not just x.

**Hardware:** an RC car with **proportional throttle** — not binary on/off, or
"varying speed" is impossible. Prefer a *slow* car: the servo moves ~20.6
px/frame and the matched-difficulty virtual target moved ~4.7 px/frame, so a
fast car simply outruns the gimbal. A marker on the car in **blue or green** —
never red, which collides with the laser.

**Geometry change:** the board is currently vertical with the camera facing it.
A car needs a horizontal play area with orange corner markers, and the camera
must see both the car and the laser dot (which will sometimes land on the car
and sometimes on the floor beside it). Sketch this before buying.

**Milestones, each independently testable:**
1. Detection only, no control — track car marker and laser dot together, log
   both, confirm the frame rate survives two objects.
2. Velocity estimation — drive at constant speed, check the estimator recovers
   it. This is the new piece.
3. Feed the estimate into HOLD's feedforward, run the existing controller
   unchanged. If 1 and 2 are solid this should mostly work; if not, the fault
   is in the estimator, not the controller.

### Part 2 — Laser data link (do this second, on the car)

On-off keying down the beam to a receiver on the car.

**Why it matters:** it gives the existing metrics physical meaning. **Dwell
becomes how long you can transmit; post-hit error becomes whether the beam
stays on the detector.**

**Spec check to do first:** measure the board width in mm, divide by ~460 px to
get mm/px. Multiply by the post-hit error (~15 px) to get beam wander. Against
a bare 5 mm photodiode this is likely marginal — which is the concrete argument
for fixing the position-dependent gain (Section 6, item 1) before attempting it.

**Hardware:** photodiode or phototransistor, transistor to switch the laser from
an Arduino digital pin, a receiving microcontroller. A few hundred bps with OOK
is very achievable.

### Part 3 — Drone (hardest; optional)

**Two structural problems, not just "harder":**
1. 3D motion, and a single camera gives no range.
2. **The feedback loop is lost.** The whole project so far closes the loop on
   where the laser dot *actually is*. There is no visible dot on a drone. This
   forces open-loop aiming — track the drone in-image and drive the gimbal by a
   calibrated angle-to-pixel mapping — which makes the gain map **mandatory**
   rather than an optimisation.

**Safety:** a laser aimed upward at a flying object is a different risk category
from one aimed at a board. Indoors, low power, never toward people or windows.

Parts 1 + 2 alone are a complete, publishable story. Decide on Part 3 afterwards.

### Adjacent — pick-and-place intercept with a robot arm

Separate project, same core skill. A robot arm grabbing an object off a moving
carrier using webcam tracking and prediction. **Arm not purchased yet**
(considering the SO-101); the blocker has been finding a conveyor substitute.

**The connection:** an RC car carrying an object *is* a conveyor substitute, and
Part 1's velocity estimation is the same machinery an arm needs to know where to
reach. Doing Part 1 first de-risks this project and defers the arm purchase.

Also on the shelf: **laser-paddle-block** — an arm holding a paddle intercepts
the gimbal's beam before it reaches the board. Reuses this exact rig.

### Publishing

The benchmark is already worth posting on its own. What distinguishes it from
typical hobby content is the method — controlled ablations, a drift control,
permutation tests, and honest negative results (the simulator was wrong three
times and got caught each time). Lead with a finding, not a build log.
