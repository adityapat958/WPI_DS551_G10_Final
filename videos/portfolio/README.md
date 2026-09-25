# Fetch mobile manipulation in Habitat 2.0 (ReplicaCAD)

| Video | What it shows |
|---|---|
| `fetch_bedroom_to_kitchen_1080p.mp4` | 1920×1080, 52 s — drawer → kitchen table, planned whole-body motion |
| `fetch_learned_skills_720p.mp4` | 1280×720 — learned PPO skills chaining nav → pick → nav → place on 4 validation episodes |

## 1. Bedroom drawer → kitchen table

![keyframes](fetch_bedroom_to_kitchen_sheet.jpg)

The Fetch mobile manipulator navigates the apartment, opens a chest-of-drawers by its handle,
picks a soup can out of the drawer, closes the drawer, carries the can upright to the kitchen table,
and places it; the can settles under Bullet physics.

### How it works (`src/demo/render_v3.py`)
Motion comes from a planning pipeline (no learned policy):
- **Navigation** — shortest paths on a navmesh recomputed after decluttering (clearance 0.38 m).
- **Arm** — damped-least-squares IK (torso + 7 arm joints) on the simulator's own forward kinematics,
  with approach-axis and finger-axis constraints (side-straddle grasps).
- **Collision-free motion planning** — Bullet contact queries on every robot link and the held object;
  each reach tries a straight Cartesian path, then via-points, then RRT-Connect in joint space with shortcut smoothing.
- **Per-frame collision audit** — every rendered frame is checked; the final render has **0 unplanned contacts**.
- **Placement** — candidate spots are filtered for a clear table surface and a collision-free reach
  (final error 0.9 cm, 0.7° tilt).
- **Camera** — GTA-style chase camera with raycast wall avoidance; PiPs show the hand depth camera
  (mounted where habitat-lab defines Fetch's arm camera) and the head RGB camera.
- Self-check report: `videos/v3/*.checks.json`.

### Reproduce
```bash
vizjob run render     # 720p   (see vizjob.sh; WPI Turing, env /scratch/apatwardhan/envs/habitat)
vizjob run final      # 1080p
```

## 2. Learned skills (`src/rl_skills/`)

Meta's Habitat 2022 Rearrangement Challenge skill policies (DD-PPO; the policies see head depth plus
proprioception) loaded into habitat-lab 0.3.3 and chained nav → pick → nav → place with no hand-written motion.
The overlay shows the active skill, object-to-goal distance, and the depth image the policy acts on.
The four clips are the full-task successes from a 60-episode `rearrange_easy` validation batch (4/60).

```bash
vizjob run rl -- --episode-ids 5 114 125 318 --pretty --third-w 1280 --third-h 720 --out-dir videos/rl_pretty
python3 src/rl_skills/make_reel.py videos/rl_pretty/ep{5,114,125,318}_SUCC.mp4 -o videos/portfolio/fetch_learned_skills_720p.mp4
```
