#!/usr/bin/env python3
"""
Chain the 2022 Rearrangement-Challenge Fetch skills (nav -> pick -> nav -> place)
inside a habitat-lab v0.3.3 RearrangePddlTask (rearrange_easy) episode and record
third-person + head RGB video.

MUST run on a GPU node (EGL). From the project root that contains ./data :

  export LD_LIBRARY_PATH=/scratch/apatwardhan/envs/habitat/lib:$LD_LIBRARY_PATH
  cd /scratch/apatwardhan/habitat_ws/WPI_DS551_G10_Final
  /scratch/apatwardhan/envs/habitat/bin/python src/rl_skills/rollout_skills.py \
      --models-dir data/models --split val --num-episodes 10 --out-dir videos/rl_skills

Login-node-safe checks (no simulator is created):
  ... rollout_skills.py --dry-config     # compose + print the hydra config only
  ... rollout_skills.py --fake-env       # run the full policy/adapter chain on fake obs
"""
import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_skill import SkillRunner  # noqa: E402
from overlay import Overlay, depth_vis  # noqa: E402
from obs_adapter import (  # noqa: E402
    K_HOLD,
    MAX_SKILL_STEPS,
    REQUIRED_V033_KEYS,
    action_to_v033,
    obs_to_2022,
    skill_done,
)

PLAN = [
    ("nav", dict(nav_to_goal=False)),  # go to the object
    ("pick", dict()),
    ("nav", dict(nav_to_goal=True)),  # go to the goal receptacle
    ("place", dict()),
]


# ---------------------------------------------------------------------------
def make_config(args):
    import habitat
    from habitat.config.default_structured_configs import (
        HeadRGBSensorConfig,
        ThirdRGBSensorConfig,
    )

    overrides = [
        f"habitat.dataset.split={args.split}",
        f"habitat.dataset.data_path=data/datasets/replica_cad/rearrange/v2/{args.split}/{args.dataset}.json.gz",
        "habitat.dataset.scenes_dir=data/replica_cad/",
        # 2022 skills were trained with CONCUR_RENDER=False (concurrent rendering
        # in v0.3.3 returns observations of the PRE-physics state => 1-step lag)
        "habitat.simulator.concur_render=False",
        "habitat.simulator.auto_sleep=False",  # 2022: AUTO_SLEEP=False
        "habitat.simulator.ctrl_freq=120.0",  # 2022: CTRL_FREQ=120
        "habitat.simulator.ac_freq_ratio=4",  # 2022: AC_FREQ_RATIO=4
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}",
        "habitat.environment.max_episode_steps=0",  # we bound the rollout ourselves
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.task.end_on_success=False",  # keep filming a few frames after success
        "habitat.task.measurements.pddl_success.must_call_stop=False",
        "habitat.task.measurements.force_terminate.max_accum_force=-1.0",
        "habitat.task.measurements.force_terminate.max_instant_force=-1.0",
        # 2022 BASE_VELOCITY for nav.pth: LIN_SPEED=ANG_SPEED=10, ALLOW_BACK, ALLOW_DYN_SLIDE
        "habitat.task.actions.base_velocity.lin_speed=10.0",
        "habitat.task.actions.base_velocity.ang_speed=10.0",
        "habitat.task.actions.base_velocity.allow_back=True",
        "habitat.task.actions.base_velocity.allow_dyn_slide=True",
        # 2022 ARM_ACTION for pick/place.pth
        "habitat.task.actions.arm_action.arm_controller=ArmRelPosAction",
        "habitat.task.actions.arm_action.grip_controller=SuctionGraspAction",
        "habitat.task.actions.arm_action.delta_pos_limit=0.0125",
        "habitat.task.actions.arm_action.arm_joint_dimensionality=7",
        "habitat.task.actions.arm_action.grasp_thresh_dist=0.15",
        "habitat.task.desired_resting_position=[0.5,0.0,1.0]",
    ] + list(args.opts)
    cfg = habitat.get_config("benchmark/rearrange/multi_task/rearrange_easy.yaml", overrides)
    with habitat.config.read_write(cfg):
        agent = cfg.habitat.simulator.agents.main_agent
        # head_depth_sensor is already present (depth_head_agent: 256x256, hfov 90,
        # normalize_depth over [0,10]) == 2022 HEAD_DEPTH_SENSOR.
        agent.sim_sensors.update(
            {
                "head_rgb_sensor": HeadRGBSensorConfig(height=args.head_res, width=args.head_res),
                "third_rgb_sensor": ThirdRGBSensorConfig(height=args.third_h or args.third_res,
                                                          width=args.third_w or args.third_res),
            }
        )
    return cfg


def check_config(cfg):
    sim = cfg.habitat.simulator
    agent = sim.agents.main_agent
    d = agent.sim_sensors.head_depth_sensor
    assert (d.height, d.width, d.hfov, d.min_depth, d.max_depth, d.normalize_depth) == (256, 256, 90, 0.0, 10.0, True), d
    assert agent.articulated_agent_type == "FetchSuctionRobot", agent.articulated_agent_type
    assert agent.articulated_agent_urdf.endswith("hab_suction.urdf")
    assert not sim.kinematic_mode, "ArmRelPosAction only moves DYNAMIC robots in v0.3.3"
    t = cfg.habitat.task
    for s in ["target_start_sensor", "goal_sensor", "joint_sensor", "is_holding_sensor",
              "relative_resting_pos_sensor", "target_start_gps_compass_sensor", "target_goal_gps_compass_sensor"]:
        assert s in t.lab_sensors, s
    assert "arm_action" in t.actions and "base_velocity" in t.actions


# ---------------------------------------------------------------------------
def to_uint8_rgb(x):
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x * (255.0 if x.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, -1)
    return x[..., :3]


def compose_frame(obs, text_lines: List[str], third_res: int):
    import cv2

    third = to_uint8_rgb(obs["third_rgb"])
    head = to_uint8_rgb(obs["head_rgb"])
    depth = to_uint8_rgb(obs["head_depth"])
    half = third.shape[0] // 2
    head = cv2.resize(head, (half, half))
    depth = cv2.resize(depth, (half, half), interpolation=cv2.INTER_NEAREST)
    right = np.concatenate([head, depth], 0)
    if right.shape[0] != third.shape[0]:
        right = cv2.resize(right, (half, third.shape[0]))
    frame = np.ascontiguousarray(np.concatenate([third, right], 1))
    y = 24
    for line in text_lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return frame


def write_video(frames, path, fps):
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)
    print(f"  wrote {path} ({len(frames)} frames)")


def _obj_goal_dist(m) -> float:
    v = m.get("object_to_goal_distance", np.nan)
    if isinstance(v, dict):
        v = min(v.values()) if v else np.nan
    try:
        return float(v)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
def run_episode(env, runners: Dict[str, SkillRunner], args):
    obs = env.reset()
    ep = env.current_episode
    missing = [k for k in REQUIRED_V033_KEYS + ["third_rgb", "head_rgb"] if k not in obs]
    assert not missing, f"missing v0.3.3 obs keys {missing}; got {list(obs.keys())}"
    frames, log = [], []
    t_total = 0
    pretty = Overlay(args.third_h or args.third_res) if args.pretty else None
    for stage, (skill, kw) in enumerate(PLAN):
        runner = runners[skill]
        runner.reset()
        prev_a = None
        n = 0
        reason = "timeout"
        while n < MAX_SKILL_STEPS[skill]:
            o22 = obs_to_2022(skill, obs, targ_idx=0, zero_nav_to_skill=args.zero_nav_to_skill, **kw)
            # 2022 TP-SRL checks termination BEFORE acting: nav on previous action,
            # pick/place on current obs.
            if n > 0 and skill_done(skill, prev_a if skill == "nav" else None, o22):
                reason = "skill_done"
                break
            a = runner.act(o22)
            holding = bool(np.asarray(obs[K_HOLD]).reshape(-1)[0] > 0.5)
            obs = env.step(action_to_v033(skill, a, holding))
            prev_a = a
            n += 1
            t_total += 1
            m = env.get_metrics()
            if t_total % args.frame_every == 0 and pretty is not None:
                frames.append(pretty(obs["third_rgb"], to_uint8_rgb(obs["head_rgb"]), depth_vis(obs["head_depth"]),
                                     stage, set(range(stage)), bool(obs[K_HOLD][0] > 0.5), _obj_goal_dist(m)))
            elif t_total % args.frame_every == 0:
                frames.append(
                    compose_frame(
                        obs,
                        [
                            f"ep {ep.episode_id}  stage {stage + 1}/4: {skill.upper()}"
                            + (" (to goal)" if kw.get("nav_to_goal") else ""),
                            f"step {n}  holding={int(obs[K_HOLD][0])}  "
                            f"obj->goal={_obj_goal_dist(m):.2f}m",
                            "2022 challenge skills (depth-only RL) in habitat v0.3.3",
                        ],
                        args.third_res,
                    )
                )
            if env.episode_over:
                reason = "episode_over"
                break
        m = env.get_metrics()
        info = dict(stage=stage, skill=skill, steps=n, reason=reason,
                    holding=int(obs[K_HOLD][0]), pddl_success=m.get("pddl_success"))
        print("   ", info)
        log.append(info)
        if env.episode_over:
            break
        if skill == "pick" and not info["holding"] and args.stop_on_fail:
            break
    # a short tail so the final state is visible
    m = env.get_metrics()
    succ = bool(m.get("pddl_success", 0.0))
    for _ in range(args.tail_frames if pretty is None else max(args.tail_frames, 60)):
        if env.episode_over:
            if pretty is None:
                break
        else:
            obs = env.step(action_to_v033("place", np.zeros(8, np.float32), bool(obs[K_HOLD][0] > 0.5)))
        if pretty is not None:
            frames.append(pretty(obs["third_rgb"], to_uint8_rgb(obs["head_rgb"]), depth_vis(obs["head_depth"]),
                                 3, set(range(4)) if succ else set(range(stage)), bool(obs[K_HOLD][0] > 0.5),
                                 _obj_goal_dist(env.get_metrics()),
                                 caption="Task complete" if succ else "Task failed", finished=succ))
        else:
            frames.append(compose_frame(obs, [f"ep {ep.episode_id}  done"], args.third_res))
    m = env.get_metrics()
    return frames, log, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="data/models")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dataset", default="rearrange_easy")
    ap.add_argument("--episode-ids", nargs="*", default=None)
    ap.add_argument("--num-episodes", type=int, default=5)
    ap.add_argument("--until-success", action="store_true")
    ap.add_argument("--out-dir", default="videos/rl_skills")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-every", type=int, default=1)
    ap.add_argument("--tail-frames", type=int, default=30)
    ap.add_argument("--head-res", type=int, default=256)
    ap.add_argument("--third-res", type=int, default=512)
    ap.add_argument("--third-w", type=int, default=0)
    ap.add_argument("--third-h", type=int, default=0)
    ap.add_argument("--only-succ", action="store_true", help="write videos only for successful episodes")
    ap.add_argument("--pretty", action="store_true", help="Roboto portfolio overlay (src/rl_skills/overlay.py)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--zero-nav-to-skill", action="store_true",
                    help="feed zeros for nav_to_skill (what the 2022 TP-SRL eval did)")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--dry-config", action="store_true")
    ap.add_argument("--fake-env", action="store_true")
    ap.add_argument("opts", nargs="*", default=[])
    args = ap.parse_args()

    if args.fake_env:
        return fake_env_smoke(args)

    cfg = make_config(args)
    check_config(cfg)
    if args.dry_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg.habitat.simulator.agents.main_agent))
        print(OmegaConf.to_yaml(cfg.habitat.task.actions))
        print("lab_sensors:", list(cfg.habitat.task.lab_sensors.keys()))
        print("concur_render", cfg.habitat.simulator.concur_render, "data_path", cfg.habitat.dataset.data_path)
        print("DRY CONFIG OK")
        return 0

    import torch
    import habitat

    device = args.device if torch.cuda.is_available() else "cpu"
    runners = {s: SkillRunner(s, args.models_dir, device, deterministic=not args.stochastic)
               for s in ("nav", "pick", "place")}
    with habitat.Env(config=cfg) as env:
        if args.episode_ids:
            env.episodes = [e for e in env.episodes if str(e.episode_id) in set(args.episode_ids)]
        n_eps = len(args.episode_ids) if args.episode_ids else args.num_episodes
        n_succ = 0
        for i in range(n_eps):
            t0 = time.time()
            frames, log, m = run_episode(env, runners, args)
            ep_id = env.current_episode.episode_id
            succ = bool(m.get("pddl_success", 0.0))
            n_succ += succ
            print(f"episode {ep_id}: pddl_success={succ} metrics="
                  f"{ {k: v for k, v in m.items() if isinstance(v, (int, float, bool))} } ({time.time() - t0:.0f}s)")
            if succ or not args.only_succ:
                write_video(frames, os.path.join(args.out_dir, f"ep{ep_id}_{'SUCC' if succ else 'fail'}.mp4"), args.fps)
            if succ and args.until_success:
                break
        print(f"success {n_succ}/{i + 1}")
    return 0


def fake_env_smoke(args):
    """Run the full adapter+policy chain on synthetic v0.3.3-shaped obs (CPU)."""
    rng = np.random.default_rng(0)

    def fake_obs(holding=0.0):
        return {
            "head_depth": rng.random((256, 256, 1), dtype=np.float32),
            "head_rgb": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "third_rgb": rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
            "obj_start_gps_compass": np.array([2.0, 0.3], np.float32),
            "obj_goal_gps_compass": np.array([3.0, -0.5], np.float32),
            "obj_start_sensor": np.array([0.4, 0.1, -0.2], np.float32),
            "obj_goal_sensor": np.array([0.5, -0.1, 0.1], np.float32),
            "joint": np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005], np.float32),
            "is_holding": np.array([holding], np.float32),
            "relative_resting_position": np.array([0.01, 0.0, 0.02], np.float32),
        }

    runners = {s: SkillRunner(s, args.models_dir, "cpu") for s in ("nav", "pick", "place")}
    for skill, kw in PLAN:
        runners[skill].reset()
        obs = fake_obs(1.0 if skill == "place" else 0.0)
        for n in range(3):
            o22 = obs_to_2022(skill, obs, **kw)
            a = runners[skill].act(o22)
            act = action_to_v033(skill, a, bool(obs["is_holding"][0]))
        print(skill, kw, {k: v.shape for k, v in o22.items()}, "->",
              {k: np.round(v, 3).tolist() for k, v in act["action_args"].items()}, act["action"],
              "done?", skill_done(skill, a, o22))
    f = compose_frame(fake_obs(), ["test"], 512)
    print("frame", f.shape, f.dtype)
    print("FAKE ENV OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
