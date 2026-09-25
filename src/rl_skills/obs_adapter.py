#!/usr/bin/env python3
"""
Adapters between habitat-lab v0.3.3 RearrangeSim / RearrangePddlTask
observations+actions and the 2022 (rearrange_challenge_2022) skill policies.

Observation mapping (v0.3.3 sensor -> 2022 key), all verified by source
comparison (see COMPAT_REPORT.md):

  head_depth (256x256x1, float, normalize_depth=True over [0,10] m)
                                  -> robot_head_depth         (identical camera 'head')
  obj_start_gps_compass[2i:2i+2]  -> object_to_agent_gps_compass  (nav to object)
  obj_goal_gps_compass[2i:2i+2]   -> object_to_agent_gps_compass  (nav to goal)
        both = [rho, -phi] of target in robot *base* frame, identical formula to
        2022 TargetOrGoalStartPointGoalSensor.
  (synthesised)                   -> nav_to_skill (8-d one-hot, 2022 PDDL action index)
  obj_start_sensor[3i:3i+3]       -> obj_start_sensor  (target START pos in EE frame)
  obj_goal_sensor[3i:3i+3]        -> obj_goal_sensor   (goal pos in EE frame)
  joint, is_holding, relative_resting_position -> same names, identical semantics.

Action mapping (2022 policy output -> v0.3.3 Env.step dict):
  nav   a[0:2] -> base_vel = [lin, ang] (clip [-1,1], x lin_speed=10 / ang_speed=10,
                 integrated over 1/ctrl_freq = 1/120 s per env step -> same as 2022)
  pick/place a[0:7] -> arm_action (ArmRelPosAction: clip[-1,1] * 0.0125 rad delta on
                 motor targets), a[7] -> grip_action (SuctionGraspAction: >=0 grasp,
                 <0 release).
"""
from typing import Dict, Optional

import numpy as np

# Index of each PDDL action in the 2022 replica_cad domain
# (habitat/tasks/rearrange/multi_task/domain_configs/replica_cad.yaml, rc22 branch):
# nav, nav_to_receptacle, pick, place, open_fridge, close_fridge, open_cab, close_cab
NAV_TO_SKILL_IDX = {
    "nav": 0,
    "nav_to_receptacle": 1,
    "pick": 2,
    "place": 3,
    "open_fridge": 4,
    "close_fridge": 5,
    "open_cab": 6,
    "close_cab": 7,
}

# v0.3.3 observation keys (single agent => no "agent_0_" prefix)
K_DEPTH = "head_depth"
K_START_GPS = "obj_start_gps_compass"
K_GOAL_GPS = "obj_goal_gps_compass"
K_START_EE = "obj_start_sensor"
K_GOAL_EE = "obj_goal_sensor"
K_JOINT = "joint"
K_HOLD = "is_holding"
K_REST = "relative_resting_position"

REQUIRED_V033_KEYS = [K_DEPTH, K_START_GPS, K_GOAL_GPS, K_START_EE, K_GOAL_EE, K_JOINT, K_HOLD, K_REST]


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _depth(obs) -> np.ndarray:
    d = _f32(obs[K_DEPTH])
    if d.ndim == 2:
        d = d[..., None]
    assert d.shape == (256, 256, 1), f"head_depth must be 256x256x1, got {d.shape}"
    # v0.3.3 HabitatSimDepthSensor with normalize_depth=True already returns
    # clip(depth, 0, 10)/10 in [0,1] -- exactly what the 2022 net was trained on.
    # Guard against a config that forgot normalize_depth:
    if d.max() > 1.0 + 1e-3:
        d = np.clip(d, 0.0, 10.0) / 10.0
    return d


def obs_to_2022(
    skill: str,
    obs: Dict[str, np.ndarray],
    targ_idx: int = 0,
    nav_to_goal: bool = False,
    nav_to_skill: Optional[str] = None,
    zero_nav_to_skill: bool = False,
) -> Dict[str, np.ndarray]:
    """Convert one (un-batched) v0.3.3 observation dict into the 2022 skill obs.

    :param targ_idx: which rearrange target (rearrange_easy has exactly one -> 0).
    :param nav_to_goal: nav only. False = navigate to the object's start position
        (before pick), True = navigate to the object's goal position (before place).
    :param nav_to_skill: nav only. PDDL skill the navigation precedes; default
        'pick' when nav_to_goal=False else 'place'.
    :param zero_nav_to_skill: nav only. Feed zeros instead of one-hot. This is what
        the official 2022 TP-SRL baseline did at eval time (AddVirtualKeys
        obs-transform, configs/methods/tp_srl.yaml); one-hot is what nav.pth saw
        during training.
    """
    out = {"robot_head_depth": _depth(obs), "joint": _f32(obs[K_JOINT]).reshape(7)}
    if skill == "nav":
        k = K_GOAL_GPS if nav_to_goal else K_START_GPS
        out["object_to_agent_gps_compass"] = _f32(obs[k]).reshape(-1, 2)[targ_idx]
        one_hot = np.zeros(8, dtype=np.float32)
        if not zero_nav_to_skill:
            name = nav_to_skill or ("place" if nav_to_goal else "pick")
            one_hot[NAV_TO_SKILL_IDX[name]] = 1.0
        out["nav_to_skill"] = one_hot
    elif skill in ("pick", "place"):
        if skill == "pick":
            out["obj_start_sensor"] = _f32(obs[K_START_EE]).reshape(-1, 3)[targ_idx]
        else:
            out["obj_goal_sensor"] = _f32(obs[K_GOAL_EE]).reshape(-1, 3)[targ_idx]
        out["is_holding"] = _f32(obs[K_HOLD]).reshape(1)
        out["relative_resting_position"] = _f32(obs[K_REST]).reshape(3)
    else:
        raise ValueError(skill)
    return out


def obs_from_sim(
    skill: str, env, head_depth: np.ndarray, targ_idx: int = 0, nav_to_goal: bool = False, **kw
) -> Dict[str, np.ndarray]:
    """Fallback: recompute the 1-D 2022 observations directly from the v0.3.3
    RearrangeSim (only needs the head_depth sensor to exist). Mirrors the 2022
    sensor code line by line (rearrange_sensors.py / nav_to_obj_sensors.py)."""
    from habitat.tasks.utils import cartesian_to_polar

    sim = env.sim  # habitat.Env
    task = env.task
    agent = sim.get_agent_data(0).articulated_agent
    grasp = sim.get_agent_data(0).grasp_mgr
    base_T = agent.base_transformation
    ee_T = agent.ee_transform()
    obj_start = np.asarray(sim.get_target_objs_start())[targ_idx]
    goal_pos = np.asarray(sim.get_targets()[1])[targ_idx]

    obs = {
        K_DEPTH: head_depth,
        K_JOINT: np.asarray(agent.arm_joint_pos, dtype=np.float32),
        K_HOLD: np.array([float(grasp.is_grasped)], dtype=np.float32),
    }
    local_ee = base_T.inverted().transform_point(ee_T.translation)
    obs[K_REST] = (np.asarray(task.desired_resting) - np.asarray(local_ee)).astype(np.float32)
    inv_ee = ee_T.inverted()
    obs[K_START_EE] = np.asarray(inv_ee.transform_point(obj_start), dtype=np.float32)
    obs[K_GOAL_EE] = np.asarray(inv_ee.transform_point(goal_pos), dtype=np.float32)
    for key, p in ((K_START_GPS, obj_start), (K_GOAL_GPS, goal_pos)):
        v = base_T.inverted().transform_point(p)
        rho, phi = cartesian_to_polar(v[0], v[1])
        obs[key] = np.array([rho, -phi], dtype=np.float32)
    return obs_to_2022(skill, obs, 0, nav_to_goal, **kw)


def action_to_v033(
    skill: str,
    a: np.ndarray,
    is_holding: bool,
    include_base: bool = True,
    include_arm: bool = True,
) -> dict:
    """Convert a 2022 policy action vector into a v0.3.3 `Env.step` action dict for
    the `fetch_suction_arm_base(_stop)` action set (arm_action + base_velocity).

    Also applies the 2022 TP-SRL skill-level grip masking
    (habitat_baselines/rl/hrl/skills/{pick,place,skill}.py, rc22 branch):
      nav   : arm frozen (zero deltas), grip keeps current hold state
      pick  : once holding, grip forced to +1 (never release)
      place : once released, grip forced to -1 (never re-grasp)
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    arm = np.zeros(7, dtype=np.float32)
    grip = np.array([1.0 if is_holding else -1.0], dtype=np.float32)
    base = np.zeros(2, dtype=np.float32)
    if skill == "nav":
        assert a.shape == (2,)
        base = np.clip(a, -1.0, 1.0)
    elif skill in ("pick", "place"):
        assert a.shape == (8,)
        arm = np.clip(a[:7], -1.0, 1.0)
        g = float(a[7])
        if skill == "pick" and is_holding:
            g = 1.0
        if skill == "place" and not is_holding:
            g = -1.0
        grip = np.array([g], dtype=np.float32)
    else:
        raise ValueError(skill)
    names, args = [], {}
    if include_arm:
        names.append("arm_action")
        args["arm_action"] = arm
        args["grip_action"] = grip
    if include_base:
        names.append("base_velocity")
        args["base_vel"] = base
    return {"action": tuple(names), "action_args": args}


# ---------------------------------------------------------------------------
# Skill termination, replicating 2022 TP-SRL (tp_srl.yaml + skills/*.py)
# ---------------------------------------------------------------------------
NAV_LIN_STOP = 0.067  # tp_srl.yaml NN_NAV.LIN_SPEED_STOP (training env used 0.1)
NAV_ANG_STOP = 0.067
AT_RESTING_THRESHOLD = 0.15
MAX_SKILL_STEPS = {"nav": 300, "pick": 200, "place": 200}


def skill_done(skill: str, a2022: np.ndarray, obs2022: Dict[str, np.ndarray]) -> bool:
    if skill == "nav":
        return abs(float(a2022[0])) < NAV_LIN_STOP and abs(float(a2022[1])) < NAV_ANG_STOP
    rest = float(np.linalg.norm(obs2022["relative_resting_position"]))
    holding = bool(obs2022["is_holding"][0] > 0.5)
    if skill == "pick":
        return holding and rest < AT_RESTING_THRESHOLD
    if skill == "place":
        return (not holding) and rest < AT_RESTING_THRESHOLD
    raise ValueError(skill)
