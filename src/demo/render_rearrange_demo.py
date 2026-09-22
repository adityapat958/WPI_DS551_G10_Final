#!/usr/bin/env python3
"""
Portfolio visualization: Fetch robot navigating a ReplicaCAD apartment,
opening a chest-of-drawers, and picking up a YCB object.

Pure habitat_sim (no habitat-lab / RL policy needed). Motion is scripted
kinematically so the demo is deterministic and clean for a portfolio reel.

Usage (on a GPU node, headless EGL build of habitat-sim):
    python src/demo/render_rearrange_demo.py --inspect          # introspect ids
    python src/demo/render_rearrange_demo.py --out videos/rearrange_demo.mp4
"""
import argparse
import os
import sys
import numpy as np

import habitat_sim
import magnum as mn

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "versioned_data")
DATA = os.path.abspath(DATA)
SCENE_DATASET = os.path.join(DATA, "replica_cad_dataset", "replicaCAD.scene_dataset_config.json")
FETCH_URDF = os.path.join(DATA, "hab_fetch", "robots", "hab_fetch.urdf")
YCB_DATASET = os.path.join(DATA, "ycb", "ycb.scene_dataset_config.json")


def make_cfg(scene_id, width, height):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = SCENE_DATASET
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True

    # single cinematic RGB camera; we drive its pose each frame via look_at
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "rgb"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [height, width]
    cam.position = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def look_at(eye, target, up=mn.Vector3(0, 1, 0)):
    """Camera transform (habitat sensor looks down -Z)."""
    return mn.Matrix4.look_at(mn.Vector3(*eye), mn.Vector3(*target), up)


def set_camera(sim, eye, target):
    sim.get_agent(0).scene_node.transformation = look_at(eye, target)


def find_ao(sim, substr):
    aom = sim.get_articulated_object_manager()
    for h in aom.get_object_handles():
        if substr.lower() in h.lower():
            return aom.get_object_by_handle(h)
    return None


def add_ycb(sim, name="002_master_chef_can"):
    otm = sim.get_object_template_manager()
    otm.load_configs(os.path.join(DATA, "ycb", "configs"))
    handles = otm.get_template_handles(name)
    if not handles:
        raise RuntimeError(f"YCB template {name} not found")
    rom = sim.get_rigid_object_manager()
    return rom.add_object_by_template_handle(handles[0])


def inspect(sim, robot, drawer):
    print("=== ROBOT links/joints ===")
    print("num joint pos:", len(robot.joint_positions))
    for lid in robot.get_link_ids():
        print(f"  link {lid}: {robot.get_link_name(lid)}")
    print("joint_positions:", np.round(robot.joint_positions, 3).tolist())
    if drawer is not None:
        print("=== DRAWER ===")
        print("handle:", drawer.handle)
        print("num joint pos:", len(drawer.joint_positions))
        for lid in drawer.get_link_ids():
            print(f"  link {lid}: {drawer.get_link_name(lid)}")
        print("joint_positions:", np.round(drawer.joint_positions, 3).tolist())


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--out", default="videos/rearrange_demo.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--drawer-joint", type=int, default=0, help="which drawer dof to open")
    ap.add_argument("--open-amt", type=float, default=0.4)
    args = ap.parse_args()

    sim = habitat_sim.Simulator(make_cfg(args.scene, args.width, args.height))

    aom = sim.get_articulated_object_manager()
    robot = aom.add_articulated_object_from_urdf(FETCH_URDF, fixed_base=False)
    robot.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    drawer = find_ao(sim, "chestOfDrawers")
    can = add_ycb(sim)

    if args.inspect:
        inspect(sim, robot, drawer)
        sim.close()
        return

    import imageio.v2 as imageio

    # --- geometry: place robot near the drawer ---
    dpos = drawer.translation
    approach_start = mn.Vector3(dpos[0] + 2.2, dpos[1], dpos[2] + 0.3)
    approach_end = mn.Vector3(dpos[0] + 1.0, dpos[1], dpos[2] + 0.3)
    # face toward drawer (-X here)
    face = mn.Quaternion.rotation(mn.Rad(mn.Deg(-90)), mn.Vector3(0, 1, 0))

    def place_robot(p):
        robot.translation = p
        robot.rotation = face

    place_robot(approach_start)

    # arm poses (indices tuned after --inspect; defaults lift shoulder/elbow)
    base_joints = list(robot.joint_positions)
    tuck = list(base_joints)
    reach = list(base_joints)
    for idx in range(min(4, len(reach))):
        reach[idx] = 0.6  # crude reach; refine per inspect

    # place can inside the drawer (will be revealed on open)
    can.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    drawer_local = mn.Vector3(dpos[0] - 0.1, dpos[1] + 0.5, dpos[2])
    can.translation = drawer_local

    frames = []

    def render(cam_eye, cam_target, label):
        set_camera(sim, cam_eye, cam_target)
        sim.step_physics(1.0 / args.fps)
        obs = sim.get_sensor_observations()
        img = np.array(obs["rgb"])[:, :, :3].copy()
        _annotate(img, label)
        frames.append(img)

    def cam_for(robot_pos, ang, dist=3.2, hgt=1.8):
        eye = (robot_pos[0] + dist * np.cos(ang), robot_pos[1] + hgt, robot_pos[2] + dist * np.sin(ang))
        tgt = (robot_pos[0], robot_pos[1] + 0.8, robot_pos[2])
        return eye, tgt

    N_NAV, N_OPEN, N_PICK, N_LIFT = 60, 45, 30, 45

    # Phase 1: navigate
    for i in range(N_NAV):
        t = i / (N_NAV - 1)
        p = mn.Vector3(lerp(approach_start[0], approach_end[0], t),
                       approach_start[1],
                       lerp(approach_start[2], approach_end[2], t))
        place_robot(p)
        e, tg = cam_for(p, ang=lerp(0.3, 0.9, t))
        render(e, tg, "1. Navigate to cabinet")

    rp = approach_end
    # Phase 2: open drawer (+ raise arm)
    for i in range(N_OPEN):
        t = i / (N_OPEN - 1)
        jp = list(drawer.joint_positions)
        jp[args.drawer_joint] = lerp(0.0, args.open_amt, t)
        drawer.joint_positions = jp
        robot.joint_positions = [lerp(tuck[k], reach[k], t) for k in range(len(tuck))]
        e, tg = cam_for(rp, ang=lerp(0.9, 1.2, t), dist=2.8)
        render(e, tg, "2. Open drawer")

    # Phase 3: reach + grasp
    for i in range(N_PICK):
        e, tg = cam_for(rp, ang=lerp(1.2, 1.5, i / (N_PICK - 1)), dist=2.4)
        render(e, tg, "3. Pick object")

    # Phase 4: lift object (can follows gripper)
    ee_lid = robot.get_link_ids()[-1]
    for i in range(N_LIFT):
        t = i / (N_LIFT - 1)
        gripper = robot.get_link_scene_node(ee_lid).absolute_translation
        can.translation = mn.Vector3(gripper[0], lerp(drawer_local[1], drawer_local[1] + 0.4, t), gripper[2])
        e, tg = cam_for(rp, ang=lerp(1.5, 2.1, t), dist=2.6)
        render(e, tg, "4. Lift & retrieve")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"WROTE {args.out}  ({len(frames)} frames)")
    sim.close()


def _annotate(img, label):
    try:
        import cv2
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(img, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 120), 2)
        cv2.putText(img, "Habitat 2.0  |  Fetch  |  ReplicaCAD", (w - 430, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
