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


def joint_map(ao):
    """name -> (offset, ndof) into ao.joint_positions."""
    m = {}
    for lid in ao.get_link_ids():
        try:
            off = ao.get_link_joint_pos_offset(lid)
            n = ao.get_link_num_joint_pos(lid)
            if n > 0:
                m[ao.get_link_name(lid)] = (off, n)
        except Exception:
            pass
    return m


def set_joint(ao, jp, name, value, jm):
    if name in jm:
        off, n = jm[name]
        for k in range(n):
            jp[off + k] = value


def link_world_pos(ao, name):
    for lid in ao.get_link_ids():
        if ao.get_link_name(lid) == name:
            return ao.get_link_scene_node(lid).absolute_translation
    return None


def inspect(sim, robot, drawer):
    print("=== ROBOT links/joints ===")
    print("num joint pos:", len(robot.joint_positions))
    for lid in robot.get_link_ids():
        print(f"  link {lid}: {robot.get_link_name(lid)}")
    print("joint dof map:", joint_map(robot))
    print("joint_positions:", np.round(robot.joint_positions, 3).tolist())
    if drawer is not None:
        print("=== DRAWER ===")
        print("handle:", drawer.handle)
        print("joint dof map:", joint_map(drawer))
        for lid in drawer.get_link_ids():
            print(f"  link {lid}: {drawer.get_link_name(lid)}")


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
    ap.add_argument("--drawer-joint", type=int, default=0, help="(legacy) drawer dof index")
    ap.add_argument("--drawer-link", default="drawer_topR", help="which drawer link to open")
    ap.add_argument("--open-amt", type=float, default=0.4)
    ap.add_argument("--stand-dist", type=float, default=1.0, help="robot standoff from chest")
    ap.add_argument("--front-x", type=float, default=1.0, help="chest front axis x")
    ap.add_argument("--front-z", type=float, default=0.0, help="chest front axis z")
    ap.add_argument("--cam-dist", type=float, default=2.6)
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

    jm = joint_map(robot)
    print("robot joint map:", jm)
    print("drawer joint map:", joint_map(drawer) if drawer else None)

    # --- geometry ---
    dpos = drawer.translation
    # which drawer to open (a specific sliding drawer link)
    djm = joint_map(drawer)
    drawer_link = args.drawer_link if args.drawer_link in djm else (
        next(iter(djm)) if djm else None)
    dworld = link_world_pos(drawer, drawer_link) if drawer_link else dpos

    # robot stands in front of the chest, facing it
    front = mn.Vector3(args.front_x, 0.0, args.front_z)  # horizontal front axis
    fn = front / max(1e-6, front.length())
    stand = mn.Vector3(dpos[0] + fn[0] * args.stand_dist, dpos[1],
                       dpos[2] + fn[2] * args.stand_dist)
    approach_start = stand + fn * 1.6
    approach_end = stand
    yaw = np.degrees(np.arctan2(-fn[0], -fn[2]))  # face toward drawer
    face = mn.Quaternion.rotation(mn.Rad(np.radians(yaw)), mn.Vector3(0, 1, 0))

    def place_robot(p):
        robot.translation = p
        robot.rotation = face

    place_robot(approach_start)

    # arm poses using named Fetch joints
    tuck = list(robot.joint_positions)
    reach = list(tuck)
    set_joint(robot, reach, "torso_lift_link", 0.35, jm)
    set_joint(robot, reach, "shoulder_pan_link", 0.0, jm)
    set_joint(robot, reach, "shoulder_lift_link", 0.9, jm)
    set_joint(robot, reach, "upperarm_roll_link", 0.0, jm)
    set_joint(robot, reach, "elbow_flex_link", 1.1, jm)
    set_joint(robot, reach, "wrist_flex_link", 0.8, jm)
    torso = list(tuck)
    set_joint(robot, torso, "torso_lift_link", 0.35, jm)

    # place can inside the target drawer
    can.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    drawer_local = mn.Vector3(dworld[0], dworld[1] + 0.05, dworld[2])
    can.translation = drawer_local

    frames = []

    def flush_video():
        if not frames:
            print("NO FRAMES to write")
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"WROTE {args.out}  ({len(frames)} frames)")

    def render(cam_eye, cam_target, label):
        set_camera(sim, cam_eye, cam_target)
        sim.step_physics(1.0 / args.fps)
        obs = sim.get_sensor_observations()
        img = np.array(obs["rgb"])[:, :, :3].copy()
        _annotate(img, label)
        frames.append(img)

    def orbit_cam(center, ang, dist, hgt):
        eye = (center[0] + dist * np.cos(ang), center[1] + hgt, center[2] + dist * np.sin(ang))
        tgt = (center[0], center[1] + 0.6, center[2])
        return eye, tgt

    # camera focuses on the midpoint of robot stand and drawer
    focus = mn.Vector3((stand[0] + dpos[0]) / 2, dpos[1], (stand[2] + dpos[2]) / 2)
    base_ang = np.arctan2(fn[2], fn[0])  # look from the robot's side

    N_NAV, N_OPEN, N_PICK, N_LIFT = 60, 45, 30, 45

    def open_drawer(t):
        if drawer_link and drawer_link in djm:
            jp = list(drawer.joint_positions)
            set_joint(drawer, jp, drawer_link, lerp(0.0, args.open_amt, t), djm)
            drawer.joint_positions = jp

    try:
        # Phase 1: navigate up to the chest
        for i in range(N_NAV):
            t = i / (N_NAV - 1)
            p = approach_start + (approach_end - approach_start) * t
            place_robot(p)
            e, tg = orbit_cam(focus, base_ang + lerp(-0.5, -0.15, t), args.cam_dist + 0.6, 1.9)
            render(e, tg, "1. Navigate to cabinet")

        # Phase 2: raise torso + open drawer
        for i in range(N_OPEN):
            t = i / (N_OPEN - 1)
            open_drawer(t)
            robot.joint_positions = [lerp(tuck[k], torso[k], t) for k in range(len(tuck))]
            e, tg = orbit_cam(focus, base_ang + lerp(-0.15, 0.1, t), args.cam_dist, 1.7)
            render(e, tg, "2. Open drawer")

        # Phase 3: reach arm into drawer
        for i in range(N_PICK):
            t = i / (N_PICK - 1)
            robot.joint_positions = [lerp(torso[k], reach[k], t) for k in range(len(tuck))]
            e, tg = orbit_cam(focus, base_ang + lerp(0.1, 0.3, t), args.cam_dist - 0.3, 1.6)
            render(e, tg, "3. Pick object")

        # Phase 4: grasp + lift (can follows gripper)
        gl = None
        for lid in robot.get_link_ids():
            if robot.get_link_name(lid) == "gripper_link":
                gl = lid
        for i in range(N_LIFT):
            t = i / (N_LIFT - 1)
            if gl is not None:
                try:
                    g = robot.get_link_scene_node(gl).absolute_translation
                    can.translation = mn.Vector3(g[0], g[1], g[2])
                except Exception:
                    pass
            # gently lower torso to lift can out and up
            robot.joint_positions = [lerp(reach[k], torso[k], t) for k in range(len(tuck))]
            e, tg = orbit_cam(focus, base_ang + lerp(0.3, 0.7, t), args.cam_dist, 1.8)
            render(e, tg, "4. Lift & retrieve")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"PHASE ERROR (writing partial video): {exc}")

    flush_video()
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
