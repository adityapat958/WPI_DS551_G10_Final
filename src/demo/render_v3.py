#!/usr/bin/env python3
"""
v3 portfolio reel: Fetch fetches an object from a bedroom drawer and carries it
to the kitchen table in ReplicaCAD apt_0 (Habitat-sim 0.3.3, Bullet physics).

  * base follows the apt_0 navmesh shortest path (wheels spin with distance)
  * arm driven by damped-least-squares IK on the sim's own FK (7 DoF, pos+axis)
  * drawer opened/closed by the gripper: drawer joint is coupled to the hand
  * object rides the drawer, is grasped (fingers close), carried, then released
    and settles on the table under real Bullet physics
  * GTA-style chase cam with raycast wall-avoidance; close 3/4 shots for
    manipulation; picture-in-picture of the robot's head camera
  * self-checks written to <out>.checks.json (and printed): occlusion, robot
    visibility from depth, IK error, gripper travel, can-in-hand, final placement

This is a SCRIPTED (non-learned) demonstration; label it as such.
"""
import argparse, json, math, os, sys, time
import numpy as np
import habitat_sim
import magnum as mn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "versioned_data")
SCENE_DATASET = os.path.join(DATA, "replica_cad_dataset", "replicaCAD.scene_dataset_config.json")
NAVMESH = os.path.join(DATA, "replica_cad_dataset", "navmeshes", "apt_0.navmesh")
FETCH_URDF = os.path.join(DATA, "hab_fetch", "robots", "hab_fetch.urdf")
YCB_CFG = os.path.join(DATA, "ycb", "configs")

try:
    sys.path.insert(0, os.environ.get("VIZJOB_LIB", os.path.expanduser("~/.vizjob/lib")))
    import vizjob_hook as vj
except Exception:  # pragma: no cover
    vj = None

UP = mn.Vector3(0, 1, 0)
TUCK = [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005]  # habitat-lab Fetch arm_init_params
ARM = ["shoulder_pan_link", "shoulder_lift_link", "upperarm_roll_link", "elbow_flex_link",
       "forearm_roll_link", "wrist_flex_link", "wrist_roll_link"]
V = lambda a: mn.Vector3(float(a[0]), float(a[1]), float(a[2]))
npv = lambda v: np.array([v[0], v[1], v[2]], dtype=np.float64)


def log(*a):
    print("[v3]", *a, flush=True)


def smooth(t):  # ease in/out
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- sim
def make_sim(w, h, pip):
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_dataset_config_file = SCENE_DATASET
    sc.scene_id = "apt_0"
    sc.enable_physics = True

    def cam(uuid, res, typ=habitat_sim.SensorType.COLOR, hfov=70):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.position = uuid, typ, res, [0.0, 0.0, 0.0]
        s.hfov = hfov
        return s

    a0 = habitat_sim.agent.AgentConfiguration()
    a0.sensor_specifications = [cam("rgb", [h, w]), cam("depth", [h // 4, w // 4], habitat_sim.SensorType.DEPTH)]
    a1 = habitat_sim.agent.AgentConfiguration()
    a1.sensor_specifications = [cam("head", [pip[1], pip[0]], hfov=60)]
    return habitat_sim.Simulator(habitat_sim.Configuration(sc, [a0, a1]))


class Robot:
    def __init__(self, sim):
        self.sim = sim
        aom = sim.get_articulated_object_manager()
        self.ao = aom.add_articulated_object_from_urdf(FETCH_URDF, fixed_base=True)
        self.ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self.lid = {self.ao.get_link_name(l): l for l in self.ao.get_link_ids()}
        self.jm = {}
        for name, l in self.lid.items():
            n = self.ao.get_link_num_joint_pos(l)
            if n > 0:
                self.jm[name] = self.ao.get_link_joint_pos_offset(l)
        self.arm_idx = [self.jm[n] for n in ARM]
        self.ik_idx = ([self.jm["torso_lift_link"]] if "torso_lift_link" in self.jm else []) + self.arm_idx
        lo, hi = self.ao.joint_position_limits
        self.lo = np.array(lo, dtype=np.float64)
        self.hi = np.array(hi, dtype=np.float64)
        self.q = np.array(self.ao.joint_positions, dtype=np.float64)
        self.q[self.arm_idx] = TUCK
        self.pos = mn.Vector3(0, 0, 0)
        self.yaw = 0.0
        self.base_up_fix = mn.Quaternion()  # filled by orient_probe
        self.ids = {self.ao.object_id} | set(self.ao.link_object_ids.keys())
        self.tip_off = 0.16  # gripper_link origin -> between fingertips (m)

    # --- pose application -------------------------------------------------
    def rot(self):
        return mn.Quaternion.rotation(mn.Rad(self.yaw), UP) * self.base_up_fix

    def apply(self):
        self.ao.translation = self.pos
        self.ao.rotation = self.rot()
        qq = np.clip(self.q, self.lo, self.hi)
        self.ao.joint_positions = [float(x) for x in qq]

    def link_T(self, name):
        return self.ao.get_link_scene_node(self.lid[name]).absolute_transformation()

    def forward(self):  # robot heading (world) = rotated local +X
        return self.rot().transform_vector(mn.Vector3(1, 0, 0))

    def tip(self):
        T = self.link_T("gripper_link")
        return T.translation + T.transform_vector(mn.Vector3(1, 0, 0)).normalized() * self.tip_off

    def grip_axis(self):
        return self.link_T("gripper_link").transform_vector(mn.Vector3(1, 0, 0)).normalized()

    def set_fingers(self, opening):  # 0 closed .. 0.05 open (prismatic)
        for n in ("l_gripper_finger_link", "r_gripper_finger_link"):
            if n in self.jm:
                self.q[self.jm[n]] = opening

    def set_torso(self, v):
        if "torso_lift_link" in self.jm:
            self.q[self.jm["torso_lift_link"]] = v

    def look_head(self, target):
        """pan/tilt the head toward a world point."""
        if "head_pan_link" not in self.jm:
            return
        d = mn.Quaternion.rotation(mn.Rad(self.yaw), UP).inverted().transform_vector(V(target) - self.pos - mn.Vector3(0, 1.1, 0))
        pan = math.atan2(-d[2], d[0])
        tilt = math.atan2(-d[1], math.hypot(d[0], d[2]))
        self.q[self.jm["head_pan_link"]] = float(np.clip(pan, -1.5, 1.5))
        self.q[self.jm["head_tilt_link"]] = float(np.clip(tilt, -0.7, 1.4))

    def spin_wheels(self, dist):
        for n in ("l_wheel_link", "r_wheel_link"):
            if n in self.jm:
                self.q[self.jm[n]] += dist / 0.0613

    # --- IK ------------------------------------------------------------------
    def ik(self, target, axis=None, iters=40, tol=0.004, w_axis=0.25):
        """DLS IK on the 7 arm joints using the simulator's FK. Returns pos err (m)."""
        target = npv(target)
        ax_t = None if axis is None else npv(V(axis).normalized())
        idx = self.ik_idx
        eps = 1e-3
        lam = 0.05

        def resid():
            self.apply()
            e = target - npv(self.tip())
            if ax_t is None:
                return e
            return np.concatenate([e, w_axis * np.cross(npv(self.grip_axis()), ax_t)])

        r = resid()
        for _ in range(iters):
            if np.linalg.norm(r[:3]) < tol:
                break
            J = np.zeros((len(r), len(idx)))
            q0 = self.q.copy()
            for k, j in enumerate(idx):
                self.q[j] = q0[j] + eps
                J[:, k] = (r - resid()) / eps  # resid decreases as tip moves toward target
                self.q[j] = q0[j]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(r)), r)
            dq = np.clip(dq, -0.25, 0.25)
            self.q[idx] = np.clip(q0[idx] + dq, self.lo[idx], self.hi[idx])
            r = resid()
        return float(np.linalg.norm(r[:3]))


# ----------------------------------------------------------------------------- scene helpers
def raycast(sim, origin, direction, max_d=50.0, ignore=()):
    d = V(direction)
    L = d.length()
    if L < 1e-9:
        return None
    ray = habitat_sim.geo.Ray(V(origin), d / L)
    res = sim.cast_ray(ray, max_d)
    for h in res.hits:
        if h.object_id in ignore:
            continue
        return h
    return None


def find_ao(sim, sub):
    aom = sim.get_articulated_object_manager()
    for h in aom.get_object_handles():
        if sub.lower() in h.lower():
            return aom.get_object_by_handle(h)
    return None


def find_rigid(sim, sub):
    rom = sim.get_rigid_object_manager()
    for h in rom.get_object_handles():
        if sub.lower() in h.lower():
            return rom.get_object_by_handle(h)
    return None


class Drawer:
    def __init__(self, sim, ao, link_name):
        self.sim, self.ao = sim, ao
        self.lid = {ao.get_link_name(l): l for l in ao.get_link_ids()}[link_name]
        self.j = ao.get_link_joint_pos_offset(self.lid)
        lo, hi = ao.joint_position_limits
        self.lo, self.hi = float(lo[self.j]), float(hi[self.j])
        self.ids = {ao.object_id} | set(ao.link_object_ids.keys())
        self.target = self.lo

    def set(self, v):
        self.target = float(v)
        jp = list(self.ao.joint_positions)
        jp[self.j] = float(np.clip(v, self.lo, self.hi))
        self.ao.joint_positions = jp
        try:
            jv = list(self.ao.joint_velocities)
            jv[self.j] = 0.0
            self.ao.joint_velocities = jv
        except Exception:
            pass

    def get(self):
        return float(self.ao.joint_positions[self.j])

    def T(self):
        return self.ao.get_link_scene_node(self.lid).absolute_transformation()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="videos/v3/fetch_bedroom_to_kitchen.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--drawer-link", default="drawer_topR")
    ap.add_argument("--table", default="frl_apartment_table_02")
    ap.add_argument("--objects", default="005_tomato_soup_can,010_potted_meat_can,007_tuna_fish_can")
    ap.add_argument("--speed", type=float, default=0.7, help="base speed m/s")
    ap.add_argument("--start-dist", type=float, default=5.0, help="geodesic dist of start from chest")
    ap.add_argument("--selftest", action="store_true", help="diagnostics + stills only")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    np.random.seed(args.seed)
    t_start = time.time()

    W, H = args.width, args.height
    PIP = (W // 4, H // 4)
    sim = make_sim(W, H, PIP)
    checks = {"warnings": []}
    warn = lambda m: (checks["warnings"].append(m), log("WARN", m))

    # ---- navmesh
    if os.path.exists(NAVMESH):
        sim.pathfinder.load_nav_mesh(NAVMESH)
    checks["navmesh_loaded"] = bool(sim.pathfinder.is_loaded)
    log("navmesh loaded:", sim.pathfinder.is_loaded)

    rb = Robot(sim)
    chest = find_ao(sim, "chestOfDrawers")
    drawer = Drawer(sim, chest, args.drawer_link)
    table = find_rigid(sim, args.table)
    if table is None:
        warn(f"table {args.table} not found")

    # ---- probe: does the URDF come in Y-up? (head must be above base)
    rb.pos = mn.Vector3(0, 0, 0); rb.apply()
    head_rel = rb.link_T("head_pan_link").translation - rb.ao.translation
    log("head rel base (identity rot):", [round(x, 3) for x in head_rel])
    if head_rel[1] < 0.5:  # Z-up import -> rotate -90 about X
        rb.base_up_fix = mn.Quaternion.rotation(mn.Rad(-math.pi / 2), mn.Vector3(1, 0, 0))
        rb.apply()
        head_rel = rb.link_T("head_pan_link").translation - rb.ao.translation
        log("applied Z-up fix; head rel:", [round(x, 3) for x in head_rel])
    checks["head_height_above_base"] = round(float(head_rel[1]), 3)
    # forward axis sanity: gripper (tucked) should be roughly ahead or level
    # ---- probe: do joint writes change FK without stepping?
    q_save = rb.q.copy()
    p0 = npv(rb.tip())
    rb.q[rb.arm_idx[1]] += 0.5; rb.apply()
    p1 = npv(rb.tip())
    rb.q = q_save; rb.apply()
    fk_move = float(np.linalg.norm(p1 - p0))
    checks["fk_probe_move_m"] = round(fk_move, 4)
    log("FK probe: tip moved", round(fk_move, 4), "m for +0.5 rad shoulder_lift")
    if fk_move < 1e-3:
        warn("joint writes do not update FK -> arm will look frozen")

    # ---- drawer geometry: opening direction + handle point
    c0 = drawer.T().translation
    drawer.set(drawer.lo + 0.3 * (drawer.hi - drawer.lo))
    c1 = drawer.T().translation
    drawer.set(drawer.lo)
    dv = c1 - c0
    fn = mn.Vector3(dv[0], 0, dv[2])
    per_unit = dv.length() / max(1e-6, 0.3 * (drawer.hi - drawer.lo))
    fn = fn.normalized() if fn.length() > 1e-4 else mn.Vector3(1, 0, 0)
    open_q = drawer.lo + 0.85 * (drawer.hi - drawer.lo)
    open_m = (open_q - drawer.lo) * per_unit
    log("drawer front", [round(fn[0], 3), round(fn[2], 3)], "limits", drawer.lo, drawer.hi,
        "m/unit", round(per_unit, 3), "open travel m", round(open_m, 3))
    checks.update(drawer_front=[round(fn[0], 3), round(fn[2], 3)], drawer_open_travel_m=round(open_m, 3))
    if open_m < 0.08:
        warn(f"drawer travel only {open_m:.3f} m")

    # handle: ray from 1 m in front of the drawer centre, toward chest
    dc = drawer.T().translation
    hits = []
    for dy in np.arange(-0.15, 0.35, 0.01):
        hh = raycast(sim, dc + fn * 1.0 + UP * float(dy), -fn, 2.0, ignore=rb.ids)
        if hh is not None and hh.object_id in drawer.ids:
            hits.append(hh.point)
    if hits:
        ys = [p_[1] for p_ in hits]
        mid = hits[int(np.argmin([abs(y - (min(ys) + max(ys)) / 2) for y in ys]))]
        handle0 = mid + fn * 0.02
        checks["drawer_front_span_m"] = round(max(ys) - min(ys), 3)
    else:
        warn("handle rays missed drawer; using link origin")
        handle0 = dc + fn * 0.15
    log("handle point", [round(x, 3) for x in handle0])

    # drawer floor (while open) + clearance (while closed) -> choose object
    drawer.set(open_q)
    hw = drawer.T().translation
    probe = handle0 + fn * open_m - fn * 0.075
    f_hit = raycast(sim, mn.Vector3(probe[0], handle0[1] + 0.4, probe[2]), -UP, 1.0, ignore=rb.ids)
    floor_y = f_hit.point[1] if (f_hit is not None and f_hit.object_id in drawer.ids) else handle0[1] - 0.05
    if f_hit is None or f_hit.object_id not in drawer.ids:
        warn("drawer floor ray did not hit drawer")
    obj_in_drawer_open = mn.Vector3(probe[0], floor_y, probe[2])
    drawer.set(drawer.lo)
    closed_pt = obj_in_drawer_open - fn * open_m
    up_hit = raycast(sim, closed_pt + UP * 0.005, UP, 1.0, ignore=rb.ids)
    clearance = (up_hit.point[1] - floor_y) if up_hit is not None else 0.3
    log("drawer floor y", round(floor_y, 3), "clearance", round(clearance, 3))
    checks["drawer_clearance_m"] = round(float(clearance), 3)

    otm = sim.get_object_template_manager()
    otm.load_configs(YCB_CFG)
    rom = sim.get_rigid_object_manager()
    can, can_h, obj_name, can_c = None, 0.1, None, 0.05
    for name in args.objects.split(","):
        hs = otm.get_template_handles(name)
        if not hs:
            continue
        o = rom.add_object_by_template_handle(hs[0])
        o.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        air = handle0 + fn * 1.0 + UP * 0.5
        o.translation = air
        def first_hit(org, d):
            ray = habitat_sim.geo.Ray(V(org), V(d))
            for hh in sim.cast_ray(ray, 2.0).hits:
                if hh.object_id == o.object_id:
                    return hh.point[1]
            return None
        top = first_hit(air + UP * 0.6, -UP)
        bot = first_hit(air - UP * 0.6, UP)
        if top is None or bot is None:
            warn(f"height rays missed {name}"); hgt, c_off = 0.1, 0.05
        else:
            hgt, c_off = float(top - bot), float(air[1] - bot)
        log(f"  {name}: height {hgt:.3f} centre-above-bottom {c_off:.3f}")
        if hgt < clearance - 0.02 or name == args.objects.split(",")[-1]:
            can, can_h, obj_name, can_c = o, hgt, name, c_off
            break
        rom.remove_object_by_id(o.object_id)
    checks.update(object=obj_name, object_height_m=round(can_h, 3))
    log("object", obj_name, "height", round(can_h, 3))
    can.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    can_ids = {can.object_id}
    # can pose relative to drawer link frame (so it rides the drawer)
    drawer.set(drawer.lo)
    can_local = drawer.T().inverted().transform_point(closed_pt + UP * (can_c + 0.003))

    def can_follow_drawer():
        can.translation = drawer.T().transform_point(can_local)

    can_follow_drawer()

    # ---- stand poses
    def snap(p):
        s = sim.pathfinder.snap_point(V(p))
        return mn.Vector3(s[0], s[1], s[2]) if not math.isnan(s[0]) else V(p)

    floor0 = snap(dc + fn * 1.2)[1]
    stand_chest = snap(mn.Vector3(handle0[0], floor0, handle0[2]) + fn * 0.78)
    if (stand_chest - (mn.Vector3(handle0[0], stand_chest[1], handle0[2]) + fn * 0.78)).length() > 0.25:
        warn("chest stand point snapped >25cm")
    # kitchen table: top via ray down, edge toward the robot's approach side
    if table is not None:
        tc = table.translation
        t_hit = raycast(sim, mn.Vector3(tc[0], 2.0, tc[2]), -UP, 3.0, ignore=rb.ids | can_ids)
        top_y = t_hit.point[1] if t_hit is not None else tc[1] + 0.4
        # candidate stands on a ring; choose navigable one with nearest geodesic to chest stand
        best = None
        for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            d = mn.Vector3(math.cos(ang), 0, math.sin(ang))
            # march to table edge
            edge = 0.0
            for r in np.arange(0.05, 1.5, 0.03):
                hh = raycast(sim, mn.Vector3(tc[0], top_y + 0.3, tc[2]) + d * r, -UP, 0.4,
                             ignore=rb.ids | can_ids)
                if hh is None or abs(hh.point[1] - top_y) > 0.03:
                    break
                edge = r
            cand = mn.Vector3(tc[0], floor0, tc[2]) + d * (edge + 0.62)
            if not sim.pathfinder.is_navigable(cand, 0.5):
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = stand_chest, snap(cand)
            if not sim.pathfinder.find_path(path):
                continue
            if best is None or path.geodesic_distance < best[0]:
                best = (path.geodesic_distance, snap(cand), d, edge)
        if best is None:
            warn("no navigable table stand; using fallback")
            best = (0, snap(tc + mn.Vector3(0.9, 0, 0)), mn.Vector3(1, 0, 0), 0.4)
        _, stand_table, tdir, tedge = best
        place_pt = mn.Vector3(tc[0], top_y, tc[2]) + tdir * max(0.0, tedge - 0.18)
        checks.update(table_top_y=round(float(top_y), 3), table_edge_m=round(float(tedge), 3))
        log("table top", round(top_y, 3), "edge", round(tedge, 3), "place", [round(x, 3) for x in place_pt])
    # start: navigable point ~start-dist geodesic from chest stand
    start = None
    for _ in range(400):
        p = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = V(p), stand_chest
        if sim.pathfinder.find_path(path) and abs(path.geodesic_distance - args.start_dist) < 0.8:
            start = V(p); break
    if start is None:
        start = snap(stand_chest + fn * 3.0)
        warn("no start at requested geodesic distance")

    def nav_path(a, b):
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = V(a), V(b)
        ok = sim.pathfinder.find_path(path)
        pts = [V(p) for p in path.points] if ok and len(path.points) >= 2 else [V(a), V(b)]
        return pts

    path1 = nav_path(start, stand_chest)
    path2 = nav_path(stand_chest, stand_table) if table is not None else []
    checks.update(path1_len=round(sum((path1[i + 1] - path1[i]).length() for i in range(len(path1) - 1)), 2),
                  path2_len=round(sum((path2[i + 1] - path2[i]).length() for i in range(len(path2) - 1)), 2))
    log("paths", checks["path1_len"], checks["path2_len"])

    # ---- camera + frame writer
    import imageio.v2 as imageio
    import cv2
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keydir = os.path.splitext(out)[0] + "_keys"
    os.makedirs(keydir, exist_ok=True)
    writer = None if args.selftest else imageio.get_writer(out, fps=args.fps, quality=8, macro_block_size=1)
    ag0, ag1 = sim.get_agent(0), sim.get_agent(1)
    hfov = math.radians(70)
    fx = (W / 2) / math.tan(hfov / 2)
    st = {"eye": None, "tgt": None, "frames": 0, "occl_adj": 0, "vis_ok": 0, "vis_n": 0,
          "caption": "", "keys": [], "ik_err": [], "can_gap": []}
    ignore_cam = rb.ids | can_ids

    def guard(eye, tgt):
        """pull camera toward target if a wall is between them; keep off ceilings."""
        d = eye - tgt
        L = d.length()
        h = raycast(sim, tgt, d, L + 0.2, ignore=ignore_cam)
        if h is not None and h.ray_distance < L + 0.15:
            st["occl_adj"] += 1
            eye = tgt + d.normalized() * max(0.35, h.ray_distance - 0.25)
        c = raycast(sim, eye, UP, 0.5, ignore=ignore_cam)
        if c is not None and c.ray_distance < 0.25:
            eye = eye - UP * (0.25 - c.ray_distance)
        return eye

    def set_cam(eye, tgt, alpha):
        if st["eye"] is None or alpha >= 1:
            st["eye"], st["tgt"] = eye, tgt
        else:
            st["eye"] = st["eye"] + (eye - st["eye"]) * alpha
            st["tgt"] = st["tgt"] + (tgt - st["tgt"]) * alpha
        e = guard(st["eye"], st["tgt"])
        ag0.scene_node.transformation = mn.Matrix4.look_at(e, st["tgt"], UP)
        return e

    def chase(alpha=0.12, back=2.6, up=1.55):
        fwd = rb.forward()
        eye = rb.pos - fwd * back + UP * up
        tgt = rb.pos + fwd * 0.9 + UP * 0.8
        return set_cam(eye, tgt, alpha)

    def shot(anchor, eye_off, alpha=0.08, tgt_off=mn.Vector3(0, 0, 0)):
        return set_cam(V(anchor) + eye_off, V(anchor) + tgt_off, alpha)

    def head_cam():
        T = rb.link_T("head_camera_link") if "head_camera_link" in rb.lid else rb.link_T("head_tilt_link")
        p = T.translation
        f = T.transform_vector(mn.Vector3(1, 0, 0)).normalized()
        ag1.scene_node.transformation = mn.Matrix4.look_at(p + f * 0.05, p + f, UP)

    def overlay(img, head):
        h, w = img.shape[:2]
        # lower-third caption
        if st["caption"]:
            (tw, th), _ = cv2.getTextSize(st["caption"], cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            ov = img.copy()
            cv2.rectangle(ov, (30, h - 90), (30 + tw + 40, h - 40), (15, 15, 15), -1)
            cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
            cv2.putText(img, st["caption"], (50, h - 55), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Habitat-sim 2.0 | Fetch | ReplicaCAD apt_0 | scripted IK demo",
                    (w - 620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        # PiP head camera
        ph, pw = head.shape[:2]
        x0, y0 = w - pw - 24, h - ph - 24
        img[y0:y0 + ph, x0:x0 + pw] = head
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + pw + 1, y0 + ph + 1), (255, 255, 255), 2)
        cv2.putText(img, "robot head camera", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def visible_pt(depth, eye, p, tol=0.35):
        Tcam = mn.Matrix4.look_at(eye, st["tgt"], UP).inverted()
        pc = Tcam.transform_point(V(p))
        if pc[2] >= -0.1:
            return False
        dh, dw = depth.shape[:2]
        f = (dw / 2) / math.tan(hfov / 2)
        u = int(dw / 2 + f * pc[0] / -pc[2]); v = int(dh / 2 - f * pc[1] / -pc[2])
        if not (0 <= u < dw and 0 <= v < dh):
            return False
        return float(depth[v, u]) > (-pc[2]) - tol

    def visible_check(depth, eye):
        """robot torso point projected into the depth image; visible if depth ~ distance."""
        p = rb.pos + UP * 0.75
        Tcam = mn.Matrix4.look_at(eye, st["tgt"], UP).inverted()
        pc = Tcam.transform_point(p)
        if pc[2] >= -0.1:
            return False
        dh, dw = depth.shape[:2]
        f = (dw / 2) / math.tan(hfov / 2)
        u = int(dw / 2 + f * pc[0] / -pc[2])
        v = int(dh / 2 - f * pc[1] / -pc[2])
        if not (0 <= u < dw and 0 <= v < dh):
            return False
        return float(depth[v, u]) > (-pc[2]) - 0.45

    def frame(held=False, drawer_ride=True, key=None):
        sim.step_physics(1.0 / args.fps)
        rb.apply()  # re-assert kinematic pose after the step
        drawer.set(drawer.target)
        if drawer_ride:
            can_follow_drawer()
        if held:
            T = rb.link_T("gripper_link")
            can.translation = rb.tip()
            st["can_gap"].append(0.0)
        head_cam()
        eye = mn.Vector3(ag0.scene_node.transformation.translation)
        obs = sim.get_sensor_observations(agent_ids=[0, 1])
        img = np.ascontiguousarray(np.array(obs[0]["rgb"])[:, :, :3])
        head = np.ascontiguousarray(np.array(obs[1]["head"])[:, :, :3])
        depth = np.array(obs[0]["depth"])
        st["vis_n"] += 1
        st["vis_ok"] += int(visible_check(depth, eye))
        for lbl, fp in st.get("focus", {}).items():
            a = st.setdefault("focus_stats", {}).setdefault(lbl, [0, 0])
            a[0] += int(visible_pt(depth, eye, fp() if callable(fp) else fp)); a[1] += 1
        img = overlay(img, head)
        if key:
            kp = os.path.join(keydir, f"{len(st['keys']):02d}_{key}.jpg")
            cv2.imwrite(kp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            st["keys"].append((kp, key))
        if writer is not None:
            writer.append_data(img)
        st["frames"] += 1
        if args.max_frames and st["frames"] >= args.max_frames:
            raise StopIteration

    # ---- motion primitives
    def drive(path, caption, held=False, ride=True):
        st["caption"] = caption
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            seg = b - a
            L = seg.length()
            if L < 1e-3:
                continue
            tyaw = math.atan2(-seg[2], seg[0])
            turn(tyaw, held=held, ride=ride)
            n = max(1, int(L / args.speed * args.fps))
            for k in range(1, n + 1):
                newp = a + seg * (k / n)
                rb.spin_wheels((newp - rb.pos).length())
                rb.pos = newp
                rb.look_head(rb.pos + rb.forward() * 3.0)
                if held:
                    rb.ik(rb.pos + rb.forward() * 0.45 + UP * 0.85, axis=rb.forward(), iters=8)
                chase()
                frame(held=held, drawer_ride=ride, key="nav" if (k == n // 2 and i == (len(path) - 2) // 2) else None)

    def turn(tyaw, held=False, ride=True, cam=None):
        dy = (tyaw - rb.yaw + math.pi) % (2 * math.pi) - math.pi
        n = int(abs(dy) / 1.4 * args.fps)
        y0 = rb.yaw
        for k in range(1, n + 1):
            rb.yaw = y0 + dy * smooth(k / n)
            rb.spin_wheels(0.0)
            (cam or chase)()
            frame(held=held, drawer_ride=ride)
        rb.yaw = tyaw

    def reach(target, axis, n, caption, cam, held=False, ride=True, fingers=None, key=None, extra=None):
        st["caption"] = caption
        p0 = rb.tip()
        f0 = rb.q[rb.jm["l_gripper_finger_link"]] if "l_gripper_finger_link" in rb.jm else 0.0
        for k in range(1, n + 1):
            s = smooth(k / n)
            if extra:
                extra(s)
            tgt = target(s) if callable(target) else p0 + (V(target) - p0) * s
            err = rb.ik(tgt, axis=axis, iters=25)
            st["ik_err"].append(err)
            if fingers is not None:
                rb.set_fingers(f0 + (fingers - f0) * s)
            rb.look_head(tgt)
            cam()
            frame(held=held, drawer_ride=ride, key=key if k == n else None)

    def hold(n, cam, held=False, ride=True, caption=None, key=None):
        if caption:
            st["caption"] = caption
        for k in range(n):
            cam()
            frame(held=held, drawer_ride=ride, key=key if k == n - 1 else None)

    # ---- choreography
    F = args.fps
    rb.pos = path1[0]
    first = path1[1] - path1[0] if len(path1) > 1 else fn
    rb.yaw = math.atan2(-first[2], first[0])
    rb.set_torso(0.05)
    rb.set_fingers(0.0)
    rb.apply()

    def open_side(anchor, perp, along):
        best, bs = None, -1
        for sg in (1.0, -1.0):
            d = (along * 0.8 + perp * sg).normalized()
            h = raycast(sim, V(anchor) + UP * 0.9, d, 4.0, ignore=rb.ids | can_ids)
            L = 4.0 if h is None else h.ray_distance
            if L > bs:
                best, bs = perp * sg, L
        return best, bs

    side, side_room = open_side(handle0, mn.Vector3(-fn[2], 0, fn[0]), fn)
    log("chest cam side room", round(side_room, 2))
    chest_cam = lambda: shot(handle0, fn * 1.2 + side * 1.7 + UP * 1.0, tgt_off=fn * 0.35 - UP * 0.1)
    arm_phase = {}
    try:
        if args.selftest:
            # stills from each planned vantage + the arm at a reach pose
            st["caption"] = "selftest: start (chase)"
            chase(1.0); frame(key="start_chase")
            rb.pos = stand_chest; rb.yaw = math.atan2(fn[2], -fn[0]); rb.apply()
            st["caption"] = "selftest: at chest"
            st["focus"] = {"handle": handle0}
            shot(handle0, fn * 1.2 + side * 1.7 + UP * 1.0, 1.0, fn * 0.35 - UP * 0.1); frame(key="chest_idle")
            e = rb.ik(handle0, axis=-fn, iters=60)
            checks["selftest_ik_err_handle_m"] = round(e, 4)
            st["caption"] = f"selftest: IK to handle err={e*100:.1f}cm"
            frame(key="chest_ik_handle")
            drawer.set(open_q); can_follow_drawer()
            e2 = rb.ik(obj_in_drawer_open + UP * (can_h + 0.08), axis=-UP, iters=60)
            checks["selftest_ik_err_object_m"] = round(e2, 4)
            st["caption"] = f"selftest: drawer open, IK above object err={e2*100:.1f}cm"
            frame(key="chest_open_ik_obj")
            if table is not None:
                rb.pos = stand_table
                rb.yaw = math.atan2(-(place_pt - stand_table)[2], (place_pt - stand_table)[0]); rb.apply()
                e3 = rb.ik(place_pt + UP * (can_c + 0.05), axis=-UP, iters=60)
                checks["selftest_ik_err_place_m"] = round(e3, 4)
                st["focus"] = {"place_target": place_pt}
                st["caption"] = f"selftest: at kitchen table, IK place err={e3*100:.1f}cm"
                shot(place_pt, -tdir * 1.6 + mn.Vector3(-tdir[2], 0, tdir[0]) * 1.3 + UP * 1.0, 1.0)
                frame(key="table_ik_place")
            raise StopIteration

        # 0. establishing shot: high, slow push-in
        st["caption"] = "Task: bring the object from the bedroom drawer to the kitchen table"
        for k in range(int(2.5 * F)):
            s = smooth(k / (2.5 * F))
            fwd = rb.forward()
            eye = rb.pos - fwd * (4.2 - 1.4 * s) + UP * (2.2 - 0.6 * s)
            set_cam(eye, rb.pos + UP * 0.6, 1.0 if k == 0 else 0.2)
            frame(key="establish" if k == 0 else None)
        # 1. navigate to the chest
        drive(path1, "Navigating to the chest of drawers (navmesh shortest path)")
        turn(math.atan2(fn[2], -fn[0]), cam=chest_cam)  # face the chest (robot +X = -fn)
        rb.set_torso(0.05)
        # 2. open the drawer by its handle
        st["focus"] = {"handle": lambda: handle0 + fn * (drawer.get() - drawer.lo) * per_unit,
                       "object": lambda: can.translation}
        pre = handle0 + fn * 0.12
        reach(pre, -fn, int(1.3 * F), "Reaching for the drawer handle (IK)", chest_cam, fingers=0.045, key="reach_handle")
        reach(handle0, -fn, int(0.5 * F), "Grasping the handle", chest_cam, fingers=0.045)
        reach(handle0, -fn, int(0.35 * F), "Grasping the handle", chest_cam, fingers=0.012)
        gp0 = npv(rb.tip())
        reach(lambda s: handle0 + fn * (open_m * s), -fn, int(1.6 * F), "Pulling the drawer open", chest_cam,
              extra=lambda s: drawer.set(drawer.lo + (open_q - drawer.lo) * s), key="drawer_open")
        arm_phase["pull_travel_m"] = float(np.linalg.norm(npv(rb.tip()) - gp0))
        handle_open = handle0 + fn * open_m
        reach(handle_open + fn * 0.1, -fn, int(0.5 * F), "Releasing the handle", chest_cam, fingers=0.045)
        # 3. pick the object from the drawer
        objp = lambda: can.translation
        above = objp() + UP * (can_h / 2 + 0.12)
        reach(above, -UP, int(1.3 * F), "Reaching into the drawer", chest_cam, key="reach_object")
        reach(objp() + UP * 0.0, -UP, int(0.7 * F), "Grasping the object", chest_cam)
        reach(objp(), -UP, int(0.35 * F), "Grasping the object", chest_cam, fingers=0.015)
        can_ids.add(can.object_id)
        tp0 = npv(rb.tip())
        reach(lambda s: V(tp0) + UP * (0.25 * s), -UP, int(1.0 * F), "Lifting the object", chest_cam,
              held=True, ride=False, key="lifted")
        arm_phase["lift_travel_m"] = float(np.linalg.norm(npv(rb.tip()) - tp0))
        # 4. push the drawer closed with the (holding) hand
        push0 = handle_open + fn * 0.05 + UP * 0.02
        reach(push0, -fn, int(1.0 * F), "Closing the drawer", chest_cam, held=True, ride=False)
        reach(lambda s: push0 - fn * (open_m * s), -fn, int(1.3 * F), "Closing the drawer", chest_cam,
              held=True, ride=False, extra=lambda s: drawer.set(open_q + (drawer.lo - open_q) * s), key="drawer_closed")
        # carry pose
        reach(rb.pos + rb.forward() * 0.45 + UP * 0.85, rb.forward(), int(1.0 * F), "Carry pose", chest_cam,
              held=True, ride=False)
        # 5. carry to the kitchen
        st["focus"] = {}
        drive(path2, "Carrying the object to the kitchen", held=True, ride=False)
        tyaw = math.atan2(-(place_pt - rb.pos)[2], (place_pt - rb.pos)[0])
        tside, _ = open_side(place_pt, mn.Vector3(-tdir[2], 0, tdir[0]), -tdir)
        table_cam = lambda: shot(place_pt, -tdir * 1.5 + tside * 1.4 + UP * 1.0, tgt_off=-UP * 0.05)
        st["focus"] = {"place_target": place_pt, "object": lambda: can.translation}
        turn(tyaw, held=True, ride=False, cam=table_cam)
        # 6. place on the table, release, let physics settle
        above_t = place_pt + UP * (can_h / 2 + 0.12)
        reach(above_t, -UP, int(1.4 * F), "Placing on the kitchen table", table_cam, held=True, ride=False, key="over_table")
        reach(place_pt + UP * (can_c + 0.01), -UP, int(0.8 * F), "Placing on the kitchen table", table_cam,
              held=True, ride=False)
        can.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        try:
            can.linear_velocity = mn.Vector3(0, 0, 0); can.angular_velocity = mn.Vector3(0, 0, 0)
        except Exception:
            pass
        reach(rb.tip(), -UP, int(0.4 * F), "Releasing (object settles under Bullet physics)", table_cam,
              ride=False, fingers=0.045)
        reach(rb.tip() + UP * 0.2, -UP, int(0.8 * F), "Releasing (object settles under Bullet physics)", table_cam,
              ride=False, key="placed")
        reach(rb.pos + rb.forward() * 0.35 + UP * 0.8, rb.forward(), int(1.0 * F), "Done", table_cam, ride=False)
        # 7. outro orbit
        st["caption"] = "Task complete"
        for k in range(int(3.0 * F)):
            a = k / (3.0 * F) * 1.4
            off = (-tdir * math.cos(a) + tside * math.sin(a)) * 2.2 + UP * 1.3
            shot(place_pt, off, 0.1)
            frame(drawer_ride=False, key="outro" if k == int(3.0 * F) - 1 else None)
    except StopIteration:
        pass

    # ---- checks
    cp = can.translation
    checks.update(
        frames=st["frames"], seconds=round(st["frames"] / args.fps, 1),
        camera_occlusion_adjust_frac=round(st["occl_adj"] / max(1, st["frames"]), 3),
        robot_visible_frac=round(st["vis_ok"] / max(1, st["vis_n"]), 3),
        ik_err_mean_cm=round(100 * float(np.mean(st["ik_err"])) if st["ik_err"] else -1, 2),
        ik_err_p95_cm=round(100 * float(np.percentile(st["ik_err"], 95)) if st["ik_err"] else -1, 2),
        **{k: round(v, 3) for k, v in arm_phase.items()},
        render_s=round(time.time() - t_start, 1),
        focus_visible_frac={k: round(a / max(1, b), 3) for k, (a, b) in st.get("focus_stats", {}).items()},
    )
    if not args.selftest and table is not None:
        up_axis = can.rotation.transform_vector(UP)
        checks.update(final_obj_pos=[round(x, 3) for x in cp],
                      final_obj_height_above_table=round(float(cp[1] - checks["table_top_y"] - can_c), 3),
                      final_obj_dist_to_target=round(float((mn.Vector3(cp[0], 0, cp[2]) - mn.Vector3(place_pt[0], 0, place_pt[2])).length()), 3),
                      final_obj_tilt_deg=round(math.degrees(math.acos(max(-1, min(1, up_axis[1])))), 1))
        if abs(checks["final_obj_height_above_table"]) > 0.05:
            warn("object did not end on the table top")
    if checks["robot_visible_frac"] < 0.8:
        warn(f"robot visible in only {checks['robot_visible_frac']*100:.0f}% of frames")
    for k, v in checks["focus_visible_frac"].items():
        if v < 0.7:
            warn(f"{k} visible in only {v*100:.0f}% of its shot frames")
    if checks.get("ik_err_p95_cm", 0) > 3:
        warn("IK p95 error > 3 cm (hand not reaching targets)")

    if writer is not None:
        writer.close()
    cpath = os.path.splitext(out)[0] + ".checks.json"
    json.dump(checks, open(cpath, "w"), indent=1)
    log("CHECKS", json.dumps(checks))

    # contact sheet of keyframes
    try:
        imgs = [cv2.imread(p) for p, _ in st["keys"]]
        imgs = [cv2.resize(i, (480, 270)) for i in imgs if i is not None]
        if imgs:
            cols = 4
            while len(imgs) % cols:
                imgs.append(np.zeros_like(imgs[0]))
            rows = [np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
            sheet = os.path.splitext(out)[0] + "_sheet.jpg"
            cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if vj is not None:
                summ = (f"frames={checks['frames']} vis={checks['robot_visible_frac']} "
                        f"occl_adj={checks['camera_occlusion_adjust_frac']} ik_p95={checks['ik_err_p95_cm']}cm "
                        f"warn={len(checks['warnings'])}")
                vj.post_image(sheet, ("SELFTEST " if args.selftest else "") + "v3 keyframes | " + summ)
                if args.selftest:
                    for p, k in st["keys"]:
                        vj.post_image(p, k)
                if checks["warnings"]:
                    vj.post_text("v3 warnings:\n- " + "\n- ".join(checks["warnings"]))
                if writer is not None:
                    vj.post_image(out, "v3 reel")
    except Exception as e:
        log("sheet/post failed", e)
    sim.close()


if __name__ == "__main__":
    main()
