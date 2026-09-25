#!/usr/bin/env python3
"""
Load the Habitat Rearrangement Challenge 2022 pretrained Fetch skills
(nav.pth / pick.pth / place.pth, trained with habitat-lab branch
`rearrange_challenge_2022`, commit 20eeee2e) into the habitat-baselines
v0.3.3 `PointNavResNetPolicy`, with `load_state_dict(strict=True)`.

Usage (CPU is fine, no simulator needed):
    python load_skill.py --models-dir /path/to/data/models [--skills nav pick place]

Only two state-dict transformations are needed (everything else is 1:1):
  1. strip the "actor_critic." prefix (2022 PPO agent wrapper).
  2. GaussianNet head: v0.3.3 fuses the 2022 `mu` and `std` Linear layers
     into one `mu_maybe_std` Linear with 2*A outputs ([mu ; log_std]):
       action_distribution.mu.{weight,bias} + action_distribution.std.{weight,bias}
         -> action_distribution.mu_maybe_std.{weight,bias}   (torch.cat dim 0)
     For pick.pth (use_std_param=True) `std` is an (8,) nn.Parameter which keeps
     its name; only mu.* -> mu_maybe_std.* is renamed.
"""
import argparse
import os
import pickle
import sys
import types
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------
# Specs extracted from ckpt["config"] (TASK_CONFIG.GYM.OBS_KEYS order is the
# exact order the 2022 PointNavResNetNet concatenated the 1-D fuse keys; see
# 2022 resnet_policy.py from_config: fuse_keys=config.TASK_CONFIG.GYM.OBS_KEYS)
# --------------------------------------------------------------------------
DEPTH_HW = 256
SKILL_SPECS: Dict[str, dict] = {
    "nav": dict(
        obs=OrderedDict(
            [
                ("robot_head_depth", (DEPTH_HW, DEPTH_HW, 1)),
                ("object_to_agent_gps_compass", (2,)),  # polar [rho, -phi], robot base frame
                ("joint", (7,)),
                ("nav_to_skill", (8,)),  # one-hot of PDDL action nav is heading for
            ]
        ),
        action_dim=2,  # BASE_VELOCITY.base_vel = [lin, ang] in [-1,1]
        use_std_param=False,
    ),
    "pick": dict(
        obs=OrderedDict(
            [
                ("robot_head_depth", (DEPTH_HW, DEPTH_HW, 1)),
                ("obj_start_sensor", (3,)),  # target start pos in EE frame (cartesian)
                ("joint", (7,)),
                ("is_holding", (1,)),
                ("relative_resting_position", (3,)),
            ]
        ),
        action_dim=8,  # ARM_ACTION: arm_action(7 joint deltas) + grip_action(1)
        use_std_param=True,
    ),
    "place": dict(
        obs=OrderedDict(
            [
                ("robot_head_depth", (DEPTH_HW, DEPTH_HW, 1)),
                ("obj_goal_sensor", (3,)),  # goal pos in EE frame (cartesian)
                ("joint", (7,)),
                ("is_holding", (1,)),
                ("relative_resting_position", (3,)),
            ]
        ),
        action_dim=8,
        use_std_param=False,
    ),
}
HIDDEN_SIZE = 512
RNN_TYPE = "LSTM"
NUM_RNN_LAYERS = 2
BACKBONE = "resnet18"  # baseplanes 32, ngroups 16, GroupNorm, no RunningMeanAndVar


# --------------------------------------------------------------------------
# Stub unpickler: ckpt["config"] is a pickled yacs habitat Config from 2022.
# --------------------------------------------------------------------------
class _Stub(dict):
    def __init__(self, *a, **k):
        super().__init__()

    def __setstate__(self, s):
        if isinstance(s, dict):
            self.update(s)
        else:
            self["__state__"] = s

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("habitat") or module.startswith("yacs"):
            return type(name, (_Stub,), {"__module__": module})
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (_Stub,), {"__module__": module})


_stub_pickle = types.ModuleType("stub_pickle")
_stub_pickle.Unpickler = _StubUnpickler
_stub_pickle.load = pickle.load
_stub_pickle.__name__ = "stub_pickle"


def load_ckpt(path: str) -> dict:
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle
    )


def _to_plain(x):
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items() if not str(k).startswith("__")}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    return x


def check_ckpt_config(skill: str, ckpt: dict) -> None:
    """Assert the pickled 2022 config agrees with SKILL_SPECS."""
    cfg = _to_plain(ckpt["config"])
    obs_keys = cfg["TASK_CONFIG"]["GYM"]["OBS_KEYS"]
    assert list(obs_keys) == list(SKILL_SPECS[skill]["obs"].keys()), (skill, obs_keys)
    assert cfg["RL"]["DDPPO"]["rnn_type"] == RNN_TYPE
    assert cfg["RL"]["DDPPO"]["num_recurrent_layers"] == NUM_RNN_LAYERS
    assert cfg["RL"]["DDPPO"]["backbone"] == BACKBONE
    assert cfg["RL"]["PPO"]["hidden_size"] == HIDDEN_SIZE
    ad = cfg["RL"]["POLICY"]["ACTION_DIST"]
    assert bool(ad["use_std_param"]) == SKILL_SPECS[skill]["use_std_param"]
    assert ad["action_activation"] == "tanh" and ad["use_log_std"]


# --------------------------------------------------------------------------
# v0.3.3 policy construction
# --------------------------------------------------------------------------
def make_policy_config(use_std_param: bool):
    from omegaconf import OmegaConf

    # Mirrors 2022 RL.POLICY.ACTION_DIST (clamp log-std to [-5, 2], then exp).
    return OmegaConf.create(
        {
            "name": "PointNavResNetPolicy",
            "action_distribution_type": "gaussian",
            "action_dist": {
                "use_log_std": True,
                "use_softplus": False,
                "log_std_init": 0.0,
                "use_std_param": use_std_param,
                "clamp_std": True,
                "min_std": 1e-6,
                "max_std": 1,
                "min_log_std": -5,
                "max_log_std": 2,
                "action_activation": "tanh",
                "scheduled_std": False,
            },
        }
    )


def make_spaces(skill: str):
    from gym import spaces

    spec = SKILL_SPECS[skill]
    obs_space = spaces.Dict(
        OrderedDict(
            (
                k,
                spaces.Box(
                    low=0.0 if len(s) == 3 else np.finfo(np.float32).min,
                    high=1.0 if len(s) == 3 else np.finfo(np.float32).max,
                    shape=s,
                    dtype=np.float32,
                ),
            )
            for k, s in spec["obs"].items()
        )
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(spec["action_dim"],), dtype=np.float32)
    return obs_space, act_space


def build_policy(skill: str):
    from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

    spec = SKILL_SPECS[skill]
    obs_space, act_space = make_spaces(skill)
    policy = PointNavResNetPolicy(
        observation_space=obs_space,
        action_space=act_space,
        hidden_size=HIDDEN_SIZE,
        num_recurrent_layers=NUM_RNN_LAYERS,
        rnn_type=RNN_TYPE,
        resnet_baseplanes=32,
        backbone=BACKBONE,
        normalize_visual_inputs=False,  # 2022: "rgb" not in obs space -> False
        force_blind_policy=False,
        policy_config=make_policy_config(spec["use_std_param"]),
        aux_loss_config=None,
        fuse_keys=list(spec["obs"].keys()),  # explicit, preserves 2022 order
    )
    return policy


def convert_state_dict(sd_2022: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in sd_2022.items():
        assert k.startswith("actor_critic."), k
        out[k[len("actor_critic."):]] = v
    ad = "action_distribution."
    if ad + "std.weight" in out:  # state-dependent std (nav, place)
        out[ad + "mu_maybe_std.weight"] = torch.cat(
            [out.pop(ad + "mu.weight"), out.pop(ad + "std.weight")], 0
        )
        out[ad + "mu_maybe_std.bias"] = torch.cat(
            [out.pop(ad + "mu.bias"), out.pop(ad + "std.bias")], 0
        )
    else:  # std is a free nn.Parameter (pick)
        out[ad + "mu_maybe_std.weight"] = out.pop(ad + "mu.weight")
        out[ad + "mu_maybe_std.bias"] = out.pop(ad + "mu.bias")
    return out


def load_skill(skill: str, models_dir: str, device="cpu") -> Tuple[torch.nn.Module, dict]:
    ckpt = load_ckpt(os.path.join(models_dir, f"{skill}.pth"))
    check_ckpt_config(skill, ckpt)
    policy = build_policy(skill)
    sd = convert_state_dict(ckpt["state_dict"])
    res = policy.load_state_dict(sd, strict=True)
    policy.to(device).eval()
    return policy, {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}


class SkillRunner:
    """Stateful wrapper: holds LSTM state + prev action for one env."""

    def __init__(self, skill: str, models_dir: str, device="cpu", deterministic=True):
        self.skill = skill
        self.policy, _ = load_skill(skill, models_dir, device)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.action_dim = SKILL_SPECS[skill]["action_dim"]
        self.reset()

    def reset(self):
        self.hidden = torch.zeros(
            1, self.policy.num_recurrent_layers, HIDDEN_SIZE, device=self.device
        )
        self.prev_action = torch.zeros(1, self.action_dim, device=self.device)
        # mask=False on the first step zeros prev_action and LSTM state (as in 2022
        # skill on_enter, which zeroes hidden state and prev action).
        self.not_done = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def act(self, obs2022: Dict[str, np.ndarray]) -> np.ndarray:
        batch = {
            k: torch.as_tensor(np.asarray(obs2022[k], dtype=np.float32), device=self.device).unsqueeze(0)
            for k in SKILL_SPECS[self.skill]["obs"]
        }
        out = self.policy.act(
            batch, self.hidden, self.prev_action, self.not_done, deterministic=self.deterministic
        )
        self.hidden = out.rnn_hidden_states
        self.prev_action = out.actions.float()
        self.not_done = torch.ones_like(self.not_done)
        return out.actions[0].cpu().numpy()


def dummy_obs(skill: str, batch: int = 2) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        k: (torch.rand((batch,) + s, generator=g) if len(s) == 3 else torch.randn((batch,) + s, generator=g))
        for k, s in SKILL_SPECS[skill]["obs"].items()
    }


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="data/models")
    ap.add_argument("--skills", nargs="+", default=["nav", "pick", "place"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    ok = True
    for skill in args.skills:
        policy, res = load_skill(skill, args.models_dir)
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[{skill}] strict load OK; missing={res['missing']} unexpected={res['unexpected']} params={n_params}")
        ok &= not res["missing"] and not res["unexpected"]
        B = 2
        obs = dummy_obs(skill, B)
        hidden = torch.zeros(B, policy.num_recurrent_layers, HIDDEN_SIZE)
        prev = torch.zeros(B, SKILL_SPECS[skill]["action_dim"])
        masks = torch.zeros(B, 1, dtype=torch.bool)
        with torch.no_grad():
            out = policy.act(obs, hidden, prev, masks, deterministic=True)
            # second step with the produced hidden state/prev action
            out2 = policy.act(obs, out.rnn_hidden_states, out.actions, torch.ones(B, 1, dtype=torch.bool), deterministic=True)
        # Independent check of the only renamed module: recompute the 2022
        # GaussianNet (separate mu / std heads) from the ORIGINAL ckpt tensors.
        with torch.no_grad():
            feats, _, _ = policy.net(obs, hidden, prev, masks)
            sd = load_ckpt(os.path.join(args.models_dir, f"{skill}.pth"))["state_dict"]
            p = "actor_critic.action_distribution."
            mu22 = torch.tanh(feats @ sd[p + "mu.weight"].T + sd[p + "mu.bias"])
            if p + "std.weight" in sd:
                ls22 = feats @ sd[p + "std.weight"].T + sd[p + "std.bias"]
            else:
                ls22 = sd[p + "std"].expand_as(mu22)
            std22 = torch.exp(torch.clamp(ls22, -5, 2))
            dist = policy.action_distribution(feats)
            dmu = (dist.mean - mu22).abs().max().item()
            dstd = (dist.stddev - std22).abs().max().item()
            v22 = feats @ sd["actor_critic.critic.fc.weight"].T + sd["actor_critic.critic.fc.bias"]
            dv = (policy.critic(feats) - v22).abs().max().item()
        print(f"[{skill}] 2022-head equivalence: max|dmu|={dmu:.2e} max|dstd|={dstd:.2e} max|dV|={dv:.2e}")
        ok &= dmu < 1e-5 and dstd < 1e-5 and dv < 1e-4
        print(
            f"[{skill}] obs shapes={ {k: tuple(v.shape) for k, v in obs.items()} }\n"
            f"[{skill}] rnn_input={policy.net.state_encoder.rnn.input_size} hidden={tuple(out.rnn_hidden_states.shape)} "
            f"action={tuple(out.actions.shape)} value={tuple(out.values.shape)}\n"
            f"[{skill}] a_t0={np.round(out.actions[0].numpy(), 3).tolist()} a_t1={np.round(out2.actions[0].numpy(), 3).tolist()}"
        )
    print("ALL OK" if ok else "KEY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
