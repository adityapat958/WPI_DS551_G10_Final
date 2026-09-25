# Pretrained 2022 Rearrangement-Challenge skills in habitat-lab 0.3.3

Meta's released Fetch skills (`nav.pth`, `pick.pth`, `place.pth`, from
`rearrange_habitat2022_challenge_baseline_v1.zip`) running in habitat-lab/baselines **v0.3.3**.

- `load_skill.py` — loads the 2022 checkpoints into v0.3.3 `PointNavResNetPolicy` with `strict=True`
  (only renames: strip `actor_critic.`, fuse `mu`/`std` heads into `mu_maybe_std`); verified numerically
  identical to the 2022 head outputs.
- `obs_adapter.py` — v0.3.3 observations → 2022 obs dicts; 2022 actions → v0.3.3 action dicts
  (BaseVelAction / ArmRelPosAction + suction grip, 2022 TP-SRL grip masking and termination rules).
- `rollout_skills.py` — hard-coded chain nav → pick → nav → place on `rearrange_easy` val episodes,
  records third-person + head RGB + head depth.

Run (GPU node): `vizjob run rl -- --num-episodes 60`. First 60-episode batch: successes in episodes
5, 114, 125, 318, … (clips in `videos/rl_skills/*_SUCC.mp4`). These are **learned policies** (depth-only RL);
per-stage failures are mostly pick misses and nav-to-goal timeouts.
