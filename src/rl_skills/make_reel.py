"""Assemble the learned-skills portfolio reel: title card -> episodes (crossfaded) -> end card.

python src/rl_skills/make_reel.py videos/rl_pretty/ep*_SUCC.mp4 -o videos/portfolio/fetch_learned_skills_720p.mp4
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "fonts")
ACCENT = (64, 196, 255)


def card(w, h, title, sub, foot, bg_frame=None):
    if bg_frame is not None:
        im = Image.fromarray(bg_frame).convert("RGB").resize((w, h))
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(18))
        im = Image.blend(im, Image.new("RGB", (w, h), (10, 10, 14)), 0.62)
    else:
        im = Image.new("RGB", (w, h), (10, 10, 14))
    dr = ImageDraw.Draw(im)
    sc = h / 720.0
    ft = ImageFont.truetype(os.path.join(FD, "Roboto-Bold.ttf"), int(54 * sc))
    fs = ImageFont.truetype(os.path.join(FD, "Roboto-Regular.ttf"), int(26 * sc))
    ff = ImageFont.truetype(os.path.join(FD, "Roboto-Regular.ttf"), int(18 * sc))
    cx = int(110 * sc)
    cy = h // 2 - int(60 * sc)
    dr.rectangle([cx - int(26 * sc), cy, cx - int(20 * sc), cy + int(118 * sc)], fill=ACCENT)
    dr.text((cx, cy - dr.textbbox((0, 0), title, font=ft)[1]), title, font=ft, fill=(255, 255, 255))
    dr.text((cx, cy + int(80 * sc)), sub, font=fs, fill=(210, 220, 230))
    if foot:
        dr.text((cx, h - int(70 * sc)), foot, font=ff, fill=(160, 170, 180))
    return np.asarray(im)


def write_still(path, img, secs, fps):
    png = path[:-4] + ".png"
    Image.fromarray(img).save(png)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-t", str(secs),
                           "-r", str(fps), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", path])


def first_frame(p):
    import io
    b = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-ss", "2", "-i", p, "-frames:v", "1",
                                 "-f", "image2pipe", "-vcodec", "png", "-"])
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", p]).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade", type=float, default=0.6)
    a = ap.parse_args()
    first = first_frame(a.clips[0])
    h, w = first.shape[:2]
    tmp = tempfile.mkdtemp()
    title = os.path.join(tmp, "title.mp4")
    end = os.path.join(tmp, "end.mp4")
    write_still(title, card(w, h, "Learned Rearrangement Skills",
                            "Navigate  →  Pick  →  Navigate  →  Place   ·   Fetch in ReplicaCAD",
                            "Habitat 2.0  ·  PPO skill policies (depth + proprioception)  ·  WPI DS551", first),
                3.0, a.fps)
    write_still(end, card(w, h, "Rearrangement Complete",
                          f"{len(a.clips)} unseen validation episodes  ·  same policies, new layouts",
                          "WPI DS551 · Group 10", first), 2.5, a.fps)
    seq = [title] + list(a.clips) + [end]
    ins = sum([["-i", p] for p in seq], [])
    f, t = [], 0.0
    prev = "[0:v]"
    for i in range(1, len(seq)):
        t += dur(seq[i - 1]) - a.fade
        out = f"[v{i}]"
        f.append(f"{prev}[{i}:v]xfade=transition=fade:duration={a.fade}:offset={t:.3f}{out}")
        prev = out
    fc = ";".join(f"[{i}:v]fps={a.fps},format=yuv420p,setsar=1[s{i}]" for i in range(len(seq)))
    fc = fc + ";" + ";".join(x.replace(f"[0:v]", "[s0]") for x in f)
    for i in range(1, len(seq)):
        fc = fc.replace(f"[{i}:v]xfade", f"[s{i}]xfade")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", fc,
                           "-map", prev, "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out])
    print("wrote", a.out, f"{dur(a.out):.1f}s")


if __name__ == "__main__":
    main()
