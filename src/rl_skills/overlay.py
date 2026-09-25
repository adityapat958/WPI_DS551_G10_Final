"""Portfolio overlay for learned-skill rollouts (Roboto, matches src/demo/render_v3.py)."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "fonts")
STAGES = ["Navigate", "Pick", "Navigate", "Place"]
CAPTIONS = [
    "Navigating to the object",
    "Picking up the object",
    "Carrying the object to its goal",
    "Placing the object",
]
ACCENT = (64, 196, 255)
OK = (80, 220, 140)


class Overlay:
    def __init__(self, h):
        sc = h / 720.0
        self.sc = sc
        f = lambda n, s: ImageFont.truetype(os.path.join(_FD, n), int(s * sc))
        self.cap = f("Roboto-Medium.ttf", 30)
        self.tag = f("Roboto-Regular.ttf", 17)
        self.lbl = f("Roboto-Medium.ttf", 15)
        self.chip = f("Roboto-Medium.ttf", 16)
        self.stat = f("Roboto-Regular.ttf", 15)

    def __call__(self, third, head_rgb, head_depth, stage, done_stages, holding, dist, caption=None,
                 finished=False):
        sc = self.sc
        h, w = third.shape[:2]
        pad = int(24 * sc)
        base = Image.fromarray(third).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)

        # top-left: stage chips
        x, y = pad, pad
        for i, name in enumerate(STAGES):
            label = f"{i + 1}  {name}"
            tb = dr.textbbox((0, 0), label, font=self.chip)
            cw, ch = tb[2] - tb[0] + int(26 * sc), tb[3] - tb[1] + int(16 * sc)
            if i in done_stages:
                fill, txt = (*OK, 215), (10, 20, 14, 255)
            elif i == stage and not finished:
                fill, txt = (*ACCENT, 230), (8, 16, 24, 255)
            else:
                fill, txt = (12, 12, 16, 150), (255, 255, 255, 170)
            dr.rounded_rectangle([x, y, x + cw, y + ch], radius=ch // 2, fill=fill)
            dr.text((x + int(13 * sc), y + int(8 * sc) - tb[1]), label, font=self.chip, fill=txt)
            x += cw + int(8 * sc)
        # stats line under chips
        stat = f"object → goal  {dist:.2f} m" + ("   ·   holding" if holding else "")
        dr.text((pad + int(4 * sc), y + int(40 * sc)), stat, font=self.stat, fill=(255, 255, 255, 210))

        # top-right tag
        tag = "Habitat 2.0  ·  Fetch  ·  Learned skills (PPO, depth-only)"
        tb = dr.textbbox((0, 0), tag, font=self.tag)
        dr.text((w - pad - (tb[2] - tb[0]), pad - tb[1]), tag, font=self.tag, fill=(255, 255, 255, 215))

        # lower-third caption
        cap = caption if caption is not None else CAPTIONS[stage]
        if cap:
            tb = dr.textbbox((0, 0), cap, font=self.cap)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            x0, y1 = pad + int(8 * sc), h - pad - int(8 * sc)
            bar = OK if finished else ACCENT
            dr.rounded_rectangle([x0 - int(18 * sc), y1 - th - int(30 * sc), x0 + tw + int(18 * sc), y1],
                                 radius=int(10 * sc), fill=(12, 12, 16, 170))
            dr.rectangle([x0 - int(18 * sc), y1 - th - int(30 * sc), x0 - int(13 * sc), y1], fill=(*bar, 230))
            dr.text((x0, y1 - th - int(15 * sc) - tb[1]), cap, font=self.cap, fill=(255, 255, 255, 255))

        # PiPs bottom-right: policy input (depth) above head RGB
        ph = int(h * 0.26)
        gap = int(10 * sc)
        slots = [(head_depth, "POLICY INPUT · HEAD DEPTH"), (head_rgb, "HEAD CAMERA · RGB")]
        py = h - pad - 2 * ph - gap
        px = w - pad - ph
        for im, lbl in slots:
            pim = Image.fromarray(im).resize((ph, ph), Image.BILINEAR)
            base.paste(pim, (px, py))
            dr.rounded_rectangle([px - 2, py - 2, px + ph + 1, py + ph + 1], radius=int(6 * sc),
                                 outline=(255, 255, 255, 230), width=2)
            lb = dr.textbbox((0, 0), lbl, font=self.lbl)
            dr.rounded_rectangle([px + int(8 * sc), py + int(8 * sc), px + int(16 * sc) + lb[2] - lb[0],
                                  py + int(14 * sc) + lb[3]], radius=int(4 * sc), fill=(12, 12, 16, 170))
            dr.text((px + int(12 * sc), py + int(10 * sc)), lbl, font=self.lbl, fill=(255, 255, 255, 255))
            py += ph + gap
        return np.asarray(Image.alpha_composite(base, ov).convert("RGB")).copy()


def depth_vis(d):
    """normalized depth [0,1] -> turbo-ish colormap uint8 rgb."""
    import cv2

    d = np.asarray(d, np.float32).reshape(d.shape[0], d.shape[1])
    u8 = np.clip(d * 255.0 * 2.0, 0, 255).astype(np.uint8)  # ~0-5 m range
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
