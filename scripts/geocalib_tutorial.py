"""spec-100 tutorial: render GeoCalib's intermediate steps on two photos.

HIGH tilt (20250822_123528, ~+16.5 deg) vs LOW tilt (20250822_112359, ~0 deg).
For each photo, dumps the per-step visualizations (raw up-field, latitude field,
confidence map, fitted horizon, roll geometry, straightened) + a numbers.json the
HTML bakes in. Output: reports/2026-07-13_geocalib_tutorial/{img,numbers.json}.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocalib import GeoCalib, viz2d
from geocalib.perspective_fields import get_up_field
from geocalib.utils import numpy_image_to_torch

SRC = Path(r"D:\Budapest2025_Google")
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-07-13_geocalib_tutorial"
IMG = OUT / "img"
PHOTOS = {"high": "20250822_123528", "low": "20250822_112359"}
_SIGN = -1.0
_UNC_TAU = 2.0


def load_rgb(stem, side=1024):
    p = next(SRC.rglob(stem + ".jp*"))
    with Image.open(p) as pil:
        rgb = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
    s = side / max(rgb.shape[:2])
    return cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))


def up_2hw(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    a = np.squeeze(a)
    if a.shape[0] != 2:
        a = np.transpose(a, (2, 0, 1))
    return a


def field_hw(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return np.squeeze(a)


def save_bare(fig, path, W):
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=110)
    plt.close(fig)


def straighten(rgb, roll):
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0)
    return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def render(tag, stem, model):
    rgb = load_rgb(stem)
    res = model.calibrate(numpy_image_to_torch(rgb).to("cpu"))
    cam, grav = res["camera"], res["gravity"]

    roll = _SIGN * float(torch.rad2deg(grav.roll))
    pitch = float(torch.rad2deg(grav.pitch))
    vfov = float(torch.rad2deg(cam.vfov)) if hasattr(cam, "vfov") else float("nan")
    roll_unc = float(torch.rad2deg(res["roll_uncertainty"]))
    pitch_unc = float(torch.rad2deg(res["pitch_uncertainty"]))

    uf = up_2hw(res["up_field"])          # raw network prediction
    H, W = uf.shape[1:]
    disp = cv2.resize(rgb, (W, H))
    fitted = up_2hw(get_up_field(cam, grav))  # reprojected from fitted params

    mx, my = float(uf[0].mean()), float(uf[1].mean())
    n = np.hypot(mx, my) or 1.0
    mx, my = mx / n, my / n
    conf = float(np.exp(-max(roll_unc, 0.0) / _UNC_TAU))
    excess = abs(roll) - 3.0
    penalty = 0.0 if (conf < 0.5 or excess <= 0) else -min(0.02 * excess, 0.15)

    DW = 460
    # 1. input
    cv2.imwrite(str(IMG / f"{tag}_1_input.jpg"),
                cv2.cvtColor(cv2.resize(disp, (DW, int(DW * H / W))), cv2.COLOR_RGB2BGR))

    # 2. raw up-field (network output)
    fig, ax = plt.subplots(figsize=(5, 5 * H / W), dpi=110); ax.imshow(disp); ax.set_axis_off()
    viz2d.plot_vector_fields([torch.from_numpy(uf)], axes=[ax], subsample=15, cmap="lime")
    save_bare(fig, IMG / f"{tag}_2_upfield.jpg", DW)

    # 3. latitude field (horizon = zero level)
    lat = np.degrees(field_hw(res["latitude_field"]))
    fig, ax = plt.subplots(figsize=(5, 5 * H / W), dpi=110)
    im = ax.imshow(lat, cmap="RdBu_r", vmin=-60, vmax=60); ax.set_axis_off()
    ax.contour(lat, levels=[0], colors="k", linewidths=2)
    save_bare(fig, IMG / f"{tag}_3_latitude.jpg", DW)

    # 4. up-confidence
    upc = field_hw(res["up_confidence"])
    fig, ax = plt.subplots(figsize=(5, 5 * H / W), dpi=110)
    ax.imshow(upc, cmap="magma", vmin=0, vmax=float(upc.max()) or 1.0); ax.set_axis_off()
    save_bare(fig, IMG / f"{tag}_4_confidence.jpg", DW)

    # 5. fitted field + horizon (LM result)
    fig, ax = plt.subplots(figsize=(5, 5 * H / W), dpi=110); ax.imshow(disp); ax.set_axis_off()
    viz2d.plot_vector_fields([torch.from_numpy(fitted)], axes=[ax], subsample=15, cmap="deepskyblue")
    try:
        viz2d.plot_horizon_lines([cam], [grav], ax=[ax], lw=2.5)
    except Exception:
        pass
    save_bare(fig, IMG / f"{tag}_5_fitted.jpg", DW)

    # 6. roll geometry: FITTED gravity direction vs screen up + angle wedge.
    # Draw the fitted roll (from the LM solve), not the raw field mean -- averaging
    # unit vectors shrinks toward vertical and ignores pitch/focal, so it under-reads.
    rr = np.radians(roll)
    vx, vy = float(np.sin(rr)), float(-np.cos(rr))  # scene "up" tilted by roll
    fig, ax = plt.subplots(figsize=(5, 5 * H / W), dpi=110); ax.imshow(disp); ax.set_axis_off()
    L = 0.40 * min(H, W); ox, oy = W * 0.5, H * 0.62
    ax.annotate("", xy=(ox, oy - L), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, ls=(0, (5, 3))))
    ax.text(ox - 8, oy - L - 10, "screen up", color="white", fontsize=11, ha="right", fontweight="bold")
    ax.annotate("", xy=(ox + vx * L, oy + vy * L), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color="lime", lw=3.5))
    ax.text(ox + vx * L + 10, oy + vy * L, "fitted up", color="lime", fontsize=11, ha="left", fontweight="bold")
    a_scene = np.degrees(np.arctan2(-vy, vx))
    ax.add_patch(mpatches.Wedge((ox, oy), L * 0.5, min(90, a_scene), max(90, a_scene),
                                color="yellow", alpha=0.35))
    ax.text(ox + 12, oy - L * 0.32, f"roll {abs(roll):.1f} deg", color="yellow", fontsize=12, fontweight="bold")
    save_bare(fig, IMG / f"{tag}_6_roll.jpg", DW)

    # 7. straightened
    st = straighten(disp, roll)
    st = cv2.putText(cv2.cvtColor(cv2.resize(st, (DW, int(DW * H / W))), cv2.COLOR_RGB2BGR),
                     f"rotated {-roll:+.1f} deg", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.imwrite(str(IMG / f"{tag}_7_straight.jpg"), st)

    return {"stem": stem, "roll": roll, "pitch": pitch, "vfov": vfov,
            "roll_unc": roll_unc, "pitch_unc": pitch_unc, "mx": mx, "my": my,
            "conf": conf, "penalty": penalty, "excess": excess,
            "a_scene": a_scene, "field_res": [int(W), int(H)]}


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    model = GeoCalib().to("cpu").eval()
    nums = {}
    for tag, stem in PHOTOS.items():
        with torch.no_grad():
            nums[tag] = render(tag, stem, model)
        print(tag, stem, "roll %.2f unc %.2f conf %.2f penalty %.3f" % (
            nums[tag]["roll"], nums[tag]["roll_unc"], nums[tag]["conf"], nums[tag]["penalty"]))
    OUT.joinpath("numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
