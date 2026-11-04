"""VIP: OpenCV art filters (CPU only). Ported from the old video/app.py."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

MAX_EDGE = 1600  # downscale huge uploads to keep filters fast


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image {path}")
    h, w = img.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _write(img: np.ndarray, out: Path) -> str:
    if not cv2.imwrite(str(out), img):
        raise RuntimeError(f"failed to write {out}")
    return str(out)


def oil(path: str, out: Path, pinsel: float = 10, radius: float = 25, **_) -> str:
    bgr = _read(path)
    try:
        res = cv2.xphoto.oilPainting(bgr, int(pinsel), int(radius))
    except (AttributeError, cv2.error):
        res = cv2.edgePreservingFilter(bgr, sigma_s=60, sigma_r=0.4)
    return _write(res, out)


def aquarell(path: str, out: Path, pinsel: float = 170, radius: float = 0.62, **_) -> str:
    bgr = _read(path)
    res = cv2.stylization(bgr, sigma_s=float(pinsel), sigma_r=float(radius))
    return _write(res, out)


def bleistift(
    path: str,
    out: Path,
    buntstift: str = "Bleistift",
    pinsel: float = 20,
    radius: float = 0.18,
    schatten: float = 0.05,
    **_,
) -> str:
    bgr = _read(path)
    gray, color = cv2.pencilSketch(bgr, sigma_s=float(pinsel), sigma_r=float(radius), shade_factor=float(schatten))
    return _write(color if buntstift == "Buntstift" else gray, out)


def point(path: str, out: Path, colors: float = 20, **_) -> str:
    img = _read(path)
    n_colors = int(max(2, min(int(colors), 64)))
    pixels = img.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    sample = pixels[rng.choice(len(pixels), size=min(20000, len(pixels)), replace=False)]
    palette = KMeans(n_clusters=n_colors, n_init=3, random_state=0).fit(sample).cluster_centers_
    palette = np.vstack((palette, 255 - palette))

    radius = max(1, int(math.ceil(max(img.shape) / 1000)))
    step = 3
    ys, xs = np.mgrid[0 : img.shape[0] : step, 0 : img.shape[1] : step]
    ys = (ys.ravel() + rng.integers(-1, 2, ys.size)) % img.shape[0]
    xs = (xs.ravel() + rng.integers(-1, 2, xs.size)) % img.shape[1]
    order = rng.permutation(ys.size)
    ys, xs = ys[order], xs[order]

    colors_at = img[ys, xs].astype(np.float32)
    nearest = palette[np.argmin(cdist(colors_at, palette), axis=1)]
    canvas = img.copy()
    for y, x, c in zip(ys, xs, nearest):
        cv2.circle(canvas, (int(x), int(y)), radius, tuple(float(v) for v in c), -1, cv2.LINE_AA)
    return _write(canvas, out)


STYLES = {"oil": oil, "aquarell": aquarell, "bleistift": bleistift, "point": point}
