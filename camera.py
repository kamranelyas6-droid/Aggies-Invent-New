"""Top camera (micron level).

Simulated sensor: a 6 MP camera (3000 x 2000 px) with a telecentric lens, 40 µm per pixel,
field 120 x 80 mm, ring light, dark background. The render includes what a real camera adds:
an unknown magnification error (removed by gauge block calibration), a small residual lens
distortion (left after grid calibration, carried as uncertainty), illumination fall off,
machining texture, optical blur, sensor noise and 8 bit quantisation.

Measurement (production algorithms, they only see the image):
  * subpixel edges: 50 % crossing on 1D profiles normal to each side, 300 profiles per side
  * sides: orthogonal regression lines; length and width from line to line distances
  * datum frame: B = left side orients the frame, C = bottom side sets the origin (A = seat)
  * holes: 360 radial profiles per hole, algebraic then geometric circle fit
  * position: diametral position error 2 * sqrt(dx^2 + dy^2) in the datum frame
  * marks: patch memory bank learned from good parts only (PatchCore style), nearest
    neighbour distance gives an anomaly heat map; regions become confocal tasks
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors

PX_NOM = 0.040                 # mm per pixel (nominal)
H_IMG, W_IMG = 2000, 3000      # rows, columns -> 80 x 120 mm field
SCALE_ERR = 3.0e-4             # true magnification error, unknown to the measurement
DIST_AMP = 0.0008              # mm, residual distortion amplitude after grid calibration
BG, FACE = 22.0, 188.0         # grey levels: background, part face
NOISE = 2.0                    # grey levels rms
BLUR = 0.7                     # pixels, optical point spread (sigma)
STRIDE = 4                     # anomaly patch size in pixels (160 µm)


# ----------------------------------------------------------------------------- rendering
def _distortion(X, Y):
    return (DIST_AMP * np.sin(2 * np.pi * X / 120.0) * np.cos(2 * np.pi * Y / 80.0),
            DIST_AMP * np.cos(2 * np.pi * X / 120.0) * np.sin(2 * np.pi * Y / 80.0))


def random_pose(rng):
    """Part origin (lower left corner) in camera world mm, and rotation in radians."""
    return (20.0 + rng.uniform(-2, 2), 15.0 + rng.uniform(-2, 2), np.radians(rng.uniform(-0.8, 0.8)))


def _window(cx, cy, half_mm, px, shape):
    """Pixel index window (rows, cols slices) around a world point."""
    H, W = shape
    c0, c1 = int((cx - half_mm) / px) - 2, int((cx + half_mm) / px) + 3
    r0, r1 = int(H - (cy + half_mm) / px) - 2, int(H - (cy - half_mm) / px) + 3
    return slice(max(r0, 0), min(r1, H)), slice(max(c0, 0), min(c1, W))


def render(truth, pose, rng, kind="part", scale_err=SCALE_ERR):
    """Render one top lit camera image (uint8, rows x cols) of the part (or a gauge block)."""
    H, W = H_IMG, W_IMG
    px = PX_NOM * (1.0 + scale_err)
    tx, ty, th = pose
    ct, st = np.cos(th), np.sin(th)
    X = ((np.arange(W, dtype=np.float32) + 0.5) * px)[None, :].repeat(H, 0)
    Y = ((H - np.arange(H, dtype=np.float32) - 0.5) * px)[:, None].repeat(W, 1)
    dX, dY = _distortion(X, Y)
    Xw, Yw = X + dX, Y + dY
    xp = ct * (Xw - tx) + st * (Yw - ty)          # part frame coordinates of every pixel
    yp = -st * (Xw - tx) + ct * (Yw - ty)
    if kind == "gauge":
        L, Wd = 50.000, 9.000
    else:
        L, Wd = truth["L"], truth["W"]
    dx = np.abs(xp - L / 2) - L / 2
    dy = np.abs(yp - Wd / 2) - Wd / 2
    sd = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2) + np.minimum(np.maximum(dx, dy), 0)
    cov = np.clip(0.5 - sd / px, 0.0, 1.0)

    # face appearance: illumination fall off, milling lay, fine random texture
    vign = 1.0 - 0.06 * ((X - 60.0) ** 2 + (Y - 40.0) ** 2) / (60.0 ** 2 + 40.0 ** 2)
    tex = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 2.5)
    tex *= 0.010 / (tex.std() + 1e-9)
    tex += 0.010 * np.sin(2 * np.pi * xp / 0.35)
    face = FACE * vign * (1.0 + tex)

    def to_world(x, y):
        return tx + ct * x - st * y, ty + st * x + ct * y

    if kind == "part":
        for d in truth.get("defects", []):
            if d["kind"] == "scratch":
                xm, ym = (d["x0"] + d["x1"]) / 2, (d["y0"] + d["y1"]) / 2
                half = 0.5 * np.hypot(d["x1"] - d["x0"], d["y1"] - d["y0"]) + 0.3
            else:
                xm, ym, half = d["x"], d["y"], 3 * d["r"] + 0.3
            wx, wy = to_world(xm, ym)
            sl = _window(wx, wy, half, px, (H, W))
            xs, ys = xp[sl], yp[sl]
            if d["kind"] == "stain":
                face[sl] *= 1.0 - d["contrast"] * np.exp(-((xs - d["x"]) ** 2 + (ys - d["y"]) ** 2) / (2 * (d["r"] / 2) ** 2))
            elif d["kind"] == "pit":
                rr = np.hypot(xs - d["x"], ys - d["y"])
                face[sl] *= 1.0 - d["contrast"] * np.clip((d["r"] - rr) / px + 0.5, 0, 1)
            elif d["kind"] == "scratch":
                ax, ay = d["x0"], d["y0"]
                bx, by = d["x1"], d["y1"]
                vx, vy = bx - ax, by - ay
                tt = np.clip(((xs - ax) * vx + (ys - ay) * vy) / (vx * vx + vy * vy), 0, 1)
                dist = np.hypot(xs - (ax + tt * vx), ys - (ay + tt * vy))
                face[sl] *= 1.0 - d["contrast"] * np.clip(1.0 - dist / (d["w"] / 2 + px / 2), 0, 1)
        # holes (through, so the dark background shows), with a three lobe roundness error
        for hid, (hx, hy, hd) in truth["holes"].items():
            wx, wy = to_world(hx, hy)
            sl = _window(wx, wy, hd / 2 + 0.4, px, (H, W))
            rx, ry = xp[sl] - hx, yp[sl] - hy
            rr = np.hypot(rx, ry)
            phi = np.arctan2(ry, rx)
            R = hd / 2 + truth.get("lobing_um", 0.0) / 1000.0 * np.cos(3 * phi + 0.7)
            cov[sl] *= 1.0 - np.clip(0.5 + (R - rr) / px, 0.0, 1.0)

    img = BG + (face - BG) * cov
    img = ndimage.gaussian_filter(img, BLUR)
    img = img + rng.normal(0.0, NOISE, img.shape).astype(np.float32) + rng.normal(0.0, 1.0, img.shape).astype(np.float32) * np.sqrt(np.maximum(img, 0) / 255.0)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- geometry helpers
def _pw(r, c):
    """Pixel index (row, col) to pixel world units (x right, y up), pixel centre at +0.5."""
    return np.c_[c + 0.5, H_IMG - r - 0.5]


def _idx(p):
    """Pixel world units to index coordinates (rows, cols) for interpolation."""
    return H_IMG - p[..., 1] - 0.5, p[..., 0] - 0.5


def fit_line(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    return c, d, np.array([-d[1], d[0]])


def intersect(c1, d1, c2, d2):
    A = np.c_[d1, -d2]
    a, _ = np.linalg.solve(A, c2 - c1)
    return c1 + a * d1


def fit_circle(P):
    """Algebraic (Kasa) start, then Gauss Newton on the geometric distance. Returns cx, cy, r, rms."""
    x, y = P[:, 0], P[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    r = np.sqrt(max(cx ** 2 + cy ** 2 - F, 1e-12))
    for _ in range(8):
        dx, dy = x - cx, y - cy
        dist = np.hypot(dx, dy)
        J = np.c_[-dx / dist, -dy / dist, -np.ones_like(x)]
        res = dist - r
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-10:
            break
    res = np.hypot(x - cx, y - cy) - r
    return cx, cy, r, float(np.sqrt(np.mean(res ** 2)))


def _side_edges(I, p0, p1, n_out, n=300, half=10.0, step=0.25):
    """Subpixel edge points along one side. p0, p1, n_out in pixel world units; bright inside."""
    s = np.linspace(0.08, 0.92, n)
    base = p0[None, :] + s[:, None] * (p1 - p0)[None, :]
    t = np.arange(-half, half + 1e-9, step)
    pts = base[:, None, :] + t[None, :, None] * n_out[None, None, :]
    rr, cc = _idx(pts)
    prof = ndimage.map_coordinates(I, [rr.ravel(), cc.ravel()], order=1).reshape(n, -1)
    inside = np.median(prof[:, :12], 1)
    outside = np.median(prof[:, -12:], 1)
    level = 0.5 * (inside + outside)
    below = prof < level[:, None]
    first = np.argmax(below, 1)
    ok = below.any(1) & (first > 0)
    k = np.nonzero(ok)[0]
    i = first[k]
    p_a, p_b = prof[k, i - 1], prof[k, i]
    tc = t[i - 1] + (p_a - level[k]) / (p_a - p_b) * step
    return base[k] + tc[:, None] * n_out[None, :]


def _hole_edges(I, centre_pw, r_px, n=360, step=0.25):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.arange(0.45 * r_px, 1.6 * r_px, step)
    u = np.c_[np.cos(ang), np.sin(ang)]
    pts = centre_pw[None, None, :] + rad[None, :, None] * u[:, None, :]
    rr, cc = _idx(pts)
    prof = ndimage.map_coordinates(I, [rr.ravel(), cc.ravel()], order=1).reshape(n, -1)
    lo = np.median(prof[:, :6], 1)
    hi = np.median(prof[:, -6:], 1)
    level = 0.5 * (lo + hi)
    above = prof > level[:, None]
    first = np.argmax(above, 1)
    ok = above.any(1) & (first > 0)
    k = np.nonzero(ok)[0]
    i = first[k]
    p_a, p_b = prof[k, i - 1], prof[k, i]
    rc = rad[i - 1] + (level[k] - p_a) / (p_b - p_a) * step
    return centre_pw[None, :] + rc[:, None] * u[k]


def _part_mask(I):
    thr = 0.5 * (np.percentile(I, 2) + np.percentile(I, 98))
    m = I > thr
    lab, n = ndimage.label(m)
    if n == 0:
        raise RuntimeError("no part found in the image")
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def measure_sides(I, part_mask):
    """Four fitted sides in pixel world units. Returns dict side -> (c, d, n, points)."""
    r, c = np.nonzero(part_mask[::4, ::4])
    P = _pw(r * 4.0, c * 4.0)
    m = P.mean(0)
    _, _, vt = np.linalg.svd(P - m, full_matrices=False)
    e1 = vt[0] if vt[0][0] > 0 else -vt[0]
    e2 = np.array([-e1[1], e1[0]])
    u, v = (P - m) @ e1, (P - m) @ e2
    umin, umax, vmin, vmax = u.min(), u.max(), v.min(), v.max()
    BL, BR = m + umin * e1 + vmin * e2, m + umax * e1 + vmin * e2
    TL, TR = m + umin * e1 + vmax * e2, m + umax * e1 + vmax * e2
    sides = {}
    for name, p0, p1, n_out in [("left", BL, TL, -e1), ("right", BR, TR, e1),
                                ("bottom", BL, BR, -e2), ("top", TL, TR, e2)]:
        pts = _side_edges(I, p0, p1, n_out)
        cc, d, nn = fit_line(pts)
        sides[name] = (cc, d, nn, pts)
    return sides


def line_distance(sa, sb):
    (ca, da, na, pa), (cb, db, nb, pb) = sa, sb
    return 0.5 * (np.mean(np.abs((pb - ca) @ na)) + np.mean(np.abs((pa - cb) @ nb)))


def measure_length(I, px_cal):
    """Gauge block calibration helper: long side length in mm (left to right lines)."""
    s = measure_sides(I.astype(np.float32), _part_mask(I))
    return line_distance(s["left"], s["right"]) * px_cal


def measure_part(img, px_cal, nominal):
    """All camera measurements from one image. Lengths in mm, in the datum frame."""
    I = img.astype(np.float32)
    pm = _part_mask(img)
    sides = measure_sides(I, pm)
    to_mm = lambda p: p * px_cal
    sides_mm = {k: (to_mm(v[0]), v[1], v[2], to_mm(v[3])) for k, v in sides.items()}
    L = line_distance(sides["left"], sides["right"]) * px_cal
    Wd = line_distance(sides["bottom"], sides["top"]) * px_cal
    cB, dB, _, _ = sides_mm["left"]
    cC, dC, _, _ = sides_mm["bottom"]
    ey = dB if dB[1] > 0 else -dB
    ex = np.array([ey[1], -ey[0]])
    O = intersect(cB, dB, cC, dC)

    def frame(p):
        q = np.atleast_2d(p) - O
        return np.c_[q @ ex, q @ ey]

    filled = ndimage.binary_fill_holes(pm)
    holes_mask = filled & ~pm
    lab, n = ndimage.label(holes_mask)
    holes = {}
    edge_world = {}
    for i in range(1, n + 1):
        comp = lab == i
        area = comp.sum()
        if area < 80:
            continue
        rr, cc = ndimage.center_of_mass(comp)
        cpw = _pw(np.array([rr]), np.array([cc]))[0]
        r_px = np.sqrt(area / np.pi)
        e_pw = _hole_edges(I, cpw, r_px)
        e_mm = e_pw * px_cal
        cx, cy, r, rms = fit_circle(e_mm)
        pf = frame([cx, cy])[0]
        hid = min(nominal["holes"], key=lambda h: np.hypot(nominal["holes"][h][0] - pf[0], nominal["holes"][h][1] - pf[1]))
        holes[hid] = {"x": float(pf[0]), "y": float(pf[1]), "d": float(2 * r), "fit_rms_um": rms * 1000,
                      "world": (float(cx), float(cy))}
        edge_world[hid] = e_mm
    for hid, h in holes.items():
        nx, ny, _ = nominal["holes"][hid]
        h["dx_um"] = (h["x"] - nx) * 1000
        h["dy_um"] = (h["y"] - ny) * 1000
        h["pos_err_um"] = 2 * np.hypot(h["dx_um"], h["dy_um"])
    angle = float(np.degrees(np.arctan2(ex[1], ex[0])))
    return {"L": float(L), "W": float(Wd), "holes": holes, "origin": O, "ex": ex, "ey": ey,
            "angle_deg": angle, "sides_mm": sides_mm, "edge_world": edge_world, "part_mask": pm,
            "frame": frame}


# ----------------------------------------------------------------------------- mark detection
def _ridge(f, sigma):
    gxx = ndimage.gaussian_filter(f, sigma, order=(0, 2))
    gyy = ndimage.gaussian_filter(f, sigma, order=(2, 0))
    gxy = ndimage.gaussian_filter(f, sigma, order=(1, 1))
    tmp = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)
    return np.maximum(0.5 * (gxx + gyy + tmp), 0) * sigma ** 2


def face_region(part_mask, margin_px=20):
    """Pixels at least margin_px from any edge or hole, and the bounding box of that region."""
    dist = ndimage.distance_transform_edt(part_mask)
    face = dist > margin_px
    rows, cols = np.nonzero(face.any(1))[0], np.nonzero(face.any(0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    r1 = r0 + (r1 - r0) // STRIDE * STRIDE
    c1 = c0 + (c1 - c0) // STRIDE * STRIDE
    return face, (r0, r1, c0, c1)


def patch_features(img, part_mask):
    face, (r0, r1, c0, c1) = face_region(part_mask)
    f = img[r0:r1, c0:c1].astype(np.float32) / 255.0
    G = ndimage.gaussian_filter
    maps = [G(f, 1.0), f - G(f, 4.0), f - G(f, 16.0), f - G(f, 48.0),
            np.sqrt(G((f - G(f, 2.0)) ** 2, 2.0)), ndimage.gaussian_gradient_magnitude(f, 1.0),
            _ridge(f, 1.0), _ridge(f, 2.0)]
    F = np.stack(maps, -1)
    h, w = (r1 - r0) // STRIDE, (c1 - c0) // STRIDE
    F = F.reshape(h, STRIDE, w, STRIDE, -1).mean((1, 3))
    valid = face[r0:r1, c0:c1].reshape(h, STRIDE, w, STRIDE).all((1, 3))
    return F, valid, (r0, c0)


class MarkDetector:
    """Memory bank of patch features from good parts only. Score = distance to nearest good patch."""
    def __init__(self, bank, mean, std, threshold):
        self.mean, self.std, self.threshold = mean, std, float(threshold)
        self.bank = bank
        self.nn = NearestNeighbors(n_neighbors=1).fit(bank)

    @classmethod
    def train(cls, good_images, good_masks, holdout_image, holdout_mask, rng, size=25000):
        feats = []
        for im, pm in zip(good_images, good_masks):
            F, valid, _ = patch_features(im, pm)
            feats.append(F[valid])
        X = np.concatenate(feats)
        mean, std = X.mean(0), X.std(0) + 1e-9
        Xn = (X - mean) / std
        idx = rng.choice(len(Xn), min(size, len(Xn)), replace=False)
        det = cls(Xn[idx], mean, std, 1.0)
        s, valid, _ = det.score(holdout_image, holdout_mask)
        det.threshold = float(np.max(s[valid]) * 1.25)
        return det

    def score(self, img, part_mask):
        F, valid, off = patch_features(img, part_mask)
        Xn = (F.reshape(-1, F.shape[-1]) - self.mean) / self.std
        d, _ = self.nn.kneighbors(Xn)
        s = d[:, 0].reshape(F.shape[:2])
        s[~valid] = 0.0
        return s, valid, off

    def regions(self, img, part_mask, px_cal, frame):
        """Connected patches above threshold -> list of marks in the part frame."""
        s, valid, (r0, c0) = self.score(img, part_mask)
        hot = ndimage.binary_closing(s > self.threshold, iterations=1) & valid
        lab, n = ndimage.label(hot)
        marks = []
        for i in range(1, n + 1):
            ii, jj = np.nonzero(lab == i)
            if ii.size < 2:
                continue
            rr = r0 + ii * STRIDE + STRIDE / 2
            cc = c0 + jj * STRIDE + STRIDE / 2
            pw = _pw(rr, cc) * px_cal
            pf = frame(pw)
            cen = pf.mean(0)
            if len(pf) >= 3:
                _, _, vt = np.linalg.svd(pf - cen, full_matrices=False)
                axis = vt[0]
            else:
                axis = np.array([1.0, 0.0])
            along = (pf - cen) @ axis
            across = (pf - cen) @ np.array([-axis[1], axis[0]])
            length = along.max() - along.min() + STRIDE * px_cal
            width = across.max() - across.min() + STRIDE * px_cal
            img_f = img.astype(np.float32)
            contrast = 0.0
            marks.append({"x": float(cen[0]), "y": float(cen[1]), "axis": axis.tolist(),
                          "length_mm": float(length), "width_mm": float(width),
                          "area_mm2": float(ii.size * (STRIDE * px_cal) ** 2),
                          "peak_score": float(s[ii, jj].max() / self.threshold),
                          "patches": int(ii.size)})
        marks.sort(key=lambda m: -m["peak_score"])
        return marks, s, (r0, c0)
