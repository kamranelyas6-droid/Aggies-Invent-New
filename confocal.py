"""Chromatic confocal sensor (sub micron level).

Simulated sensor, numbers from the AMSI sensor selection (Micro Epsilon confocalDT):
  IFS2405-0.3 pen: 300 µm measuring range, 6 µm spot, 10 nm resolution, ±0.15 µm linearity
  IFD2410 controller: up to 25 kHz
  assumed noise at 25 kHz: 20 nm rms
Motion: X gantry with a 1 µm glass scale; short Y slide with a 0.1 µm encoder.
A through hole is far outside the 300 µm range, so it returns no signal (NaN): that dropout
is what the edge scan uses to find the bore wall.

Four scan types, each answering one kind of requirement:
  bore_scan      360 radial passes across a hole wall -> diameter and roundness (lateral, ~1 µm)
  flatness_grid  short line averages on a grid over the face -> flatness (height, sub micron)
  texture_patch  1 x 1 mm areal map -> Sa, Sq, Sz on the S-L surface (ISO 25178)
  defect_scan    oriented strip or patch over a camera mark -> depth of pits and scratches
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from . import surface as S

RATE_HZ = 25000.0
NOISE_UM = 0.02
LIN_UM = 0.15
RANGE_UM = 300.0
SPOT_MM = 0.006
X_SCALE_MM = 0.001
Y_ENC_MM = 0.0001
MOVE_S = 0.35                 # settle and move between grid points


class Surface:
    """True top face height (µm) as a function of part frame x, y (mm). NaN in holes and off the part."""
    def __init__(self, truth, seed):
        rng = np.random.default_rng(seed)
        self.t = truth
        K = 60
        lam = np.exp(rng.uniform(np.log(0.008), np.log(0.12), K))       # 8 to 120 µm wavelengths
        ang = rng.normal(np.pi / 2, 0.35, K)                              # milling lay across x
        k = 2 * np.pi / lam
        self.kx, self.ky = k * np.cos(ang), k * np.sin(ang)
        self.ph = rng.uniform(0, 2 * np.pi, K)
        sq = truth["Sa_um"] / np.sqrt(2 / np.pi)                          # Gaussian: Sa = 0.798 Sq
        self.a = sq / np.sqrt(K / 2)

    def height(self, x, y):
        t = self.t
        x = np.asarray(x, float); y = np.asarray(y, float)
        u, v = (x - 40.0) / 40.0, (y - 25.0) / 25.0
        z = t["bow_um"] * (u ** 2 + v ** 2) / 2 + t["twist_um"] * u * v
        rough = np.zeros_like(x)
        for kx, ky, ph in zip(self.kx, self.ky, self.ph):
            rough += np.cos(kx * x + ky * y + ph)
        z = z + self.a * rough
        for d in t.get("defects", []):
            if d["kind"] == "pit":
                rr = np.hypot(x - d["x"], y - d["y"])
                z = z - d["depth_um"] * np.clip(1 - (rr / d["r"]) ** 2, 0, None)
            elif d["kind"] == "scratch":
                ax, ay, bx, by = d["x0"], d["y0"], d["x1"], d["y1"]
                vx, vy = bx - ax, by - ay
                tt = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
                dist = np.hypot(x - (ax + tt * vx), y - (ay + tt * vy))
                z = z - d["depth_um"] * np.clip(1 - dist / (d["w"] / 2), 0, None)
        off = (x < 0) | (x > t["L"]) | (y < 0) | (y > t["W"])
        for hid, (hx, hy, hd) in t["holes"].items():
            rx, ry = x - hx, y - hy
            R = hd / 2 + t.get("lobing_um", 0.0) / 1000.0 * np.cos(3 * np.arctan2(ry, rx) + 0.7)
            off |= np.hypot(rx, ry) < R
        return np.where(off, np.nan, z)


class Confocal:
    def __init__(self, surf, rng):
        self.surf, self.rng, self.points, self.seconds = surf, rng, 0, 0.0

    def read(self, xr, yr):
        """Heights (µm) at reported stage positions xr, yr (mm, part frame)."""
        xr = np.asarray(xr, float); yr = np.asarray(yr, float)
        xt = xr + self.rng.uniform(-X_SCALE_MM / 2, X_SCALE_MM / 2, xr.shape)
        yt = yr + self.rng.uniform(-Y_ENC_MM / 2, Y_ENC_MM / 2, yr.shape)
        s = SPOT_MM / 3
        Z = np.stack([self.surf.height(xt + ox, yt + oy) for ox, oy in [(0, 0), (s, 0), (-s, 0), (0, s), (0, -s)]])
        ok = np.isfinite(Z).sum(0) >= 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z = np.nanmean(Z, 0)
        z = z + LIN_UM * np.sin(2 * np.pi * z / RANGE_UM) + self.rng.normal(0, NOISE_UM, z.shape)
        z = np.where(ok, z, np.nan)
        self.points += z.size
        self.seconds += z.size / RATE_HZ
        return z


# ----------------------------------------------------------------------------- scan types
def bore_scan(sensor, centre_xy, d_nom, correction_um=0.0, rays=360, band=0.12, step=0.001):
    """Radial passes across the wall of a hole. centre_xy comes from the camera hand off."""
    p0, s0 = sensor.points, sensor.seconds
    cx, cy = centre_xy
    R = d_nom / 2
    ang = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    rad = np.arange(R - band, R + band + 1e-12, step)
    X = cx + rad[None, :] * np.cos(ang)[:, None]
    Y = cy + rad[None, :] * np.sin(ang)[:, None]
    X = np.round(X / X_SCALE_MM) * X_SCALE_MM                    # positions as the scales report them
    Y = np.round(Y / Y_ENC_MM) * Y_ENC_MM
    Z = sensor.read(X, Y)
    sensor.seconds += rays * 0.004                                # turnaround per pass
    fin = np.isfinite(Z)
    first = np.argmax(fin, 1)
    ok = fin.any(1) & (first > 0)
    k = np.nonzero(ok)[0]
    i = first[k]
    ex = 0.5 * (X[k, i - 1] + X[k, i])
    ey = 0.5 * (Y[k, i - 1] + Y[k, i])
    from .camera import fit_circle
    fx, fy, fr, rms = fit_circle(np.c_[ex, ey])
    d = 2 * fr + correction_um / 1000.0
    res_raw = (np.hypot(ex - fx, ey - fy) - fr) * 1000
    th = np.arctan2(ey - fy, ex - fx)
    cols = [np.ones_like(th)] + [f(n * th) for n in range(1, 16) for f in (np.cos, np.sin)]
    Hm = np.stack(cols, 1)
    coef = np.linalg.lstsq(Hm, res_raw, rcond=None)[0]
    res_um = Hm @ coef - coef[0]                                  # 15 UPR roundness profile
    # overview map for display (10 µm grid)
    half = R + 0.3
    gstep = max(0.010, round(2 * half / 250, 3))                  # overview grid, at most ~250 x 250 points
    g = np.arange(-half, half + 1e-9, gstep)
    GX, GY = np.meshgrid(cx + g, cy + g)
    M = sensor.read(GX, GY)
    return {"kind": "bore", "d_mm": float(d), "centre": (float(fx), float(fy)), "roundness_um": float(res_um.max() - res_um.min()),
            "edge_angles": th, "edge_res_um": res_um, "edge_res_raw_um": res_raw, "upr": 15, "edges_used": int(k.size), "rays": rays,
            "map": M, "map_x": cx + g, "map_y": cy + g, "points": sensor.points - p0, "seconds": sensor.seconds - s0,
            "step_um": step * 1000}


def flatness_grid(sensor, truth_nominal, nx=9, ny=6, inset=5.0, keep_out=1.5, line_mm=0.2):
    """Grid of short line averages on the top face; least squares plane; flatness = PV of residuals.
    (Least squares flatness is never smaller than minimum zone flatness, so it is conservative.)"""
    p0, s0 = sensor.points, sensor.seconds
    L, W = truth_nominal["L"], truth_nominal["W"]
    xs, ys = np.linspace(inset, L - inset, nx), np.linspace(inset, W - inset, ny)
    pts = []
    for y in ys:
        for x in xs:
            if all(np.hypot(x - hx, y - hy) > hd / 2 + keep_out for hx, hy, hd in truth_nominal["holes"].values()):
                pts.append((x, y))
    pts = np.array(pts)
    z = []
    line = np.arange(-line_mm / 2, line_mm / 2 + 1e-12, 0.001)
    for x, y in pts:
        zz = sensor.read(np.round((x + line) / X_SCALE_MM) * X_SCALE_MM, np.full_like(line, y))
        z.append(np.nanmean(zz))
        sensor.seconds += MOVE_S
    z = np.array(z)
    A = np.c_[np.ones(len(pts)), pts]
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    res = z - A @ coef
    gx, gy = np.linspace(0, L, 161), np.linspace(0, W, 101)
    GX, GY = np.meshgrid(gx, gy)
    Mi = griddata(pts, res, (GX, GY), method="cubic")
    Mn = griddata(pts, res, (GX, GY), method="nearest")
    Mi = np.where(np.isfinite(Mi), Mi, Mn)
    for hx, hy, hd in truth_nominal["holes"].values():
        Mi[np.hypot(GX - hx, GY - hy) < hd / 2] = np.nan
    return {"kind": "flatness", "flatness_um": float(res.max() - res.min()), "points_xy": pts, "residual_um": res,
            "map": Mi, "map_x": gx, "map_y": gy, "n_points": int(len(pts)),
            "points": sensor.points - p0, "seconds": sensor.seconds - s0}


def texture_patch(sensor, centre_xy, size=1.0, step=0.002, lc_mm=0.25):
    """Areal patch. Form removed (plane), then a Gaussian L filter (cut off lc) gives the S-L surface."""
    p0, s0 = sensor.points, sensor.seconds
    cx, cy = centre_xy
    g = np.arange(-size / 2, size / 2 + 1e-12, step)
    GX, GY = np.meshgrid(cx + g, cy + g)
    Z = sensor.read(np.round(GX / X_SCALE_MM) * X_SCALE_MM, GY)
    Rf, _, valid = S.remove_form(Z)
    sigma_px = 0.1874 * lc_mm / step                       # ISO 16610-21 Gaussian weighting
    waviness = ndimage.gaussian_filter(np.nan_to_num(Rf), sigma_px, mode="reflect")
    SL = Rf - waviness
    tex = S.areal_parameters(SL, valid)
    return {"kind": "texture", "texture": tex, "Sa_um": tex["Sa_um"], "map": SL, "raw": Z,
            "map_x": cx + g, "map_y": cy + g, "lc_mm": lc_mm, "step_um": step * 1000,
            "points": sensor.points - p0, "seconds": sensor.seconds - s0}


def _section_depth(t, z, core_frac=0.35):
    """Depth of a groove in one cross section: straight reference through the outer parts of the
    profile (undisturbed surface), depth = reference minus the minimum of a 3 point median."""
    ok = np.isfinite(z)
    if ok.sum() < 20:
        return np.nan, None
    tt, zz = t[ok], z[ok]
    outer = np.abs(tt) > core_frac * np.abs(tt).max()
    c = np.polyfit(tt[outer], zz[outer], 1)
    r = zz - np.polyval(c, tt)
    rm = ndimage.median_filter(r, 3)
    i = int(np.argmin(rm))
    return float(-rm[i]), float(tt[i])


def defect_scan(sensor, mark, margin=0.25):
    """Confocal check of one camera mark.
    Elongated marks (scratches): cross sections every 0.1 mm along the mark, 1 µm step across.
    Compact marks (pits, stains): areal map, 2 to 5 µm step, then pit / scratch / bump analysis."""
    p0, s0 = sensor.points, sensor.seconds
    ax = np.asarray(mark["axis"], float); ax /= np.linalg.norm(ax)
    nrm = np.array([-ax[1], ax[0]])
    elongated = mark["length_mm"] >= 4 * max(mark["width_mm"], 0.16)
    if elongated:
        Ls = mark["length_mm"] + 2 * margin
        Lt = max(mark["width_mm"] + 2 * margin, 0.6)
        s = np.arange(-Ls / 2, Ls / 2 + 1e-12, 0.1)
        t = np.arange(-Lt / 2, Lt / 2 + 1e-12, 0.001)
        SS, TT = np.meshgrid(s, t, indexing="ij")
        X = mark["x"] + SS * ax[0] + TT * nrm[0]
        Y = mark["y"] + SS * ax[1] + TT * nrm[1]
        Z = sensor.read(np.round(X / X_SCALE_MM) * X_SCALE_MM, np.round(Y / Y_ENC_MM) * Y_ENC_MM)
        sensor.seconds += len(s) * 0.004
        depths, pos = [], []
        R = np.full_like(Z, np.nan)
        for k in range(len(s)):
            d, tpos = _section_depth(t, Z[k])
            depths.append(d); pos.append(tpos)
            ok = np.isfinite(Z[k])
            if ok.sum() > 20:
                outer = ok & (np.abs(t) > 0.35 * np.abs(t).max())
                c = np.polyfit(t[outer], Z[k][outer], 1)
                R[k] = Z[k] - np.polyval(c, t)
        depths = np.array(depths)
        sig = S.robust_sigma(R[:, np.abs(t) > 0.35 * np.abs(t).max()])
        real = np.isfinite(depths) & (depths > 5 * sig)
        k0 = int(np.nanargmax(depths)) if np.isfinite(depths).any() else 0
        worst = float(depths[k0]) if real.any() else 0.0
        length = float(real.sum() * 0.1)
        defects = []
        if real.any():
            defects.append({"id": 1, "type": "scratch", "depth_um": round(worst, 3), "length_um": round(length * 1000, 1),
                            "at_s_mm": round(float(s[k0]), 2), "severity_x_sigma": round(worst / sig, 1) if sig else None})
        prof = {"axis": "across", "dist_mm": t, "z_um": R[k0]}
        an = {"defects": defects, "defect_summary": {"count": len(defects), "worst_depth_um": round(worst, 3)},
              "sigma_bg_um": sig, "sections": int(len(s)), "depth_along": depths}
        out = {"mode": "cross sections", "map": R.T, "map_s": s, "map_t": t, "step_um": 1.0, "section_pitch_mm": 0.1}
    else:
        Ls = max(mark["length_mm"], mark["width_mm"]) + 2 * margin
        step = float(np.clip(np.ceil(1000 * Ls / 600) / 1000, 0.002, 0.005))
        s = np.arange(-Ls / 2, Ls / 2 + 1e-12, step)
        t = s.copy()
        SS, TT = np.meshgrid(s, t)
        X = mark["x"] + SS * ax[0] + TT * nrm[0]
        Y = mark["y"] + SS * ax[1] + TT * nrm[1]
        Z = sensor.read(np.round(X / X_SCALE_MM) * X_SCALE_MM, np.round(Y / Y_ENC_MM) * Y_ENC_MM)
        an = S.analyze_height_map(Z, step * 1000)
        worst = float(an["defect_summary"]["worst_depth_um"])
        R, _, _ = S.remove_form(Z)
        an["sigma_bg_um"] = S.robust_sigma(R[np.isfinite(R)])
        if an["defects"]:
            d0 = an["defects"][0]
            row = min(max(int(round(d0["y_um"] / (step * 1000))), 0), R.shape[0] - 1)
            prof = {"axis": "across", "dist_mm": s, "z_um": R[row, :]}
        else:
            prof = {"axis": "across", "dist_mm": s, "z_um": R[R.shape[0] // 2, :]}
        out = {"mode": "areal map", "map": R, "map_s": s, "map_t": t, "step_um": step * 1000}
    finding = "no measurable depth (cosmetic)" if worst <= 0 else ", ".join(sorted({d["type"] for d in an["defects"]}))
    out.update({"kind": "defect", "depth_um": worst, "analysis": an, "axis": ax.tolist(), "centre": (mark["x"], mark["y"]),
                "profile": prof, "finding": finding, "points": sensor.points - p0, "seconds": sensor.seconds - s0})
    return out
