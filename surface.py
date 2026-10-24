"""
Confocal sub-micron analysis engine for AMSI.

Input:  a confocal height map Z(x, y) in micrometres (the native output of a chromatic
        confocal sensor: one height per sampled point; NaN where the surface dropped out).
Output: one JSON-serializable dict with
          - form      : fitted datum and flatness (PV of residuals)
          - texture   : ISO 25178 areal parameters on the form-removed surface
          - defects   : a list of pits, scratches and bumps, each with depth, area, position
          - decision  : per-characteristic PASS / FAIL / REVIEW using an ISO 14253-1 guard band

Depends only on numpy and scipy, so it drops into any Python backend. The UI consumes the
returned dict as JSON; nothing here imports the rest of the AMSI code.

References:
  ISO 25178-2  areal surface texture parameters (Sa, Sq, Sz, Sp, Sv, Ssk, Sku)
  ISO 25178-3  specification operators (S-F surface = form removed)
  ISO 14253-1  decision rule with a guard band g = k * u_c (here k = 1.65, ~95%)
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def robust_sigma(values):
    """Background roughness sigma that ignores defects: 1.4826 x median absolute deviation."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    mad = np.median(np.abs(v - np.median(v)))
    return float(1.4826 * mad) if mad > 0 else float(v.std())


# --------------------------------------------------------------------------- form removal
def remove_form(Z_um, order="plane", mask=None):
    """Fit and subtract the nominal form so only texture and defects remain (ISO 25178 S-F surface).
    order: 'plane' for a flat face, 'poly2' for a gently curved face.
    Returns (residual_um, fitted_um, valid_mask)."""
    Z = np.asarray(Z_um, float)
    valid = np.isfinite(Z)
    if mask is not None:
        valid &= mask
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    xf, yf, zf = xx[valid], yy[valid], Z[valid]
    if order == "plane":
        A = np.c_[np.ones_like(xf), xf, yf]
    else:  # second order (removes bow and twist)
        A = np.c_[np.ones_like(xf), xf, yf, xf * xf, xf * yf, yf * yf]
    coef, *_ = np.linalg.lstsq(A, zf, rcond=None)
    for _ in range(2):                                   # robust refit: ignore defect points
        r = zf - A @ coef
        keep = np.abs(r - np.median(r)) <= 3.0 * robust_sigma(r)
        if keep.sum() > A.shape[1] * 10:
            coef, *_ = np.linalg.lstsq(A[keep], zf[keep], rcond=None)
    if order == "plane":
        fit = coef[0] + coef[1] * xx + coef[2] * yy
    else:
        fit = coef[0] + coef[1] * xx + coef[2] * yy + coef[3] * xx * xx + coef[4] * xx * yy + coef[5] * yy * yy
    R = np.where(valid, Z - fit, np.nan)
    return R, fit, valid


# --------------------------------------------------------------------------- ISO 25178 texture
def areal_parameters(R_um, valid=None):
    """ISO 25178-2 field parameters on a form-removed height map R (micrometres)."""
    R = np.asarray(R_um, float)
    v = np.isfinite(R) if valid is None else (valid & np.isfinite(R))
    r = R[v]
    n = r.size
    if n == 0:
        return {}
    Sq = float(np.sqrt(np.mean(r ** 2)))
    Sa = float(np.mean(np.abs(r)))
    Sp = float(np.max(r))
    Sv = float(-np.min(r))
    out = {
        "Sa_um": Sa,                               # mean absolute height (overall roughness)
        "Sq_um": Sq,                               # root mean square height
        "Sp_um": Sp,                               # highest peak above the mean plane
        "Sv_um": Sv,                               # deepest valley below the mean plane
        "Sz_um": float(Sp + Sv),                   # total height, peak to valley
        "Ssk": float(np.mean(r ** 3) / Sq ** 3) if Sq > 0 else 0.0,   # skewness (<0 = pitted, >0 = peaked)
        "Sku": float(np.mean(r ** 4) / Sq ** 4) if Sq > 0 else 0.0,   # kurtosis (>3 = spiky)
        "points": int(n),
    }
    return out


# --------------------------------------------------------------------------- defect detection
def _ridge_valley(R, sigma):
    """Most-negative eigenvalue of the smoothed Hessian: strong where there is a thin valley
    (a scratch). Returns an array that is large and positive inside dark line-like grooves."""
    Rf = np.nan_to_num(R)
    gxx = ndimage.gaussian_filter(Rf, sigma, order=(0, 2))
    gyy = ndimage.gaussian_filter(Rf, sigma, order=(2, 0))
    gxy = ndimage.gaussian_filter(Rf, sigma, order=(1, 1))
    tmp = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0.0))
    lam_max = 0.5 * (gxx + gyy + tmp)              # largest Hessian eigenvalue; valley => large positive
    return lam_max


def detect_defects(R_um, pixel_um, valid=None, Sq_um=None,
                   pit_k=5.0, bump_k=5.0, scratch_k=4.0, min_area_um2=None):
    """Find pits, bumps and scratches in a form-removed height map.
    A feature counts only if it is deeper/higher than k times the background roughness Sq,
    so ordinary texture is never flagged. Returns a list of defect dicts (positions in µm,
    measured from the top-left of the map)."""
    R = np.asarray(R_um, float)
    v = np.isfinite(R) if valid is None else (valid & np.isfinite(R))
    if Sq_um is None:
        Sq_um = robust_sigma(R[v])
    px = float(pixel_um)
    area_px_min = max(4, int((min_area_um2 or max((3 * px) ** 2, 300.0)) / (px * px)))   # at least ~3 spots
    defects = []
    taken = np.zeros(R.shape, bool)

    def is_new(cx, cy, tol_um):
        return all(abs(cx - d["x_um"]) > tol_um or abs(cy - d["y_um"]) > tol_um for d in defects)

    def components(binary, kind):                     # (re-bound to carry the dedup guard)
        nonlocal defects, taken
        binary = binary & v
        binary = ndimage.binary_closing(binary, iterations=1)
        lab, n = ndimage.label(binary)
        for i in range(1, n + 1):
            m = lab == i
            if m.sum() < area_px_min:
                continue
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean() * px, xs.mean() * px
            if not is_new(cx, cy, 3 * px) or (m & taken).any():   # already found in an earlier pass
                continue
            taken |= ndimage.binary_dilation(m, iterations=2)
            vals = R[m]
            h_um, w_um = (ys.max() - ys.min() + 1) * px, (xs.max() - xs.min() + 1) * px
            length = max(h_um, w_um); width = max(min(h_um, w_um), px)
            grow = max(3, int(0.6 * np.sqrt(m.sum() / np.pi)))     # ring at about 1.6 x the feature radius
            ring = ndimage.binary_dilation(m, iterations=grow + 3) & ~ndimage.binary_dilation(m, iterations=grow) & v
            ref = float(np.median(R[ring])) if ring.any() else 0.0
            if kind == "bump":
                extent = float(vals.max() - ref); key = "height_um"
            else:
                extent = float(ref - vals.min()); key = "depth_um"
            if extent < (bump_k if kind == "bump" else pit_k) * Sq_um:   # must stay deep against its own ring
                continue
            aspect = length / width
            typ = "scratch" if (kind == "pit" and aspect >= 3.0) else kind
            defects.append({
                "type": typ, "x_um": round(cx, 2), "y_um": round(cy, 2), key: round(extent, 3),
                "area_um2": round(int(m.sum()) * px * px, 1), "length_um": round(length, 1),
                "width_um": round(width, 1), "aspect_ratio": round(aspect, 1),
                "severity_x_Sq": round(extent / Sq_um, 1) if Sq_um > 0 else None})

    components(R < -pit_k * Sq_um, "pit")           # deep pits and (by aspect) scratches
    components(R > bump_k * Sq_um, "bump")           # raised bumps / burrs
    # dedicated thin-scratch pass: Hessian valley lines that the depth threshold alone can miss
    ridge = _ridge_valley(R, sigma=max(1.0, 2.0))
    rv = ridge[v]
    if rv.size:
        thr = rv.mean() + scratch_k * rv.std()
        n0 = len(defects)
        components((ridge > thr) & (R < -pit_k * Sq_um), "pit")
        for d in defects[n0:]:
            if d["type"] == "pit":
                d["type"] = "scratch"
    # sort worst first
    defects.sort(key=lambda d: d.get("depth_um", d.get("height_um", 0)), reverse=True)
    for k, d in enumerate(defects, 1):
        d["id"] = k
    return defects


# --------------------------------------------------------------------------- decision rule
def guard_band_decision(value, limit, u_c, k=1.65, lower=False):
    """ISO 14253-1 conformance with a guard band. Returns PASS / FAIL / REVIEW.
    value, limit, u_c in the same units. lower=False => conform when value <= limit (an upper limit,
    e.g. a roughness ceiling or a max allowed defect depth)."""
    g = k * u_c
    if not lower:
        if value <= limit - g:
            return "PASS"
        if value > limit + g:
            return "FAIL"
        return "REVIEW"
    else:
        if value >= limit + g:
            return "PASS"
        if value < limit - g:
            return "FAIL"
        return "REVIEW"


# --------------------------------------------------------------------------- top level
def analyze_height_map(Z_um, pixel_um, tol=None, form="plane", u=None):
    """Full sub-micron analysis of one confocal height map.
    Z_um      : 2D array of heights in micrometres (NaN = dropout).
    pixel_um  : lateral sampling, micrometres per pixel.
    tol       : optional dict of acceptance limits, e.g.
                {"Sa_um": 0.8, "flatness_um": 5.0, "max_defect_depth_um": 10.0}
    u         : optional dict of measurement uncertainties (1 sigma, µm) for the guard band;
                sensible defaults are used when missing.
    Returns a JSON-serializable dict."""
    Z = np.asarray(Z_um, float)
    R, fit, valid = remove_form(Z, order=form)
    rv = R[valid]
    flatness = float(np.max(rv) - np.min(rv)) if rv.size else None       # raw peak to valley (includes defects)
    # robust flatness = form of the bulk surface with pits/scratches excluded.
    # MAD gives a defect-insensitive sigma; keep pixels within 4 sigma of the median, PV of those.
    if rv.size:
        med = np.median(rv)
        sigma = 1.4826 * np.median(np.abs(rv - med)) or (rv.std() or 1e-9)
        bulk = rv[np.abs(rv - med) <= 4 * sigma]
        flatness_rob = float(bulk.max() - bulk.min()) if bulk.size else flatness
    else:
        flatness_rob = None
    tex = areal_parameters(R, valid)
    sigma_bg = robust_sigma(rv)
    defects = detect_defects(R, pixel_um, valid, Sq_um=sigma_bg)
    worst = max((d.get("depth_um", 0.0) for d in defects), default=0.0)

    result = {
        "scale": "sub-micron (confocal)",
        "map": {"rows": int(Z.shape[0]), "cols": int(Z.shape[1]), "pixel_um": float(pixel_um),
                "field_um": [round(Z.shape[1] * pixel_um, 1), round(Z.shape[0] * pixel_um, 1)],
                "valid_fraction": round(float(valid.mean()), 3)},
        "form": {"model": form, "flatness_um": round(flatness, 3) if flatness is not None else None,
                 "flatness_robust_um": round(flatness_rob, 3) if flatness_rob is not None else None},
        "texture": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tex.items()},
        "defects": defects,
        "defect_summary": {"count": len(defects),
                           "pits": sum(d["type"] == "pit" for d in defects),
                           "scratches": sum(d["type"] == "scratch" for d in defects),
                           "bumps": sum(d["type"] == "bump" for d in defects),
                           "worst_depth_um": round(worst, 3)},
    }

    if tol:
        u = u or {}
        u_tex = u.get("Sa_um", 0.04)             # confocal Sa uncertainty (default from AMSI calibration)
        u_flat = u.get("flatness_um", 1.25)
        u_dep = u.get("max_defect_depth_um", 1.51)
        checks = []
        if "Sa_um" in tol and "Sa_um" in tex:
            checks.append(_row("Surface roughness Sa", tex["Sa_um"], tol["Sa_um"], u_tex, "um"))
        if "flatness_um" in tol and flatness_rob is not None:
            checks.append(_row("Flatness (form, defects excluded)", flatness_rob, tol["flatness_um"], u_flat, "um"))
        if "max_defect_depth_um" in tol:
            checks.append(_row("Worst defect depth", worst, tol["max_defect_depth_um"], u_dep, "um"))
        verdicts = [c["verdict"] for c in checks]
        overall = "FAIL" if "FAIL" in verdicts else ("REVIEW" if "REVIEW" in verdicts else "PASS")
        result["decision"] = {"overall": overall, "checks": checks}
    return result


def _row(name, value, limit, u_c, unit, lower=False):
    v = guard_band_decision(value, limit, u_c, lower=lower)
    return {"characteristic": name, "measured": round(float(value), 3), "limit": float(limit),
            "uncertainty": float(u_c), "guard_band": round(1.65 * u_c, 3), "unit": unit, "verdict": v}


# --------------------------------------------------------------------------- line profile (point pen on the gantry)
def analyze_profile(z_um, step_um, tol=None):
    """A single confocal line scan (what the point pen gives as the carriage sweeps).
    Returns waviness-removed roughness Ra/Rz and the deepest groove it crosses."""
    z = np.asarray(z_um, float)
    x = np.arange(z.size) * step_um
    v = np.isfinite(z)
    A = np.c_[np.ones(v.sum()), x[v]]
    c, *_ = np.linalg.lstsq(A, z[v], rcond=None)
    r = z - (c[0] + c[1] * x)
    r = r[v]
    Ra = float(np.mean(np.abs(r)))
    Rz = float(np.max(r) - np.min(r))
    # deepest groove: reference = median of the outer thirds, depth = reference - filtered minimum
    k = max(3, int(round(0.02 / (step_um / 1000) )) | 1) if step_um else 5
    med = ndimage.median_filter(r, size=min(k, max(3, r.size // 5)) | 1)
    depth = float(np.median(np.r_[r[:r.size // 3], r[-r.size // 3:]]) - med.min())
    out = {"scale": "sub-micron (confocal line)", "Ra_um": round(Ra, 4), "Rz_um": round(Rz, 4),
           "deepest_groove_um": round(depth, 3), "points": int(v.sum()), "length_um": round(x[-1], 1)}
    if tol and "Ra_um" in tol:
        out["decision"] = _row("Roughness Ra", Ra, tol["Ra_um"], 0.02, "um")
    return out


if __name__ == "__main__":
    import json
    rng = np.random.default_rng(0)
    Z = 0.4 * rng.standard_normal((200, 200))          # ~0.4 µm roughness
    Z += np.linspace(0, 3, 200)[None, :]               # slight tilt (form)
    Z[90:95, 40:170] -= 8.0                             # a scratch, 8 µm deep
    yy, xx = np.mgrid[0:200, 0:200]
    Z[(xx - 150) ** 2 + (yy - 60) ** 2 < 36] -= 25.0    # a pit, 25 µm deep
    res = analyze_height_map(Z, pixel_um=5.0,
                             tol={"Sa_um": 0.8, "flatness_um": 5.0, "max_defect_depth_um": 10.0})
    print(json.dumps(res, indent=2))
