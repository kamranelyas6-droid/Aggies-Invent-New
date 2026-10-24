"""Instrument calibration, done the way it would be done on the cell, but on the simulated sensors.
The results feed the capability registry used by engine.py, so every expanded uncertainty U in
the demo comes from a calibration run, not from a typed number.

Camera (CAM)
  scale      gauge block 50.000 mm imaged 5 times at random poses -> mm per pixel
  repeat     reference plate imaged 5 times at random poses -> largest standard deviation
  distortion residual field left after grid calibration (RMS over the field)
  thermal    6061 aluminium, 23.6e-6 /K, part temperature known to ±0.5 K (uniform), 80 mm
Confocal edge scan (CCF-EDGE)
  ring gauge Ø 2.0000 mm scanned 5 times -> bias (applied as a correction) and repeatability
Confocal height scan (CCF-HEIGHT)
  step height standard 10.000 µm scanned 5 times -> height scale factor and repeatability;
  flatness and roughness repeated on the reference plate for task repeatability
"""
from __future__ import annotations
import json, os, time
from datetime import date
import numpy as np
from . import camera as CAM, confocal as CF, part as P

DATA = os.path.join(os.path.dirname(__file__), "data")
CAL_JSON = os.path.join(DATA, "calibration.json")
BANK_NPZ = os.path.join(DATA, "mark_bank.npz")


class _Ring(CF.Surface):
    """Lapped ring gauge top face: flat, Sa 0.01 µm, one perfectly round Ø 2.0000 mm bore."""
    def __init__(self, seed):
        super().__init__({"Sa_um": 0.01, "bow_um": 0.0, "twist_um": 0.0, "defects": [], "L": 20.0, "W": 20.0,
                          "holes": {"R": (10.0, 10.0, 2.0)}, "lobing_um": 0.0}, seed)


class _Step(CF.Surface):
    """Step height standard: 10.000 µm step at x = 10 mm, polished (Sa 0.005 µm)."""
    def __init__(self, seed):
        super().__init__({"Sa_um": 0.005, "bow_um": 0.0, "twist_um": 0.0, "defects": [], "L": 20.0, "W": 20.0,
                          "holes": {}, "lobing_um": 0.0}, seed)

    def height(self, x, y):
        z = super().height(x, y)
        return z + np.where(np.asarray(x) > 10.0, 10.0, 0.0)


def run(progress=None, n=5):
    say = progress or (lambda msg, f: None)
    rng = np.random.default_rng(2026)
    t0 = time.time()
    # ---------------- camera scale from the gauge block
    say("Camera: imaging the 50 mm gauge block", 0.05)
    Lpx = []
    for k in range(n):
        img = CAM.render({}, CAM.random_pose(rng), rng, kind="gauge")
        Lpx.append(CAM.measure_length(img, 1.0))
    Lpx = np.array(Lpx)
    px_cal = 50.000 / Lpx.mean()
    u_scale = (Lpx.std(ddof=1) / np.sqrt(n)) / Lpx.mean() * 80.0 * 1000       # µm over 80 mm
    # residual distortion RMS over the field (what the grid calibration leaves)
    gx, gy = np.meshgrid(np.linspace(0, 120, 241), np.linspace(0, 80, 161))
    dx, dy = CAM._distortion(gx, gy)
    u_dist = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)) * 1000 * np.sqrt(2))   # two edges per length
    u_thermal = P.PART["cte_per_K"] * 80.0 * 1000 * 0.5 / np.sqrt(3)
    u_cert_gb = 0.05
    u_cal_cam = float(np.sqrt(u_scale ** 2 + u_dist ** 2 + u_thermal ** 2 + u_cert_gb ** 2))

    # ---------------- camera repeatability and verification on the reference plate
    say("Camera: repeat measurements of the reference plate", 0.25)
    rows, imgs, masks = [], [], []
    for k in range(n):
        img = CAM.render(P.REFERENCE, CAM.random_pose(rng), rng)
        m = CAM.measure_part(img, px_cal, P.PART)
        row = [m["L"], m["W"]]
        for h in sorted(m["holes"]):
            row += [m["holes"][h]["d"], m["holes"][h]["x"], m["holes"][h]["y"]]
        rows.append(row)
        imgs.append(img); masks.append(m["part_mask"])
    rows = np.array(rows) * 1000
    u_rep_cam = float(rows.std(0, ddof=1).max())
    truth_vec = [P.REFERENCE["L"], P.REFERENCE["W"]]
    for h in sorted(P.REFERENCE["holes"]):
        x, y, d = P.REFERENCE["holes"][h]
        truth_vec += [d, x, y]
    verify_cam = float(np.abs(rows.mean(0) - np.array(truth_vec) * 1000).max())

    # ---------------- mark detector, trained on good parts only
    say("Camera: learning what a good surface looks like", 0.45)
    det = CAM.MarkDetector.train(imgs[:3], masks[:3], imgs[3], masks[3], rng)
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(BANK_NPZ, bank=det.bank.astype(np.float32), mean=det.mean, std=det.std, threshold=det.threshold)

    # ---------------- confocal edge scan on the ring gauge
    say("Confocal: scanning the Ø 2 mm ring gauge", 0.65)
    ds = []
    for k in range(n):
        sensor = CF.Confocal(_Ring(100 + k), rng)
        off = rng.uniform(-0.005, 0.005, 2)
        ds.append(CF.bore_scan(sensor, (10.0 + off[0], 10.0 + off[1]), 2.0)["d_mm"] * 1000)
    ds = np.array(ds)
    bias_edge = float(ds.mean() - 2000.0)
    u_rep_edge = float(ds.std(ddof=1))
    u_cert_ring = 0.15
    u_cal_edge = float(np.sqrt(u_cert_ring ** 2 + (u_rep_edge / np.sqrt(n)) ** 2))

    # ---------------- confocal height scale on the step height standard
    say("Confocal: scanning the 10 µm step height standard", 0.80)
    steps = []
    xs = np.arange(8.0, 12.0 + 1e-12, 0.001)
    for k in range(n):
        sensor = CF.Confocal(_Step(200 + k), rng)
        z = sensor.read(xs, np.full_like(xs, 10.0 + 0.1 * k))
        steps.append(np.nanmean(z[xs > 10.2]) - np.nanmean(z[xs < 9.8]))
    steps = np.array(steps)
    kz = float(10.000 / steps.mean())
    u_cert_step = 0.015
    u_cal_h = float(np.sqrt(u_cert_step ** 2 + (steps.std(ddof=1) / np.sqrt(n)) ** 2))   # scale factor kz is applied
    # task repeatability on the reference plate: flatness and roughness
    fl, sa = [], []
    for k in range(3):
        sensor = CF.Confocal(CF.Surface(P.REFERENCE, 300), rng)
        fl.append(CF.flatness_grid(sensor, P.PART)["flatness_um"] * kz)
        sa.append(CF.texture_patch(sensor, (30.0, 25.0))["Sa_um"] * kz)
    u_rep_h = float(max(np.std(fl, ddof=1), np.std(sa, ddof=1), steps.std(ddof=1)))

    today = date.today()
    due = today.replace(year=today.year + 1)
    cal = {
        "date": today.isoformat(), "due": due.isoformat(), "seconds": round(time.time() - t0, 1),
        "camera": {"px_cal_mm": px_cal, "px_true_mm": CAM.PX_NOM * (1 + CAM.SCALE_ERR), "gauge_lengths_px": Lpx.tolist(),
                   "u_scale_um": u_scale, "u_dist_um": u_dist, "u_thermal_um": u_thermal, "u_cert_um": u_cert_gb,
                   "u_cal_um": u_cal_cam, "u_repeat_um": u_rep_cam, "verify_max_error_um": verify_cam,
                   "mark_threshold": det.threshold},
        "edge": {"ring_d_um": ds.tolist(), "bias_um": bias_edge, "correction_um": -bias_edge, "u_cert_um": u_cert_ring,
                 "u_cal_um": u_cal_edge, "u_repeat_um": u_rep_edge},
        "height": {"steps_um": steps.tolist(), "kz": kz, "u_cert_um": u_cert_step, "u_cal_um": u_cal_h,
                   "flatness_repeat_um": fl, "Sa_repeat_um": sa, "u_repeat_um": u_rep_h},
    }
    with open(CAL_JSON, "w") as f:
        json.dump(cal, f, indent=1)
    say("Calibration complete", 1.0)
    return cal


def load():
    if not os.path.exists(CAL_JSON) or not os.path.exists(BANK_NPZ):
        return run()
    with open(CAL_JSON) as f:
        return json.load(f)


def detector():
    z = np.load(BANK_NPZ)
    return CAM.MarkDetector(z["bank"].astype(np.float64), z["mean"], z["std"], float(z["threshold"]))


def registry(cal):
    """Capability registry in the schema of the project's instruments.json."""
    c, e, h = cal["camera"], cal["edge"], cal["height"]
    common = {"enabled": True, "calibrated_on": cal["date"], "calibration_due": cal["due"], "access": ["visible"]}
    return [
        dict(common, id="CAM", technique="Top camera, telecentric, 40 µm per pixel (micron level)",
             characteristics=["size", "position"], geometries=["external", "bore"], surfaces=["diffuse", "reflective"],
             field_mm=120, resolution_um=2.0, u_cal_um=round(c["u_cal_um"], 4), u_repeat_um=round(c["u_repeat_um"], 4),
             cycle_s=2, certificate="AMSI-SIM-GB50",
             reference="Gauge block 50.000 mm (ISO 3650) and distortion grid; simulated calibration chain"),
        dict(common, id="CCF-EDGE", technique="Confocal edge scan, lateral (about 1 µm)",
             characteristics=["size"], geometries=["bore"], surfaces=["diffuse", "reflective"],
             field_mm=100, resolution_um=1.0, u_cal_um=round(e["u_cal_um"], 4), u_repeat_um=round(e["u_repeat_um"], 4),
             cycle_s=8, certificate="AMSI-SIM-RING2",
             reference="Ring gauge Ø 2.0000 mm; simulated calibration chain"),
        dict(common, id="CCF-HEIGHT", technique="Confocal height scan (sub micron level)",
             characteristics=["flatness", "texture", "defect"], geometries=["plane"], surfaces=["diffuse", "reflective"],
             field_mm=100, resolution_um=0.01, u_cal_um=round(h["u_cal_um"], 4), u_repeat_um=round(h["u_repeat_um"], 4),
             cycle_s=12, certificate="AMSI-SIM-STEP10",
             reference="Step height standard 10.000 µm; simulated calibration chain"),
    ]
