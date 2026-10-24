"""One complete inspection: camera survey -> mark discovery -> planning with the project's engine
-> confocal scans only where the engine asks for them -> one verdict with evidence.

The engine (engine.py from the project repository) decides everything that is a decision:
which instrument may measure which requirement (4:1 uncertainty rule, field, surface, calibration
dates), when a result is too close to call and needs a better instrument, and the final
Conforming / Nonconforming / Indeterminate class. This module only supplies measured values when
the engine asks for them ("Awaiting measurement"), exactly as the repository README describes for
a hardware adapter.
"""
from __future__ import annotations
import copy, math, time
import numpy as np
import engine
from . import camera as CAM, confocal as CF, part as P, calibration as CALIB

LEVEL = {"CAM": "Micron (camera)", "CCF-EDGE": "Sub micron (confocal edge, lateral)", "CCF-HEIGHT": "Sub micron (confocal height)"}
TRAVEL_MM_S = 50.0
SETTLE_S = 0.3


def _camera_values(m):
    holes = m["holes"]
    pos = max(holes[h]["pos_err_um"] for h in ("H1", "H2", "H3", "H4") if h in holes)
    return {"F01": m["L"] * 1000, "F02": m["W"] * 1000, "F03": holes["H1"]["d"] * 1000, "F04": pos,
            "F05": holes["H5"]["d"] * 1000}


def run_inspection(scenario, ratio=4, registration_u=0.02, environment_u=0.02, seed_offset=0, progress=None):
    say = progress or (lambda msg, f: None)
    t_wall = time.time()
    truth = P.SCENARIOS[scenario]
    cal = CALIB.load()
    det = CALIB.detector()
    instruments = CALIB.registry(cal)
    px_cal = cal["camera"]["px_cal_mm"]
    rng = np.random.default_rng(truth["seed"] + seed_offset)

    # ---------------------------------------------------------------- 1. camera survey
    say("Camera: capturing the top view", 0.05)
    pose = CAM.random_pose(rng)
    img = CAM.render(truth, pose, rng)
    say("Camera: measuring edges, holes and the datum frame", 0.15)
    m = CAM.measure_part(img, px_cal, P.PART)
    say("Camera: scanning the surface for marks", 0.25)
    marks, score, score_off = det.regions(img, m["part_mask"], px_cal, m["frame"])
    cam_vals = _camera_values(m)

    # camera to stage hand off: measured part pose (origin and angle) versus the true pose
    O, ex = m["origin"], m["ex"]
    ang = math.atan2(ex[1], ex[0])
    T = np.eye(4)
    T[:2, :2] = [[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]
    T[:3, 3] = [O[0], O[1], 0.0]
    d_ang = pose[2] - ang
    d_org = np.array([pose[0], pose[1]]) - O
    handoff_err_um = float(np.hypot(*d_org) * 1000)

    # ---------------------------------------------------------------- 2. requirements (+ discovered marks)
    features = copy.deepcopy(P.REQUIREMENTS)
    mark_by_id = {}
    for k, mk in enumerate(marks, 1):
        roi = max(mk["length_mm"], mk["width_mm"]) + 0.5
        f = P.defect_feature(k, mk["x"], mk["y"], roi)
        features.append(f)
        mark_by_id[f["id"]] = mk

    # ---------------------------------------------------------------- 3. confocal, on demand
    surf = CF.Surface(truth, truth["seed"] * 7 + seed_offset)
    sensor = CF.Confocal(surf, np.random.default_rng(truth["seed"] * 13 + seed_offset))
    kz = cal["height"]["kz"]
    corr = cal["edge"]["correction_um"]
    loc_mm = engine.uncertainty(instruments[0], registration_u, environment_u) / 1000.0
    jobs = []

    def acquire(fid, iid):
        f = next(x for x in features if x["id"] == fid)
        if iid == "CAM":
            return cam_vals[fid], None
        say(f"Confocal: {f['label']}", 0.35 + 0.5 * len(jobs) / max(1, len(features)))
        if iid == "CCF-EDGE":
            hid = "H1" if fid == "F03" else "H5"
            hx, hy = m["holes"][hid]["x"], m["holes"][hid]["y"]
            r = CF.bore_scan(sensor, (hx, hy), P.PART["holes"][hid][2], correction_um=corr)
            r.update(hole=hid, handoff_xy=(hx, hy))
            val = r["d_mm"] * 1000
        elif fid == "F06":
            r = CF.flatness_grid(sensor, P.PART)
            r["flatness_um"] *= kz; r["residual_um"] = r["residual_um"] * kz; r["map"] = r["map"] * kz
            val = r["flatness_um"]
        elif fid == "F07":
            r = CF.texture_patch(sensor, tuple(f["xyz_mm"][:2]))
            val = r["Sa_um"] * kz
            r["Sa_um"] = val
        else:
            r = CF.defect_scan(sensor, mark_by_id[fid])
            r["depth_um"] *= kz
            val = r["depth_um"]
        r.update(feature_id=fid, instrument_id=iid, value_um=float(val), label=f["label"], xy=tuple(f["xyz_mm"][:2]))
        jobs.append(r)
        return float(val), r

    observations = [{"feature_id": fid, "instrument_id": "CAM", "value_um": float(v)} for fid, v in cam_vals.items()]
    survey = f"Top camera, 40 µm per pixel, telecentric, calibrated (U = {loc_mm * 1000:.2f} µm)"
    report = None
    for _ in range(12):
        report = engine.run(features, instruments, ratio=ratio, registration_u=registration_u, environment_u=environment_u,
                            localization_error_mm=loc_mm, observations=observations, transform=T.tolist(),
                            survey_adapter=survey)
        waiting = [r for r in report["results"] if r["status"] == "Awaiting measurement"]
        if not waiting:
            break
        for r in waiting:
            val, _ = acquire(r["feature_id"], r["instrument_id"])
            observations.append({"feature_id": r["feature_id"], "instrument_id": r["instrument_id"], "value_um": val})

    # ---------------------------------------------------------------- 4. timeline (cell time, not computer time)
    say("Writing the report", 0.95)
    timeline = [{"step": "Camera survey (capture + measure)", "instrument": "CAM", "start": 0.0, "seconds": 2.0}]
    tcur, here = 2.0, (P.PART["L"] / 2, P.PART["W"] / 2)
    for j in jobs:
        dest = j.get("handoff_xy") or j["xy"]
        move = math.hypot(dest[0] - here[0], dest[1] - here[1]) / TRAVEL_MM_S + SETTLE_S
        timeline.append({"step": "Move", "instrument": "move", "start": tcur, "seconds": move})
        tcur += move
        timeline.append({"step": f"{j['feature_id']} {j['label']}", "instrument": j["instrument_id"], "start": tcur,
                         "seconds": float(j["seconds"])})
        tcur += float(j["seconds"])
        here = dest

    # ---------------------------------------------------------------- 5. one table for both levels
    feat_by_id = {f["id"]: f for f in features}
    rows = []
    for r in report["results"]:
        f = feat_by_id[r["feature_id"]]
        row = {"Feature": f["id"], "What": f["label"], "Requirement": f["requirement"], "Status": r["status"],
               "Instrument": r.get("instrument_id", ""), "Level": LEVEL.get(r.get("instrument_id", ""), ""),
               "Measured": r.get("value_um"), "U_um": r.get("U_um"), "Reason": r.get("reason", "")}
        rows.append(row)
    statuses = [r["Status"] for r in rows]
    if "Nonconforming" in statuses:
        verdict = "REJECT"
    elif any(s not in ("Conforming", "Nonconforming") for s in statuses):
        verdict = "REVIEW"
    else:
        verdict = "ACCEPT"
    refined = [e["feature_id"] for e in report["events"] if e["state"] == "REFINE"]
    return {
        "scenario": scenario, "truth": truth, "verdict": verdict, "rows": rows, "report": report, "features": features,
        "instruments": instruments, "calibration": cal, "camera": {"image": img, "meas": m, "marks": marks, "score": score,
        "score_off": score_off, "threshold": det.threshold, "values": cam_vals, "pose": pose, "px_cal": px_cal},
        "jobs": jobs, "timeline": timeline, "cycle_s": tcur, "refined": refined, "handoff_err_um": handoff_err_um,
        "policy": {"ratio": ratio, "registration_u": registration_u, "environment_u": environment_u, "loc_mm": loc_mm},
        "compute_s": time.time() - t_wall,
    }
