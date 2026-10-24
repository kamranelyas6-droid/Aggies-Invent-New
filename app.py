"""AMSI demo: adaptive multi scale inspection. Run:  streamlit run app.py"""
import hashlib, io, json, os, sys, types
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(HERE, "amsi")):
    # files were uploaded without their folders: treat this folder as the amsi package
    _pkg = types.ModuleType("amsi"); _pkg.__path__ = [HERE]; sys.modules["amsi"] = _pkg
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from amsi import pipeline as PL, figures as FG, part as P, calibration as CALIB

RIG_IMG = next((p for p in (os.path.join(HERE, "assets", "rig_v4_labeled.png"), os.path.join(HERE, "rig_v4_labeled.png"))
                if os.path.exists(p)), None)

st.set_page_config(page_title="AMSI · Adaptive Multi Scale Inspection", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.verdict {border-radius: 10px; padding: 16px 22px; color: white; font-size: 30px; letter-spacing: 1px;}
.verdict small {display:block; font-size: 16px; opacity: .92; letter-spacing: 0;}
.step {background:#f3f5f8; border-radius:10px; padding:14px 16px; height: 100%;}
.step h4 {margin: 0 0 6px 0; color:#1f4e79;}
.pill {display:inline-block; padding:2px 10px; border-radius: 999px; color: white; font-size: 13px;}
</style>""", unsafe_allow_html=True)

STATUS_BG = {"Conforming": "#dcfce7", "Nonconforming": "#fee2e2", "Indeterminate": "#fef3c7", "Blocked": "#e5e7eb",
             "Awaiting measurement": "#e5e7eb"}
VERDICT = {"ACCEPT": ("#16a34a", "Every requirement is proven in tolerance, including its measurement uncertainty."),
           "REJECT": ("#dc2626", "At least one requirement is proven out of tolerance."),
           "REVIEW": ("#d97706", "At least one requirement could not be decided by any instrument in the cell.")}


def fmt(feature, v):
    if v is None:
        return "—"
    c = feature["characteristic"]
    if c == "size":
        return f"{v / 1000:.4f} mm"
    if c == "position":
        return f"⌀ {v / 1000:.4f} mm"
    if c == "flatness":
        return f"{v:.2f} µm"
    if c == "texture":
        return f"Sa {v:.3f} µm"
    if c == "defect":
        return "no depth" if v <= 0 else f"{v:.1f} µm deep"
    return f"{v:.3f} µm"


def colour_status(col):
    return [f"background-color: {STATUS_BG.get(v, '')}" for v in col]


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("AMSI demo")
    st.caption("Camera at micron level. Confocal at sub micron level. One verdict.")
    scenario = st.radio("1 · Choose a part", list(P.SCENARIOS), index=2)
    st.caption(P.SCENARIOS[scenario]["desc"])
    run_clicked = st.button("2 · Run inspection", type="primary", width="stretch")
    with st.expander("Accuracy policy"):
        ratio = st.slider("Tolerance / maximum expanded uncertainty", 2, 10, 4,
                          help="4 means an instrument may measure a requirement only if its U is at most a quarter of the tolerance.")
        registration = st.number_input("Local registration standard uncertainty (µm)", 0.0, 5.0, 0.02, format="%.3f")
        environment = st.number_input("Environment standard uncertainty (µm)", 0.0, 5.0, 0.02, format="%.3f")
        seed = st.number_input("Random seed offset (new noise, same part)", 0, 999, 0)
    with st.expander("Calibration"):
        cal = CALIB.load()
        st.write(f"Calibrated {cal['date']}, due {cal['due']}")
        for r in CALIB.registry(cal):
            from engine import uncertainty
            st.write(f"{r['id']}: U = {uncertainty(r, registration, environment):.3f} µm (k = 2)")
        if st.button("Re-run calibration (about 40 s)"):
            with st.spinner("Calibrating the camera and the confocal sensor"):
                CALIB.run()
            st.session_state.pop("res", None)
            st.rerun()
    st.caption("Sensors are simulated so the demo runs anywhere. The measurement algorithms, the calibration "
               "procedure and the decision engine are the real software.")

policy = dict(scenario=scenario, ratio=ratio, registration=registration, environment=environment, seed=int(seed))
fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()

st.title("AMSI · Adaptive Multi Scale Inspection")
st.caption("Interpret the drawing → survey with the camera → measure what the camera can prove → switch to the "
           "confocal sensor only where needed → one verdict with evidence")

if run_clicked:
    with st.status("Inspecting " + scenario.lower(), expanded=True) as box:
        bar = st.progress(0.0)
        def progress(msg, frac):
            box.write(msg); bar.progress(min(max(frac, 0.0), 1.0))
        res = PL.run_inspection(scenario, ratio=ratio, registration_u=registration, environment_u=environment,
                                seed_offset=int(seed), progress=progress)
        bar.progress(1.0)
        box.update(label=f"Inspection complete: {res['verdict']}", state="complete", expanded=False)
    st.session_state["res"] = (fingerprint, res)

if "res" not in st.session_state:
    cols = st.columns(5)
    steps = [("1 · Requirements", "The GD&T requirements come from the CAD model: sizes, hole positions, flatness, roughness."),
             ("2 · Camera survey", "One top image, 40 µm per pixel. Edges, holes and the datum frame to about 3 µm. Marks are flagged."),
             ("3 · Scale switch", "A requirement stays on the camera only if the camera's uncertainty is at most a quarter of its tolerance."),
             ("4 · Confocal scans", "Only the requirements that need it: tight holes, flatness, roughness, depth of every mark."),
             ("5 · Verdict", "Accept, reject or review, with every number, its uncertainty and a hash linked evidence trail.")]
    for c, (h, t) in zip(cols, steps):
        c.markdown(f"<div class='step'><h4>{h}</h4>{t}</div>", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([3, 2])
    a.image(RIG_IMG, caption="The inspection rig: camera and confocal pen on one carriage, end clamps and a clamping table")
    b.info("Choose a part on the left and press Run inspection. Start with the defective part to see every kind of "
           "decision, then try the near limit part to see the automatic switch from camera to confocal.")
    st.stop()

old, res = st.session_state["res"]
if old != fingerprint:
    st.warning("The settings changed. Press Run inspection to refresh the results.")

feat = {f["id"]: f for f in res["features"]}
rows = res["rows"]
bg, msg = VERDICT[res["verdict"]]
n_conf = sum(r["Status"] == "Conforming" for r in rows)
n_non = sum(r["Status"] == "Nonconforming" for r in rows)
n_conf_jobs = len(res["jobs"])
top = st.columns([2.2, 1, 1, 1, 1])
top[0].markdown(f"<div class='verdict' style='background:{bg}'>{res['verdict']}<small>{res['scenario']}: {msg}</small></div>",
                unsafe_allow_html=True)
top[1].metric("Requirements checked", len(rows))
top[2].metric("In tolerance", n_conf)
top[3].metric("Out of tolerance", n_non)
top[4].metric("Confocal scans", n_conf_jobs, help="Only what the camera could not prove")

VIEWS = ["1 · Camera survey (micron)", "2 · Scale switch", "3 · Confocal scans (sub micron)", "4 · Final report", "How it works"]
view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]
st.write("")

# --------------------------------------------------------------------------- 1 camera
if view == VIEWS[0]:
    cam = res["camera"]
    m = cam["meas"]
    heat = st.toggle("Show the mark heat map", value=True)
    st.plotly_chart(FG.camera_view(res, heat), width="stretch")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("What the camera measured")
        final = {r["Feature"]: r for r in rows}
        tbl = []
        U_cam = next(c["U_um"] for c in res["report"]["candidates"] if c["instrument_id"] == "CAM")
        for fid, v in cam["values"].items():
            f = feat[fid]
            r = final[fid]
            if r["Instrument"] == "CAM":
                dec = r["Status"]
            elif fid in res["refined"]:
                dec = "Too close to the limit → confocal"
            else:
                dec = "Camera U too large → confocal"
            tbl.append({"Feature": fid, "Requirement": f["requirement"], "Camera value": fmt(f, v), "Camera decision": dec})
        df = pd.DataFrame(tbl)
        st.dataframe(df.style.apply(lambda c: [f"background-color: {STATUS_BG.get(v, '#ede9fe')}" for v in c], subset=["Camera decision"]),
                     hide_index=True, width="stretch")
        st.caption(f"Every camera value carries U = {U_cam:.2f} µm (k = 2). Part found at {m['angle_deg']:+.3f}° in the field. "
                   "Datum frame from the fitted sides: B (left side) orients it, C (bottom side) sets the origin, A is the seating face.")
    with right:
        st.subheader("Marks found on the surface")
        if cam["marks"]:
            mk = [{"Mark": f"D{k}", "Where (x, y) mm": f"({x['x']:.2f}, {x['y']:.2f})",
                   "Size (mm)": f"{x['length_mm']:.2f} × {x['width_mm']:.2f}", "Score": f"{x['peak_score']:.1f} ×"}
                  for k, x in enumerate(cam["marks"], 1)]
            st.dataframe(pd.DataFrame(mk), hide_index=True, width="stretch")
        else:
            st.write("No marks above the threshold.")
        st.caption("Score is in multiples of the alarm threshold. A camera sees that a mark is there, but not how deep it is: "
                   "a stain and a scratch can look the same, so every mark goes to the confocal.")
    with st.expander("What the camera gives, and how accurate it is"):
        c = res["calibration"]["camera"]
        st.markdown(f"""
- Image: 3000 × 2000 pixels, telecentric lens, 40 µm per pixel, field 120 × 80 mm, ring light on a dark background.
- Edges: 300 brightness profiles across every side, each edge found at the 50 % level between part and background to a
  fraction of a pixel; straight lines fitted through them. Holes: 360 profiles each and a circle fit.
- Calibration (simulated run, {res['calibration']['date']}): gauge block 50.000 mm imaged 5 times gives the true pixel size
  {c['px_cal_mm'] * 1000:.4f} µm.
- Uncertainty components (standard, µm): scale {c['u_scale_um']:.2f}, residual distortion {c['u_dist_um']:.2f},
  temperature ±0.5 K on aluminium {c['u_thermal_um']:.2f}, gauge certificate {c['u_cert_um']:.2f},
  repeatability {c['u_repeat_um']:.2f}.
- Expanded uncertainty used by the planner: U = {U_cam:.2f} µm (k = 2). Verification on the reference plate: largest
  error {c['verify_max_error_um']:.2f} µm.
- Mark detector: learned from good parts only; a surface patch is flagged when it is further from every good patch than
  the threshold set on an unseen good part.
""")

# --------------------------------------------------------------------------- 2 scale switch
if view == VIEWS[1]:
    st.info(f"Rule: an instrument may measure a requirement only if its expanded uncertainty U is at most 1/{ratio} of the "
            "tolerance. The fastest instrument that qualifies goes first. If its result is too close to call, the next "
            "better instrument measures again. A mark found by the camera always goes to the confocal for its depth.")
    a, b = st.columns([3, 2])
    a.plotly_chart(FG.budget_chart(res), width="stretch")
    with b:
        st.plotly_chart(FG.part_map(res, "instrument"), width="stretch")
    cand = res["report"]["candidates"]
    dec = []
    for f in res["features"]:
        r = next(x for x in rows if x["Feature"] == f["id"])
        caps = [c for c in cand if c["feature_id"] == f["id"] and "unsupported characteristic" not in c["reason"]
                and "incompatible geometry" not in c["reason"]]
        budget = f["tolerance_um"] / ratio
        cams = [c for c in caps if c["instrument_id"] == "CAM"]
        if f["id"].startswith("D"):
            why = "Mark found by the camera → confocal measures its depth"
        elif f["id"] in res["refined"]:
            v = res["camera"]["values"][f["id"]]
            why = f"Camera {v / 1000:.4f} ± {cams[0]['U_um'] / 1000:.4f} mm touches the limit → confocal measured again"
        elif r["Instrument"] == "CAM":
            why = f"Camera qualifies: U {cams[0]['U_um']:.2f} ≤ {budget:.2f} µm, and it is fastest"
        elif cams and not cams[0]["eligible"]:
            why = f"Camera U {cams[0]['U_um']:.2f} > budget {budget:.2f} µm → confocal"
        else:
            why = "Height requirement → only the confocal can measure it"
        dec.append({"Feature": f["id"], "What": f["label"], "Tolerance": f"{f['tolerance_um']:g} µm",
                    "Budget": f"{budget:.3f} µm", "Measured by": FG.INSTR_NAME.get(r["Instrument"], r["Instrument"]),
                    "Why": why})
    st.dataframe(pd.DataFrame(dec), hide_index=True, width="stretch",
                 column_config={"Why": st.column_config.TextColumn(width="large"),
                                "What": st.column_config.TextColumn(width="medium")})
    st.caption(f"Hand off: the camera's datum frame tells the confocal where each region is (position error of the hand off "
               f"in this run: {res['handoff_err_um']:.1f} µm; every scan window has a margin of 0.12 mm or more).")

# --------------------------------------------------------------------------- 3 confocal
if view == VIEWS[2]:
    jobs = res["jobs"]
    if not jobs:
        st.write("No confocal scans were needed.")
    else:
        names = [f"{j['feature_id']} · {j['label']}" for j in jobs]
        pick = st.selectbox("Confocal scan", names, index=0)
        j = jobs[names.index(pick)]
        f = feat[j["feature_id"]]
        r = next(x for x in rows if x["Feature"] == j["feature_id"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Result", fmt(f, r["Measured"]), help=f"U = {r['U_um']:.3f} µm (k = 2)")
        c2.markdown(f"<span class='pill' style='background:{FG.STATUS_COLOR.get(r['Status'], '#666')}'>{r['Status']}</span>"
                    f"<br><small>Requirement {f['requirement']}<br>U = {r['U_um']:.3f} µm (k = 2)</small>", unsafe_allow_html=True)
        c3.metric("Points measured", f"{j['points']:,}")
        c4.metric("Scan time on the cell", f"{j['seconds']:.1f} s")
        lf, rf = FG.job_figures(j)
        a, b = st.columns(2)
        a.plotly_chart(lf, width="stretch")
        b.plotly_chart(rf, width="stretch")
        if j["kind"] == "bore":
            st.caption(f"360 radial passes across the wall of {j['hole']}, 1 µm steps. The sensor gets no signal over the open "
                       f"hole (it is far outside the 300 µm range), so the wall is where the signal starts. Circle fit: "
                       f"Ø {j['d_mm']:.5f} mm. Started from the camera's position for the hole: "
                       f"({j['handoff_xy'][0]:.4f}, {j['handoff_xy'][1]:.4f}) mm.")
        elif j["kind"] == "flatness":
            st.caption(f"{j['n_points']} points, each the average of a 0.2 mm line (201 readings) so roughness does not "
                       "count as flatness. Least squares plane removed; flatness is the peak to valley of what is left. "
                       "Least squares flatness is never smaller than the minimum zone value, so the decision is on the safe side.")
        elif j["kind"] == "texture":
            t = j["texture"]
            st.dataframe(pd.DataFrame([{"Sa (µm)": round(j["Sa_um"], 3), "Sq (µm)": round(t["Sq_um"], 3), "Sz (µm)": round(t["Sz_um"], 2),
                                        "Ssk": round(t["Ssk"], 2), "Sku": round(t["Sku"], 2), "points": t["points"]}]), hide_index=True)
            st.caption(f"ISO 25178: plane removed (F operator), Gaussian L filter with a {j['lc_mm']} mm cut off, 2 µm sampling.")
        else:
            an = j["analysis"]
            st.caption(f"{j['mode'].capitalize()}. Background roughness σ = {an['sigma_bg_um']:.2f} µm; anything counts as a defect "
                       f"only if it is deeper than 5 σ ({5 * an['sigma_bg_um']:.1f} µm) against its own surroundings. Finding: "
                       f"{j['finding']}.")
    with st.expander("What the confocal sensor gives, and how accurate it is"):
        e, h = res["calibration"]["edge"], res["calibration"]["height"]
        st.markdown(f"""
- Principle: white light through a lens that focuses every colour at a different height. Only the colour in focus on
  the surface returns, so its wavelength is the height. One height per point, up to 25,000 points per second.
- Sensor (Micro Epsilon confocalDT IFS2405-0.3 with IFD2410): 300 µm range, 6 µm spot, 10 nm resolution, ±0.15 µm linearity.
- Motion: X carriage with a 1 µm glass scale, short Y slide with a 0.1 µm encoder.
- Height calibration: step height standard 10.000 µm scanned 5 times, scale factor {h['kz']:.4f}, repeatability
  {h['u_repeat_um'] * 1000:.1f} nm. U for height tasks = {next(c['U_um'] for c in res['report']['candidates'] if c['instrument_id'] == 'CCF-HEIGHT'):.3f} µm.
- Edge calibration: ring gauge Ø 2.0000 mm scanned 5 times, bias {e['bias_um']:+.3f} µm (corrected), repeatability
  {e['u_repeat_um']:.3f} µm. U for hole diameters = {next(c['U_um'] for c in res['report']['candidates'] if c['instrument_id'] == 'CCF-EDGE'):.3f} µm.
""")

# --------------------------------------------------------------------------- 4 report
if view == VIEWS[3]:
    out = []
    for r in rows:
        f = feat[r["Feature"]]
        out.append({"Feature": r["Feature"], "What": r["What"], "Requirement": r["Requirement"],
                    "Level": r["Level"], "Measured": fmt(f, r["Measured"]),
                    "U (k=2)": f"{r['U_um']:.3f} µm" if r["U_um"] else "—", "Status": r["Status"]})
    df = pd.DataFrame(out)
    st.dataframe(df.style.apply(colour_status, subset=["Status"]), hide_index=True, width="stretch")
    a, b = st.columns([2, 3])
    a.plotly_chart(FG.part_map(res, "status"), width="stretch")
    b.plotly_chart(FG.timeline(res), width="stretch")
    with st.expander("Evidence trail (every step, hash linked, from the decision engine)"):
        ev = pd.DataFrame(res["report"]["events"])
        keep = [c for c in ["sequence", "feature_id", "state", "instrument_id", "reason", "value_um", "U_um", "status", "hash"] if c in ev]
        ev = ev[keep].copy()
        ev["hash"] = ev["hash"].str[:12] + "…"
        st.dataframe(ev, hide_index=True, width="stretch", height=320)
    export = dict(res["report"])
    export.update(verdict=res["verdict"], scenario=res["scenario"], rows=rows, calibration=res["calibration"],
                  cycle_time_s=res["cycle_s"], confocal_jobs=[{k: v for k, v in j.items() if isinstance(v, (int, float, str))} for j in res["jobs"]])
    c1, c2 = st.columns(2)
    c1.download_button("Download the evidence (JSON)", json.dumps(export, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                       "amsi_inspection_evidence.json", "application/json", width="stretch")
    c2.download_button("Download the decisions (CSV)", df.to_csv(index=False), "amsi_decisions.csv", "text/csv", width="stretch")

# --------------------------------------------------------------------------- how it works
if view == VIEWS[4]:
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f3f5f8", fontname="Times New Roman"];
  cad [label="CAD + GD&T\\nrequirements"]; cam [label="Camera survey\\n40 µm/pixel, U ≈ 3 µm", fillcolor="#dbeafe"];
  plan [label="Planner (engine.py)\\nU ≤ tolerance / 4 ?", fillcolor="#fef3c7"];
  done [label="Decided by\\nthe camera", fillcolor="#dcfce7"]; conf [label="Confocal scans\\nedge U ≈ 0.7 µm\\nheight U ≈ 0.06 µm", fillcolor="#fce7f3"];
  rep [label="Verdict +\\nevidence", fillcolor="#e0e7ff"];
  cad -> cam -> plan; plan -> done [label="yes, and clear"]; plan -> conf [label="no, too close,\\nor a mark"];
  done -> rep; conf -> rep; }""")
    st.markdown("""
1. Requirements. The drawing's GD&T is turned into a list of requirements (here typed in `amsi/part.py`; on the real
   cell it comes from the FreeCAD GD&T export). Each has a characteristic, a tolerance and a location.
2. Camera survey (micron level). One top image measures the outline, every hole and the datum frame in about two seconds,
   and a mark detector learned from good parts flags anything unusual on the surface.
3. Scale switch. The planner from the project repository (`engine.py`) checks every instrument against every requirement:
   characteristic, geometry, surface, field, calibration dates, and the 4:1 rule on expanded uncertainty. The camera keeps
   a requirement only if it qualifies and its result is clearly inside or outside the limits.
4. Confocal scans (sub micron level). The camera's datum frame tells the confocal exactly where to go. It scans only the
   regions it was sent to: a tight hole wall, a grid for flatness, a 1 mm patch for roughness, and every flagged mark.
5. Verdict. Every requirement ends Conforming or Nonconforming only when its whole uncertainty interval is on one side of
   the limit; otherwise it is flagged for review. Every step is recorded in a hash linked evidence trail.
""")
    st.image(RIG_IMG, caption="Rig: gantry with camera and confocal pen, end clamps, clamping lift and turn table")
