"""All charts of the demo (Plotly). Heat maps carry their units on the colour bar."""
from __future__ import annotations
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from . import camera as CAM, part as P

FONT = dict(family="Times New Roman, Georgia, serif", size=14, color="#111")
INSTR_COLOR = {"CAM": "#2563eb", "CCF-EDGE": "#7c3aed", "CCF-HEIGHT": "#db2777", "move": "#cbd5e1"}
INSTR_NAME = {"CAM": "Camera", "CCF-EDGE": "Confocal edge scan", "CCF-HEIGHT": "Confocal height scan", "move": "Move"}
STATUS_COLOR = {"Conforming": "#16a34a", "Nonconforming": "#dc2626", "Indeterminate": "#d97706",
                "Blocked": "#6b7280", "Awaiting measurement": "#6b7280"}


def _layout(fig, h=480, **kw):
    fig.update_layout(template="plotly_white", font=FONT, height=h, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.14, x=0, font=dict(size=13)), **kw)
    return fig


def _circle(cx, cy, r, n=120):
    a = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def _to_world(m, x, y):
    p = m["origin"] + x * m["ex"] + y * m["ey"]
    return p[0], p[1]


def _rect(cx, cy, axis, length, width):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    n = np.array([-a[1], a[0]])
    c = np.array([cx, cy])
    pts = [c + sa * length / 2 * a + sn * width / 2 * n for sa, sn in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]]
    return [p[0] for p in pts], [p[1] for p in pts]


# ------------------------------------------------------------------------------ camera
def camera_view(res, heat=True, ds=3):
    cam = res["camera"]
    img, m, px_ = cam["image"], cam["meas"], cam["px_cal"]
    H, W = img.shape
    small = img[::ds, ::ds][::-1]
    xs = (np.arange(0, W, ds) + 0.5) * px_
    ys = (H - np.arange(0, H, ds) - 0.5) * px_
    fig = px.imshow(small, x=xs, y=ys[::-1], binary_string=True)
    fig.update_yaxes(autorange=True, scaleanchor="x", title="y (mm, camera)")
    fig.update_xaxes(title="x (mm, camera)")
    if heat:
        s, (r0, c0) = cam["score"] / cam["threshold"], cam["score_off"]
        hs, ws = s.shape
        hx = (c0 + np.arange(ws) * CAM.STRIDE + CAM.STRIDE / 2 + 0.5) * px_
        hy = (H - (r0 + np.arange(hs) * CAM.STRIDE + CAM.STRIDE / 2) - 0.5) * px_
        z = np.where(s >= 0.6, s, np.nan)
        fig.add_trace(go.Heatmap(z=z, x=hx, y=hy, colorscale="Inferno", zmin=0.6, zmax=max(3.0, float(np.nanmax(s))),
                                 opacity=0.65, colorbar=dict(title="mark score<br>(× threshold)", len=0.6),
                                 hovertemplate="score %{z:.2f} × threshold<extra></extra>"))
    for name, (c, d, n, pts) in m["sides_mm"].items():
        t = (pts - c) @ d
        a, b = c + t.min() * d, c + t.max() * d
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color="#ef4444", width=2),
                                 name="fitted sides", legendgroup="sides", showlegend=name == "left", hoverinfo="skip"))
    for k, (hid, h) in enumerate(sorted(m["holes"].items())):
        cx, cy = h["world"]
        X, Y = _circle(cx, cy, h["d"] / 2)
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", line=dict(color="#22d3ee", width=2), name="fitted holes",
                                 legendgroup="holes", showlegend=k == 0, hoverinfo="skip"))
        fig.add_annotation(x=cx, y=cy - h["d"] / 2 - 1.9, text=f"{hid}  Ø {h['d']:.4f}", showarrow=False,
                           font=dict(color="#e0f2fe", size=12), bgcolor="rgba(15,23,42,0.75)")
    O, ex, ey = m["origin"], m["ex"], m["ey"]
    for v, lab, shift in [(ex, "datum C, x", (3.5, -2.2)), (ey, "datum B, y", (-1.0, 2.4))]:
        e = O + 12 * v
        fig.add_annotation(x=e[0], y=e[1], ax=O[0], ay=O[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowwidth=2.5, arrowcolor="#facc15", text="")
        fig.add_annotation(x=e[0] + shift[0], y=e[1] + shift[1], text=lab, showarrow=False, font=dict(color="#facc15", size=12),
                           bgcolor="rgba(15,23,42,0.75)")
    for k, mk in enumerate(cam["marks"], 1):
        wx, wy = _to_world(m, mk["x"], mk["y"])
        ax_w = mk["axis"][0] * m["ex"] + mk["axis"][1] * m["ey"]
        X, Y = _rect(wx, wy, ax_w, mk["length_mm"] + 1.2, mk["width_mm"] + 1.2)
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", line=dict(color="#fb923c", width=2.5), name="marks → confocal",
                                 legendgroup="marks", showlegend=k == 1, hoverinfo="skip"))
        fig.add_annotation(x=wx + (mk["length_mm"] / 2 + 2.6) * ax_w[0], y=wy + (mk["length_mm"] / 2 + 2.6) * ax_w[1] + 0.6, text=f"D{k}", showarrow=False,
                           font=dict(color="#fdba74", size=14), bgcolor="rgba(15,23,42,0.75)")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return _layout(fig, 640, title="Top camera image with measurements" + (" and mark heat map" if heat else ""))


# ------------------------------------------------------------------------------ scale switch
def budget_chart(res):
    feats = {f["id"]: f for f in res["features"]}
    ratio = res["policy"]["ratio"]
    rows = [c for c in res["report"]["candidates"]
            if "unsupported characteristic" not in c["reason"] and "incompatible geometry" not in c["reason"]]
    fig = go.Figure()
    for iid in ["CAM", "CCF-EDGE", "CCF-HEIGHT"]:
        rr = [c for c in rows if c["instrument_id"] == iid]
        if not rr:
            continue
        fig.add_trace(go.Bar(x=[c["feature_id"] for c in rr], y=[c["U_um"] for c in rr], name=f"{INSTR_NAME[iid]}  U",
                             marker_color=INSTR_COLOR[iid], text=[f"{c['U_um']:.2f}" for c in rr], textposition="outside",
                             hovertemplate="%{x}: U = %{y:.3f} µm<extra>" + INSTR_NAME[iid] + "</extra>"))
    ids = [f for f in feats]
    fig.add_trace(go.Scatter(x=ids, y=[feats[f]["tolerance_um"] / ratio for f in ids], mode="markers",
                             marker=dict(symbol="line-ew-open", size=46, color="#111", line=dict(width=4, color="#111")),
                             name=f"budget = tolerance / {ratio}", hovertemplate="%{x}: budget %{y:.3f} µm<extra></extra>"))
    tv = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    fig.update_yaxes(type="log", title="expanded uncertainty U, k = 2 (µm)", tickvals=tv, ticktext=[f"{v:g}" for v in tv],
                     range=[np.log10(0.03), np.log10(80)])
    fig.update_xaxes(title="requirement", categoryorder="array", categoryarray=ids)
    return _layout(fig, 460, barmode="group", title="Who is allowed to measure? A bar below the black line qualifies")


# where each fixed requirement is drawn on the part map (mm): outside edges for length and width
DISPLAY_XY = {"F01": (40.0, -4.5), "F02": (-4.5, 25.0), "F03": (15.0, 16.5), "F04": (40.0, 44.0), "F05": (40.0, 28.5),
              "F06": (72.0, 44.0), "F07": (30.0, 28.5)}


def part_map(res, color_by="instrument"):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=P.PART["L"], y1=P.PART["W"], line=dict(color="#334155", width=2),
                  fillcolor="#e2e8f0", layer="below")
    for hid, (hx, hy, hd) in P.PART["holes"].items():
        fig.add_shape(type="circle", x0=hx - hd / 2, y0=hy - hd / 2, x1=hx + hd / 2, y1=hy + hd / 2,
                      line=dict(color="#334155"), fillcolor="#ffffff", layer="below")
    final = {r["Feature"]: r for r in res["rows"]}
    for j in res["jobs"]:
        col = INSTR_COLOR[j["instrument_id"]] if color_by == "instrument" else STATUS_COLOR.get(final[j["feature_id"]]["Status"], "#999")
        if j["kind"] == "bore":
            cx, cy = j["handoff_xy"]
            r = P.PART["holes"][j["hole"]][2] / 2
            for rr in (r - 0.12, r + 0.12):
                X, Y = _circle(cx, cy, rr)
                fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", line=dict(color=col, width=2), showlegend=False, hoverinfo="skip"))
        elif j["kind"] == "flatness":
            fig.add_trace(go.Scatter(x=j["points_xy"][:, 0], y=j["points_xy"][:, 1], mode="markers",
                                     marker=dict(size=6, color=col), showlegend=False, hovertemplate="flatness point<extra></extra>"))
        elif j["kind"] == "texture":
            cx, cy = j["xy"]
            X, Y = _rect(cx, cy, (1, 0), 1.0, 1.0)
            fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", line=dict(color=col, width=3), showlegend=False, hoverinfo="skip"))
        elif j["kind"] == "defect":
            Ls = j["map_s"][-1] - j["map_s"][0]
            Lt = j["map_t"][-1] - j["map_t"][0]
            X, Y = _rect(j["centre"][0], j["centre"][1], j["axis"], Ls, Lt)
            fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", line=dict(color=col, width=3), showlegend=False, hoverinfo="skip"))
    for f in res["features"]:
        r = final[f["id"]]
        col = INSTR_COLOR.get(r["Instrument"], "#999") if color_by == "instrument" else STATUS_COLOR.get(r["Status"], "#999")
        x, y = DISPLAY_XY.get(f["id"], tuple(f["xyz_mm"][:2]))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f["id"]], textposition="top center",
                                 marker=dict(size=14, color=col, line=dict(color="#fff", width=1.5)), showlegend=False,
                                 hovertemplate=f"{f['id']} {f['label']}<br>{r['Instrument']} · {r['Status']}<extra></extra>"))
    keys = (["CAM", "CCF-EDGE", "CCF-HEIGHT"] if color_by == "instrument" else ["Conforming", "Nonconforming", "Indeterminate"])
    for k in keys:
        col = INSTR_COLOR.get(k) if color_by == "instrument" else STATUS_COLOR[k]
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=col),
                                 name=INSTR_NAME.get(k, k)))
    fig.update_xaxes(range=[-8, 88], title="x (mm, part frame)", showgrid=False)
    fig.update_yaxes(range=[-9, 59], scaleanchor="x", title="y (mm)", showgrid=False)
    title = "Where each requirement was measured" if color_by == "instrument" else "Result on the part"
    return _layout(fig, 430, title=title)


# ------------------------------------------------------------------------------ confocal jobs
def _heat(z, x, y, title, unit="µm", diverging=True, zmax=None, xlab="x (mm)", ylab="y (mm)", equal=True):
    z = np.asarray(z, float)
    step = max(1, int(np.ceil(max(z.shape) / 320)))
    z, x, y = z[::step, ::step], np.asarray(x)[::step], np.asarray(y)[::step]
    kw = {}
    if diverging:
        m = zmax or float(np.nanpercentile(np.abs(z), 99.5) or 1.0)
        kw = dict(colorscale="RdBu_r", zmid=0, zmin=-m, zmax=m)
    else:
        kw = dict(colorscale="Viridis")
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorbar=dict(title=unit), hovertemplate="%{x:.3f}, %{y:.3f} mm<br>%{z:.3f} " + unit + "<extra></extra>", **kw))
    fig.update_xaxes(title=xlab, showgrid=False, zeroline=False)
    fig.update_yaxes(title=ylab, showgrid=False, zeroline=False, **({"scaleanchor": "x"} if equal else {}))
    fig.update_layout(plot_bgcolor="#1f2937")
    return _layout(fig, 430, title=title)


def job_figures(j):
    if j["kind"] == "bore":
        left = _heat(j["map"], j["map_x"], j["map_y"], f"Height map around hole {j['hole']} (no signal inside the hole)",
                     diverging=False)
        cx, cy = j["centre"]
        X, Y = _circle(cx, cy, j["d_mm"] / 2)
        left.add_trace(go.Scatter(x=X, y=Y, mode="lines", line=dict(color="#22d3ee", width=2), name=f"fitted Ø {j['d_mm']:.5f} mm"))
        ang = np.degrees(np.mod(j["edge_angles"], 2 * np.pi))
        o = np.argsort(ang)
        right = go.Figure()
        right.add_trace(go.Scatter(x=ang[o], y=j["edge_res_raw_um"][o], mode="markers", marker=dict(size=3, color="#94a3b8"),
                                   name="edge points (1 µm step)"))
        right.add_trace(go.Scatter(x=ang[o], y=j["edge_res_um"][o], mode="lines", line=dict(color="#7c3aed", width=3),
                                   name=f"roundness profile, {j['upr']} UPR"))
        right.update_xaxes(title="angle around the hole (degrees)", range=[0, 360], dtick=45)
        right.update_yaxes(title="radial deviation (µm)")
        _layout(right, 430, title=f"Roundness {j['roundness_um']:.2f} µm from {j['edges_used']} edge points")
        return left, right
    if j["kind"] == "flatness":
        left = _heat(j["map"], j["map_x"], j["map_y"], f"Top face height after removing the best plane (flatness {j['flatness_um']:.2f} µm)")
        left.add_trace(go.Scatter(x=j["points_xy"][:, 0], y=j["points_xy"][:, 1], mode="markers",
                                  marker=dict(size=7, color="#111", line=dict(color="#fff", width=1)), name="measured points",
                                  text=[f"{v:+.2f} µm" for v in j["residual_um"]], hovertemplate="%{text}<extra></extra>"))
        Z = j["map"][::2, ::2]
        right = go.Figure(go.Surface(z=Z, x=j["map_x"][::2], y=j["map_y"][::2], colorscale="RdBu_r", cmid=0,
                                     colorbar=dict(title="µm")))
        right.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1.6, y=1, z=0.45),
                                       xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="µm"))
        return left, _layout(right, 430, title="3D view (height exaggerated about 3000 times)")
    if j["kind"] == "texture":
        left = _heat(j["map"], j["map_x"], j["map_y"], f"Roughness map, 1 × 1 mm, S-L surface (Sa {j['Sa_um']:.3f} µm)")
        Z = j["map"][::4, ::4]
        right = go.Figure(go.Surface(z=Z, x=j["map_x"][::4], y=j["map_y"][::4], colorscale="RdBu_r", cmid=0, colorbar=dict(title="µm")))
        right.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35),
                                       xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="µm"))
        return left, _layout(right, 430, title="3D view of the surface texture")
    # defect
    d = max(j["depth_um"], 1.0)
    left = _heat(j["map"], j["map_s"], j["map_t"], f"Height map over the mark ({j['mode']})", zmax=d,
                 xlab="along the mark (mm)", ylab="across (mm)", equal=j["mode"] != "cross sections")
    pr = j["profile"]
    right = go.Figure()
    right.add_trace(go.Scatter(x=pr["dist_mm"], y=pr["z_um"], mode="lines", line=dict(color="#1f4e79", width=2), name="profile"))
    right.add_hline(y=0, line=dict(color="#64748b", dash="dash"), annotation_text="undisturbed surface", annotation_position="bottom right")
    if j["depth_um"] > 0:
        zz = np.asarray(pr["z_um"], float)
        i = int(np.nanargmin(zz))
        right.add_annotation(x=pr["dist_mm"][i], y=0, ax=pr["dist_mm"][i], ay=zz[i], xref="x", yref="y", axref="x", ayref="y",
                             showarrow=True, arrowhead=3, arrowside="end+start", arrowcolor="#dc2626", arrowwidth=2, text="")
        right.add_annotation(x=pr["dist_mm"][i], y=zz[i] / 2, text=f"  {j['depth_um']:.1f} µm deep", showarrow=False,
                             xanchor="left", font=dict(color="#dc2626", size=15))
    right.update_xaxes(title="distance across the mark (mm)")
    right.update_yaxes(title="height (µm)")
    return left, _layout(right, 430, title=f"Profile across the mark: {j['finding']}")


# ------------------------------------------------------------------------------ report
def timeline(res):
    tl = res["timeline"]
    fig = go.Figure()
    seen = set()
    for k, t in enumerate(tl):
        iid = t["instrument"]
        fig.add_trace(go.Bar(y=[t["step"] if iid != "move" else f"move {k}"], x=[t["seconds"]], base=[t["start"]], orientation="h",
                             marker_color=INSTR_COLOR[iid], name=INSTR_NAME[iid], showlegend=iid not in seen,
                             hovertemplate=f"{t['step']}: {t['seconds']:.1f} s<extra></extra>"))
        seen.add(iid)
    labels = [t["step"] if t["instrument"] != "move" else f"move {k}" for k, t in enumerate(tl)]
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1],
                     tickvals=[l for l in labels if not l.startswith("move")])
    fig.update_xaxes(title="time on the cell (s)")
    return _layout(fig, 60 + 28 * len(tl), title=f"Inspection timeline: {res['cycle_s']:.0f} s in total")
