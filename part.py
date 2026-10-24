"""Demo part, its GD&T requirements and three part conditions (the ground truth the simulated
sensors observe). Part frame: origin at the lower left corner of the top view, x along the
80 mm length, y along the 50 mm width, z up from datum A (the seating face). Units: mm unless
the key ends in _um."""

TOP_Z = 5.0
PART = {
    "name": "AMSI demo plate (6061 aluminium)",
    "L": 80.0, "W": 50.0, "T": 5.0,
    "holes": {"H1": (15.0, 12.0, 6.0), "H2": (65.0, 12.0, 6.0),
              "H3": (15.0, 38.0, 6.0), "H4": (65.0, 38.0, 6.0),
              "H5": (40.0, 25.0, 2.0)},
    "cte_per_K": 23.6e-6,
}

# No surface defect may be deeper than this (drawing note).
DEFECT_LIMIT_UM = 10.0


def _feat(fid, char, label, requirement, geometry, tol_um, nominal_um, roi_mm, xy, datums=""):
    """One requirement in the schema of the project's engine.py, plus two display fields."""
    return {"id": fid, "characteristic": char, "geometry": geometry, "surface": "diffuse",
            "access": "visible", "datums": datums, "modifier": "RFS", "tolerance_um": tol_um,
            "nominal_um": nominal_um, "roi_mm": roi_mm, "xyz_mm": [xy[0], xy[1], TOP_Z],
            "label": label, "requirement": requirement}


REQUIREMENTS = [
    _feat("F01", "size", "Length", "80.000 ± 0.100 mm", "external", 200, 80000, 80, (40, 25)),
    _feat("F02", "size", "Width", "50.000 ± 0.100 mm", "external", 200, 50000, 50, (40, 25)),
    _feat("F03", "size", "Hole H1 diameter", "Ø 6.000 ± 0.025 mm", "bore", 50, 6000, 8, (15, 12)),
    _feat("F04", "position", "Position of the four Ø 6 holes", "⌀ 0.100 | A | B | C", "bore", 100, 0, 60,
          (40, 25), datums="A|B|C"),
    _feat("F05", "size", "Dowel hole H5 diameter", "Ø 2.000 ± 0.004 mm", "bore", 8, 2000, 3, (40, 25)),
    _feat("F06", "flatness", "Top face flatness", "0.005 mm", "plane", 5, 0, 80, (40, 25)),
    _feat("F07", "texture", "Top face roughness", "Sa ≤ 0.8 µm", "plane", 0.8, 0, 1, (30, 25)),
]


def defect_feature(k, x, y, roi_mm):
    """A requirement created at run time for a mark the camera found (checked for depth)."""
    return _feat(f"D{k}", "defect", f"Mark D{k} found by the camera",
                 f"depth ≤ {DEFECT_LIMIT_UM:.0f} µm", "plane", DEFECT_LIMIT_UM, 0,
                 round(roi_mm, 2), (round(x, 2), round(y, 2)))


# Ground truth for each part condition. Holes: (x, y, diameter) in mm. bow/twist: form of the
# top face (µm). Sa: areal roughness (µm). lobing: three lobe roundness error of every hole (µm).
SCENARIOS = {
    "Good part": {
        "desc": "Every feature is in tolerance. The camera sees a faint stain; the confocal "
                "shows it has no depth, so it is only cosmetic.",
        "L": 80.012, "W": 49.991,
        "holes": {"H1": (15.006, 12.004, 6.006), "H2": (65.003, 11.996, 6.004),
                  "H3": (14.997, 38.005, 6.008), "H4": (65.008, 38.002, 6.005),
                  "H5": (40.002, 25.001, 2.0011)},
        "bow_um": 2.2, "twist_um": 0.6, "Sa_um": 0.40, "lobing_um": 0.20,
        "defects": [{"kind": "stain", "x": 56.0, "y": 31.0, "r": 1.3, "contrast": 0.10}],
        "seed": 11,
    },
    "Near limit part": {
        "desc": "Hole H1 is 1.5 µm inside its upper limit, too close for the camera to decide, so the "
                "confocal measures it again. A shallow pit is checked for depth.",
        "L": 80.041, "W": 50.022,
        "holes": {"H1": (15.010, 12.006, 6.0235), "H2": (65.004, 12.003, 6.007),
                  "H3": (15.006, 38.008, 6.006), "H4": (65.009, 37.995, 6.009),
                  "H5": (40.004, 24.998, 2.0026)},
        "bow_um": 4.0, "twist_um": 0.8, "Sa_um": 0.66, "lobing_um": 0.30,
        "defects": [{"kind": "pit", "x": 27.0, "y": 41.0, "r": 0.15, "depth_um": 6.0, "contrast": 0.35},
                    {"kind": "stain", "x": 58.0, "y": 22.0, "r": 1.2, "contrast": 0.10}],
        "seed": 22,
    },
    "Defective part": {
        "desc": "A deep scratch, a deep pit, an oversize dowel hole, a shifted hole and an out of "
                "flat top face. One stain is harmless.",
        "L": 80.030, "W": 49.985,
        "holes": {"H1": (15.004, 12.003, 6.005), "H2": (65.006, 12.002, 6.007),
                  "H3": (15.045, 38.040, 6.006), "H4": (65.002, 37.996, 6.004),
                  "H5": (40.003, 25.002, 2.0062)},
        "bow_um": 6.5, "twist_um": 1.2, "Sa_um": 0.52, "lobing_um": 0.30,
        "defects": [{"kind": "scratch", "x0": 22.0, "y0": 20.0, "x1": 34.0, "y1": 31.0, "w": 0.06,
                     "depth_um": 14.0, "contrast": 0.30},
                    {"kind": "pit", "x": 60.0, "y": 20.0, "r": 0.18, "depth_um": 22.0, "contrast": 0.40},
                    {"kind": "stain", "x": 52.0, "y": 42.0, "r": 1.4, "contrast": 0.11}],
        "seed": 33,
    },
}

# A defect free part used only for calibration and for training the camera's mark detector.
REFERENCE = {
    "L": 80.000, "W": 50.000,
    "holes": {k: v for k, v in PART["holes"].items()},
    "bow_um": 1.0, "twist_um": 0.3, "Sa_um": 0.40, "lobing_um": 0.0, "defects": [], "seed": 1,
}
