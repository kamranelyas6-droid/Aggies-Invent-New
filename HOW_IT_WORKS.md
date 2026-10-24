# How AMSI works, from start to verdict

AMSI (Adaptive Multi Scale Inspection) proves every requirement on a part at the scale that requirement needs:
- a camera for everything that can be proven at micron level;
- a chromatic confocal sensor only where sub micron evidence is needed.

The decisions are made by the planner from the project repository (`engine.py`). Every number in this document comes from the demo software (version 1.0, calibration of 24 Oct 2026).

## 0. Before the first part: calibration
| Instrument | Reference | What the calibration gives | Expanded uncertainty U (k = 2) |
|---|---|---|---|
| Camera | Gauge block 50.000 mm, imaged 5 times at random positions | Pixel size 40.012 µm. Uncertainty components (µm): scale 0.56, residual distortion 0.80, temperature ±0.5 K on aluminium 0.55, certificate 0.05, repeatability 0.76 | 2.95 µm |
| Confocal edge scan | Ring gauge Ø 2.0000 mm, scanned 5 times | Bias −0.013 µm (applied as a correction), repeatability 0.019 µm, certificate 0.15 µm | 0.65 µm |
| Confocal height scan | Step height standard 10.000 µm, scanned 5 times | Height scale factor 0.9969, repeatability 2.4 nm, certificate 0.015 µm | 0.064 µm |

The planner adds the standard uncertainty components in quadrature:

u_c = sqrt(u_cal² + u_repeat² + (resolution / √12)² + u_registration² + u_environment²)

and uses U = 2·u_c. These values become the capability registry, in the same format as the repository's `instruments.json`.

## 1. Requirements
The GD&T of the drawing becomes a list of requirements. In the demo they are typed in `amsi/part.py`; on the cell they come from the FreeCAD GD&T export.

| ID | Requirement | Tolerance | Budget at 4:1 |
|---|---|---|---|
| F01 | Length 80.000 ± 0.100 mm | 200 µm | 50 µm |
| F02 | Width 50.000 ± 0.100 mm | 200 µm | 50 µm |
| F03 | Hole H1 Ø 6.000 ± 0.025 mm | 50 µm | 12.5 µm |
| F04 | Position of the four Ø 6 holes, ⌀ 0.100 relative to A, B, C | 100 µm | 25 µm |
| F05 | Dowel hole H5 Ø 2.000 ± 0.004 mm | 8 µm | 2 µm |
| F06 | Top face flatness 0.005 mm | 5 µm | 1.25 µm |
| F07 | Top face roughness Sa ≤ 0.8 µm | 0.8 µm | 0.2 µm |
| Dn | Any mark the camera finds: depth ≤ 10 µm (drawing note) | 10 µm | 2.5 µm |

## 2. Loading
The part sits on the clamping table and the table jaws grip it. The end clamps close on its ends; then the table jaws open and the table drops away (rig version 4). The part is always held by something.

## 3. Camera survey (micron level)
One top lit image: 3000 × 2000 pixels, telecentric lens, 40 µm per pixel, a 120 × 80 mm field, about 2 s.

What the image gives:
- **Outline.** 300 brightness profiles across each side. Each edge is placed at the 50 % level between part and background, to a fraction of a pixel, and a straight line is fitted through each side. Length and width are the distances between opposite lines.
- **Datum frame.** The left side (datum B) orients the frame, the bottom side (datum C) sets the origin, and the seating face is datum A.
- **Holes.** 360 radial profiles per hole, an algebraic circle fit, then a geometric fit. This gives each diameter and centre. Position error = 2·√(dx² + dy²) in the datum frame.
- **Marks.** A detector learned from good parts only. Every 160 µm surface patch is compared with all good patches, and its distance to the nearest one gives the heat map. Regions above the threshold become new requirements D1, D2, …

The camera can see that a mark exists, but not how deep it is: a stain and a scratch can look the same.

## 4. The scale switch
For every requirement, the planner checks every instrument against these conditions:
- the characteristic and geometry it supports;
- surface and access;
- validated field, including the localisation margin;
- calibration dates;
- the 4:1 rule: U ≤ tolerance / 4.

The fastest instrument that qualifies measures first. The engine then decides on the uncertainty interval:
- **Conforming:** the whole interval [value − U, value + U] is inside the limits.
- **Nonconforming:** the whole interval is outside the limits.
- **Indeterminate:** the interval touches a limit. The engine then asks the next instrument with a smaller U to measure again. If none is left, the requirement goes to review.

A requirement that no instrument can measure is blocked, never guessed.

In the demo this produces three triggers for the confocal:
- **Budget.** The camera's U = 2.95 µm is above the dowel hole budget of 2 µm, so F05 goes to the confocal edge scan.
- **Too close to call.** On the near limit part the camera reads H1 at 6.0229 ± 0.0030 mm; the interval touches 6.025, so the confocal measures it again: 6.0235 mm, Conforming.
- **Kind of requirement or mark.** Flatness, roughness and mark depth are height requirements, and only the confocal measures height.

You can watch the rule work: set the ratio slider to 2 and the dowel hole stays with the camera (budget 4 µm); set it to 4 and it moves to the confocal.

## 5. Hand off
The camera's datum frame gives the part position on the stage, so each region is sent to the confocal as stage coordinates. In the demo runs the hand off error is about 1 µm, and every scan window has at least a 0.12 mm margin.

## 6. Confocal scans (sub micron level)
White light passes through a lens that focuses each colour at a different height. Only the colour in focus on the surface comes back, so its wavelength gives the height.

The sensor is a Micro Epsilon confocalDT IFS2405-0.3 pen with an IFD2410 controller:
- 300 µm range;
- 6 µm spot;
- 10 nm resolution;
- ±0.15 µm linearity;
- up to 25 kHz.

Motion comes from an X carriage with a 1 µm glass scale and a short Y slide with a 0.1 µm encoder. The sensor scans only where it was sent:

| Scan | What it measures | How | Points / time (defective part) | Gives |
|---|---|---|---|---|
| Edge scan | Hole diameter and roundness | 360 radial passes across the wall, 1 µm steps. An open hole returns no signal (far outside the range), so the wall is where the signal starts. Circle fit, 15 UPR roundness | 154,881 / 7.6 s | F05: Ø 2.0063 mm (U 0.65 µm), roundness about 1 µm |
| Flatness grid | Form of the top face | 50 points, each the mean of a 0.2 mm line (201 readings) so roughness does not count as form. Least squares plane removed, peak to valley (conservative against minimum zone) | 10,050 / 17.9 s | F06: 5.65 µm |
| Roughness patch | Surface texture | 1 × 1 mm at 2 µm. Plane removed, Gaussian L filter with a 0.25 mm cut off (ISO 25178 S-L surface) | 251,001 / 10.0 s | F07: Sa 0.488 µm, plus Sq, Sz, Ssk, Sku |
| Mark scan | Depth of each mark | Scratches: cross sections every 0.1 mm along the mark at 1 µm steps. Pits and stains: areal map at 2 to 5 µm. A defect counts only if it is deeper than 5 σ of the surrounding roughness, measured against its own surroundings | 191,178 to 237,169 / 8.3 to 9.5 s | D1 pit 23.4 µm, D2 scratch 14.7 µm, D3 stain no depth |

## 7. Verdict and evidence
The overall verdict follows from the individual requirements:
- **REJECT:** any requirement is Nonconforming.
- **REVIEW:** any requirement is Indeterminate or Blocked.
- **ACCEPT:** everything else.

The engine records every step in a hash linked event trail: interpretation, survey, localisation, technique selection, local frame, measurement with U and certificate, refinement, and completion. The trail and the decisions download as JSON and CSV.

| Part | Verdict | Confocal scans | Time on the cell |
|---|---|---|---|
| Good part | ACCEPT (the stain has no depth, so it is cosmetic) | 4 | 51 s |
| Near limit part | ACCEPT (H1 refined by the confocal) | 6 | 68 s |
| Defective part | REJECT: F04 position 119.7 µm, F05 Ø 2.0063 mm, F06 5.65 µm, D1 23.4 µm, D2 14.7 µm | 6 | 69 s |

## 8. Other faces
The demo inspects the top face. On the rig, the end clamps turn the part to 90° and 180° for the sides and the bottom, and the hand over to the clamping table frees the clamped ends. Each new face runs the same steps 3 to 7, in the same datum frame.

## 9. What is real and what is simulated
**Real (the production code):**
- edge and circle fitting, the datum frame and the mark detector;
- the confocal analysis: form removal, ISO 25178 parameters, defect depth and roundness;
- the calibration procedure and the uncertainty budget;
- the planner and decision rule (the repository's `engine.py`);
- the evidence trail.

**Simulated, so that the demo runs on any laptop:**
- the camera image (`camera.render`);
- the confocal readings (`confocal.Surface` and `Confocal.read`);
- the calibration artefacts.

**On the real cell,** replace those three with the camera driver and the sensor driver. Nothing downstream changes.
