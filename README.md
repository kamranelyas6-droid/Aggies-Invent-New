# AMSI demo software

AMSI (Adaptive Multi Scale Inspection) proves each tolerance at the right scale: a camera at micron level and a chromatic confocal sensor at sub micron level. You get one verdict, with heat maps, numbers, uncertainties and an evidence trail. The decision engine is `engine.py` from the team repository (github.com/ajielee1975-arch/Aggie-Invent-LANL-UI), with three small backward compatible changes; its 15 original tests still pass.

The full system explanation is in `docs/HOW_IT_WORKS.md`.

## 1. Deploy on your own laptop (5 minutes)
You need Python 3.10 or newer (3.11 recommended).

```bash
# 1. unzip AMSI_demo_software.zip, then open a terminal in the AMSI-demo folder
cd AMSI-demo

# 2. create a clean environment
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. install
python -m pip install -r requirements.txt

# 4. start
python -m streamlit run app.py
```

The browser opens at http://localhost:8501. If it does not, open that address yourself.

The calibration ships with the software, so the first start is immediate. "Re-run calibration" in the sidebar repeats it in about 40 s.

## 2. Deploy online for the judges (free public link)
1. Put this folder in a GitHub repository. It can be its own repository, or a subfolder such as `amsi_demo/` inside the team repository.
2. Go to share.streamlit.io and sign in with GitHub.
3. Click "Create app", choose the repository and branch, and set the main file path to `app.py` (or `amsi_demo/app.py` for the subfolder).
4. Under advanced settings choose Python 3.11, then deploy.
5. Share the link that appears. Each run takes about 10 to 12 s on the server.

Keep the `.streamlit/` folder and `amsi/data/` with the app: they hold the theme and the calibration.

## 3. Use it (what to click, what to look at)
1. In the sidebar choose "Defective part" and press "Run inspection". A progress box lists each step, and the run takes about 10 s.
2. Read the top row. The coloured verdict (ACCEPT, REJECT or REVIEW) is followed by four numbers: requirements checked, in tolerance, out of tolerance, and confocal scans.
3. **1 · Camera survey (micron).** The camera image shows:
   - fitted sides (red) and fitted holes (cyan), with their diameters;
   - the datum frame (yellow);
   - the marks found (orange boxes, D1 to D3), with the mark heat map, which you can switch off.

   Below the image: every camera value with the camera's decision, and the list of marks.
4. **2 · Scale switch.** The bar chart shows each instrument's uncertainty U next to each requirement's budget (black line). A bar below the line qualifies. The map shows where each requirement was measured, and the table says why in one line.
5. **3 · Confocal scans (sub micron).** Pick a scan from the list:
   - F05 hole: height map and roundness plot.
   - F06 flatness: heat map and 3D view.
   - F07 roughness: S-L map and the ISO 25178 parameters.
   - D1 pit and D2 scratch: depth profiles with the measured depth.
   - D3 stain: no measurable depth.

   Each scan shows its result ± U, the number of points measured and the scan time.
6. **4 · Final report.** One table holds both levels, followed by the part map coloured by result, the inspection timeline, the evidence trail and the JSON and CSV downloads.
7. Run "Near limit part" and open 2 · Scale switch: F03 reads "touches the limit → confocal measured again".
8. Run "Good part": the verdict is ACCEPT, and the stain is shown to be cosmetic.
9. Optional: in "Accuracy policy" set the ratio to 2 and run again. The dowel hole F05 now stays with the camera, which shows the switching rule at work.

## 4. Check it
```bash
python -m pytest -q tests/          # 9 tests: every part condition, decisions, accuracy, and the app itself
```

## 5. Files
| Path | What it is |
|---|---|
| `app.py` | The Streamlit interface |
| `engine.py` | The team's planner and decision engine (with texture/defect characteristics, transform and survey arguments) |
| `amsi/part.py` | The demo plate, its GD&T requirements and the three part conditions |
| `amsi/camera.py` | Camera simulation and measurement: subpixel edges, circle fits, datum frame, mark detector |
| `amsi/confocal.py` | Confocal simulation and the four scan types (edge, flatness, roughness, mark) |
| `amsi/surface.py` | Height map analysis: form removal, ISO 25178 parameters, pits, scratches, bumps |
| `amsi/calibration.py` | Calibration on the gauge block, ring gauge and step height standard → instrument registry |
| `amsi/pipeline.py` | One inspection: camera survey → engine → confocal on demand → verdict |
| `amsi/figures.py` | All charts and heat maps |
| `amsi/data/` | Stored calibration and the learned good surface model |
| `tests/` | Pipeline tests, the app smoke test, and the screenshot script used for the video |
| `docs/HOW_IT_WORKS.md` | The full explanation of the system |

## 6. Honest limits
- The camera image and the confocal readings are simulated so the demo runs anywhere. The measurement algorithms, the calibration procedure, the uncertainty budget and the decisions are the real software. On the cell, replace `camera.render` and `confocal.Confocal.read` with the device drivers.
- Flatness uses a least squares plane, which is never smaller than minimum zone flatness, so the decision is on the safe side.
- Roughness uses plane removal and a Gaussian L filter. A certified roughness result would also set the S filter and the evaluation area by the drawing's specification.
- The demo inspects the top face. The rig turns the part for the other faces, and each face repeats the same steps.
