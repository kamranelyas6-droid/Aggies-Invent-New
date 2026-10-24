"""Smoke test of the Streamlit app: landing page renders, an inspection runs, every view renders without errors."""
import os, sys
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_runs_every_view():
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Defective part")
    at.sidebar.button[0].click().run()
    assert not at.exception
    for v in ["1 · Camera survey (micron)", "2 · Scale switch", "3 · Confocal scans (sub micron)", "4 · Final report", "How it works"]:
        at.session_state["view"] = v
        at.run()
        assert not at.exception, v
