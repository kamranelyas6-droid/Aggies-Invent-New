"""End to end checks: every demo part must give the expected decisions, through the project's engine."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from amsi import pipeline as PL

EXPECT = {
    "Good part": ("ACCEPT", {}),
    "Near limit part": ("ACCEPT", {"F03": "CCF-EDGE"}),
    "Defective part": ("REJECT", {"F04": "Nonconforming", "F05": "Nonconforming", "F06": "Nonconforming"}),
}


@pytest.fixture(scope="module")
def results():
    return {k: PL.run_inspection(k) for k in EXPECT}


@pytest.mark.parametrize("name", list(EXPECT))
def test_verdict(results, name):
    assert results[name]["verdict"] == EXPECT[name][0]


def test_dowel_hole_always_goes_to_confocal(results):
    for r in results.values():
        row = next(x for x in r["rows"] if x["Feature"] == "F05")
        assert row["Instrument"] == "CCF-EDGE"


def test_near_limit_hole_is_refined(results):
    r = results["Near limit part"]
    assert "F03" in r["refined"]
    assert next(x for x in r["rows"] if x["Feature"] == "F03")["Instrument"] == "CCF-EDGE"


def test_defective_failures(results):
    rows = {x["Feature"]: x for x in results["Defective part"]["rows"]}
    for fid, status in EXPECT["Defective part"][1].items():
        assert rows[fid]["Status"] == status
    deep = [x for x in rows.values() if x["Feature"].startswith("D") and x["Status"] == "Nonconforming"]
    assert len(deep) == 2                       # the pit and the scratch


def test_stains_are_cosmetic(results):
    for r in results.values():
        stains = [j for j in r["jobs"] if j["kind"] == "defect" and j["depth_um"] == 0.0]
        assert len(stains) >= 1


def test_measurement_accuracy(results):
    r = results["Defective part"]
    rows = {x["Feature"]: x for x in r["rows"]}
    assert abs(rows["F05"]["Measured"] - 2006.2) < 0.5          # µm, confocal edge
    assert abs(rows["F04"]["Measured"] - 120.4) < 5.0           # µm, camera position
