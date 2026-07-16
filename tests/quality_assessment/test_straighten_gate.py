"""ut for the subject-aware straighten gate (spec-101 S3)."""

from sim_bench.quality_assessment.straighten_gate import GateConfig, decide

CFG = GateConfig()  # defaults: conf 0.5, gate 3deg, area 0.70, person 0.15


def test_near_level_is_not_tilted():
    r = decide(1.0, 0.9, 1000, 800, None, CFG)
    assert not r.straighten and r.reason == "not_tilted"


def test_low_confidence_is_not_touched():
    r = decide(12.0, 0.3, 1000, 800, None, CFG)
    assert not r.straighten and r.reason == "not_tilted"


def test_landscape_small_tilt_straightens():
    # 5 deg on a 3:2 landscape keeps well over 70% -> straighten
    r = decide(5.0, 0.9, 1500, 1000, None, CFG)
    assert r.straighten and r.reason == "straighten" and r.retained_area >= 0.70


def test_tall_portrait_big_tilt_declined_on_area():
    # ~16 deg on a tall narrow frame blows the area floor
    r = decide(16.0, 0.9, 458, 1024, None, CFG)
    assert not r.straighten and r.reason == "declined_area"


def test_prominent_person_clipped_is_declined():
    # 5 deg keeps area > floor so the PERSON rule is what bites. Prominent person
    # (16.5% of frame) hugs the right edge (0.96) > the crop's right edge (~0.94).
    person = (0.66, 0.25, 0.30, 0.55)
    r = decide(5.0, 0.9, 1500, 1000, person, CFG)
    assert not r.straighten and r.reason == "declined_person"


def test_prominent_person_safely_inside_straightens():
    # prominent (16%) but centred and well inside the crop -> straighten
    person = (0.34, 0.25, 0.32, 0.50)
    r = decide(5.0, 0.9, 1500, 1000, person, CFG)
    assert r.straighten and r.reason == "straighten"


def test_small_background_person_does_not_block():
    # person below the prominence threshold (0.06% of frame) is ignored even at the edge
    person = (0.95, 0.5, 0.02, 0.03)
    r = decide(5.0, 0.9, 1500, 1000, person, CFG)
    assert r.straighten


def test_prominence_threshold_is_config():
    person = (0.66, 0.25, 0.30, 0.55)               # 16.5% of frame, clips the crop
    lax = GateConfig(prominent_person_frac=0.30)    # 16.5% < 30% -> not prominent -> ignored
    assert decide(5.0, 0.9, 1500, 1000, person, lax).straighten
