import pytest
import imaging.tilt_shift_blur as mod

CASES = [
    ("band_by_focus", (1,)),
    ("radius_by_ratio", (1,)),
    ("intermediate_high_depth", (1,)),
    ("smooth_transition", (1,)),
    ("orientation_from_exif", (1,)),
    ("preview_before_export", (1,)),
    ("load", (1,)),
    ("blur", (1,)),
    ("compose", (1,)),
    ("export", (1,)),
    ("validate_tiltshift", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
