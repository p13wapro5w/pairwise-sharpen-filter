import pytest
import render.sharpen_filter as mod

CASES = [
    ("luma_sharpen", ([0.2, 0.5, 0.3], 0.6)),
    ("radius_scale", (1.0, 1080, 2160)),
    ("contrast_adaptive", (0.8, 0.9)),
    ("post_denoise", (1, 2)),
    ("chroma_skip", (1, 2)),
    ("ui_mask", (1, 2)),
    ("taa_order", (1, 2)),
    ("pre_resize", (1, 2)),
    ("highlight_clamp", (1.2, 1.0)),
    ("negative_clamp", (-0.2, 0.0)),
    ("kernel_build", (3, 0.4)),
    ("amount_curve", (0.5, 0.7)),
    ("threshold_band", (0.1, 0.1)),
    ("alpha_preserve", (0.5, 0.3)),
    ("edge_limit", (0.9, 0.2)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
