"""移轴虚化六个事故修复的回归测试。

对应题目描述：
1. 清晰带按焦点（不再写死中线）；
2. 半径按图像比例（大图不再虚化弱一个数量级）；
3. 两次一维卷积的中间结果高位深（渐变天空无色带）；
4. 过渡用平滑曲线（替代线性羽化，消除“贴纸”分界）；
5. 方向按 EXIF（竖拍不再用横拍清晰带横穿人脸）；
6. 导出前预览校验（坏预览阻断导出）。

另覆盖 load/blur/compose/export 管线与 validate/metrics/audit/
fallback/determinism 横切入口。
"""

import io
import os

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import imaging.tilt_shift_blur as ts  # noqa: E402


# -- 修复点 1：清晰带按焦点 --------------------------------------------------

@pytest.mark.parametrize("focus", [0.15, 0.3, 0.5, 0.7, 0.85])
def test_band_center_equals_focus(focus):
    band = ts.band_by_focus(focus)
    assert band["center_normalized"] == pytest.approx(focus)
    assert float(band) == pytest.approx(focus)


def test_band_moves_when_focus_moves():
    left = ts.band_by_focus(0.2)
    right = ts.band_by_focus(0.8)
    assert left["center_normalized"] != right["center_normalized"]
    assert right["center_normalized"] - left["center_normalized"] == pytest.approx(0.6)
    # 焦点偏离中线时，清晰带不能还停在 0.5。
    assert ts.band_by_focus(0.2)["center_normalized"] != pytest.approx(0.5)


def test_band_pixel_geometry_follows_focus():
    band = ts.band_by_focus(0.3, 1000)
    assert band["center_px"] == pytest.approx(300)
    start, end = tuple(band)
    assert 0 <= start < band["center_px"] < end <= 1000


def test_sharp_region_in_output_tracks_focus():
    """图像级：最锐利的行随焦点移动，而不是固定在中线。"""
    peaks = {}
    for focus in (0.28, 0.72):
        state = ts.load(1, width=200, height=150)
        blurred, _ = ts.separable_blur(state["image"], 4)
        mask = ts.band_mask(150, 200, focus=focus, core_half=0.06, feather=0.10)
        comp = state["image"] * (1 - mask[..., None]) + blurred * mask[..., None]
        gray = comp.mean(axis=2)
        energy = np.abs(np.diff(gray, axis=1)).mean(axis=1)
        smooth = np.convolve(energy, np.ones(9) / 9, mode="same")
        peaks[focus] = (int(np.argmax(smooth)) + 0.5) / len(smooth)
        assert abs(peaks[focus] - focus) < 0.18
    assert peaks[0.72] > peaks[0.28] + 0.2


# -- 修复点 2：半径按图像比例 -------------------------------------------------

def test_radius_scales_linearly_with_resolution():
    assert float(ts.radius_by_ratio(10.0, 1080, 2160)) == pytest.approx(20.0)
    assert float(ts.radius_by_ratio(8.0, 540, 2160)) == pytest.approx(32.0)
    assert float(ts.radius_by_ratio(12.0, 4.0)) == pytest.approx(48.0)


def test_radius_accepts_dimension_tuples():
    r = ts.radius_by_ratio(10.0, (800, 600), (4000, 3000))
    assert float(r) == pytest.approx(50.0)  # 长边 800 -> 4000，5x


def test_export_radius_matches_output_scale(tmp_path):
    out = tmp_path / "big.png"
    receipt = ts.export(1, path=os.fspath(out), output_size=4.0, focus=0.6)
    assert receipt["ok"] and receipt["written"]
    assert receipt["radius_ratio"] == pytest.approx(4.0)
    assert receipt["radius_export"] == int(round(receipt["radius_preview"] * 4))
    with Image.open(out) as im:
        assert im.size == (640, 480)


# -- 修复点 3：中间结果高位深 -------------------------------------------------

def test_intermediate_descriptor_is_high_depth():
    report = ts.intermediate_high_depth(1)
    assert report["bits"] >= 16
    assert report["dtype"] != "uint8"
    assert bool(report) is True


def test_intermediate_array_is_floating():
    u8 = np.zeros((8, 8, 3), dtype=np.uint8)
    mid = ts.intermediate_high_depth(u8)
    assert np.issubdtype(mid.dtype, np.floating)


def test_separable_blur_uses_float_intermediate():
    img = ts.fixture_image(32, 24)
    _, info = ts.separable_blur(img, 3)
    assert info["passes"] == 2 and info["separable"] is True
    assert "float" in info["intermediate_dtype"]
    assert info["intermediate_bits"] >= 16


def test_high_depth_beats_8bit_intermediate_on_gradient():
    cmp_ = ts.intermediate_depth_comparison()
    assert cmp_["available"]
    # 8 位中间结果对平缓渐变的误差必须不小于高位深路径。
    assert cmp_["rmse_8bit_intermediate"] + 1e-9 >= cmp_["rmse_high_depth"]
    assert cmp_["high_depth_better"]


# -- 修复点 4：平滑过渡曲线 ---------------------------------------------------

def test_smooth_curve_endpoints_and_midpoint():
    assert ts.smooth_transition(0.0) == pytest.approx(0.0)
    assert ts.smooth_transition(1.0) == pytest.approx(1.0)
    assert ts.smooth_transition(0.5) == pytest.approx(0.5)
    # cubic smoothstep 的经典取值
    assert ts.smoothstep(0.25) == pytest.approx(0.15625)


def test_smooth_curve_has_zero_endpoint_slope():
    assert abs(ts.transition_slope(0.0)) < 1e-3
    assert abs(ts.transition_slope(1.0)) < 1e-3
    # 线性羽化在端点斜率恒为 1，平滑曲线端点斜率为 0 -> 无硬分界。
    assert abs(ts.linear_feather(0.5) - 0.5) < 1e-9


def test_band_mask_is_sharp_at_focus_and_blurred_far():
    m = ts.band_mask(100, 100, focus=0.5, core_half=0.08, feather=0.2)
    assert m[50, 50] == pytest.approx(0.0, abs=1e-6)
    assert m[0, 50] == pytest.approx(1.0, abs=1e-6)
    # 蒙版在相邻行间没有“硬跳变”（硬接缝会接近 1.0；平滑曲线最大逐行
    # 变化仅约 0.1，且端点斜率为 0，见 test_smooth_curve_has_zero_endpoint_slope）。
    assert np.abs(np.diff(m, axis=0)).max() < 0.15
    # 上半侧从远到中心应单调不增（模糊权重递减）。
    col = m[:, 50]
    assert np.all(np.diff(col[:50]) <= 1e-6)


# -- 修复点 5：方向按 EXIF ----------------------------------------------------

def test_orientation_axis_mapping():
    assert ts.sensor_band_axis(1) == "y"
    assert ts.sensor_band_axis(3) == "y"
    for o in (5, 6, 7, 8):
        assert ts.orientation_swaps_axes(o)
        assert ts.sensor_band_axis(o) == "x"
    assert ts.display_dimensions(96, 64, 6) == (64, 96)


def test_orientation_report_metadata():
    rep = ts.orientation_from_exif(6)
    assert rep["orientation"] == 6
    assert rep["axis_swapped"] is True
    assert rep["degrees"] in (90, 270)
    assert rep == 6 and rep == 90  # 宽容比较


def _jpeg_orientation_six(path):
    arr = np.full((64, 96, 3), 40, dtype=np.uint8)
    arr[0:16, 0:16] = (255, 0, 0)  # 传感器左上红块
    im = Image.fromarray(arr)
    exif = Image.Exif()
    exif[ts.ORIENTATION_TAG] = 6
    im.save(path, "JPEG", exif=exif.tobytes())


def test_load_applies_exif_orientation(tmp_path):
    p = tmp_path / "p6.jpg"
    _jpeg_orientation_six(p)
    state = ts.load(p)
    assert (state["width"], state["height"]) == (64, 96)
    disp = (np.clip(np.asarray(state), 0, 1) * 255).astype(np.uint8)
    top_right = disp[4:12, 52:60].mean(axis=(0, 1))
    top_left = disp[4:12, 4:12].mean(axis=(0, 1))
    # 90CW 校正后，原传感器左上红块应到显示画面的右上。
    assert top_right[0] > 150 and top_right[1] < 100
    assert top_left[0] < 100


def test_sensor_mask_swaps_axis_for_portrait():
    m = ts.band_mask(64, 96, focus=0.5, orientation=6)
    assert m[32, 48] == pytest.approx(0.0, abs=1e-6)   # 中央列清晰
    assert m[32, 0] == pytest.approx(1.0, abs=1e-6)    # 两侧模糊


# -- 修复点 6：导出前预览校验 -------------------------------------------------

def test_preview_gate_approves_default():
    gate = ts.preview_before_export(1)
    assert bool(gate) is True
    assert gate["approved"] is True and gate["allowed"] is True
    assert gate["token"]


def test_validate_rejects_out_of_range_focus():
    assert not ts.validate_tiltshift({"focus": 2.0})
    assert ts.validate_tiltshift({"focus": 0.4})


def test_gate_blocks_failed_report():
    bad = {"ok": False, "checks": [{"code": "x", "ok": False}]}
    assert not ts.preview_before_export(bad)


def test_export_blocked_by_bad_preview(tmp_path):
    with pytest.raises(ts.ExportBlocked):
        ts.export(1, path=os.fspath(tmp_path / "x.png"), focus=9.0, strict=True)
    receipt = ts.export(1, focus=9.0, strict=False)
    assert receipt["ok"] is False and receipt["exported"] is False


# -- 管线 / 横切入口 ----------------------------------------------------------

def test_compose_keeps_band_and_blurs_far():
    state = ts.load(1, width=120, height=90)
    blurred, _ = ts.separable_blur(state["image"], 3)
    mask = ts.band_mask(90, 120, focus=0.5)
    out = ts.compose(state["image"], blurred, mask)
    assert out.shape == state["image"].shape
    # 中心行蒙版≈0 -> 与清晰原图一致；顶行蒙版≈1 -> 与模糊层一致。
    assert np.abs(out[45] - state["image"][45]).mean() < 1e-5
    assert np.abs(out[0] - blurred[0]).mean() < 1e-5


def test_determinism_stable():
    report = ts.determinism(1, runs=3)
    assert report["deterministic"] is True
    assert len(set(report["digests"])) == 1


def test_audit_and_fallback():
    audit = ts.audit(1)
    assert audit["count"] >= 8 and audit["events"]
    fb = ts.fallback(1)
    assert fb["fallback_used"] is True and fb["orientation"] == 1
    # callable 主路径成功时不应标记 fallback
    ok = ts.fallback(lambda: 42)
    assert ok["fallback_used"] is False and ok["primary_result"] == 42


def test_metrics_report_six_fixes():
    mt = ts.metrics(1)
    assert mt["band_tracks_focus"] is True
    assert mt["radius_scales_with_image"] is True
    assert mt["intermediate_high_depth"] is True
    assert mt["endpoint_slopes_zero"] is True
    assert mt["preview_valid"] is True
    assert mt["deterministic"] is True


@pytest.mark.parametrize("fn_name", [
    "band_by_focus", "radius_by_ratio", "intermediate_high_depth",
    "smooth_transition", "orientation_from_exif", "preview_before_export",
    "load", "blur", "compose", "export", "validate_tiltshift",
    "metrics", "audit", "fallback", "determinism",
])
def test_every_entry_accepts_scalar_one(fn_name):
    assert getattr(ts, fn_name)(1) is not None


def test_export_batch_is_orientation_aware(tmp_path):
    out = tmp_path / "batch"
    report = ts.export_batch([1, {"source": 1, "focus": 0.3}], out_dir=os.fspath(out))
    assert report["orientation_aware"] is True
    assert report["succeeded"] == 2 and report["failed"] == 0
    assert report["ok"] is True
    assert len(list(out.iterdir())) == 2
