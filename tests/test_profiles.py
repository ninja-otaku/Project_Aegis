"""Smoke tests for the game profile system.

Verifies profile JSON schema compliance, fallback behaviour on missing profiles,
and the ROI cropping pipeline.  No real camera, network, or API keys required.
Run with:  pytest tests/
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_PROFILE_KEYS = {
    "game_name",
    "system_prompt",
    "frame_diff_threshold",
    "roi_strategy",
    "roi_crops",
}

_VALID_ROI_STRATEGIES = {"full_frame", "horizontal_stack", "vertical_stack", "grid"}


def _make_processor() -> "FrameProcessor":
    """Construct a FrameProcessor with mock intake and provider."""
    from engine.processor import FrameProcessor

    intake = MagicMock()
    provider = MagicMock()
    return FrameProcessor(
        intake=intake,
        provider=provider,
        interval_ms=2000,
        history_max=10,
    )


# ---------------------------------------------------------------------------
# JSON schema compliance for all built-in profiles
# ---------------------------------------------------------------------------

def test_all_profiles_are_valid_json():
    """Every .json file in profiles/ must be parseable."""
    profiles_dir = Path("profiles")
    assert profiles_dir.is_dir(), "profiles/ directory not found — run from repo root"

    for path in profiles_dir.glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{path.name} is not valid JSON: {exc}")


def test_all_profiles_have_required_keys():
    """Every profile must contain all required schema keys."""
    profiles_dir = Path("profiles")
    assert profiles_dir.is_dir(), "profiles/ directory not found — run from repo root"

    for path in profiles_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = _REQUIRED_PROFILE_KEYS - set(data.keys())
        assert not missing, (
            f"{path.name} is missing required keys: {missing}"
        )


def test_all_profiles_have_valid_roi_strategy():
    """roi_strategy must be one of the recognised values."""
    profiles_dir = Path("profiles")

    for path in profiles_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        strategy = data.get("roi_strategy")
        assert strategy in _VALID_ROI_STRATEGIES, (
            f"{path.name}: unknown roi_strategy '{strategy}'. "
            f"Valid options: {_VALID_ROI_STRATEGIES}"
        )


def test_all_profiles_have_valid_threshold():
    """frame_diff_threshold must be a positive number."""
    profiles_dir = Path("profiles")

    for path in profiles_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        threshold = data.get("frame_diff_threshold")
        assert isinstance(threshold, (int, float)), (
            f"{path.name}: frame_diff_threshold must be numeric, got {type(threshold)}"
        )
        assert threshold > 0, (
            f"{path.name}: frame_diff_threshold must be positive, got {threshold}"
        )


def test_profile_roi_crops_have_required_keys():
    """Each crop in roi_crops must have x, y, w, h keys."""
    profiles_dir = Path("profiles")

    for path in profiles_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, crop in data.get("roi_crops", {}).items():
            if name.startswith("_"):
                continue  # skip comment keys
            for coord in ("x", "y", "w", "h"):
                assert coord in crop, (
                    f"{path.name}: crop '{name}' is missing key '{coord}'"
                )


# ---------------------------------------------------------------------------
# _load_profile — fallback behaviour
# ---------------------------------------------------------------------------

def test_load_profile_falls_back_to_default_on_missing_profile():
    """If the requested profile does not exist, _load_profile_by_name falls back to default.json."""
    proc = _make_processor()
    result = proc.reload_profile("__nonexistent_game__")
    assert result == "default", (
        f"Expected 'default' after fallback, got '{result}'"
    )


def test_load_profile_loads_known_profile():
    """reload_profile with a valid profile name should load that profile."""
    proc = _make_processor()
    result = proc.reload_profile("league_of_legends")
    assert result == "league_of_legends", (
        f"Expected 'league_of_legends', got '{result}'"
    )


def test_load_profile_sets_system_prompt():
    """After loading a profile, _profile_system_prompt should reflect the JSON value."""
    proc = _make_processor()
    proc.reload_profile("league_of_legends")

    profiles_dir = Path("profiles")
    data = json.loads((profiles_dir / "league_of_legends.json").read_text())
    assert proc._profile_system_prompt == data["system_prompt"]


def test_load_profile_sets_frame_diff_threshold():
    """After loading a profile, _profile_frame_diff_threshold should match the JSON value."""
    proc = _make_processor()
    proc.reload_profile("valorant")

    data = json.loads(Path("profiles/valorant.json").read_text())
    assert proc._profile_frame_diff_threshold == pytest.approx(data["frame_diff_threshold"])


# ---------------------------------------------------------------------------
# _apply_roi_cropping
# ---------------------------------------------------------------------------

def _solid_frame(h: int = 120, w: int = 160, channels: int = 3) -> np.ndarray:
    """Return a solid-colour BGR frame for cropping tests."""
    return np.full((h, w, channels), 64, dtype=np.uint8)


def test_roi_cropping_full_frame_returns_unchanged():
    """With roi_strategy='full_frame', the original frame is returned unchanged."""
    proc = _make_processor()
    proc.reload_profile("default")  # default.json uses full_frame

    frame = _solid_frame()
    result = proc._apply_roi_cropping(frame)

    # Should be the exact same object (no copy) for full_frame strategy.
    assert result is frame, "full_frame strategy should return the original frame object"


def test_roi_cropping_horizontal_stack_reduces_area():
    """horizontal_stack with valid crops should return a frame smaller than the original."""
    proc = _make_processor()
    proc.reload_profile("league_of_legends")

    frame = _solid_frame(h=200, w=400)
    result = proc._apply_roi_cropping(frame)

    original_pixels = frame.shape[0] * frame.shape[1]
    result_pixels   = result.shape[0] * result.shape[1]
    assert result_pixels <= original_pixels, (
        f"ROI-cropped frame ({result_pixels}px) is larger than original ({original_pixels}px)"
    )


def test_roi_cropping_returns_ndarray():
    """_apply_roi_cropping must always return a numpy ndarray."""
    proc = _make_processor()
    for profile in ("default", "league_of_legends", "valorant"):
        proc.reload_profile(profile)
        frame = _solid_frame()
        result = proc._apply_roi_cropping(frame)
        assert isinstance(result, np.ndarray), (
            f"Profile '{profile}': _apply_roi_cropping returned {type(result)}"
        )


def test_roi_cropping_empty_crops_returns_frame_unchanged():
    """If roi_crops is empty the original frame is returned regardless of strategy."""
    proc = _make_processor()
    # Manually set horizontal_stack with no crops.
    proc._roi_strategy = "horizontal_stack"
    proc._roi_crops = {}

    frame = _solid_frame()
    result = proc._apply_roi_cropping(frame)
    assert result is frame
