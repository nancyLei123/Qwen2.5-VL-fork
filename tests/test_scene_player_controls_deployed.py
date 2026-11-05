"""Ensure locally published trials do not silently retain an older camera UI."""
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["Ntrial3", "60minutes_1", "RoomScan3c", "Fuchs", "Fuchs22", "Fuchs333"]


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = {}
        self.options = {}
        self.current = None
        self.display = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "button":
            self.current = attrs.get("id")
            self.buttons[self.current] = ""
        elif tag == "select":
            self.display = attrs.get("id") == "displayQuality"
        elif tag == "option" and self.display:
            self.current = attrs["value"]
            self.options[self.current] = ""

    def handle_data(self, value):
        if self.current is not None:
            target = self.options if self.display else self.buttons
            target[self.current] += value

    def handle_endtag(self, tag):
        if tag in ("button", "option"):
            self.current = None
        elif tag == "select":
            self.display = False


@pytest.mark.parametrize("trial", TRIALS)
def test_deployed_controls_and_complete_bundle(trial):
    output = ROOT / "runs" / trial / "environment_reconstruction"
    if not (output / "scene_playback_data.json").is_file():
        pytest.skip("Local research artifacts are not present in a code-only checkout.")
    controls = Controls()
    controls.feed((output / "playback.html").read_text())
    for key, label in (("explore", "Around camera"), ("cameraContext", "Nearby route"),
                       ("follow", "Aria viewpoint"), ("recenter", "Reset view")):
        assert controls.buttons.get(key) == label
    assert controls.options == {"high": "High / Retina", "balanced": "Balanced"}
    player = (output / "scene_playback.js").read_text()
    assert "const recordedViewFrame = new RecordedViewFrame(THREE)" in player
    assert "const display = figure ? figureDisplay : 'point'" in player
    assert "updateRecordedViewFrame(time)" in player
    assert 'id="viewFrameNote"' in (output / "playback.html").read_text()
    for name in ("scene_playback.js", "scene_playback_core.mjs", "scene_playback_quality.mjs",
                 "scene_route_navigation.mjs", "scene_camera_view.mjs", "scene_spatial_overview.mjs",
                 "scene_pose_figure.mjs", "scene_surface_visibility.mjs", "scene_view_footprint.mjs"):
        assert (output / name).read_bytes() == (ROOT / "scripts/templates" / name).read_bytes()
