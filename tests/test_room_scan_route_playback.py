"""Room adapter must preserve exact frame timing and existing scientific assets."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_aria_scene_playback as player
import build_room_scan_route_playback as room


def test_encoded_review_copy_requires_exact_frame_count_and_clock(monkeypatch):
    info = {"streams": [{"nb_frames": "44490", "avg_frame_rate": "30/1", "duration": "1483.000000"}]}
    monkeypatch.setattr(room.subprocess, "check_output", lambda *a, **kw: json.dumps(info).encode())
    assert room.validate_video("unused.mp4", 44490, 30)["nb_frames"] == "44490"
    with pytest.raises(ValueError, match="frame count"):
        room.validate_video("unused.mp4", 44489, 30)
    info["streams"][0]["duration"] = "1482.729395838"
    with pytest.raises(ValueError, match="duration"):
        room.validate_video("unused.mp4", 44490, 30)


def test_room_publish_preserves_model_video_route_and_other_trials(tmp_path, monkeypatch):
    env = tmp_path / "runs/RoomScan3c/environment_reconstruction"
    env.mkdir(parents=True)
    preserved = {
        env / "scene_playback_data.json": '{"trial_id":"RoomScan3c","labels_not_applicable":true}',
        env / "splat.ply": "preserved reconstruction",
        env / "aria_room_scan.mp4": "preserved video",
        env / "quality_20261029/index.html": "preserved quality comparison",
        tmp_path / "runs/Ntrial3_labels.json": "preserved manual labels",
    }
    for path, text in preserved.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(room, "ENV", env)
    monkeypatch.setattr(player, "ROOT", tmp_path)
    room.publish()
    page = (env / "playback.html").read_text()
    assert 'href="quality_20261029/index.html"' in page
    assert "Around camera" in page
    assert "Nearby route" in page
    assert "Aria viewpoint" in page
    assert 'id="spatialPlot"' in page
    assert 'id="figurePanel"' in page
    assert 'id="roomView"' in page
    assert (env / "scene_pose_figure.mjs").read_text() == (player.TEMPLATES / "scene_pose_figure.mjs").read_text()
    assert "Manual Labels</a>" not in page
    assert 'url=playback.html' in (env / "index.html").read_text()
    for path, text in preserved.items():
        assert path.read_text() == text
