import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pd_insighter_environment_navigation import adapt_aria_only
from run_registered_reconstruction import plan


def test_dual_view_is_not_changed(tmp_path):
    page = '<video id="downwardVideo" src="existing.mp4"></video>'
    assert adapt_aria_only(page, tmp_path, "full") == page


def test_missing_camera_never_loads_or_blocks_seeking(tmp_path):
    (tmp_path / "trial_manifest.json").write_text(json.dumps({"media_mode": "aria_only"}))
    page = '''<head></head><nav>Navigation</nav><video id="downwardVideo" src=""></video>
    <script>downwardVideo.src = data.downward_session_video_uri;
    async function seekVideoPrecisely(videoEl, targetSeconds) { await doSeek(); }
    function syncFollower(force = false) { synchronize(); }
    downwardVideo.play();</script>'''
    output = adapt_aria_only(page, tmp_path, "full")
    assert 'src=""' not in output
    assert "downwardVideo.src =" not in output
    assert "if (!videoEl.getAttribute('src')) return;" in output
    assert "if (!downwardVideo.getAttribute('src')) return;" in output
    assert "Promise.resolve()" in output
    assert "Aria-only recording" in output


def test_parkinson_review_uses_available_camera_clock(tmp_path):
    (tmp_path / "trial_manifest.json").write_text('{"media_mode":"aria_only"}')
    page = "const now = downwardVideo.currentTime || 0; downwardVideo.paused ? playBoth() : pauseBoth();"
    output = adapt_aria_only(page, tmp_path, "parkinson")
    assert "const now = ariaVideo.currentTime || 0" in output
    assert "ariaVideo.paused ?" in output
    page = "ariaVideo.addEventListener('timeupdate', () => syncVideos(ariaVideo));"
    output = adapt_aria_only(page, tmp_path, "parkinson")
    assert "ariaVideo.addEventListener('timeupdate', () => { updatePlayhead(); });" in output


def test_submap_plan_overlaps_and_does_not_claim_recording_end():
    jobs = plan("Fuchs", 1152.8)["jobs"]
    assert jobs[0]["start_s"] == 2
    assert jobs[-1]["end_s"] == 1151.3
    assert all(j["end_s"] - j["start_s"] <= 178 for j in jobs)
    assert all(a["end_s"] - b["start_s"] == 20 for a, b in zip(jobs, jobs[1:]))
    assert all(j["status"] == "pending" for j in jobs)
