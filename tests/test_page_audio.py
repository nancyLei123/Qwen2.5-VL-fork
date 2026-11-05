import json
from pathlib import Path
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pd_insighter_audio import audio_availability, inject_page_audio


def test_injection_is_idempotent_and_preserves_analysis(tmp_path):
    document = '''<html><head></head><body><video id="ariaVideo"></video>
    <script>const analysis = {"step_length": 0.45}; ariaVideo.muted = false;</script></body></html>'''
    first = inject_page_audio(document, tmp_path)
    assert inject_page_audio(first, tmp_path) == first
    assert first.count('id="pd-page-audio"') == 1
    assert 'const analysis = {"step_length": 0.45}' in first
    assert 'ariaVideo.muted = true;' in first


def test_no_player_no_control(tmp_path):
    document = '<html><head></head><body>Trial library</body></html>'
    assert inject_page_audio(document, tmp_path) == document


def test_probes_exact_environment_asset_not_stale_flag(tmp_path):
    video = tmp_path / 'review.mp4'
    video.touch()
    (tmp_path / 'scene_playback_data.json').write_text(json.dumps({'video_url': video.name, 'video_has_audio': True}))
    with patch('pd_insighter_audio._probe_audio', return_value=False) as probe:
        assert audio_availability('', tmp_path) is False
        assert probe.call_args.args[0] == str(video)


def test_session_asset_resolves_query_string(tmp_path):
    video = tmp_path / 'aria.mp4'
    video.touch()
    document = '{"aria_session_video_uri": "aria.mp4?v=12"}'
    with patch('pd_insighter_audio._probe_audio', return_value=True):
        assert audio_availability(document, tmp_path) is True


def test_unknown_media_not_misrepresented_as_silent(tmp_path):
    assert audio_availability('{"aria_session_video_uri": "missing.mp4"}', tmp_path) is None


def test_static_video_source_takes_priority_over_neighboring_playback_payload(tmp_path):
    video = tmp_path / 'original.mp4'
    video.touch()
    (tmp_path / 'scene_playback_data.json').write_text('{"video_url":"other.mp4"}')
    with patch('pd_insighter_audio._probe_audio', return_value=True) as probe:
        assert audio_availability('<video id="ariaVideo" src="original.mp4"></video>', tmp_path) is True
        assert probe.call_args.args[0] == str(video)


def test_checked_in_scene_does_not_bind_a_second_audio_handler():
    script = (Path(__file__).resolve().parents[1] / 'scripts/templates/scene_playback.js').read_text()
    assert "$('audioToggle').onclick" not in script


def test_all_deployed_primary_video_pages_have_the_current_controller():
    from pd_insighter_environment_navigation import VIEW_FILES
    from pd_insighter_audio import TEMPLATE
    root = Path(__file__).resolve().parents[1] / 'runs'
    if not root.is_dir():
        pytest.skip('Private deployed pages absent in this code-only checkout')
    checked = 0
    for directory in root.iterdir():
        pages = [*(directory / name for name in VIEW_FILES),
                 directory / 'environment_reconstruction/playback.html',
                 directory / 'environment_reconstruction/index.html']
        for path in pages:
            if not path.is_file():
                continue
            text = path.read_text()
            if '<video id="ariaVideo"' not in text:
                continue
            checked += 1
            assert text.count('id="pd-page-audio"') == 1, path
            assert TEMPLATE.read_text() in text, path
    if not checked:
        pytest.skip('No deployed primary-video pages available in this checkout')
