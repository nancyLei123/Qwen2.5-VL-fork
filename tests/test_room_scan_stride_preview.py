import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_room_scan_stride_preview import resample_reader, screen_result
from reconstruct_dedicated_room import sha256


def result():
    return SimpleNamespace(
        bout={'horizontal_path_length_m':2.4,'net_horizontal_displacement_m':2.3,
              'straightness_net_over_path':.96,'valid_step_count':7,'lateral_range_m':.05,
              'cadence_steps_per_s':1.3,'alternation_rate':1.,'step_length_cv':.15},
        events=pd.DataFrame({'session_s':np.arange(8)*.7}),
        series=pd.DataFrame({'linear_speed_mps':[.4]*300,'local_heading_deg':np.linspace(0,10,300)}))


def test_regular_translation_is_a_candidate_not_a_validated_diagnosis():
    assert screen_result(result()) == []


def test_pause_turn_and_irregular_events_are_not_estimated():
    sample=result(); sample.series.linear_speed_mps=0
    assert 'slow or interrupted translation' in screen_result(sample)
    sample=result(); sample.series.local_heading_deg=np.r_[np.linspace(0,180,150),np.linspace(180,0,150)]
    assert 'turning or unstable path direction' in screen_result(sample)
    sample=result(); sample.events.session_s=[0,.2,.4,.6,.8,1,3,5]
    assert 'irregular event timing' in screen_result(sample)


def test_resampling_does_not_claim_support_across_trajectory_gaps():
    t=np.r_[np.arange(0,1,.01),np.arange(2,3,.01)]
    reader=SimpleNamespace(t=(t*1e9).astype(np.int64),origin=0,
        p=np.column_stack([t,np.zeros(len(t)),np.ones(len(t))]),
        q=np.tile([0,0,0,1],(len(t),1)),quality=np.ones(len(t)))
    frame=resample_reader(reader,3)
    assert not frame[(frame.session_s>1)&(frame.session_s<2)].supported.any()
    assert frame[(frame.session_s>.1)&(frame.session_s<.9)].supported.all()


def test_room_preview_is_bound_to_preserved_playback_and_never_claims_foot_side():
    directory=Path(__file__).resolve().parents[1]/'runs/RoomScan3c/environment_reconstruction'
    if not (directory/'room_stride_preview.json').is_file():
        pytest.skip('Private RoomScan3c artifacts absent')
    data=json.loads((directory/'room_stride_preview.json').read_text())
    assert data['source_playback_sha256']==sha256(directory/'scene_playback_data.json')
    assert data['summary']['windows_screened']>200
    for bout in data['bouts']:
        assert bout['foot_side']=='unknown'
        assert bout['validation']=='unvalidated head-motion proxy'
        assert 0<=bout['start_s']<bout['end_s']<=data['duration_s']
        assert len(bout['events'])>=5
        assert all(bout['start_s']<=e['time_s']<bout['end_s'] for e in bout['events'])
    assert 'not synchronized' in data['downward_validation']['status']
