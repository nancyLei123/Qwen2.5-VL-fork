import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

spec = importlib.util.spec_from_file_location('spatial', Path(__file__).parents[1] / 'scripts/build_scene_spatial_overview.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_landmark_quality_distance_and_world_frame():
    frame = pd.DataFrame({
        'graph_uid': ['a'] * 7, 'px_world': [1,2,3,4,20,np.nan,7],
        'py_world': [0]*7, 'pz_world': [0]*7,
        'inv_dist_std': [.001,.008,.001,-.001,.001,.001,.001],
        'dist_std': [.001,.001,.02,.001,.001,.001,np.nan],
    })
    points, quality = module.filter_chunk(frame,'a',cKDTree([[0,0,0]]))
    assert quality == 2
    np.testing.assert_array_equal(points,[[1,0,0]])
    frame.loc[0,'graph_uid'] = 'b'
    with pytest.raises(ValueError,match='graph_uid'):
        module.filter_chunk(frame,'a',cKDTree([[0,0,0]]))


def test_display_sample_is_deterministic_and_uses_real_landmarks():
    points = np.random.default_rng(1).normal(size=(1000,3))
    first, count = module.display_sample(points,limit=100)
    second, _ = module.display_sample(points,limit=100)
    np.testing.assert_array_equal(first,second)
    assert len(first)==100 and count>100
    originals = set(map(tuple,points))
    assert all(tuple(p) in originals for p in first)
    duplicate, count = module.display_sample(np.array([[.01,.01,.01],[.02,.02,.02]]))
    assert count==1
    np.testing.assert_array_equal(duplicate,[[.01,.01,.01]])
    with pytest.raises(ValueError):
        module.display_sample(np.zeros((0,3)))


def test_export_preserves_playback_and_records_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    output=tmp_path/'runs/T/environment_reconstruction'
    output.mkdir(parents=True)
    data=json.dumps({'trial_id':'T','provenance':{'graph_uid':'a'},'poses':[[0,0,0,0,0,0,1]]})
    (output/'scene_playback_data.json').write_text(data)
    source=tmp_path/'points.csv.gz'
    pd.DataFrame({'graph_uid':['a','a'],'px_world':[1,2],'py_world':[0,0],
        'pz_world':[1,1],'inv_dist_std':[.001,.001],'dist_std':[.001,.001]}).to_csv(source,index=False)
    result=json.loads(module.export('T',source).read_text())
    assert result['filtering']['display_count']==2
    assert result['playback_sha256']==hashlib.sha256(data.encode()).hexdigest()
    assert result['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert (output/'scene_playback_data.json').read_text()==data


def test_deployed_landmarks_match_playback_if_available():
    output=module.ROOT/'runs/60minutes_1/environment_reconstruction'
    if not (output/'spatial_overview_data.json').exists():
        pytest.skip('No deployed spatial overview')
    exported=json.loads((output/'spatial_overview_data.json').read_text())
    data=(output/'scene_playback_data.json').read_bytes()
    assert exported['playback_sha256']==hashlib.sha256(data).hexdigest()
    assert exported['graph_uid']==json.loads(data)['provenance']['graph_uid']
    points=np.asarray(exported['points'])
    assert len(points)==120_000 and np.isfinite(points).all()
