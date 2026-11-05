import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'docs/reproduction' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


importer = load('import_fuchs_data')
relocation = load('relocate_copy')


@pytest.fixture
def handoff(tmp_path):
    package = tmp_path / 'package'
    repo = tmp_path / 'workspace/Qwen2.5-VL-fork'
    server = repo / 'scripts/pd_insighter_server.py'
    server.parent.mkdir(parents=True)
    server.write_text('# current GitHub source\n')
    records = []
    names = [f'{trial}/index.html' for trial in importer.TRIALS]
    names += ['pd_insighter_trial_selector.html', 'Fuchs/manual_labels/segments.json']
    for name in names:
        relative = str(importer.PREFIX / name)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}' if name.endswith('.json') else '<html>test</html>')
        records.append({'path': relative, 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    records.append({'path': 'workspace/Qwen2.5-VL-fork/scripts/private_snapshot.py',
                    'bytes': 0, 'sha256': 'not imported'})
    manifest = package / importer.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'files': records}))
    return package, repo, records


def test_import_is_dry_by_default_and_copies_no_source(handoff):
    package, repo, records = handoff
    assert importer.import_data(package, repo) == 5
    assert not (repo / 'runs').exists()
    importer.import_data(package, repo, apply=True)
    assert (repo / 'runs/Fuchs/index.html').is_file()
    assert not (repo / 'scripts/private_snapshot.py').exists()
    assert (repo / 'scripts/pd_insighter_server.py').read_text() == '# current GitHub source\n'
    assert (package / records[0]['path']).read_text() == '<html>test</html>'
    with pytest.raises(FileExistsError):
        importer.import_data(package, repo, apply=True)


def test_changed_data_rejected_before_copy(handoff):
    package, repo, records = handoff
    source = package / records[0]['path']
    source.write_text('<html>FAIL</html>')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        importer.import_data(package, repo, apply=True)
    assert not (repo / 'runs').exists()


@pytest.mark.parametrize('bad', ['/tmp/private', '../private', 'workspace/../private', 'x\\..\\private'])
def test_unsafe_manifest_rejected(handoff, bad):
    package, repo, records = handoff
    records.append({'path': bad})
    (package / importer.MANIFEST).write_text(json.dumps({'files': records}))
    with pytest.raises(ValueError, match='Unsafe manifest'):
        importer.import_data(package, repo, apply=True)
    assert not (repo / 'runs').exists()


def test_missing_and_symlink_data_rejected(handoff):
    package, repo, records = handoff
    source = package / records[0]['path']
    source.unlink()
    with pytest.raises(ValueError, match='Missing data'):
        importer.import_data(package, repo)
    source.symlink_to(repo / 'scripts/pd_insighter_server.py')
    with pytest.raises(ValueError, match='Symlink'):
        importer.import_data(package, repo)


def test_data_only_relocation_keeps_source_and_annotations(handoff):
    package, repo, _ = handoff
    importer.import_data(package, repo, apply=True)
    server = repo / 'scripts/pd_insighter_server.py'
    server.write_text(relocation.ORIGINAL)
    data = repo / 'runs/Fuchs/summary.json'
    data.write_text(json.dumps({'path': relocation.ORIGINAL + '/example.mp4'}))
    labels = repo / 'runs/Fuchs/manual_labels/segments.json'
    labels.write_text(json.dumps({'note': relocation.ORIGINAL}))
    before = labels.read_bytes()
    assert len(relocation.relocate(repo.parent, data_only=True)) == 1
    assert len(relocation.relocate(repo.parent, apply=True, data_only=True)) == 1
    assert server.read_text() == relocation.ORIGINAL
    assert labels.read_bytes() == before
    assert str(repo.parent) in data.read_text()
    assert relocation.relocate(repo.parent, data_only=True) == []
