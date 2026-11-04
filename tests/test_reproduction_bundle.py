import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_pd_reproduction_bundle import archive_group, excluded_reason, sha256

spec = importlib.util.spec_from_file_location("relocate_copy", Path(__file__).resolve().parents[1] / "docs/reproduction/relocate_copy.py")
relocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relocation)


def test_sensitive_and_disposable_files_excluded():
    for path in (".env", "x/token.json", "x/.git/config", "a.vrs.enc", "x/__pycache__/a.pyc", "a.key"):
        assert excluded_reason(Path(path))
    assert excluded_reason(Path("x/closed_loop_trajectory.csv")) is None


def test_relocation_is_copy_only_and_preserves_annotations(tmp_path):
    server = tmp_path / "Qwen2.5-VL-fork/scripts/pd_insighter_server.py"
    server.parent.mkdir(parents=True)
    original = f'ROOT = "{relocation.ORIGINAL}"'
    server.write_text(original)
    labels = tmp_path / "Qwen2.5-VL-fork/runs/Ntrial10/parkinson_motion_review/manual_segments.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps({"note": relocation.ORIGINAL, "segments": []}))
    before = labels.read_bytes()
    assert len(relocation.relocate(tmp_path)) == 1
    assert server.read_text() == original
    assert len(relocation.relocate(tmp_path, True)) == 1
    assert str(tmp_path) in server.read_text()
    assert labels.read_bytes() == before
    assert (tmp_path / ".relocation_originals/Qwen2.5-VL-fork/scripts/pd_insighter_server.py").read_text() == original
    with pytest.raises(ValueError):
        relocation.relocate(Path(relocation.ORIGINAL), True)


def test_archive_checksums_and_no_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = source / "label.json"
    data.write_text('{"segments":[]}')
    before = data.read_bytes()
    stat = data.stat()
    entries = [{"group": "00_code", "path": "label.json", "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}]
    output = tmp_path / "output"
    report = archive_group(source, output, "00_code", entries)[0]
    archive = output / "archives" / report["archive"]
    with zipfile.ZipFile(archive) as pack:
        assert pack.read("workspace/label.json") == before
        checks = json.loads(pack.read("_package_checksums/" + archive.name + ".json"))
        assert checks[0]["sha256"] == sha256(data)
    assert data.read_bytes() == before
    assert report["upload_status"] == "NOT_UPLOADED"
    with pytest.raises(FileExistsError):
        archive_group(source, output, "00_code", entries)


def test_archive_rejects_changed_sources(tmp_path):
    data = tmp_path / "source.json"
    data.write_text('{}')
    entries = [{"group": "test", "path": data.name, "bytes": 1, "mtime_ns": data.stat().st_mtime_ns}]
    with pytest.raises(RuntimeError, match="Source changed"):
        archive_group(tmp_path, tmp_path / "output", "test", entries)
