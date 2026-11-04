import importlib.util
import json
from pathlib import Path
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("snapshot", Path(__file__).parents[1] / "scripts/stage_pd_onedrive_snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


@pytest.mark.skipif(sys.platform != "darwin", reason="APFS clone test")
def test_clone_is_not_a_hardlink(tmp_path):
    source, target = tmp_path / "source", tmp_path / "copy"
    source.write_bytes(b"original")
    snapshot.clone(source, target)
    assert source.stat().st_ino != target.stat().st_ino
    source.write_bytes(b"changed")
    assert target.read_bytes() == b"original"


def test_existing_snapshot_refused(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        snapshot.stage(tmp_path, tmp_path / "unused", output)


def test_unsafe_inventory_refused(tmp_path):
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps({"files": [{"path": "../outside"}]}))
    with pytest.raises(ValueError, match="Unsafe"):
        snapshot.stage(tmp_path, inv, tmp_path / "snapshot")


def test_changed_source_refused(tmp_path):
    source = tmp_path / "labels.json"
    source.write_text("[]")
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps({"files": [{"path": source.name, "bytes": 7, "mtime_ns": source.stat().st_mtime_ns}]}))
    with pytest.raises(RuntimeError, match="Source changed"):
        snapshot.stage(tmp_path, inv, tmp_path / "snapshot")
