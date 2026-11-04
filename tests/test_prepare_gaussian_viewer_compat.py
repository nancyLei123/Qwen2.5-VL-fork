import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("viewer_compat", ROOT / "scripts/prepare_gaussian_viewer_compat.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RENDERER = ROOT / "vendor/gaussian-viewer/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js"


def fixture_ply(newline=b"\n"):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(9)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "confidence"]
    lines = [b"ply", b"format binary_little_endian 1.0", b"comment preserve me", b"element vertex 2"]
    lines += [f"property float {name}".encode() for name in names]
    lines += [b"end_header"]
    rows = []
    for row in range(2):
        values = [float((row + 1) * (i + 1)) for i in range(len(names))]
        values[names.index("opacity")] = 2.0
        values[names.index("rot_0")] = 1.0
        for i in range(1, 4):
            values[names.index(f"rot_{i}")] = 0.0
        for i in range(3):
            values[names.index(f"scale_{i}")] = -2.0
        values[names.index("ny")] = -0.0
        for channel in range(3):
            for term in range(3):
                values[names.index(f"f_rest_{channel * 3 + term}")] = (row + 1) * (channel * 10 + term + 1) / 64
        rows.append(struct.pack(f"<{len(names)}f", *values))
    return newline.join(lines) + newline + b"".join(rows), names, rows


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_channel_major_padding_is_lossless(newline):
    original, names, rows = fixture_ply(newline)
    compat = MODULE.pad_sh1_to_sh2(original)
    header, body = compat.split(b"end_header" + newline, 1)
    output_names = [line.decode().split()[2] for line in header.splitlines() if line.startswith(b"property")]
    assert len(output_names) == len(names) + 15
    assert b"comment preserve me" + newline in header
    for index, row in enumerate(rows):
        out_row = body[index * len(output_names) * 4:(index + 1) * len(output_names) * 4]
        for field, name in enumerate(names):
            target = name
            if name.startswith("f_rest_"):
                channel, term = divmod(int(name[7:]), 3)
                target = f"f_rest_{channel * 8 + term}"
            offset = output_names.index(target) * 4
            assert out_row[offset:offset + 4] == row[field * 4:field * 4 + 4]
        for channel in range(3):
            for term in range(3, 8):
                offset = output_names.index(f"f_rest_{channel * 8 + term}") * 4
                assert out_row[offset:offset + 4] == b"\0" * 4
    assert MODULE.verify_padding(original, compat)["verified_bitwise_preservation"]


@pytest.mark.parametrize("mutation", [
    lambda b: b.replace(b"binary_little_endian", b"ascii"),
    lambda b: b.replace(b"binary_little_endian", b"binary_big_endian"),
    lambda b: b.replace(b"property float confidence", b"property double confidence"),
    lambda b: b.replace(b"property float confidence", b"property list uchar float confidence"),
    lambda b: b.replace(b"end_header", b"element face 0\nend_header"),
    lambda b: b.replace(b"f_rest_8", b"f_rest_9"),
    lambda b: b.replace(b"property float x\n", b"property float y\n"),
    lambda b: b.replace(b"property float x\n", b"property float other_x\n"),
    lambda b: b[:-1],
    lambda b: b + b"\0",
    lambda b: b.replace(b"end_header", b"no_end"),
])
def test_unsupported_or_malformed_input_is_rejected(mutation):
    original, _, _ = fixture_ply()
    with pytest.raises(ValueError):
        MODULE.pad_sh1_to_sh2(mutation(original))


def test_already_sh2_input_is_not_reconverted():
    original, _, _ = fixture_ply()
    with pytest.raises(ValueError, match="nine ordered SH1"):
        MODULE.pad_sh1_to_sh2(MODULE.pad_sh1_to_sh2(original))


def test_noncontiguous_sh1_properties_are_rejected():
    original, _, _ = fixture_ply()
    original = original.replace(b"property float f_rest_4\nproperty float f_rest_5\n",
                                b"property float confidence\nproperty float f_rest_5\n")
    original = original.replace(b"property float confidence\nend_header",
                                b"property float f_rest_4\nend_header")
    with pytest.raises(ValueError):
        MODULE.pad_sh1_to_sh2(original)


@pytest.mark.parametrize("field", ["x", "opacity", "f_rest_8", "f_rest_3"])
def test_verifier_detects_changes_to_geometry_existing_terms_and_padding(field):
    original, _, _ = fixture_ply()
    compat = MODULE.pad_sh1_to_sh2(original)
    parsed = MODULE.parse_ply(compat)
    damaged = bytearray(compat)
    offset = len(compat) - len(parsed["body"]) + parsed["properties"].index(field) * 4
    damaged[offset] ^= 1
    with pytest.raises(ValueError):
        MODULE.verify_padding(original, bytes(damaged))


def write_source(root):
    original, _, _ = fixture_ply()
    source = root / MODULE.DEFAULT_SOURCE
    source.parent.mkdir(parents=True)
    source.write_bytes(original)
    return source, original


def test_prepare_writes_new_asset_and_auditable_mapping_only(tmp_path):
    source, original = write_source(tmp_path)
    before = MODULE.sha256(source)
    manifest_path = MODULE.prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    replacement = manifest["replacements"][MODULE.DEFAULT_SOURCE.as_posix()]
    assert isinstance(replacement, str)
    assert replacement == (MODULE.DEFAULT_OUTPUT / "splat_1500_sh2_compat.ply").as_posix()
    assert not Path(replacement).is_absolute() and ".." not in Path(replacement).parts
    target = tmp_path / replacement
    record = manifest["assets"][0]
    assert source.read_bytes() == original
    assert MODULE.sha256(source) == before == record["original_sha256"] == record["original_sha256_after"]
    assert MODULE.sha256(target) == record["compat_sha256"]
    assert record["zero_coefficients_added_per_vertex"] == 15
    assert record["source_sh_degree"] == 1 and record["compat_sh_degree"] == 2
    assert manifest["coefficient_mapping"]["f_rest_3"] == "f_rest_8"
    assert manifest["coefficient_mapping"]["f_rest_6"] == "f_rest_16"
    assert any("does not restore SH3" in item for item in manifest["limitations"])
    assert MODULE.verify_padding(original, target.read_bytes())["non_sh_vertex_bytes_sha256"] == record["non_sh_vertex_bytes_sha256"]
    assert set(path for path in tmp_path.rglob("*") if path.is_file()) == {source, target, manifest_path}
    with pytest.raises(FileExistsError):
        MODULE.prepare(tmp_path)
    assert record["compat_sha256"] == MODULE.sha256(target)


@pytest.mark.parametrize("output", ["../escape", "/tmp/unsafe", ".", "gaussian_splatting_trial"])
def test_unsafe_output_rejected(tmp_path, output):
    source, original = write_source(tmp_path)
    with pytest.raises(ValueError):
        MODULE.prepare(tmp_path, output_relative=output)
    assert source.read_bytes() == original


def test_source_change_aborts_before_output(tmp_path, monkeypatch):
    source, original = write_source(tmp_path)
    monkeypatch.setattr(MODULE, "sha256", lambda path: "changed")
    with pytest.raises(RuntimeError, match="Original SHA changed"):
        MODULE.prepare(tmp_path)
    assert source.read_bytes() == original
    assert not (tmp_path / MODULE.DEFAULT_OUTPUT).exists()


def test_source_change_after_output_prevents_manifest(tmp_path, monkeypatch):
    source, original = write_source(tmp_path)
    real_hash = MODULE.sha256
    source_checks = 0

    def changing_hash(path):
        nonlocal source_checks
        if path == source:
            source_checks += 1
            if source_checks == 2:
                return "changed"
        return real_hash(path)

    monkeypatch.setattr(MODULE, "sha256", changing_hash)
    with pytest.raises(RuntimeError, match="no integration manifest"):
        MODULE.prepare(tmp_path)
    assert not (tmp_path / MODULE.DEFAULT_OUTPUT / "manifest.json").exists()
    assert source.read_bytes() == original


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "scene"
    root.mkdir()
    source, original = write_source(root)
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside scene_root"):
        MODULE.prepare(root, output_relative="outside/compat")
    assert source.read_bytes() == original
    assert not (tmp_path / "compat").exists()


NODE_RENDERER_CHECK = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [rendererPath, threePath, originalPath, compatPath, ...newerPaths] = process.argv.slice(1);
const G = await import(pathToFileURL(rendererPath));
const THREE = await import(pathToFileURL(threePath));
function parse(path) {
    const b = fs.readFileSync(path);
    return G.PlyParser.parseToUncompressedSplatArray(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), 2);
}
const original = parse(originalPath), compat = parse(compatPath), newer = newerPaths.map(parse);
assert.equal(original.sphericalHarmonicsDegree, 1);
assert.equal(compat.sphericalHarmonicsDegree, 2);
assert.ok(newer.length > 0);
for (const scene of newer) assert.equal(scene.sphericalHarmonicsDegree, 2);
assert.equal(original.splatCount, compat.splatCount);
for (let i = 0; i < original.splatCount; i++) {
    // The local parser rearranges channel-major PLY fields into degree-grouped storage.
    assert.deepEqual(compat.splats[i].slice(0, original.splats[i].length), original.splats[i]);
    assert.deepEqual(compat.splats[i].slice(original.splats[i].length), Array(15).fill(0));
}
const buffer = a => G.SplatBuffer.generateFromUncompressedSplatArrays([a], 1, 0, new THREE.Vector3());
const buffers = {original: buffer(original), compat: buffer(compat), newer: newer.map(buffer)};
assert.equal(buffers.original.getSplatCount(), buffers.compat.getSplatCount());
const holder = {splatRenderMode: G.SplatRenderMode.ThreeD, dynamicScene: false,
    enableOptionalEffects: true, halfPrecisionCovariancesOnGPU: false, devicePixelRatio: 1,
    gpuAcceleratedSort: false, integerBasedSort: false, antialiased: false,
    maxScreenSpaceSplatSize: 1024, logLevel: G.LogLevel.None, sphericalHarmonicsDegree: 2,
    sceneFadeInRateMultiplier: 1, kernel2DSize: 0.3};
G.Viewer.prototype.createSplatMesh.call(holder);
const mesh = holder.splatMesh;
function check(list, expected) {
    mesh.build(list, list.map((_, i) => ({splatAlphaRemovalThreshold: 1, visible: i === 0})), false, false);
    assert.equal(mesh.minSphericalHarmonicsDegree, expected);
    assert.equal(mesh.material.uniforms.sphericalHarmonicsDegree.value, expected);
    assert.equal(mesh.material.vertexShader.includes('if (sphericalHarmonicsDegree >= 2)'), expected === 2);
}
try {
    check(buffers.newer, 2);
    check([...buffers.newer, buffers.original], 1);
    check([...buffers.newer, buffers.compat], 2);
    check([buffers.compat, ...buffers.newer], 2);
} finally { mesh.dispose(); }
console.log(JSON.stringify({legacyCount: original.splatCount, newerScenes: newer.length,
    newerCount: newer.reduce((n, scene) => n + scene.splatCount, 0),
    retainedLegacyCount: buffers.compat.getSplatCount(), originalMixedMinSH: 1, compatMixedMinSH: 2}));
"""


@pytest.mark.skipif(not RENDERER.is_file() or shutil.which("node") is None, reason="Local renderer/Node absent")
@pytest.mark.parametrize("local_assets", [False, True], ids=["synthetic", "ntrial3"])
def test_local_renderer_preserves_sh2_with_compat(tmp_path, local_assets):
    if local_assets:
        source = MODULE.DEFAULT_SCENE_ROOT / MODULE.DEFAULT_SOURCE
        scene_names = ["sitting_context", "connecting_walk_gap", "upstairs_context",
                       "corridor_walking", "downstairs_context", "final_walk_gap"]
        newer_paths = [MODULE.DEFAULT_SCENE_ROOT / f"gaussian_submaps/{name}/trained_7000_fullres/splat_7000.ply"
                       for name in scene_names]
        if not source.is_file() or not all(path.is_file() for path in newer_paths):
            pytest.skip("Ntrial3 assets absent")
        original = source.read_bytes()
    else:
        original, _, _ = fixture_ply()
        source = tmp_path / "original.ply"
        source.write_bytes(original)
        newer = tmp_path / "newer.ply"
        padded = MODULE.pad_sh1_to_sh2(original)
        # Ensure the higher-degree fixture has a nonzero SH2 term, not only padding.
        layout = MODULE.parse_ply(padded)
        changed = bytearray(padded)
        offset = len(padded) - len(layout["body"]) + layout["properties"].index("f_rest_3") * 4
        struct.pack_into("<f", changed, offset, 0.125)
        newer.write_bytes(changed)
        newer_paths = [newer]
    compat = tmp_path / "compat.ply"
    compat.write_bytes(MODULE.pad_sh1_to_sh2(original))
    three = ROOT / "vendor/gaussian-viewer/node_modules/three/build/three.module.js"
    result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", NODE_RENDERER_CHECK,
                             str(RENDERER), str(three), str(source), str(compat), *map(str, newer_paths)],
                            text=True, capture_output=True, timeout=60, check=True)
    result = json.loads(result.stdout)
    assert result["compatMixedMinSH"] == 2
    assert result["newerScenes"] == len(newer_paths)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original).digest()
