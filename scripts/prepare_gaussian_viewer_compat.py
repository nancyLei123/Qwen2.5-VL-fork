#!/usr/bin/env python3
"""Create a lossless SH1-to-SH2 PLY copy for local GaussianSplats3D playback.

Only a new compatibility asset and manifest are written. This does not restore
SH3 support, change scene quality settings, or update any playback references.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_ROOT = ROOT / "runs/Ntrial3/environment_reconstruction"
DEFAULT_SOURCE = Path("gaussian_splatting_trial/trained_proof/splat_1500.ply")
DEFAULT_OUTPUT = Path("quality_comparison_20261022/viewer_compat")
MAX_HEADER_BYTES = 65536
LIMITATIONS = [
    "Only binary_little_endian 1.0, one vertex element, and float32 scalar properties are supported.",
    "Input must have exactly nine contiguous, ordered f_rest_0 through f_rest_8 properties (SH1).",
    "Adds zero SH2 terms only; does not restore SH3 support or recover trained higher-degree terms.",
    "Prevents this SH1 asset from lowering a shared mesh below SH2; other lower-degree assets can still do so.",
    "Does not change geometry, opacity, DC color, existing SH1 terms, vertex order, or rendering settings.",
    "No templates, renderer, playback data, or original assets are updated; integration is opt-in.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ply(data: bytes) -> dict:
    """Read only the narrow, fixed-width OpenSplat layout without decoding floats."""
    stream = io.BytesIO(data)
    lines = []
    while stream.tell() < MAX_HEADER_BYTES:
        line = stream.readline(MAX_HEADER_BYTES - stream.tell() + 1)
        if not line or stream.tell() > MAX_HEADER_BYTES:
            raise ValueError("Missing or oversized PLY header")
        lines.append(line)
        if line.rstrip(b"\r\n") == b"end_header":
            break
    else:
        raise ValueError("Missing or oversized PLY header")
    if lines[0].rstrip(b"\r\n") != b"ply":
        raise ValueError("Not a PLY file")
    properties, property_lines = [], []
    count = None
    format_seen = False
    for index, line in enumerate(lines[1:-1], start=1):
        words = line.decode("ascii").split()
        if not words:
            raise ValueError("Empty PLY header line")
        if words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "format":
            if format_seen or count is not None or words != ["format", "binary_little_endian", "1.0"]:
                raise ValueError("Only binary_little_endian 1.0 is supported")
            format_seen = True
        elif words[0] == "element":
            if not format_seen or count is not None or len(words) != 3 or words[1] != "vertex":
                raise ValueError("Exactly one vertex element is required")
            count = int(words[2])
        elif words[0] == "property":
            if count is None or len(words) != 3 or words[1] != "float":
                raise ValueError("Only float32 scalar vertex properties are supported")
            properties.append(words[2])
            property_lines.append(index)
        else:
            raise ValueError("Unsupported PLY header directive")
    if count is None or count <= 0 or not properties or len(set(properties)) != len(properties):
        raise ValueError("Invalid vertex count or duplicate/missing properties")
    body = memoryview(data)[stream.tell():]
    if len(body) != count * len(properties) * 4:
        raise ValueError("PLY payload size does not match its vertex layout")
    return {"lines": lines, "properties": properties, "property_lines": property_lines,
            "vertex_count": count, "body": body}


def sh1_start(ply: dict) -> int:
    names = ply["properties"]
    required = {"x", "y", "z", "opacity", *(f"f_dc_{i}" for i in range(3)),
                *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4))}
    if not required.issubset(names):
        raise ValueError("Missing required Gaussian properties")
    rest = [name for name in names if name.startswith("f_rest_")]
    if rest != [f"f_rest_{i}" for i in range(9)]:
        raise ValueError("Expected exactly nine ordered SH1 f_rest properties")
    start = names.index("f_rest_0")
    if names[start:start + 9] != rest:
        raise ValueError("SH1 properties must be contiguous")
    indexes = ply["property_lines"][start:start + 9]
    if indexes != list(range(indexes[0], indexes[0] + 9)):
        raise ValueError("SH1 property header lines must be contiguous")
    return start


def pad_sh1_to_sh2(data: bytes) -> bytes:
    source = parse_ply(data)
    start = sh1_start(source)
    first_line = source["property_lines"][start]
    newline = b"\r\n" if source["lines"][first_line].endswith(b"\r\n") else b"\n"
    sh_lines = [f"property float f_rest_{i}".encode("ascii") + newline for i in range(24)]
    header = b"".join(source["lines"][:first_line] + sh_lines + source["lines"][first_line + 9:])
    old_stride = len(source["properties"]) * 4
    new_stride = old_stride + 15 * 4
    prefix = start * 4
    body = bytearray(source["vertex_count"] * new_stride)
    src = source["body"]
    for row in range(source["vertex_count"]):
        old, new = row * old_stride, row * new_stride
        body[new:new + prefix] = src[old:old + prefix]
        # PLY stores each color channel's coefficients contiguously, not RGB triplets.
        for channel in range(3):
            src_offset = old + prefix + channel * 3 * 4
            dst_offset = new + prefix + channel * 8 * 4
            body[dst_offset:dst_offset + 12] = src[src_offset:src_offset + 12]
        body[new + prefix + 96:new + new_stride] = src[old + prefix + 36:old + old_stride]
    return header + body


def verify_padding(original: bytes, compat: bytes) -> dict:
    source, target = parse_ply(original), parse_ply(compat)
    start = sh1_start(source)
    names = source["properties"]
    expected = names[:start] + [f"f_rest_{i}" for i in range(24)] + names[start + 9:]
    if target["properties"] != expected or target["vertex_count"] != source["vertex_count"]:
        raise ValueError("Compatibility layout or vertex count changed unexpectedly")
    non_sh_hash, sh1_hash = hashlib.sha256(), hashlib.sha256()
    old_stride, new_stride, prefix = len(names) * 4, len(expected) * 4, start * 4
    for row in range(source["vertex_count"]):
        src = source["body"][row * old_stride:(row + 1) * old_stride]
        dst = target["body"][row * new_stride:(row + 1) * new_stride]
        if src[:prefix] != dst[:prefix] or src[prefix + 36:] != dst[prefix + 96:]:
            raise ValueError("Non-SH vertex bytes changed")
        non_sh_hash.update(src[:prefix])
        non_sh_hash.update(src[prefix + 36:])
        sh1_hash.update(src[prefix:prefix + 36])
        for channel in range(3):
            old, new = prefix + channel * 12, prefix + channel * 32
            if src[old:old + 12] != dst[new:new + 12] or dst[new + 12:new + 32] != bytes(20):
                raise ValueError("Existing SH1 terms changed or SH2 padding is not zero")
    return {"vertex_count": source["vertex_count"], "non_sh_vertex_bytes_sha256": non_sh_hash.hexdigest(),
            "preserved_sh1_bytes_sha256": sh1_hash.hexdigest(), "verified_bitwise_preservation": True}


def inside_root(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Asset paths must be relative to scene_root without '..'")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Asset path must stay inside scene_root")
    return path


def prepare(scene_root=DEFAULT_SCENE_ROOT, source_relative=DEFAULT_SOURCE, output_relative=DEFAULT_OUTPUT) -> Path:
    root = Path(scene_root).resolve(strict=True)
    source = inside_root(root, Path(source_relative))
    output = inside_root(root, Path(output_relative))
    if source.is_relative_to(output):
        raise ValueError("Output must not contain the source asset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    original = source.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    compat = pad_sh1_to_sh2(original)
    invariants = verify_padding(original, compat)
    if sha256(source) != before:
        raise RuntimeError("Original SHA changed during conversion")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / f"{source.stem}_sh2_compat.ply"
    with destination.open("xb") as stream:
        stream.write(compat)
    compat_hash = sha256(destination)
    if compat_hash != hashlib.sha256(compat).hexdigest():
        raise RuntimeError("Compatibility file failed read-back hash verification")
    after = sha256(source)
    if after != before:
        raise RuntimeError("Original SHA changed; no integration manifest will be published")
    source_rel, compat_rel = source.relative_to(root).as_posix(), destination.relative_to(root).as_posix()
    manifest = {
        "schema_version": 1,
        "purpose": "Prevent legacy SH1 from lowering a mixed GaussianSplats3D mesh below SH2",
        "path_base": "environment_reconstruction (scene_root)",
        "replacements": {source_rel: compat_rel},
        "assets": [{"original_ply": source_rel, "compat_ply": compat_rel,
                    "original_sha256": before, "original_sha256_after": after, "compat_sha256": compat_hash,
                    "original_size_bytes": len(original), "compat_size_bytes": len(compat),
                    "source_sh_degree": 1, "compat_sh_degree": 2,
                    "source_rest_coefficients": 9, "compat_rest_coefficients": 24,
                    "zero_coefficients_added_per_vertex": 15, **invariants}],
        "coefficient_mapping": {f"f_rest_{c * 3 + i}": f"f_rest_{c * 8 + i}"
                                for c in range(3) for i in range(3)},
        "limitations": LIMITATIONS,
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--source-ply", type=Path, default=DEFAULT_SOURCE, help="Relative to scene-root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="New directory relative to scene-root")
    args = parser.parse_args()
    print(prepare(args.scene_root, args.source_ply, args.output_dir))


if __name__ == "__main__":
    main()
