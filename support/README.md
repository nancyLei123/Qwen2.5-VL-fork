# Supporting Source Snapshots

`workspace/` preserves supporting source that lives beside the main application
in the original research workspace:

- `VRS_Pre_Processing/scripts/`: the current extraction scripts, including the
  locally modified video extractor.
- `projectaria_tools/scripts/`: local helper scripts only, not the full Aria SDK.
- `scripts/`: original workspace helpers, including the chunk/MPS shell workflow.
- `tools/OpenSplat-1.2.2/`: local reconstruction source with upstream and vendor
  license files. Build products, installed libraries, models, and datasets are absent.

For recomputation, restore these source directories to the corresponding paths
under the new workspace root, beside `Qwen2.5-VL-fork/`. Do not overwrite an existing
checkout without reviewing its differences. Viewing saved results does not require
building OpenSplat or restoring these optional source snapshots.

`SOURCE_MANIFEST.json` records workspace-relative paths and SHA-256 digests for
these copied files. The original sources were not moved or modified.

Project Aria Tools and the VRS CLI remain external dependencies. The full historical
source snapshots are available in the restricted OneDrive workspace; npm packages
are restored from `vendor/gaussian-viewer/package-lock.json`. See each project's
own installation instructions rather than copying Mac binaries to another platform.

The OpenSplat source is from [WebODM/OpenSplat](https://github.com/WebODM/OpenSplat)
with local research modifications; its `LICENSE.txt` and vendored component licenses
apply independently of this repository's Qwen license. Other helpers originate from
the user's workspace; their inclusion does not relicense third-party code or data.
No proprietary Metashape executable, license, credential, or research media is included.
