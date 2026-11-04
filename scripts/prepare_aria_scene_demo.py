"""Create a separate capture page without modifying the research playback."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs/Ntrial3/environment_reconstruction'
    source = (output / 'scene_playback.js').read_text()
    source = "import {createDemoCapture} from './scene_demo_capture.mjs';\n" + source
    anchor = '  viewer.controls.enableDamping = false;'
    assert source.count(anchor) == 1
    source = source.replace(anchor, "  viewer.renderer.setPixelRatio(2.5);\n" + anchor)
    anchor = '  let lastPaint = 0;'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '  const demoCapture = createDemoCapture(video);\n' + anchor)
    anchor = "    $('sceneStage').dataset.rendered = String(renderedOnce);"
    assert source.count(anchor) == 1
    source = source.replace(anchor, '    demoCapture.paint(viewer.renderer.domElement);\n' + anchor)
    (output / 'presentation_playback.js').write_text(source)
    html = (output / 'playback.html').read_text().replace('src="scene_playback.js"', 'src="presentation_playback.js"')
    (output / 'presentation_capture.html').write_text(html)
    (output / 'scene_demo_capture.mjs').write_text((root / 'scripts/templates/scene_demo_capture.mjs').read_text())
    print(output / 'presentation_capture.html')


if __name__ == '__main__':
    main()
