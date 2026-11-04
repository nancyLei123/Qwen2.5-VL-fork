"""Build a compact screenshot-only page using the unchanged playback engine."""

from pathlib import Path


STYLE = """
/* Fixed presentation layout; this does not alter the research playback page. */
body{min-width:1440px}
main{width:1440px;max-width:none;padding:24px;gap:16px;grid-template-columns:minmax(0,1fr) 400px;grid-template-areas:'hero hero' 'views views' 'timeline context' 'note note'}
main>.nav,main>details{display:none}
.hero{grid-area:hero;padding:0 4px}.hero h1{font-size:32px;margin:0 0 4px}.hero p{font-size:15px;margin:0}
.workspace{grid-area:views;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);align-items:stretch;gap:16px;--scene-size:400px;--video-size:350px}
.heading{padding:12px 16px;min-height:48px;align-items:center}.toolbar{padding:8px 16px;min-height:48px}
.toolbar button{font-size:12px;padding:7px 10px}
.scene-card .stage,.video-card .stage{height:400px}
.scene-card .square{width:400px}.video-card .square{width:350px}
.caption{padding:9px 16px;min-height:46px;margin:0;font-size:12px}
.timeline{grid-area:timeline;padding:18px;align-self:stretch}
.transport{gap:8px;margin-bottom:12px}.transport>div{gap:8px}.transport label{display:none}.time{font-size:18px}
.timeline-help,.gap-details,.timeline>p.small{display:none}
.timeline-row{grid-template-columns:144px minmax(0,1fr);gap:12px;margin:14px 0}
.track{height:27px}.timeline-label{font-size:12px}.ticks{font-size:11px}
.legend{margin:12px 0 0 156px;gap:12px;font-size:12px}.coverage-key{gap:8px 15px;font-size:11px;margin-top:12px}
.availability{font-size:12px;margin-top:12px}
.context{grid-area:context;grid-template-columns:minmax(0,1fr);gap:10px;padding:18px;align-content:start}
.context h2{font-size:17px}.context svg{height:190px;margin-top:8px}.context p.small{font-size:11px;line-height:1.4;margin:5px 0}
.readout{font-size:12px;line-height:1.6}.readout h2,.readout .notice,.scene-link,#poseStatus{display:none}.readout strong{font-size:15px}
.snapshot-note{grid-area:note;color:var(--muted);font-size:12px;margin:0 4px;line-height:1.4}
"""


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/Ntrial3/environment_reconstruction"
    html = (output / "playback.html").read_text()
    html = html.replace("</style>", STYLE + "\n</style>", 1)
    html = html.replace(
        "<h1>Synchronized 3D Playback</h1>",
        "<h1>PD Insighter / NTrial3 / Synchronized 3D Playback</h1>",
        1,
    )
    html = html.replace(
        "</main>",
        '<p class="snapshot-note">Research prototype: static surroundings, not a reconstructed body. '
        'The original video clock synchronizes the 3D camera pose, action tracks and top-down route. '
        'Blur and missing surfaces remain.</p></main>',
        1,
    )
    target = output / "presentation_sync_snapshot.html"
    target.write_text(html)
    print(target)


if __name__ == "__main__":
    main()
