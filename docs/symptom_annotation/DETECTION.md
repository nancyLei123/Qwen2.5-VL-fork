# Automatic symptom detection (baseline)

The detector proposes time intervals where a Parkinson's-like movement event may be happening.
The proposals show up on the Symptom Annotation timeline as predictions, and I review them
against the synchronized video: accept, edit, reject, or leave for later.

This is a signal-processing baseline. It is not a trained model, it has not been validated
against reviewed labels, and it does not diagnose anything. Time without a prediction is not
evidence that nothing happened there.

## Run it

With trial data in `runs/`:

```bash
python scripts/run_symptom_detection.py
python scripts/run_symptom_detection.py --apply
```

The first command is a dry run: it runs the detector and lists the candidates, and writes
nothing. `--apply` saves each trial's predictions (see [Storage](#storage)). Useful options:

- `--trials NAME ...` to limit it to some trials
- `--symptom-store DIR` to write somewhere other than `runs/` (use the same value for the server)
- `--config FILE` for a JSON file with overrides of thresholds or labels
- `--report FILE` to also write a Markdown summary of candidates per trial
- `--verbose` to print every evidence item
- `--no-video`, `--no-audio` to skip those sources

Running it again with the same detector version and configuration gives the same result and the
same run ID. The downward video analysis takes about a minute per ten minutes of video the first
time and is cached next to the predictions. Recordings, analysis results, manual annotations and
review decisions are only read.

After `--apply`, reload the Symptom Annotation page. If the page was built before this feature
existed, run `python scripts/update_symptom_annotation_pages.py --apply` once and restart the
server.

## What it reads

Everything comes from a trial's saved outputs under `runs/<trial>/` and is placed on the
dashboard session clock (seconds from the first frame of the session videos, the clock of Full
Session and Symptom Annotation).

| Source | File | Kind | Used for |
| --- | --- | --- | --- |
| Head track | `environment_reconstruction/scene_playback_data.json` (MPS SLAM camera pose per video frame, 30 Hz, with per-frame tracking status). Fallback: the ~15 Hz x/y/z samples in `session_dashboard_manifest.json`, without orientation | measured | speed, net travel, stepping rhythm, sway, turning, height |
| Downward video | the downward session video, dense optical flow at its native 15 fps on a 160x90 copy | measured image motion | whether the legs and body keep moving |
| Audio | `downward_sync/aria_sync_audio.wav`, or the Aria session video's audio track through ffmpeg | supporting | loaded and reported, not scored (see below) |
| Activity labels | `continuous_model_segments` in the session manifest | inferred | context only (a "sitting" label lowers a score) |
| Step proxies | `stride_review.events` | inferred | reported as available; not scored |

Pose frame `i` is shown by video frame `i`, so its session time is `i / fps`. Frames without a
tracked pose stay empty, and gaps longer than 0.5 s are not interpolated. The downward video is
only analysed inside the recorded window from `downward_attachment`; padding and dark frames are
masked, so black padding can never count as "no leg movement". An Aria-only trial simply has no
downward source.

Measured and inferred are kept apart on purpose. Step rate and step length are estimated from
head motion (one vertical head bob per step). They are not foot-contact measurements.

Features, all on a 20 Hz grid:

- `speed`: horizontal head speed, 0.5 s average
- `progress_speed`, `progress_fast`: straight-line distance covered over 2 s and 1 s, which
  ignores sway and stepping in place
- `travel_extent`, `vertical_extent`: how much ground and height the head covers within 8 s
- `bob`, `bob_amp`, `step_rate_hz`: vertical head movement in the 0.6-3.2 Hz stepping band, its
  amplitude, and the rate of its peaks
- `sway_amp`: horizontal movement that is not travel
- `yaw_rate`, `path_turn_rate`: head rotation about the vertical axis and change of the direction
  of travel
- `height`, `vertical_speed`: slow height trend (stairs, sitting down, standing up)
- `flow_smooth`, `flow_step_amp`: mean optical-flow magnitude in the downward view and its
  stepping-band amplitude

## What it detects

Walking context is measured, not taken from the activity labels: net travel of at least
0.07 m/s while the head covers at least 0.45 m of ground or height within 8 s. The second
condition matters. Without it, someone shifting in a chair looks like slow walking.

| Label | Pattern | Main evidence |
| --- | --- | --- |
| Freezing-like episode | Walking was under way, net progress stays below 0.045 m/s for 2-30 s, and the body keeps moving in place | head sway and stepping-band head bob during the stop; leg and body motion in the downward view |
| Gait hesitation | A 0.8-3 s interruption inside walking, or movement in place for 1.2-6 s right before walking starts | drop in travel, walking on both sides, movement in place |
| Shuffling-like gait | Level walking for at least 5 s with an estimated step length under 0.30 m while the step rate stays at 1.1 steps/s or more | step length estimate, kept rhythm, downward view confirms movement |
| Slowed gait | Level walking for at least 5 s below 0.40 m/s that is not shuffling-like | speed, a stepping rhythm, straightness, downward view |

Things that are deliberately not reported, and counted in each run's summary instead:

- a stop with a quiet body (an ordinary stop)
- a stop longer than 30 s (walking ended)
- a stop with a head height change of 25 cm or more (sitting down, standing up)
- movement in place that goes on for the whole 6 s before walking (busy at a counter, not a start attempt)
- the pause between two slow steps (travel before it was under 0.15 m/s)
- gait bouts that are mostly turning, cover less than 1.5 m, or change head height (moving around a chair or table)

Not detectable with these recordings:

- **Tremor-like movement.** I looked for a 3.5-7 Hz component in the head track and in the
  downward optical flow while the body was otherwise still. There is none that stands out. The
  head barely carries limb tremor, the downward video is 15 fps (so 7.5 Hz is its limit), and the
  hands are often out of view. No tremor label is produced.
- **Footsteps from audio.** Audio onsets near head-bob step times are no more frequent than at
  random offsets, and onset rate and loudness are the same whether the person moves or stands.
  Speech and room noise dominate. Audio therefore carries no weight in any score.

## Score and evidence strength

Each candidate gets an **uncalibrated detection score from 0 to 100**. It is not a probability.
A score of 80 does not mean an 80% chance of a symptom.

How it is computed:

1. Each evidence item has a fixed maximum number of points and one measurement. The measurement
   earns a share of the points through a linear ramp between two thresholds. Example: for a
   freezing-like stop, "leg movement in the downward view" has 40 points; image motion at this
   trial's quiet-standing level times 1.4 earns 0, at its walking level it earns all 40.
2. Evidence that could not be measured (no downward footage for the interval) earns 0 but stays in
   the total, so a candidate nobody could see properly cannot score high. The reason is listed as
   a limitation.
3. Penalties are subtracted for ordinary explanations and poor data: turning in place, a change of
   head height, stairs, a "sitting" activity label, missing tracking, and the two measured sources
   disagreeing (head moves, downward view is quiet).
4. Score = 100 x (points earned - penalties) / points possible, limited to 0-100.

Evidence strength:

- **High**: score of 70 or more and both measured sources (head track and downward video) each
  earned at least 40% of their own points
- **Medium**: score of 45 or more
- **Low**: score of 30-44. Below 30 a candidate is not reported.

The quiet-standing and walking levels used in the ramps are measured per trial, because camera
mounting and lighting change the optical-flow scale between sessions.

Every prediction stores its evidence items (text, value, points, maximum, source), the sources
that contributed, the measurements, and its limitations. The page shows all of them.

Weights and thresholds are in `DEFAULT_CONFIG` in `scripts/symptom_detection.py`, together with
the label names, colours and an `enabled` switch per label. A `--config` file can override any of
them. The configuration is hashed into the run ID, so results from different settings never
overwrite each other.

## Review on the page

On the Symptom Annotation page, the Symptoms row gets its own selector next to the Activity one
(the Activity selector is unchanged and still only chooses activity labels):

- **Both**: predictions in a lane at the top of the Symptoms track, manual annotations below
- **Manual**: predictions hidden
- **Automatic**: predictions only

Predictions are outlined, dashed and hatched. Manual annotations are solid. Clicking a prediction
moves both videos to its start and opens a panel with the label, time range, score, evidence
strength, every evidence item with its points, and the limitations. From there:

- **Replay** plays the interval with one second before and after
- **Accept** creates a manual annotation with the suggested label and range
- **Edit** opens the normal editor with the suggestion filled in; saving accepts it with the changes
- **Reject** marks it as incorrect; the prediction stays in the run and is drawn struck through
- **Needs review** leaves it for later
- **Clear decision** undoes a decision (earlier decisions stay in the history)

`[` and `]` step through predictions. Low-evidence predictions can be hidden from the settings
menu, which also shows the detector run in use. Drag across the Symptoms track as before to
annotate anything the detector missed.

Accepting does not make the detector right. It records my judgement of one candidate.

## Storage

All under the symptom store (`runs/` by default, git-ignored), next to the manual annotations and
separate from them:

```
<store>/<trial>/symptom_annotations/
  annotations.json                 manual annotations (unchanged format; see below)
  detection_reviews.json           my decisions on predictions
  detections/
    <run_id>.json                  one detector run, never edited by review
    <run_id>.json.bak              the previous result of the same run ID, if it changed
    cache/                         optical-flow cache
```

- `detections/<run_id>.json` (`pd-insighter/symptom-detections@1`): detector name, version,
  configuration and its hash, the sources used, per-trial reference levels, the label set, and
  the predictions with evidence. The run ID is `motion-baseline-<version>-<config hash>`. A new
  version or configuration writes a new file, so earlier predictions stay available to compare.
  The page shows the newest run and lets me pick another one when there are several.
- `detection_reviews.json` (`pd-insighter/symptom-detection-reviews@1`): per run and prediction,
  the decision, reviewer, time, an optional note, a snapshot of what the detector said (label,
  range, score), what was accepted if it was edited, and the history of earlier decisions.
  Rejections are kept for false-positive analysis. Saves are revision-checked like annotations.
- A manual annotation created by accepting a prediction is an ordinary manual annotation
  (`source: manual`) with one extra field, `origin`, naming the run and prediction and whether it
  was edited. Deleting that annotation later does not touch the prediction or the decision.

To go back to a state without predictions, delete the `detections/` folder and
`detection_reviews.json` for the trial. Manual annotations are not affected.

API (served by `scripts/pd_insighter_server.py`):

| Method | Path | Body / query |
| --- | --- | --- |
| GET | `/api/symptom-detections?trial_id=NAME[&run_id=ID]` | saved runs, one run, review decisions |
| POST | `/api/symptom-detection-reviews` | `{"trial_id", "base_revision", "run_id", "detection_id", "decision", ...}` |

## Evaluation against reviewed annotations

```bash
python scripts/evaluate_symptom_detection.py
```

Until reviewed annotations exist it reports that there is nothing to evaluate. It does not
produce numbers from nothing. Once they exist it reports, per trial and per symptom type:

- event level: true positives, false positives, false negatives, precision, recall, F1, mean IoU,
  and start/end timing errors. Predictions and annotations are paired one to one, best overlap
  first, and a pair needs an IoU of at least `--min-iou` (default 0.1).
- time level, separately: seconds of agreement on a 0.1 s grid, with precision, recall, F1 and IoU
- the same event metrics at score thresholds 0, 10, ... 90
- pooled counts over trials, and the share of scored predictions that were confirmed per score band

Rules that keep the numbers honest:

- Reviewed time is the union of my annotations of any label ("No Target Symptom" marks time I
  looked at and found clear). A prediction in time I have not reviewed is `unscored`, not a false
  positive. A prediction I rejected is a false positive wherever it is.
- "Uncertain" and "Not Observable" annotations are left out.
- Annotations created by accepting a prediction are not independent: the detector cannot miss
  them. They are counted, and every event metric is also given for independent annotations only.
- Annotation labels match a predicted type by ID or name; `--map prediction=annotation` adds others.

Calibrating the score needs enough independently reviewed predictions in each score band. The
per-band confirmed share is the starting point for that; until then the score stays uncalibrated.

## Limitations

- The thresholds were set by looking at signals and sample footage from three sessions of
  simulated symptoms. Those sessions cannot also serve as an independent test of the thresholds.
- Much of the walking in those sessions is very slow, so "slowed gait" covers a lot of it. That
  is what the reference value says, not a judgement about cause. Small rooms and furniture slow
  anyone down.
- Head sway while standing can be talking, reaching or looking around. Without downward footage a
  freezing-like candidate stays at medium strength at most.
- Turning in place and a real turning hesitation look alike in these signals. Turns lower the
  score instead of removing the candidate.
- Sitting is not measured directly. The travel-extent check removes most seated movement; the
  activity label is used only as a penalty because it is itself an estimate.
- The downward camera is worn on the body, so its image also moves with the trunk.

## Data that would improve it

The detector uses what the processed trial package contains. The raw recordings hold more:

| Data | Where | What it would add |
| --- | --- | --- |
| MPS closed-loop trajectory at 1 kHz with device velocity, angular velocity and a per-sample quality score | `mps_<recording>_vrs/slam/closed_loop_trajectory.csv` in the raw package | sharper step timing, a per-sample tracking-quality weight, and head acceleration up to the tremor band instead of 30 Hz poses |
| Aria IMU streams (1 kHz and 800 Hz accelerometer and gyroscope) | the VRS file | heel-strike impacts for real step detection, and the freezing index (3-8 Hz versus 0.5-3 Hz power) used in the freezing-of-gait literature; needs `projectaria_tools` |
| Barometer | the VRS file | independent stair and sit/stand evidence |
| 7-channel 48 kHz audio | the VRS file or the extracted audio track | possibly separable footsteps through beamforming toward the floor; uncertain |
| Original 15 fps downward recording | raw package | less compression than the re-encoded session video; small gain |
| MPS hand tracking | not computed for these recordings | hand tremor and arm swing; would need a new MPS request |

The first two would help most. None is required to run the baseline.

## Files

- `scripts/symptom_detection_signals.py`: loads the sources and computes the features
- `scripts/symptom_detection.py`: the detectors, scoring, taxonomy and configuration
- `scripts/run_symptom_detection.py`: the command above
- `scripts/symptom_detection_store.py`: storage of runs and review decisions
- `scripts/evaluate_symptom_detection.py`: evaluation
- `tests/test_symptom_detection.py`: detection on synthetic motion, time mapping, storage, API, evaluation
- `tests/e2e/symptom_predictions.spec.mjs`: the review workflow in the browser
