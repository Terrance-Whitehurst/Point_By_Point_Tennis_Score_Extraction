# Spec: Point-by-Point CSV Extraction

**Status:** Draft  
**Last Updated:** 2026-04-26  
**Owner:** Planning Lead  

---

## 1. Goal / Non-Goals

### Goal

Transform the existing scoreboard OCR pipeline (one row per *scoreboard change event*) into a **row-per-point CSV** that matches the format established in `test_vids/1/Clip1.csv`. Each row represents one complete rally — from serve to point conclusion — with accurate timestamps and the score *as displayed during that rally*.

**Success looks like:** Running the pipeline on Clip1 produces a CSV where Game_Score and Set_Score match `Clip1.csv` 100%, and Point_Start/Point_End are within ±3 seconds of ground truth.

### Non-Goals (v1)

- **No VLM fine-tuning.** Score reading is not the bottleneck. Fine-tuning deferred to M5 (contingent only).
- **No sub-second timestamp accuracy** for Point_Start or Point_End. ±3s tolerance is acceptable in v1.
- **No serve-motion or pose detection.** Point_Start is approximated via fixed offset; pose detection is future work.
- **No doubles support.** Singles only.
- **No real-time processing.** Post-processing batch mode only.

---

## 2. Target Schema

### Columns (must match exactly)

| Column | Type | Format | Notes |
|---|---|---|---|
| `Point_Start` | string | `MM:SS` | Timestamp when serve motion begins (approximated in v1) |
| `Point_End` | string | `MM:SS` or placeholder | Timestamp when rally physically ends |
| `Server` | string | Player last name | Who is serving this point |
| `Returner` | string | Player last name | Who is returning this point |
| `Game_Score` | string | See below | Score displayed *during* this point (before outcome) |
| `Set_Score` | string | `X-Y` | Set score at the time of this point (server sets first) |

### Timestamp format

`MM:SS` — zero-padded. Examples: `00:00`, `02:19`, `01:05`.  
All timestamps are relative to the **start of the video clip**, not the match.

### Game_Score formats by situation

| Situation | Format | Example |
|---|---|---|
| Normal play | `PP-PP` where PP ∈ {0, 15, 30, 40} | `15-30` |
| Deuce | `40-40` | `40-40` |
| Advantage server | `AD-40` | `AD-40` |
| Advantage returner | `40-AD` | `40-AD` |
| Tiebreak | Integer scores `X-Y` | `6-5`, `0-0` |

### Set_Score format

`X-Y` where X = server's sets won, Y = returner's sets won. Server is always listed first in both Game_Score and Set_Score.

### Edge Cases

| Case | Handling |
|---|---|
| **End-of-clip incomplete point** | `Point_End` = last available timestamp + `"(Placeholder — clip ends)"`. Match convention from `Clip1.csv`. |
| **Game boundary point** | `Game_Score` = pre-transition score (e.g., `40-0`). Next row reflects new game with `0-0`. Set_Score updates on the row *after* the game-winning point. |
| **Set completion** | Same as game boundary. Set_Score on the game-winning row is still the pre-set score; the new set score appears starting on the next point row. |
| **Empty trailing row** | Do not emit. `Clip1.csv` has a blank trailing row — strip it on output. |

---

## 3. Pipeline Changes

### Current pipeline output (event-stream)

```
frame_num, timestamp_sec, player1_name, player2_name,
set_score_p1, set_score_p2, game_score_p1, game_score_p2,
point_score_p1, point_score_p2, server, returner
```

One row fires every time the scoreboard changes (SSIM drop → settle → OCR). This is the **input** to the new post-processing stage.

### New post-processing stage: `src/inference/point_builder.py`

Add a standalone module (not modifying the core pipeline) that consumes the event-stream CSV and produces the point-by-point CSV.

#### Pairing logic

For a sorted event stream `E[0], E[1], ..., E[N]`, each point row is derived from consecutive pairs:

```
Point row i:
  Game_Score  = format(E[i].game_score_p1, E[i].game_score_p2)
  Set_Score   = format(E[i].set_score_p1, E[i].set_score_p2, server=E[i].server)
  Server      = E[i].server
  Returner    = E[i].returner
  Point_Start = E[i].timestamp_sec + START_OFFSET   → format as MM:SS
  Point_End   = E[i+1].timestamp_sec - LAG_OFFSET   → format as MM:SS
               (clamped: Point_End >= Point_Start)
```

For the **last event** `E[N]` (no `E[N+1]`):
```
  Point_End = E[N].timestamp_sec + "(Placeholder — clip ends)"
```

#### First row handling

- `Point_Start` for the very first point: `00:00` (beginning of clip).
- `Game_Score` / `Set_Score`: values from `E[0]` (first OCR reading).
- If the clip starts mid-rally (score is already non-zero), this is still valid — the row represents that ongoing point.

#### Server-flip detection at game boundaries

When the game score resets to 0-0 (detected as: `E[i+1].game_score_p1 == 0` and `E[i+1].game_score_p2 == 0` after a non-zero game score), validate that server/returner swap. Cross-check against the OCR-read server field. If OCR disagrees with the expected flip, **trust the state machine** and flag the discrepancy in a `warnings` log column (or stderr).

#### Set_Score orientation

Always render Set_Score as `{server_sets}-{returner_sets}`. Derive from `set_score_p1`/`set_score_p2` using the `server` field to know which player is p1.

---

## 4. Broadcaster-Lag Offset Model

### The problem

Broadcasters update the on-screen scoreboard **1–3 seconds after** the physical point ends. So `E[i+1].timestamp_sec` (when the new score appears) is later than `Point_End` (when the ball actually went out/net).

### Formula

```
Point_End = E[i+1].timestamp_sec - LAG_OFFSET
```

Clamp: `Point_End = max(Point_End, Point_Start + 1)` to prevent negative-duration points.

### Calibration procedure

1. Take each of the 3 corrected matches (`configs/corrected_test_label/match1..3`).
2. Identify the timestamp in the event-stream when each score transition fires.
3. Compare against a manually-annotated ground-truth point-end time (if available) or use `Clip1.csv` as the one reference we have.
4. Compute: `lag_i = E[i+1].timestamp_sec - ground_truth_point_end_i` for each point.
5. `LAG_OFFSET = mean(lag_i)` across all measured points on that broadcaster's feed.

### Default (before calibration)

`LAG_OFFSET = 2.0` seconds.

### Storage

Store as a config value. Two options (use whichever is simplest to implement first):
- CLI argument: `--broadcaster-lag 2.0`
- Config file: `configs/broadcaster_offsets.json` with keys per broadcaster name (e.g., `"ESPN": 1.8`, `"Tennis Channel": 2.3`, `"default": 2.0`)

`START_OFFSET` (time from score appearance to next serve): default `3.0` seconds. Also configurable.

---

## 5. Point_Start Derivation

Ranked by complexity — implement (a) for v1, consider (b)/(c) only after M4 validation.

### (a) v1: Fixed offset from prior score appearance ✅

```
Point_Start = E[i].timestamp_sec + START_OFFSET
```

`START_OFFSET ≈ 3–5s` — the time between the scoreboard stabilizing after the previous point and the next serve beginning (replay, camera cut, player preparation).

Calibrate `START_OFFSET` on corrected matches the same way as `LAG_OFFSET`. Default: `3.0s`.

**Exception — first point of clip:**
```
Point_Start = "00:00"
```

### (b) Future: Audio/motion cues

Detect crowd noise drop, umpire call ("Quiet please"), or rapid motion onset in the video frame near the baseline. Out of scope for v1.

### (c) Future: Serve pose detection

Use a pose estimation model to detect ball toss keypoint. Provides ±0.5s accuracy. Out of scope for v1.

---

## 6. State-Machine Validation Layer

The existing `validate_transition()` in `scoreboard_ocr.py` has basic checks. Enhance it (or add a separate validator in `point_builder.py`) with the following rules.

### Legal point score progressions

**Normal game:**
```
(0,0) → (15,0) | (0,15)
(15,0) → (30,0) | (15,15)
(0,15) → (15,15) | (0,30)
... [standard 0/15/30/40/game ladder]
(40,40) → (AD,40) | (40,AD)   # deuce
(AD,40) → (game,_) | (40,40)  # ad server wins or back to deuce
(40,AD) → (_,game) | (40,40)  # ad returner wins or back to deuce
```

**Tiebreak:**
Points increment by 1 from any integer ≥ 0. Server alternates every 2 points (after the first point). First to 7+ with 2-point lead wins.

### Legal game score progressions

- Increment by exactly 1 for the winner.
- At game boundary: both game scores reset to 0; set score increments by 1 for winner.
- Tiebreak is played at 6-6 in a set.

### Recovery strategy (on invalid transition)

1. **Skip** the bad reading — treat `E[i]` as a duplicate of `E[i-1]` and discard it.
2. If **2+ consecutive invalid readings**: emit a warning, mark those rows with `INVALID_TRANSITION` flag in an optional `_debug` column, and re-anchor to the next reading that produces a valid state.
3. If **3+ identical consecutive readings**: flag as OCR drift (scoreboard may be stuck or SSIM threshold too low). Emit a `DRIFT_DETECTED` warning.

### OCR drift detection

```python
if E[i] == E[i-1] == E[i-2]:
    emit_warning("DRIFT_DETECTED at frame {E[i].frame_num}")
```

Drift means the change-gate is firing on non-changes, or OCR is producing the same hallucinated output repeatedly. Do not emit duplicate point rows for drift events.

---

## 7. Data & Labels

### What we have now

| Asset | Path | Format | Notes |
|---|---|---|---|
| 3 corrected event-stream matches | `configs/corrected_test_label/match1..3/` | Event-stream CSV | Ground truth for OCR accuracy |
| 1 reference point-by-point CSV | `test_vids/1/Clip1.csv` | Point-by-point CSV | 6-point clip, Alcaraz vs De Minaur, 0-1 → 1-1 sets |
| Test video | `test_vids/1/` | Video | Matches `Clip1.csv` |

### What would unlock better validation

1. **Convert the 3 corrected matches to point-by-point format.** Manually or semi-manually annotate `Point_Start`/`Point_End` for each point in those matches. Store at `test_vids/<match_id>/points.csv` using the same schema as `Clip1.csv`.
2. **More clips from different broadcasters.** Each new broadcaster feed may have different lag offsets and scoreboard layouts. Even 1 additional broadcaster would let us test offset calibration generalization.

### Label storage convention

```
test_vids/
  1/
    Clip1.csv          # existing reference
  2/
    points.csv         # future: match2 ground truth
  3/
    points.csv         # future: match3 ground truth
```

---

## 8. Milestones

### M1 — Schema Converter *(start here)*

**What:** Post-processing script `src/inference/point_builder.py` that takes an event-stream CSV as input and outputs a point-by-point CSV.

**How:** Implement the pairing logic from §3. Use fixed default offsets (`LAG_OFFSET=2.0`, `START_OFFSET=3.0`). No calibration yet.

**Acceptance criteria:**
- Output CSV has exactly the 6 columns: `Point_Start, Point_End, Server, Returner, Game_Score, Set_Score`
- Output has the correct number of rows (one per score transition, plus one for the final placeholder)
- Column names match `Clip1.csv` exactly
- Can be run as: `python -m src.inference.point_builder --input reports/match.csv --output reports/match_points.csv`

---

### M2 — Lag Calibration

**What:** Measure broadcaster lag on `Clip1.csv` and the 3 corrected matches. Derive calibrated offsets.

**How:** Compare `Point_End` timestamps from M1 output (using default 2s offset) against `Clip1.csv` ground truth. Compute per-point lag. Update `configs/broadcaster_offsets.json`.

**Acceptance criteria:**
- `Point_End` timestamps within ±2s of ground truth on `Clip1.csv`
- `LAG_OFFSET` and `START_OFFSET` values documented in `configs/broadcaster_offsets.json`

---

### M3 — State-Machine Hardening

**What:** Enhance `validate_transition()` and/or add validation layer in `point_builder.py` with recovery logic, tiebreak support, and drift detection.

**How:** Implement rules from §6. Add `--debug` flag to output a `_debug` column with `OK`, `INVALID_TRANSITION`, or `DRIFT_DETECTED`.

**Acceptance criteria:**
- No illegal score sequences in output on all 3 corrected matches
- Recovery from single-frame OCR errors without dropping valid surrounding points
- Tiebreak points format correctly as integers (e.g., `6-5`)
- Drift detection logs a warning when 3+ identical consecutive events appear

---

### M4 — End-to-End Validation

**What:** Run the full pipeline on the Clip1 video and compare output to `test_vids/1/Clip1.csv`.

**How:** `python -m src.inference.scoreboard_ocr --video test_vids/1/<clip>.mp4 | python -m src.inference.point_builder` → diff against `Clip1.csv`.

**Acceptance criteria:**
- `Game_Score` matches `Clip1.csv` on 100% of rows
- `Set_Score` matches `Clip1.csv` on 100% of rows
- `Server` / `Returner` match on 100% of rows
- `Point_Start` within ±3s of ground truth on all rows
- `Point_End` within ±3s of ground truth on all rows (excluding placeholder row)

---

### M5 — Modal VLM Fine-Tune *(contingent — do not start unless trigger criteria met)*

**What:** Fine-tune a small VLM (Qwen2-VL 2B or PaliGemma) on Modal using scoreboard crops + ground truth labels.

**Trigger:** See §10. Only execute if score-read errors remain high after M3.

---

## 9. Open Questions

These are unresolved ambiguities. Flag to user for decisions before M4.

| # | Question | Impact |
|---|---|---|
| OQ-1 | **What exactly counts as `Point_Start`?** First serve motion? Ball toss? Umpire "play"? The v1 offset is an approximation — we need a ground-truth definition for proper validation. | Affects M4 acceptance criteria |
| OQ-2 | **How to handle replays and challenges?** During hawk-eye or replay, the scoreboard is static — our pipeline naturally ignores these (no change event fires). But if a challenge reverses a point, the score sequence may look like an illegal backward transition. | Affects §6 recovery logic |
| OQ-3 | **Placeholder convention for incomplete points.** `Clip1.csv` uses `02:20(Placeholder because we arent testing on full video just a clip rn)`. Should we standardize this to a fixed string like `"END_OF_CLIP"` for machine readability, or preserve the human-readable note? | Affects M1 output format |
| OQ-4 | **Multi-set clips.** The set_score column handles this structurally, but Set_Score orientation (server's sets first) may need to track server changes across sets. | Affects §3 Set_Score orientation |
| OQ-5 | **Doubles format.** Not in scope for v1, but does the scoreboard OCR even attempt to read doubles server indicators? Flag for future spec. | Future scope |
| OQ-6 | **`Clip1.csv` has a blank trailing row.** Should we strip it on output, or preserve it for compatibility? | Affects M1 output |

---

## 10. Fine-Tuning Trigger Criteria

After completing M3, measure the following metrics on the validation set (corrected matches + `Clip1.csv`). **Only proceed to M5 if any trigger condition is met.**

| Metric | Trigger Threshold | Notes |
|---|---|---|
| Score-read field error rate | > 5% of fields incorrect | Covers game_score, set_score, point_score fields combined |
| Server detection accuracy | < 90% | Server indicator is small and easily missed |
| Player name extraction failure | > 10% of readings | OCR hallucination or layout mismatch |
| Systematic error pattern | Same field wrong on same layout | Random errors → fix prompting. Systematic errors → fine-tune |

**Error types that indicate fine-tuning will help:**
- Consistent digit confusion on a specific broadcaster's font (e.g., 1↔7, 4↔9)
- Server dot/indicator missed on specific scoreboard layouts
- Score column confusion (game vs. set column swapped)

**Error types that fine-tuning will NOT fix:**
- Temporal segmentation errors (wrong Point_Start/End) → fix offset calibration
- Illegal score transitions from OCR drift → fix state-machine recovery
- Random single-frame hallucinations → fix SSIM threshold / settling time

If fine-tuning is triggered, the recommended target model is **Qwen2-VL 2B** on **Modal** (A10G), using scoreboard crops from the corrected matches as training data.

---

## Appendix: File Paths Reference

| Path | Purpose |
|---|---|
| `src/inference/scoreboard_ocr.py` | Existing pipeline — event-stream output |
| `src/inference/point_builder.py` | **New** — post-processing stage (M1) |
| `configs/broadcaster_offsets.json` | **New** — lag/start offset config (M2) |
| `configs/corrected_test_label/match1..3/` | Ground truth event-stream CSVs |
| `test_vids/1/Clip1.csv` | Reference point-by-point CSV (6 points) |
| `test_vids/<id>/points.csv` | Future ground-truth point-by-point CSVs |
| `reports/` | Pipeline output CSVs |
