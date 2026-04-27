# Spec: Post-Read Frame Skip Optimization

## Goal

After `scoreboard_ocr.py` commits a score-change row, skip RF-DETR detection on the next ~10 seconds of video. Nothing scoreable happens between Point_End and the next Point_Start (the broadcaster doesn't update faster than that, the players are walking), so running the detector during that window is wasted compute.

## Expected speedup

Measured baseline on `test_vids/1/Clip1.mp4` (3525 frames, 25 fps, MPS):
- Total runtime: ~3 min
- 6 score changes detected
- RF-DETR runs every frame (~50 ms/frame on MPS) — this is the bottleneck

With a 10s post-read skip:
- Frames skipped per match-second-of-action: 6 × 10s × 25fps = **1500 frames (~42% of total)**
- Estimated runtime: ~1:50 min — **~25–30% faster**

Speedup scales with score-change density. Sparser matches benefit less; rally-heavy clips benefit more.

## Files involved

- `src/inference/scoreboard_ocr.py` — all changes live here
- Existing related concept: `--cooldown-frames` (different semantic; do **not** repurpose)

## Read first

Before editing, read these in order:

1. `src/inference/scoreboard_ocr.py:1222–1310` — `process_video` main loop
2. `src/inference/scoreboard_ocr.py:1313–1410` — `_handle_ocr_read` (where rows get appended to `self._results`)
3. `src/inference/scoreboard_ocr.py:849–926` — `ChangeDetector` state machine (so you understand why cooldown is *not* what we want)

The existing `--cooldown-frames` (default 30 = 1.2s) suppresses *change-detection triggers* during cooldown but still runs RF-DETR on every frame. We are adding a separate, longer "hard skip" window that bypasses detection entirely.

## Implementation

### 1. New CLI flag

Add a new flag in the argparse section (around `src/inference/scoreboard_ocr.py:1602` next to `--cooldown-frames`):

```python
parser.add_argument(
    "--post-read-skip-seconds",
    type=float,
    default=10.0,
    help=(
        "Seconds to skip after a successful score-change read. During this "
        "window, RF-DETR detection is bypassed (only frame counter advances). "
        "Set to 0 to disable. (default: 10.0)"
    ),
)
```

Thread it through `ScoreboardOCRPipeline.__init__` (around line 1167) and store as `self.post_read_skip_seconds: float`.

### 2. Skip-window state in `process_video`

Modify the main loop at `src/inference/scoreboard_ocr.py:1258`:

```python
skip_until_frame: int = -1  # frame_idx at which detection resumes

while True:
    if frame_idx < skip_until_frame:
        # Hard-skip window: demux only, no decode, no detection.
        if not cap.grab():
            break
        frame_idx += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    # ── Stage 1: detect + crop ────────────────────────────────
    crop = self.detector.detect_and_crop(frame)
    # ... (existing code unchanged through _handle_ocr_read call)

    # ── Stage 2: VLM OCR (only when change detected) ─────────
    rows_before = len(self._results)
    if settled_crop is not None:
        vlm_calls += 1
        self._handle_ocr_read(settled_crop, frame_idx, fps)

    # If a real row was committed, arm the skip window.
    if len(self._results) > rows_before and self.post_read_skip_seconds > 0:
        skip_until_frame = frame_idx + int(self.post_read_skip_seconds * fps)
        logger.debug(
            "Post-read skip armed: frames %d → %d (%.1fs)",
            frame_idx,
            skip_until_frame,
            self.post_read_skip_seconds,
        )

    frame_idx += 1
    # ... (existing progress logging unchanged)
```

### Why `cap.grab()` instead of `cap.read()`

- `cap.read()` = demux + decode + return frame array
- `cap.grab()` = demux only; no decode, no allocation
- Inside the skip window we never use the frame, so `grab()` is ~5× cheaper and adds to the speedup
- `frame_idx` still increments correctly because each grab consumes one frame from the demuxer

### 3. Reset skip window when detector resets

If `self.detector.reset()` is called (line 1273, after prolonged absence of detections), do NOT reset `skip_until_frame` — the skip is independent of detector smoothing state. Leave it.

## Edge cases to handle

1. **Skip window crosses end of video** — `cap.grab()` returns False at EOF; the existing `break` path handles it.
2. **`post_read_skip_seconds == 0`** — disables the optimization (the `> 0` guard above ensures `skip_until_frame` is never armed). Should run identically to today's behavior.
3. **First score change** — works correctly because `skip_until_frame` starts at -1 (always less than `frame_idx`).
4. **Score change committed at `frame_idx == N - 1`** (last frame) — skip arms but loop will exit on next `cap.grab()`. No special handling needed.
5. **`frame_idx` accuracy for output CSV** — must remain correct because `_handle_ocr_read` writes `timestamp_sec=round(frame_idx / fps, 2)`. Both `cap.read()` and `cap.grab()` advance the demuxer by exactly one frame, so `frame_idx += 1` stays aligned.

## Do NOT

- Do **not** use `cap.set(cv2.CAP_PROP_POS_FRAMES, target)` to seek. On H.264 streams this seeks to the nearest keyframe and silently desyncs `frame_idx` from the actual position. Use the `cap.grab()` loop above.
- Do **not** change `--cooldown-frames` semantics or default. It serves a different purpose (suppressing repeat SSIM triggers during scoreboard animation tail) and downstream code may depend on the current behavior.
- Do **not** import anything from `point_builder.py` to share the constant. Keep `scoreboard_ocr.py` self-contained.

## Verification

After implementing, run both with and without the optimization on the existing test clip and verify:

```bash
# Disabled (baseline)
time uv run python -m src.inference.scoreboard_ocr \
    --video test_vids/1/Clip1.mp4 \
    --output-csv reports/clip1_no_skip.csv \
    --ocr-backend easyocr \
    --device mps \
    --post-read-skip-seconds 0

# Enabled (default 10s)
time uv run python -m src.inference.scoreboard_ocr \
    --video test_vids/1/Clip1.mp4 \
    --output-csv reports/clip1_skip10.csv \
    --ocr-backend easyocr \
    --device mps

# Diff the outputs — they MUST be identical
diff reports/clip1_no_skip.csv reports/clip1_skip10.csv
```

## Acceptance criteria

- [ ] `--post-read-skip-seconds` flag exists and threads through to the main loop
- [ ] Default value is `10.0`
- [ ] `--post-read-skip-seconds 0` produces byte-identical output to current `main` branch on `test_vids/1/Clip1.mp4`
- [ ] Default run produces byte-identical output to `--post-read-skip-seconds 0` on `test_vids/1/Clip1.mp4` (because no two real score changes happen within 10s of each other in this clip)
- [ ] Default run is measurably faster (target: ≥20% wall-time reduction on this clip)
- [ ] Frame counter (`frame_idx` in CSV `frame_num` column) matches the no-skip output exactly for every committed row
- [ ] `pytest tests/ -v` still passes (no new tests required, but nothing should break)
