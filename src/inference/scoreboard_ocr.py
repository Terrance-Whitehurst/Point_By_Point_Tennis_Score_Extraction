"""
Tennis Scoreboard OCR Pipeline.

Two-stage pipeline for extracting score data from tennis match videos:
  Stage 1: RF-DETR scoreboard detection + SSIM change detection (every frame)
  Stage 2: FastVLM vision-language OCR (only when a change is detected)

Usage:
    # Standalone CLI
    python -m src.inference.scoreboard_ocr --video path/to/video.mp4

    # Programmatic
    from src.inference.scoreboard_ocr import ScoreboardOCRPipeline
    pipeline = ScoreboardOCRPipeline(...)
    results = pipeline.process_video("match.mp4", "scores.csv")

Output CSV columns:
    frame_num, timestamp_sec, player1_name, player2_name,
    set_score_p1, set_score_p2, game_score_p1, game_score_p2,
    point_score_p1, point_score_p2, server, returner
"""

from __future__ import annotations

import argparse
import csv
import enum
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv
import torch

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SCOREBOARD_CLASS_NAMES: list[str] = ["scoreboards"]
CANONICAL_CROP_SIZE: tuple[int, int] = (384, 128)  # (width, height)
VALID_TENNIS_POINTS: set[str] = {"0", "15", "30", "40", "AD", ""}
MAX_GAMES_IN_SET: int = 13  # Tiebreak can reach 7-6
MAX_SETS_IN_MATCH: int = 5


# ── State Machine ─────────────────────────────────────────────────────────────


class _PipelineState(enum.Enum):
    """State machine states for change detection."""

    IDLE = "idle"
    SETTLING = "settling"
    COOLDOWN = "cooldown"


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class ScoreReading:
    """Parsed scoreboard reading from a single frame."""

    player1: str
    player2: str
    sets_p1: int
    sets_p2: int
    games_p1: int
    games_p2: int
    points_p1: str
    points_p2: str
    server: str
    returner: str

    @staticmethod
    def _norm_pt(pt: str) -> str:
        """Normalize point for comparison: treat '' as '0'."""
        return pt if pt else "0"

    def score_changed(self, other: Optional["ScoreReading"]) -> bool:
        """Return True if the score portion differs from *other*."""
        if other is None:
            return True
        return (
            self.sets_p1 != other.sets_p1
            or self.sets_p2 != other.sets_p2
            or self.games_p1 != other.games_p1
            or self.games_p2 != other.games_p2
            or self._norm_pt(self.points_p1) != self._norm_pt(other.points_p1)
            or self._norm_pt(self.points_p2) != self._norm_pt(other.points_p2)
            or self.server != other.server
        )


@dataclass
class CSVRow:
    """One row of the output CSV."""

    frame_num: int
    timestamp_sec: float
    player1_name: str
    player2_name: str
    set_score: str  # e.g. "1-0"
    game_score: str  # e.g. "5-3"
    point_score: str  # e.g. "30-15"
    server: str
    returner: str

    def as_dict(self) -> dict:
        """Convert to dict matching CSV fieldnames."""
        return {
            "frame_num": self.frame_num,
            "timestamp_sec": self.timestamp_sec,
            "player1_name": self.player1_name,
            "player2_name": self.player2_name,
            "set_score": self.set_score,
            "game_score": self.game_score,
            "point_score": self.point_score,
            "server": self.server,
            "returner": self.returner,
        }


CSV_FIELDNAMES: list[str] = list(CSVRow.__dataclass_fields__.keys())


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FastVLM Scoreboard Reader
# ══════════════════════════════════════════════════════════════════════════════


class FastVLMReader:
    """Wrapper around FastVLM for scoreboard OCR.

    Supports two backends:
      1. mlx-vlm  — optimised for Apple Silicon (preferred)
      2. transformers — cross-platform fallback

    The model is loaded once at init and kept in memory for the lifetime of
    the pipeline.
    """

    PROMPT: str = (
        "You are reading a tennis match scoreboard. Extract the following "
        "information from this scoreboard image and return it as a single "
        "JSON object. Do not include any other text.\n\n"
        "Required fields:\n"
        '- "player1": name of the top player (string, uppercase)\n'
        '- "player2": name of the bottom player (string, uppercase)\n'
        '- "sets_p1": sets won by player 1 (integer)\n'
        '- "sets_p2": sets won by player 2 (integer)\n'
        '- "games_p1": games in current set for player 1 (integer)\n'
        '- "games_p2": games in current set for player 2 (integer)\n'
        '- "points_p1": current game points for player 1 '
        '(string: "0","15","30","40","AD","")\n'
        '- "points_p2": current game points for player 2 '
        '(string: "0","15","30","40","AD","")\n'
        '- "server": name of the serving player (string, uppercase)\n\n'
        "If a field is not visible or unclear, use null.\n\n"
        "Example output:\n"
        '{"player1":"SINNER","player2":"DJOKOVIC","sets_p1":1,"sets_p2":0,'
        '"games_p1":5,"games_p2":3,"points_p1":"30","points_p2":"15",'
        '"server":"SINNER"}'
    )

    def __init__(self, model_path: str) -> None:
        self.model_path: str = model_path
        self._backend: str = "none"
        self._model = None
        self._processor = None
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Try mlx-vlm first, fall back to transformers."""
        # Attempt 1: mlx-vlm (Apple Silicon optimised)
        try:
            from mlx_vlm import load as mlx_load  # type: ignore[import-untyped]

            self._model, self._processor = mlx_load(self.model_path)
            self._backend = "mlx"
            logger.info("FastVLM loaded via mlx-vlm (%s)", self.model_path)
            return
        except ImportError:
            logger.debug("mlx-vlm not available, trying transformers")
        except Exception as exc:
            logger.warning("mlx-vlm load failed (%s), trying transformers", exc)

        # Attempt 2: HuggingFace transformers
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForVision2Seq,
                AutoProcessor,
            )

            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_path, torch_dtype=torch.float16
            )
            self._backend = "transformers"
            logger.info("FastVLM loaded via transformers (%s)", self.model_path)
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("transformers load failed: %s", exc)

        raise RuntimeError(
            f"Cannot load FastVLM from '{self.model_path}'. "
            "Install mlx-vlm (`pip install mlx-vlm`) or "
            "transformers (`pip install transformers`)."
        )

    # ── Inference ──────────────────────────────────────────────────────────

    def read(self, crop_bgr: npt.NDArray[np.uint8]) -> dict | None:
        """Run VLM on a BGR scoreboard crop. Returns parsed dict or None."""
        from PIL import Image

        pil_image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        try:
            raw_response = self._infer(pil_image)
        except Exception as exc:
            logger.warning("FastVLM inference error: %s", exc)
            return None

        return self._parse_response(raw_response)

    def _infer(self, pil_image) -> str:
        """Dispatch to the active backend."""

        if self._backend == "mlx":
            from mlx_vlm import generate as mlx_generate  # type: ignore[import-untyped]

            # mlx-vlm generate() accepts a file path or PIL Image depending
            # on version.  Write to a temp file for maximum compatibility.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp, format="PNG")
                tmp_path = tmp.name
            try:
                result = mlx_generate(
                    self._model,
                    self._processor,
                    self.PROMPT,
                    image=tmp_path,
                    max_tokens=256,
                )
            finally:
                os.unlink(tmp_path)
            # mlx_vlm.generate returns a GenerationResult dataclass
            if hasattr(result, "text"):
                return result.text
            return str(result)

        elif self._backend == "transformers":
            inputs = self._processor(
                images=pil_image,
                text=self.PROMPT,
                return_tensors="pt",
            )
            # Move inputs to same device as model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = self._model.generate(**inputs, max_new_tokens=256)
            return self._processor.decode(output_ids[0], skip_special_tokens=True)

        raise RuntimeError(f"Unknown VLM backend: {self._backend}")

    # ── Response parsing ───────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response: str) -> dict | None:
        """Extract and validate JSON from the VLM response string."""
        # Find the first JSON object in the response
        json_match = re.search(r"\{[^{}]*\}", response)
        if json_match is None:
            logger.warning("No JSON object in VLM response: %.200s", response)
            return None

        try:
            data: dict = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error (%s) in response: %.200s", exc, response)
            return None

        # Verify all required keys are present
        required_keys = {
            "player1",
            "player2",
            "sets_p1",
            "sets_p2",
            "games_p1",
            "games_p2",
            "points_p1",
            "points_p2",
            "server",
        }
        missing = required_keys - data.keys()
        if missing:
            logger.warning("VLM response missing keys: %s", missing)
            return None

        return data


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 (alt) — EasyOCR Scoreboard Reader
# ══════════════════════════════════════════════════════════════════════════════


class EasyOCRReader:
    """Scoreboard OCR using EasyOCR with spatial parsing.

    EasyOCR returns word bounding boxes with text and confidence.  Words are
    split into top/bottom rows (one per player), then assigned to columns
    (name, games, points) by x-position.

    Server detection uses the green dot, which EasyOCR may or may not pick up.
    As a fallback, the server dot is detected via colour analysis on the left
    side of each row.
    """

    def __init__(self) -> None:
        import easyocr  # type: ignore[import-untyped]

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader initialised")

    def read(self, crop_bgr: npt.NDArray[np.uint8]) -> dict | None:
        """Run EasyOCR on a BGR scoreboard crop. Returns parsed dict or None."""
        # Preprocess: grayscale + 2x upscale + CLAHE contrast enhancement
        # to reliably detect white text on blue backgrounds
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray_large = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray_large)

        results = self._reader.readtext(enhanced)
        if len(results) < 3:
            logger.warning("EasyOCR: too few detections (%d)", len(results))
            return None
        # Pass original crop for server dot colour detection (needs BGR)
        return self._parse_results(results, crop_bgr)

    def _parse_results(
        self, results: list, crop_bgr: npt.NDArray[np.uint8]
    ) -> dict | None:
        """Parse EasyOCR results into structured score data."""
        # Use bounding boxes from results to determine image height
        all_ys = [bbox[2][1] for bbox, _, _ in results]
        max_y = max(all_ys) if all_ys else crop_bgr.shape[0]
        min_y = min(bbox[0][1] for bbox, _, _ in results) if results else 0
        mid_y = (min_y + max_y) / 2

        top_items: list[tuple[float, str, float]] = []
        bot_items: list[tuple[float, str, float]] = []

        for bbox, text, conf in results:
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            text = text.strip()
            if not text:
                continue
            if cy < mid_y:
                top_items.append((cx, text, conf))
            else:
                bot_items.append((cx, text, conf))

        if not top_items or not bot_items:
            logger.warning("EasyOCR: could not split into two rows")
            return None

        row1 = self._parse_row(sorted(top_items, key=lambda x: x[0]))
        row2 = self._parse_row(sorted(bot_items, key=lambda x: x[0]))
        if row1 is None or row2 is None:
            logger.warning("EasyOCR: failed to parse one or both rows")
            return None

        name1, numbers1 = row1
        name2, numbers2 = row2

        # Detect server via green dot colour on left side of each row
        server = self._detect_server_by_colour(crop_bgr, name1, name2)

        games1, sets1, points1 = self._interpret_numbers(numbers1)
        games2, sets2, points2 = self._interpret_numbers(numbers2)

        return {
            "player1": name1.upper(),
            "player2": name2.upper(),
            "sets_p1": sets1,
            "sets_p2": sets2,
            "games_p1": games1,
            "games_p2": games2,
            "points_p1": points1,
            "points_p2": points2,
            "server": server.upper() if server else None,
        }

    @staticmethod
    def _parse_row(
        items: list[tuple[float, str, float]],
    ) -> tuple[str, list[str]] | None:
        """Parse sorted (cx, text, conf) items from one row.

        Returns (player_name, [number_strings]) or None.
        """
        name_parts: list[str] = []
        numbers: list[str] = []

        for _cx, text, _conf in items:
            cleaned = text.strip("|.,;:")
            if not cleaned:
                continue
            if cleaned.isdigit():
                numbers.append(cleaned)
            elif cleaned.upper() == "AD":
                numbers.append("AD")
            elif (
                cleaned.replace("-", "")
                .replace("'", "")
                .replace(".", "")
                .replace(" ", "")
                .isalpha()
            ):
                if len(cleaned) >= 2:
                    name_parts.append(cleaned)

        if not name_parts or not numbers:
            return None
        return " ".join(name_parts), numbers

    @staticmethod
    def _detect_server_by_colour(
        crop_bgr: npt.NDArray[np.uint8], name1: str, name2: str
    ) -> str | None:
        """Detect the server indicator (green/yellow dot) by colour.

        Checks the left ~12% of the top and bottom halves for green-ish pixels.
        """
        h, w = crop_bgr.shape[:2]
        left_strip = max(1, int(w * 0.20))
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # Bright green/yellow dot: H 25-85, S > 100, V > 150
        lower = np.array([25, 100, 150])
        upper = np.array([85, 255, 255])

        top_region = hsv[: h // 2, :left_strip]
        bot_region = hsv[h // 2 :, :left_strip]

        top_green = int(cv2.inRange(top_region, lower, upper).sum()) // 255
        bot_green = int(cv2.inRange(bot_region, lower, upper).sum()) // 255

        threshold = 10  # minimum green pixels to count as a dot
        top_has = top_green > threshold
        bot_has = bot_green > threshold

        if top_has and not bot_has:
            return name1
        elif bot_has and not top_has:
            return name2
        return None

    @staticmethod
    def _interpret_numbers(nums: list[str]) -> tuple[int, int, str]:
        """Interpret number strings as (games, sets_won, point_score)."""
        _VALID_POINTS = {"0", "15", "30", "40", "AD"}

        def _to_int(s: str, default: int = 0) -> int:
            try:
                return int(s)
            except ValueError:
                return default

        if len(nums) >= 3:
            # GAMES | SETS_WON | POINTS
            games = _to_int(nums[0])
            sets_won = _to_int(nums[1])
            pt = nums[2] if nums[2] in _VALID_POINTS else ""
            return games, sets_won, pt
        elif len(nums) == 2:
            games = _to_int(nums[0])
            second = nums[1]
            if second in _VALID_POINTS:
                return games, 0, second
            return games, _to_int(second), ""
        elif len(nums) == 1:
            val = nums[0]
            # Single number: if it's a standard point (15/30/40) it's likely
            # the point score with the games column missed by OCR.
            if val in ("15", "30", "40", "AD"):
                return 0, 0, val
            return _to_int(val), 0, ""
        return 0, 0, ""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 (alt) — Tesseract OCR Scoreboard Reader
# ══════════════════════════════════════════════════════════════════════════════


class TesseractReader:
    """Scoreboard OCR using Tesseract with spatial word-position parsing.

    Uses ``image_to_data`` to get word bounding boxes, then assigns each
    word to a column (name / games / points) based on its x-position.
    The server indicator (dot/bullet) is detected by its small size and
    position to the left of the name.

    Expected scoreboard layout (two rows):
        [dot?] PLAYER_NAME  |  GAMES  |  POINTS
    """

    # Characters Tesseract might produce for the server dot
    _DOT_CHARS: set[str] = {"*", "°", "·", "•", "®", "©", "e", "."}

    def __init__(self) -> None:
        import pytesseract  # noqa: F401 — verify available

        self._tesseract = pytesseract

    def read(self, crop_bgr: npt.NDArray[np.uint8]) -> dict | None:
        """Run Tesseract on a BGR scoreboard crop. Returns parsed dict or None."""
        from PIL import Image

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 3
        gray_large = cv2.resize(
            gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
        )

        pil_img = Image.fromarray(gray_large)
        data = self._tesseract.image_to_data(
            pil_img, config="--psm 6", output_type=self._tesseract.Output.DICT
        )
        return self._parse_spatial(data, w * scale, h * scale)

    def _parse_spatial(self, data: dict, img_w: int, img_h: int) -> dict | None:
        """Parse Tesseract word boxes into structured score data."""
        # Collect non-empty words with positions
        words: list[dict] = []
        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if not txt:
                continue
            words.append(
                {
                    "text": txt,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": int(data["conf"][i]),
                    "cx": data["left"][i] + data["width"][i] / 2,
                }
            )

        if len(words) < 3:
            logger.warning("Tesseract: too few words detected (%d)", len(words))
            return None

        # Split words into top row and bottom row by y-center
        mid_y = img_h / 2
        top_words = [w for w in words if (w["y"] + w["h"] / 2) < mid_y]
        bot_words = [w for w in words if (w["y"] + w["h"] / 2) >= mid_y]

        if not top_words or not bot_words:
            logger.warning("Tesseract: could not split into two rows")
            return None

        row1 = self._parse_row(top_words)
        row2 = self._parse_row(bot_words)
        if row1 is None or row2 is None:
            logger.warning("Tesseract: failed to parse one or both rows")
            return None

        name1, has_dot1, nums1 = row1
        name2, has_dot2, nums2 = row2

        if has_dot1 and not has_dot2:
            server = name1
        elif has_dot2 and not has_dot1:
            server = name2
        else:
            server = None

        games1, sets1, points1 = self._interpret_numbers(nums1)
        games2, sets2, points2 = self._interpret_numbers(nums2)

        return {
            "player1": name1.upper(),
            "player2": name2.upper(),
            "sets_p1": sets1,
            "sets_p2": sets2,
            "games_p1": games1,
            "games_p2": games2,
            "points_p1": points1,
            "points_p2": points2,
            "server": server.upper() if server else None,
        }

    def _parse_row(self, words: list[dict]) -> tuple[str, bool, list[str]] | None:
        """Parse words from one scoreboard row.

        Returns (player_name, has_server_dot, [number_strings]) or None.
        """
        # Sort by x position
        words = sorted(words, key=lambda w: w["x"])

        has_dot = False
        name_parts: list[str] = []
        number_strs: list[str] = []

        for w in words:
            txt = w["text"].strip("|.,;:")
            if not txt:
                continue

            # Detect server dot: small width, single char
            if txt in self._DOT_CHARS and w["w"] < 60 and len(txt) <= 1:
                has_dot = True
                continue

            # Skip very low-confidence short garbage
            if w["conf"] < 30 and len(txt) <= 3:
                continue

            # Normalize common OCR misreads of '0'
            normalized = self._fix_zero(txt)

            if normalized.isdigit():
                number_strs.append(normalized)
            elif normalized.upper() == "AD":
                number_strs.append("AD")
            elif (
                normalized.replace("-", "").replace("'", "").replace(".", "").isalpha()
            ):
                # Only accept as name if it looks like a real word (>= 2 chars, has uppercase)
                if len(normalized) >= 2 and any(c.isupper() for c in normalized):
                    name_parts.append(normalized)
                elif len(normalized) == 1 and w["conf"] >= 80:
                    name_parts.append(normalized)
            # else: skip garbage/noise

        if not name_parts or not number_strs:
            return None

        # Clean name: keep only the longest word(s) that look like real names
        # Filter out stray 1-2 char OCR artifacts ("A", "QO", etc.)
        clean_parts = [p for p in name_parts if len(p) >= 3]
        if not clean_parts:
            clean_parts = name_parts  # fallback to originals
        name = " ".join(clean_parts)
        return name, has_dot, number_strs

    @staticmethod
    def _fix_zero(txt: str) -> str:
        """Fix Tesseract misreads of '0' as 'O', 'o', 'ko', etc."""
        upper = txt.upper()
        # Single 'O' → '0'
        if upper == "O":
            return "0"
        # 'OO' → '00' (unlikely but handle)
        if upper == "OO":
            return "00"
        # 'ko', 'Oo' etc when it should be a number
        if len(txt) <= 3 and not any(c.isdigit() for c in txt):
            # Check if it looks like a garbled zero
            if upper in ("KO", "OO", "CO", "IO", "LO", "GO"):
                return "0"
        return txt

    @staticmethod
    def _interpret_numbers(nums: list[str]) -> tuple[int, int, str]:
        """Interpret number strings as (games, sets_won, point_score).

        Scoreboard column layouts observed:
          2 cols: GAMES | POINTS           (during a game)
          2 cols: GAMES | SETS_WON         (between games, no active point)
          3 cols: GAMES | SETS_WON | POINTS
        """
        _VALID_POINTS = {"0", "15", "30", "40", "AD"}

        def _to_int(s: str, default: int = 0) -> int:
            try:
                return int(s)
            except ValueError:
                return default

        if len(nums) >= 3:
            # GAMES | SETS_WON | POINTS
            games = _to_int(nums[0])
            sets_won = _to_int(nums[1])
            pt = nums[2] if nums[2] in _VALID_POINTS else ""
            return games, sets_won, pt
        elif len(nums) == 2:
            games = _to_int(nums[0])
            second = nums[1]
            # If second value is a valid tennis point → it's the point score
            if second in _VALID_POINTS:
                return games, 0, second
            # Otherwise it's likely a set score (small int, no active point)
            return games, _to_int(second), ""
        elif len(nums) == 1:
            return _to_int(nums[0]), 0, ""
        return 0, 0, ""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Scoreboard Detector (RF-DETR + EMA Smoothing)
# ══════════════════════════════════════════════════════════════════════════════


class ScoreboardDetector:
    """RF-DETR scoreboard detector with EMA bounding-box smoothing.

    Loads the fine-tuned RF-DETR checkpoint (trained in
    ``src/training/train_scoreboard_detection.py``), runs per-frame inference,
    smooths the bounding box with an exponential moving average, then crops
    and resizes the scoreboard region to a canonical size for downstream
    comparison.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        ema_alpha: float = 0.3,
        padding: float = 0.1,
        crop_size: tuple[int, int] = CANONICAL_CROP_SIZE,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device: torch.device = device
        self.ema_alpha: float = ema_alpha
        self.padding: float = padding
        self.crop_size: tuple[int, int] = crop_size
        self.confidence_threshold: float = confidence_threshold
        self._smoothed_bbox: npt.NDArray[np.float64] | None = None
        self._model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load RF-DETR with fine-tuned scoreboard detection head."""
        from rfdetr import RFDETRBase  # type: ignore[import-untyped]

        model = RFDETRBase()
        model.model.reinitialize_detection_head(
            num_classes=3
        )  # 2 dataset categories + 1 background
        model.model.class_names = SCOREBOARD_CLASS_NAMES

        ckpt: dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict: dict = ckpt["state_dict"]
        cleaned: dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.model.model.load_state_dict(cleaned)
        model.model.model.to(self.device)
        model.model.device = self.device
        model.model.model.eval()

        logger.info("RF-DETR scoreboard detector loaded from %s", checkpoint_path)
        return model

    def detect_and_crop(
        self, frame: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8] | None:
        """Detect scoreboard → smooth bbox → crop → resize.

        Args:
            frame: BGR frame from cv2.VideoCapture.

        Returns:
            Resized scoreboard crop (BGR, ``self.crop_size``), or ``None`` if
            no scoreboard was detected.
        """
        detections: sv.Detections = self._model.predict(
            frame, threshold=self.confidence_threshold
        )

        if detections is None or len(detections) == 0:
            return None

        # Drop metadata that can cause ByteTrack / indexing issues
        detections.data.pop("source_image", None)
        detections.data.pop("source_shape", None)

        # Pick the best detection: blend of confidence and area
        if len(detections) > 1:
            areas: npt.NDArray[np.float64] = (
                detections.xyxy[:, 2] - detections.xyxy[:, 0]
            ) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            blended = detections.confidence * 0.5 + (areas / areas.max()) * 0.5
            best_idx: int = int(np.argmax(blended))
            bbox: npt.NDArray = detections.xyxy[best_idx]
        else:
            bbox = detections.xyxy[0]

        # EMA smoothing
        if self._smoothed_bbox is None:
            self._smoothed_bbox = bbox.astype(np.float64)
        else:
            self._smoothed_bbox = (
                self.ema_alpha * bbox.astype(np.float64)
                + (1 - self.ema_alpha) * self._smoothed_bbox
            )

        # Add padding
        x1, y1, x2, y2 = self._smoothed_bbox
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * self.padding, h * self.padding

        # Clamp to frame dimensions
        fh, fw = frame.shape[:2]
        x1_p = max(0, int(x1 - pad_x))
        y1_p = max(0, int(y1 - pad_y))
        x2_p = min(fw, int(x2 + pad_x))
        y2_p = min(fh, int(y2 + pad_y))

        crop: npt.NDArray[np.uint8] = frame[y1_p:y2_p, x1_p:x2_p]
        if crop.size == 0:
            return None

        resized: npt.NDArray[np.uint8] = cv2.resize(
            crop, self.crop_size, interpolation=cv2.INTER_AREA
        )
        return resized

    def reset(self) -> None:
        """Reset EMA state (call when scoreboard disappears for a while)."""
        self._smoothed_bbox = None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Change Detector (SSIM + State Machine)
# ══════════════════════════════════════════════════════════════════════════════


class ChangeDetector:
    """SSIM-based scoreboard change detector with settle / cooldown.

    State machine::

        IDLE ──(SSIM drop)──► SETTLING ──(N frames)──► [return crop] ──► COOLDOWN ──(M frames)──► IDLE
                                                                             ▲
                                                          (M frames) ────────┘
    """

    def __init__(
        self,
        ssim_threshold: float = 0.92,
        settle_frames: int = 15,
        cooldown_frames: int = 30,
        no_detection_reset_frames: int = 30,
    ) -> None:
        self.ssim_threshold: float = ssim_threshold
        self.settle_frames: int = settle_frames
        self.cooldown_frames: int = cooldown_frames
        self.no_detection_reset_frames: int = no_detection_reset_frames

        self._state: _PipelineState = _PipelineState.IDLE
        self._prev_crop: npt.NDArray[np.uint8] | None = None
        self._counter: int = 0
        self.frames_without_detection: int = 0

    def update(
        self, crop: npt.NDArray[np.uint8] | None
    ) -> npt.NDArray[np.uint8] | None:
        """Feed a crop (or ``None``).  Returns the settled crop when ready."""
        # ── No scoreboard in frame ────────────────────────────────────────
        if crop is None:
            self.frames_without_detection += 1
            if self.frames_without_detection >= self.no_detection_reset_frames:
                self._reset()
            return None

        self.frames_without_detection = 0

        # ── IDLE: check for change ────────────────────────────────────────
        if self._state is _PipelineState.IDLE:
            if self._prev_crop is None:
                # Very first crop — force a read after settling
                self._prev_crop = crop.copy()
                self._state = _PipelineState.SETTLING
                self._counter = 0
                logger.debug("First scoreboard detected, entering SETTLING")
                return None

            if self._has_changed(crop):
                logger.debug("Score change detected, entering SETTLING")
                self._state = _PipelineState.SETTLING
                self._counter = 0

            self._prev_crop = crop.copy()
            return None

        # ── SETTLING: wait for animation to finish ────────────────────────
        if self._state is _PipelineState.SETTLING:
            self._counter += 1
            if self._counter >= self.settle_frames:
                logger.debug("Settle complete, triggering OCR read")
                self._prev_crop = crop.copy()
                self._state = _PipelineState.COOLDOWN
                self._counter = 0
                return crop.copy()
            return None

        # ── COOLDOWN: suppress duplicate triggers ─────────────────────────
        if self._state is _PipelineState.COOLDOWN:
            self._counter += 1
            if self._counter >= self.cooldown_frames:
                logger.debug("Cooldown complete, returning to IDLE")
                self._state = _PipelineState.IDLE
                self._prev_crop = crop.copy()
            return None

        return None  # pragma: no cover

    def _has_changed(self, current_crop: npt.NDArray[np.uint8]) -> bool:
        """Return ``True`` when SSIM between current and previous drops below threshold."""
        from skimage.metrics import structural_similarity as ssim

        prev_gray: npt.NDArray = cv2.cvtColor(
            self._prev_crop,
            cv2.COLOR_BGR2GRAY,  # type: ignore[arg-type]
        )
        curr_gray: npt.NDArray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
        score: float = ssim(prev_gray, curr_gray)
        logger.debug("SSIM = %.4f (threshold %.4f)", score, self.ssim_threshold)
        return score < self.ssim_threshold

    def _reset(self) -> None:
        """Full reset after prolonged scoreboard absence."""
        self._state = _PipelineState.IDLE
        self._prev_crop = None
        self._counter = 0


# ══════════════════════════════════════════════════════════════════════════════
# Score Validation
# ══════════════════════════════════════════════════════════════════════════════


def validate_score(data: dict) -> bool:
    """Sanity-check VLM output values. Returns ``True`` if all checks pass."""
    ok = True

    for key in ("points_p1", "points_p2"):
        val = str(data.get(key) or "")
        if val not in VALID_TENNIS_POINTS and not val.isdigit():
            logger.warning("Validation: %s='%s' is not a valid point value", key, val)
            ok = False

    for key in ("games_p1", "games_p2"):
        val = data.get(key)
        if val is not None:
            try:
                iv = int(val)
                if iv < 0 or iv > MAX_GAMES_IN_SET:
                    logger.warning(
                        "Validation: %s=%d out of range [0, %d]",
                        key,
                        iv,
                        MAX_GAMES_IN_SET,
                    )
                    ok = False
            except (ValueError, TypeError):
                logger.warning("Validation: %s=%r is not an integer", key, val)
                ok = False

    for key in ("sets_p1", "sets_p2"):
        val = data.get(key)
        if val is not None:
            try:
                iv = int(val)
                if iv < 0 or iv > MAX_SETS_IN_MATCH:
                    logger.warning(
                        "Validation: %s=%d out of range [0, %d]",
                        key,
                        iv,
                        MAX_SETS_IN_MATCH,
                    )
                    ok = False
            except (ValueError, TypeError):
                logger.warning("Validation: %s=%r is not an integer", key, val)
                ok = False

    return ok


# ── Point ordering for transition checks ──────────────────────────────────
_POINT_ORDER: dict[str, int] = {"": -1, "0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def validate_transition(prev: "ScoreReading", curr: "ScoreReading") -> bool:
    """Check that the transition from *prev* to *curr* is legal in tennis.

    Returns ``True`` if the transition is plausible.  Logs a warning and
    returns ``False`` for impossible transitions like 40-15 → 15-0.
    """
    # Set score can only increase by 1 for one player at a time
    ds1 = curr.sets_p1 - prev.sets_p1
    ds2 = curr.sets_p2 - prev.sets_p2
    if ds1 < 0 or ds2 < 0 or ds1 > 1 or ds2 > 1 or (ds1 == 1 and ds2 == 1):
        logger.warning(
            "Invalid set transition: %d-%d → %d-%d",
            prev.sets_p1,
            prev.sets_p2,
            curr.sets_p1,
            curr.sets_p2,
        )
        return False

    # If a set changed, game scores should reset to 0-0
    if ds1 == 1 or ds2 == 1:
        if curr.games_p1 != 0 or curr.games_p2 != 0:
            # Allow the first reading after set change to show the new set
            # starting — game scores should be 0-0 but the OCR might catch
            # a brief transitional frame. Log but allow.
            logger.debug(
                "Set changed but games not 0-0 (%d-%d), allowing",
                curr.games_p1,
                curr.games_p2,
            )
        return True

    # Game score: can only change by 1 for one player, or stay the same
    dg1 = curr.games_p1 - prev.games_p1
    dg2 = curr.games_p2 - prev.games_p2
    if dg1 < 0 or dg2 < 0 or dg1 > 1 or dg2 > 1 or (dg1 == 1 and dg2 == 1):
        logger.warning(
            "Invalid game transition: %d-%d → %d-%d",
            prev.games_p1,
            prev.games_p2,
            curr.games_p1,
            curr.games_p2,
        )
        return False

    # If a game changed, point scores should reset
    if dg1 == 1 or dg2 == 1:
        # The winner must have been in a winning position (40 or AD) or
        # tiebreak score. Allow flexible point reset.
        return True

    # Point-level transition within the same game
    # Normalize empty points to "0" — empty means 0-0 (scoreboard not showing points)
    pp1 = prev.points_p1 if prev.points_p1 else "0"
    pp2 = prev.points_p2 if prev.points_p2 else "0"
    cp1 = curr.points_p1 if curr.points_p1 else "0"
    cp2 = curr.points_p2 if curr.points_p2 else "0"

    # Tiebreak: points are integers, just check one goes up by 1
    if _is_tiebreak_score(prev) or _is_tiebreak_score(curr):
        return True  # Tiebreak scoring is more flexible, trust VLM

    # Normal game: exactly one player's point should advance
    p1_changed = pp1 != cp1
    p2_changed = pp2 != cp2

    if not p1_changed and not p2_changed:
        # No point change but score_changed() triggered — could be server change
        return True

    # Deuce situations
    if pp1 == "40" and pp2 == "40":
        # Can go to AD-40, 40-AD, or stay (re-read)
        valid = (cp1 == "AD" and cp2 == "40") or (cp1 == "40" and cp2 == "AD")
        if not valid:
            logger.warning(
                "Invalid deuce transition: %s-%s → %s-%s", pp1, pp2, cp1, cp2
            )
            return False
        return True

    if pp1 == "AD" and pp2 == "40":
        # AD player wins game (game score change) or back to deuce
        valid = cp1 == "40" and cp2 == "40"
        if not valid:
            logger.warning("Invalid AD transition: %s-%s → %s-%s", pp1, pp2, cp1, cp2)
            return False
        return True

    if pp1 == "40" and pp2 == "AD":
        valid = cp1 == "40" and cp2 == "40"
        if not valid:
            logger.warning("Invalid AD transition: %s-%s → %s-%s", pp1, pp2, cp1, cp2)
            return False
        return True

    # Normal progression: one player goes up, other stays the same
    if p1_changed and p2_changed:
        logger.warning(
            "Both point scores changed in one step: %s-%s → %s-%s",
            pp1,
            pp2,
            cp1,
            cp2,
        )
        return False

    # Check the changed player advanced correctly: 0→15→30→40
    if p1_changed:
        if not _valid_point_advance(pp1, cp1):
            logger.warning("Invalid point advance for P1: %s → %s", pp1, cp1)
            return False
    if p2_changed:
        if not _valid_point_advance(pp2, cp2):
            logger.warning("Invalid point advance for P2: %s → %s", pp2, cp2)
            return False

    return True


def _valid_point_advance(prev_pt: str, curr_pt: str) -> bool:
    """Check a single player's point went up by exactly one step."""
    order = {"0": "15", "15": "30", "30": "40"}
    return order.get(prev_pt) == curr_pt


def _is_tiebreak_score(reading: "ScoreReading") -> bool:
    """Heuristic: we're in a tiebreak if games are 6-6 or points are plain ints."""
    if reading.games_p1 == 6 and reading.games_p2 == 6:
        return True
    # Points that are plain integers not in standard set → tiebreak
    for pt in (reading.points_p1, reading.points_p2):
        if pt and pt not in VALID_TENNIS_POINTS and pt.isdigit():
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════


class ScoreboardOCRPipeline:
    """Two-stage tennis scoreboard OCR pipeline.

    Stage 1 (every frame):
        RF-DETR scoreboard detection → EMA bbox smoothing → crop + resize →
        SSIM change detection with settle/cooldown state machine.

    Stage 2 (on change only):
        FastVLM reads the settled scoreboard crop → structured JSON → CSV row.

    Args:
        scoreboard_model_path: Path to RF-DETR fine-tuned checkpoint.
        fastvlm_model_path: HuggingFace model ID or local path for FastVLM.
        device: Torch device (auto-detected if ``None``).
        ssim_threshold: SSIM below this triggers change detection.
        settle_frames: Frames to wait after change before reading.
        cooldown_frames: Frames to suppress after a read.
        crop_padding: Fractional padding around detected scoreboard bbox.
        confidence_threshold: RF-DETR detection confidence threshold.
    """

    def __init__(
        self,
        scoreboard_model_path: str = "models/scoreboard_detection/checkpoint_best_total.pth",
        fastvlm_model_path: str = "mlx-community/FastVLM-0.5B-4bit",
        device: torch.device | None = None,
        ssim_threshold: float = 0.92,
        settle_frames: int = 15,
        cooldown_frames: int = 30,
        crop_padding: float = 0.1,
        confidence_threshold: float = 0.5,
        ocr_backend: str = "tesseract",
    ) -> None:
        if device is None:
            device = self._auto_device()
        self.device: torch.device = device
        logger.info("Initializing ScoreboardOCRPipeline on device=%s", device)

        # Stage 1 components
        self.detector = ScoreboardDetector(
            checkpoint_path=scoreboard_model_path,
            device=device,
            padding=crop_padding,
            confidence_threshold=confidence_threshold,
        )
        self.change_detector = ChangeDetector(
            ssim_threshold=ssim_threshold,
            settle_frames=settle_frames,
            cooldown_frames=cooldown_frames,
        )

        # Stage 2 component — OCR reader
        if ocr_backend == "easyocr":
            self.vlm_reader = EasyOCRReader()
        elif ocr_backend == "tesseract":
            self.vlm_reader = TesseractReader()
        else:
            self.vlm_reader = FastVLMReader(model_path=fastvlm_model_path)

        # Tracking
        self._last_score: ScoreReading | None = None
        self._results: list[CSVRow] = []

    # ── Device auto-detection ──────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> torch.device:
        """Pick the best available accelerator."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── Public API ─────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_csv_path: str = "reports/scoreboard_log.csv",
    ) -> list[CSVRow]:
        """Run the full pipeline on a video file.

        Args:
            video_path: Path to the input video.
            output_csv_path: Where to write the output CSV.

        Returns:
            List of ``CSVRow`` objects (one per detected score change).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec: float = total_frames / fps

        logger.info(
            "Video: %s (%d frames, %.1f fps, %.1fs)",
            video_path,
            total_frames,
            fps,
            duration_sec,
        )

        self._results = []
        self._last_score = None
        frame_idx: int = 0
        vlm_calls: int = 0

        try:
            while True:
                ret: bool
                frame: npt.NDArray[np.uint8]
                ret, frame = cap.read()
                if not ret:
                    break

                # ── Stage 1: detect + crop ────────────────────────────────
                crop = self.detector.detect_and_crop(frame)

                # Reset detector smoothing after prolonged absence
                if (
                    self.change_detector.frames_without_detection
                    >= self.change_detector.no_detection_reset_frames
                ):
                    self.detector.reset()

                # ── Stage 1: change detection state machine ───────────────
                settled_crop = self.change_detector.update(crop)

                # ── Stage 2: VLM OCR (only when change detected) ─────────
                if settled_crop is not None:
                    vlm_calls += 1
                    self._handle_ocr_read(settled_crop, frame_idx, fps)

                frame_idx += 1

                # Progress logging every 1000 frames
                if frame_idx % 1000 == 0:
                    pct = frame_idx / max(total_frames, 1) * 100
                    logger.info(
                        "Progress: %d/%d (%.1f%%) — %d VLM calls, %d changes",
                        frame_idx,
                        total_frames,
                        pct,
                        vlm_calls,
                        len(self._results),
                    )
        finally:
            cap.release()

        # Write results
        self._write_csv(output_csv_path)

        logger.info(
            "Complete: %d frames, %d VLM calls, %d score changes → %s",
            frame_idx,
            vlm_calls,
            len(self._results),
            output_csv_path,
        )
        return self._results

    # ── Internal helpers ───────────────────────────────────────────────────

    def _handle_ocr_read(
        self,
        crop: npt.NDArray[np.uint8],
        frame_idx: int,
        fps: float,
    ) -> None:
        """Run VLM on a settled crop, validate, and maybe append a CSV row."""
        score_data = self.vlm_reader.read(crop)

        if score_data is None:
            logger.warning("Frame %d: VLM returned no parseable result", frame_idx)
            return

        if not validate_score(score_data):
            logger.warning(
                "Frame %d: validation failed, skipping: %s", frame_idx, score_data
            )
            return

        # Build ScoreReading — guard against null values from VLM
        server_raw = score_data.get("server")
        p1_raw = score_data.get("player1")
        p2_raw = score_data.get("player2")

        if p1_raw is None or p2_raw is None:
            logger.warning(
                "Frame %d: player name(s) are null (p1=%r, p2=%r)",
                frame_idx,
                p1_raw,
                p2_raw,
            )
            return

        p1 = str(p1_raw).upper()
        p2 = str(p2_raw).upper()
        server = str(server_raw).upper() if server_raw else ""
        returner = p2 if server == p1 else p1

        # Convert numeric fields — treat None as 0 for sets/games, "" for points
        def _safe_int(val, default: int = 0) -> int:
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        reading = ScoreReading(
            player1=p1,
            player2=p2,
            sets_p1=_safe_int(score_data.get("sets_p1")),
            sets_p2=_safe_int(score_data.get("sets_p2")),
            games_p1=_safe_int(score_data.get("games_p1")),
            games_p2=_safe_int(score_data.get("games_p2")),
            points_p1=str(score_data.get("points_p1") or ""),
            points_p2=str(score_data.get("points_p2") or ""),
            server=server,
            returner=returner,
        )

        # Only log if score actually changed (catch Stage 1 false positives)
        if not reading.score_changed(self._last_score):
            logger.debug("Frame %d: VLM confirmed no actual score change", frame_idx)
            return

        # Validate transition against tennis rules
        if self._last_score is not None:
            if not validate_transition(self._last_score, reading):
                logger.warning(
                    "Frame %d: invalid score transition, skipping", frame_idx
                )
                return

        row = CSVRow(
            frame_num=frame_idx,
            timestamp_sec=round(frame_idx / fps, 2),
            player1_name=reading.player1,
            player2_name=reading.player2,
            set_score=f"{reading.sets_p1}-{reading.sets_p2}",
            game_score=f"{reading.games_p1}-{reading.games_p2}",
            point_score=f"{reading.points_p1}-{reading.points_p2}",
            server=reading.server,
            returner=reading.returner,
        )
        self._results.append(row)
        self._last_score = reading

        logger.info(
            "[Frame %d / %.1fs] %s vs %s | Sets %d-%d | Games %d-%d | "
            "Points %s-%s | Serving: %s",
            frame_idx,
            frame_idx / fps,
            reading.player1,
            reading.player2,
            reading.sets_p1,
            reading.sets_p2,
            reading.games_p1,
            reading.games_p2,
            reading.points_p1,
            reading.points_p2,
            reading.server,
        )

    def _write_csv(self, output_path: str) -> None:
        """Write all accumulated results to a CSV file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for row in self._results:
                writer.writerow(row.as_dict())

        logger.info("CSV written: %s (%d rows)", output_path, len(self._results))

    # ── Point clipping ────────────────────────────────────────────────────

    def clip_points(
        self,
        video_path: str,
        output_dir: str = "data/interim/point_clips",
        padding_before: float = 1.0,
        padding_after: float = 0.5,
    ) -> list[dict]:
        """Clip the video into individual point segments based on score changes.

        Each point runs from the previous score change to the current one.
        The first point starts from the beginning of the video.

        Args:
            video_path: Path to the source video.
            output_dir: Directory to write point clip files.
            padding_before: Seconds of padding before the point start.
            padding_after: Seconds of padding after the point end.

        Returns:
            List of dicts with point metadata (start, end, score, clip path).
        """
        import subprocess

        if len(self._results) < 2:
            logger.warning("Need at least 2 score changes to clip points")
            return []

        os.makedirs(output_dir, exist_ok=True)

        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        points: list[dict] = []

        for i in range(1, len(self._results)):
            prev_row = self._results[i - 1]
            curr_row = self._results[i]

            # Point starts at previous score change, ends at current
            start_sec = max(0.0, prev_row.timestamp_sec - padding_before)
            end_sec = min(duration, curr_row.timestamp_sec + padding_after)

            point_num = i
            # Build a descriptive filename
            score_label = f"{curr_row.game_score}_{curr_row.point_score}".replace(
                "-", "v"
            )
            clip_name = f"point_{point_num:03d}_{score_label}.mp4"
            clip_path = os.path.join(output_dir, clip_name)

            # Use ffmpeg for fast, accurate clipping
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_sec:.2f}",
                "-to",
                f"{end_sec:.2f}",
                "-i",
                video_path,
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                clip_path,
            ]

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logger.warning("ffmpeg clip failed for point %d: %s", point_num, exc)
                continue

            point_info = {
                "point_num": point_num,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "duration_sec": round(end_sec - start_sec, 2),
                "score_before": f"{prev_row.game_score} ({prev_row.point_score})",
                "score_after": f"{curr_row.game_score} ({curr_row.point_score})",
                "server": curr_row.server,
                "clip_path": clip_path,
            }
            points.append(point_info)

            logger.info(
                "Point %d: %.1fs–%.1fs (%.1fs) | %s → %s | %s",
                point_num,
                start_sec,
                end_sec,
                end_sec - start_sec,
                point_info["score_before"],
                point_info["score_after"],
                clip_path,
            )

        # Write point index CSV
        index_path = os.path.join(output_dir, "point_index.csv")
        with open(index_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "point_num",
                    "start_sec",
                    "end_sec",
                    "duration_sec",
                    "score_before",
                    "score_after",
                    "server",
                    "clip_path",
                ],
            )
            writer.writeheader()
            for p in points:
                writer.writerow(p)

        logger.info(
            "Clipped %d points → %s (index: %s)",
            len(points),
            output_dir,
            index_path,
        )
        return points


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for standalone CLI usage."""
    parser = argparse.ArgumentParser(
        description="Tennis Scoreboard OCR Pipeline — detect score changes in match video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--scoreboard-model",
        default="models/scoreboard_detection/checkpoint_best_total.pth",
        help="Path to RF-DETR scoreboard detection checkpoint",
    )
    parser.add_argument(
        "--fastvlm-model",
        default="mlx-community/FastVLM-0.5B-4bit",
        help="FastVLM model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/scoreboard_log.csv",
        help="Path for output CSV file",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=15,
        help="Frames to wait after change detected before reading",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=30,
        help="Frames to suppress after a successful read",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.92,
        help="SSIM threshold — below this triggers change detection",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.1,
        help="Fractional padding around scoreboard bounding box",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--ocr-backend",
        default="easyocr",
        choices=["easyocr", "tesseract", "fastvlm"],
        help="OCR backend: easyocr (recommended), tesseract, or fastvlm (VLM)",
    )
    parser.add_argument(
        "--clip-points",
        action="store_true",
        help="Clip individual points from the video based on score changes",
    )
    parser.add_argument(
        "--clip-output-dir",
        default="data/interim/point_clips",
        help="Directory for point clip output files",
    )
    parser.add_argument(
        "--clip-padding-before",
        type=float,
        default=1.0,
        help="Seconds of padding before each point clip",
    )
    parser.add_argument(
        "--clip-padding-after",
        type=float,
        default=0.5,
        help="Seconds of padding after each point clip",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main() -> None:
    """CLI entry point for ``python -m src.inference.scoreboard_ocr``."""
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device: torch.device | None = None
    if args.device is not None:
        device = torch.device(args.device)

    pipeline = ScoreboardOCRPipeline(
        scoreboard_model_path=args.scoreboard_model,
        fastvlm_model_path=args.fastvlm_model,
        device=device,
        ssim_threshold=args.ssim_threshold,
        settle_frames=args.settle_frames,
        cooldown_frames=args.cooldown_frames,
        crop_padding=args.crop_padding,
        ocr_backend=args.ocr_backend,
    )

    results = pipeline.process_video(
        video_path=args.video,
        output_csv_path=args.output_csv,
    )

    # Point clipping
    points: list[dict] = []
    if args.clip_points and len(results) >= 2:
        points = pipeline.clip_points(
            video_path=args.video,
            output_dir=args.clip_output_dir,
            padding_before=args.clip_padding_before,
            padding_after=args.clip_padding_after,
        )

    # Summary
    print(f"\n{'=' * 64}")
    print(" Scoreboard OCR Pipeline Complete")
    print(f" Score changes detected: {len(results)}")
    print(f" Output CSV: {args.output_csv}")
    if results:
        print(
            f" First change: frame {results[0].frame_num} ({results[0].timestamp_sec}s)"
        )
        print(
            f" Last change:  frame {results[-1].frame_num} ({results[-1].timestamp_sec}s)"
        )
    if points:
        print(f" Points clipped: {len(points)} → {args.clip_output_dir}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
