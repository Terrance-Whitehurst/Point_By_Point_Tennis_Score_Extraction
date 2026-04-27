"""
point_builder — Point-by-Point CSV Post-Processor
===================================================

Converts the event-stream CSV produced by ``scoreboard_ocr.py`` (one row per
scoreboard change) into a point-by-point CSV whose schema matches
``test_vids/1/Clip1.csv``.

Input schema (scoreboard_ocr.py output):
    frame_num, timestamp_sec, player1_name, player2_name,
    set_score, game_score, point_score, server, returner

    Where:
        set_score   — "p1_sets-p2_sets"        e.g. "0-1"
        game_score  — "p1_games-p2_games"       e.g. "3-2"  (games in set)
        point_score — "p1_pts-p2_pts"           e.g. "30-15" (within game)

Output schema (6 columns, exact order):
    Point_Start, Point_End, Server, Returner, Game_Score, Set_Score

    Where:
        Game_Score — derived from point_score, server's score first
        Set_Score  — derived from set_score,   server's score first
        Point_Start / Point_End — MM:SS timestamps (see pairing logic)

Pairing logic (for sorted events E[0..N], row i):
    Point_Start = "00:00"                          for i == 0
                  MM:SS(E[i].timestamp_sec + START_OFFSET)  otherwise
    Point_End   = MM:SS(E[i+1].timestamp_sec − LAG_OFFSET)  for i < N
                  clamped to >= Point_Start + 1 s
                  "{MM:SS(E[N].timestamp_sec)}(Placeholder — clip ends)"
                  for the last row (i == N)

Defaults:
    LAG_OFFSET   = 2.0 s  (--broadcaster-lag)
    START_OFFSET = 3.0 s  (--start-offset)

    Defaults are overridden by ``configs/broadcaster_offsets.json`` when it
    exists, using the ``--broadcaster`` key (falls back to ``"default"`` block).
    Explicit ``--broadcaster-lag`` / ``--start-offset`` flags always take
    highest precedence.

CLI:
    python -m src.inference.point_builder \\
        --input  reports/match.csv \\
        --output reports/match_points.csv

    # With broadcaster-specific calibration:
    python -m src.inference.point_builder \\
        --input  reports/match.csv \\
        --output reports/match_points.csv \\
        --broadcaster test_vids/1

    # Fully explicit (overrides config):
    python -m src.inference.point_builder \\
        --input  reports/match.csv \\
        --output reports/match_points.csv \\
        --broadcaster-lag 1.4 --start-offset 3.4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_LAG_OFFSET: float = 2.0    # seconds to subtract from next timestamp
DEFAULT_START_OFFSET: float = 3.0  # seconds to add to current timestamp

# Default config file path (relative to cwd); loaded automatically if present
DEFAULT_CONFIG_PATH: str = "configs/broadcaster_offsets.json"

# Output column names — must match Clip1.csv exactly
OUTPUT_FIELDNAMES: list[str] = [
    "Point_Start",
    "Point_End",
    "Server",
    "Returner",
    "Game_Score",
    "Set_Score",
]

# Standard tennis point-ladder values (anything else ⇒ tiebreak)
_TENNIS_POINT_LADDER: frozenset[str] = frozenset({"0", "15", "30", "40", "AD", ""})

# Ordered standard-ladder list and index map (used by validate_transition)
_POINT_LADDER: list[str] = ["0", "15", "30", "40"]
_POINT_LADDER_IDX: dict[str, int] = {v: i for i, v in enumerate(_POINT_LADDER)}

# Placeholder suffix for the final row of a clip
_PLACEHOLDER_SUFFIX: str = "(Placeholder because we arent testing on full video just a clip rn)"


# ── Data types ────────────────────────────────────────────────────────────────


class EventRow(NamedTuple):
    """One row from the scoreboard_ocr.py event-stream CSV."""

    frame_num: int
    timestamp_sec: float
    player1_name: str
    player2_name: str
    set_score: str    # "p1_sets-p2_sets",  e.g. "0-1"
    game_score: str   # "p1_games-p2_games", e.g. "3-2"  (games in current set)
    point_score: str  # "p1_pts-p2_pts",     e.g. "30-15"
    server: str
    returner: str


class PointRow(NamedTuple):
    """One row for the output point-by-point CSV."""

    Point_Start: str
    Point_End: str
    Server: str
    Returner: str
    Game_Score: str
    Set_Score: str


# ── Pure helper functions ──────────────────────────────────────────────────────


def sec_to_mmss(seconds: float) -> str:
    """Format a non-negative duration (seconds) as zero-padded MM:SS.

    Args:
        seconds: Duration in seconds.

    Returns:
        String in the form ``"MM:SS"``, e.g. ``"02:05"``.

    Examples:
        >>> sec_to_mmss(0)
        '00:00'
        >>> sec_to_mmss(65.7)
        '01:06'
        >>> sec_to_mmss(140.0)
        '02:20'
    """
    total_sec = max(0, int(round(seconds)))
    mm, ss = divmod(total_sec, 60)
    return f"{mm:02d}:{ss:02d}"


def _names_match(a: str, b: str) -> bool:
    """Case-insensitive, whitespace-stripped name equality."""
    return a.strip().upper() == b.strip().upper()


def normalize_name(s: str) -> str:
    """Convert an OCR player name to display Title Case.

    Applies ``str.title()`` which correctly handles common tennis-name
    patterns::

        "ALCARAZ"   → "Alcaraz"
        "DE MINAUR" → "De Minaur"
        "O'BRIEN"   → "O'Brien"
        "MCDONALD"  → "Mcdonald"  # minor edge-case; acceptable for v1

    The original casing is preserved internally for the state machine
    (server-flip checks use :func:`_names_match` which is case-insensitive).
    This function is only called when writing the final output row.

    Args:
        s: Raw player name string from the event stream.

    Returns:
        Title-cased string, e.g. ``"De Minaur"``.
    """
    return s.strip().title()


def parse_score_pair(score_str: str) -> tuple[str, str]:
    """Split a ``"X-Y"`` score string into ``(x, y)`` component strings.

    Handles edge cases:
        - ``""``   → ``("", "")``
        - ``"-"``  → ``("", "")``
        - ``"0-"`` → ``("0", "")``
        - ``"AD-40"`` → ``("AD", "40")``

    Args:
        score_str: Raw score string, e.g. ``"30-15"``.

    Returns:
        Tuple ``(p1_value, p2_value)`` as raw strings.
    """
    if "-" in score_str:
        parts = score_str.split("-", 1)
        return parts[0].strip(), parts[1].strip()
    return score_str.strip(), ""


def is_tiebreak_score(p1_pts: str, p2_pts: str) -> bool:
    """Return True when either point value falls outside the standard ladder.

    Heuristic: if either value is not in {0, 15, 30, 40, AD, ""}, we are in
    a tiebreak and the raw integer strings should be passed through.

    Args:
        p1_pts: Player-1 point string.
        p2_pts: Player-2 point string.

    Returns:
        ``True`` if this looks like a tiebreak, ``False`` otherwise.
    """
    return (
        p1_pts not in _TENNIS_POINT_LADDER
        or p2_pts not in _TENNIS_POINT_LADDER
    )


def format_game_score(server_pts: str, returner_pts: str) -> str:
    """Format the within-game score string with the server's score first.

    Normalizes empty strings to ``"0"``.  For tiebreak scores (integers not
    on the standard ladder) the values are passed through verbatim.

    Args:
        server_pts:   Server's raw point value.
        returner_pts: Returner's raw point value.

    Returns:
        Formatted string such as ``"30-15"``, ``"AD-40"``, or ``"7-5"``.

    Examples:
        >>> format_game_score("0", "0")
        '0-0'
        >>> format_game_score("AD", "40")
        'AD-40'
        >>> format_game_score("7", "5")
        '7-5'
        >>> format_game_score("", "")
        '0-0'
    """
    sp = server_pts.strip() if server_pts else ""
    rp = returner_pts.strip() if returner_pts else ""

    # Normalize empty → "0"
    sp = sp if sp else "0"
    rp = rp if rp else "0"

    return f"{sp}-{rp}"


def build_game_score(
    point_score: str,
    player1_name: str,
    server: str,
) -> str:
    """Derive the output ``Game_Score`` from an event row.

    Reads the ``point_score`` field (e.g. ``"30-15"`` meaning p1 has 30, p2
    has 15) and re-orders so the server's score is always first.

    Args:
        point_score:  Combined point score string from the event stream.
        player1_name: Name of player 1 in the event stream.
        server:       Name of the serving player.

    Returns:
        Formatted game score string with server's score first.
    """
    p1_pts, p2_pts = parse_score_pair(point_score)

    if _names_match(server, player1_name):
        server_pts, returner_pts = p1_pts, p2_pts
    else:
        server_pts, returner_pts = p2_pts, p1_pts

    return format_game_score(server_pts, returner_pts)


def build_set_score(
    game_score: str,
    player1_name: str,
    server: str,
) -> str:
    """Derive the output ``Set_Score`` from an event row.

    Reads the ``game_score`` field (e.g. ``"0-1"`` meaning p1 has 0 games, p2
    has 1 game in the current set) and re-orders so the server's score is
    always first.  This is games-in-current-set, NOT match-level sets won.

    Args:
        game_score:   Combined games-in-set score string from the event stream.
        player1_name: Name of player 1 in the event stream.
        server:       Name of the serving player.

    Returns:
        Set score string with server's score first, e.g. ``"0-1"``.
    """
    p1_sets, p2_sets = parse_score_pair(game_score)

    if _names_match(server, player1_name):
        return f"{p1_sets}-{p2_sets}"
    else:
        return f"{p2_sets}-{p1_sets}"


def validate_transition(prev: EventRow, curr: EventRow) -> tuple[bool, str]:
    """Check whether the score transition from *prev* to *curr* is legal.

    Covers the following rule-sets:

    * **Rule 6 — prev is a boundary row:** if ``prev.point_score`` is ``"-"``
      or ``""`` (a row that marks a game/set reset rather than an in-progress
      point), any following state is valid — the state machine has just
      re-anchored.
    * **Rule 2 — curr is a game reset ("0-0" or "-"):** ``game_score`` for
      the current event must have incremented by exactly 1 for one side, or
      both games must have reset to 0 (set-boundary).
    * **Rule 5 — tiebreak:** detected via :func:`is_tiebreak_score`; exactly
      one integer score must increment by 1.
    * **Rule 4 — deuce / advantage:**
        * From ``40-40``: next must be ``AD-40`` or ``40-AD``.
        * From ``AD-X`` / ``X-AD``: next must be ``40-40`` or a game reset
          (the reset case is already caught by Rule 2).
    * **Rule 3 — standard ladder (``0/15/30/40``):** exactly one side
      increments by one step; the other side is unchanged.

    False-positive tolerance is intentionally high — unrecognised score
    formats return ``(True, "OK")`` rather than raising.

    Args:
        prev: The previous :class:`EventRow`.
        curr: The current :class:`EventRow`.

    Returns:
        ``(True, "OK")`` for a valid transition; ``(False, reason)`` otherwise.
    """
    prev_p1, prev_p2 = parse_score_pair(prev.point_score)
    curr_p1, curr_p2 = parse_score_pair(curr.point_score)

    # ── Rule 6: prev is a boundary / reset row ────────────────────────────
    if prev.point_score in ("-", ""):
        return True, "OK"

    # ── Rule 2: curr is a game reset ──────────────────────────────────────
    curr_is_reset = curr.point_score in ("-", "") or (
        curr_p1 == "0" and curr_p2 == "0"
    )
    if curr_is_reset:
        prev_g1, prev_g2 = parse_score_pair(prev.game_score)
        curr_g1, curr_g2 = parse_score_pair(curr.game_score)
        try:
            pg1, pg2 = int(prev_g1 or "0"), int(prev_g2 or "0")
            cg1, cg2 = int(curr_g1 or "0"), int(curr_g2 or "0")
        except ValueError:
            return True, "OK"  # unparseable games — don't flag
        delta1, delta2 = cg1 - pg1, cg2 - pg2
        # Exactly one side gains one game
        if (delta1 == 1 and delta2 == 0) or (delta1 == 0 and delta2 == 1):
            return True, "OK"
        # Set boundary: both game counts reset to 0, but only if games were
        # non-trivial before (prevents false-positive when game_score was
        # already '0-0' and nothing changed).
        if cg1 == 0 and cg2 == 0 and (pg1 > 0 or pg2 > 0):
            return True, "OK"
        return False, (
            f"game_score did not increment at game boundary: "
            f"{prev.game_score!r} → {curr.game_score!r}"
        )

    # ── Rule 5: tiebreak ──────────────────────────────────────────────────
    if is_tiebreak_score(prev_p1, prev_p2) or is_tiebreak_score(curr_p1, curr_p2):
        try:
            pp1, pp2 = int(prev_p1 or "0"), int(prev_p2 or "0")
            cp1, cp2 = int(curr_p1 or "0"), int(curr_p2 or "0")
        except ValueError:
            return True, "OK"  # mixed tiebreak/normal format — don't flag
        delta1, delta2 = cp1 - pp1, cp2 - pp2
        if (delta1 == 1 and delta2 == 0) or (delta1 == 0 and delta2 == 1):
            return True, "OK"
        return False, (
            f"tiebreak score did not increment by 1: "
            f"{prev.point_score!r} → {curr.point_score!r}"
        )

    # ── Rule 4: deuce ─────────────────────────────────────────────────────
    if prev_p1 == "40" and prev_p2 == "40":
        if (curr_p1 == "AD" and curr_p2 == "40") or (
            curr_p1 == "40" and curr_p2 == "AD"
        ):
            return True, "OK"
        return False, (
            f"from deuce expected AD-40 or 40-AD, got: {curr.point_score!r}"
        )

    # ── Rule 4: advantage ─────────────────────────────────────────────────
    if prev_p1 == "AD" or prev_p2 == "AD":
        # Valid outcomes: back to deuce (40-40); game-end reset caught above
        if curr_p1 == "40" and curr_p2 == "40":
            return True, "OK"
        return False, (
            f"from advantage {prev.point_score!r} expected 40-40 or game reset, "
            f"got: {curr.point_score!r}"
        )

    # ── Rule 3: standard ladder ───────────────────────────────────────────
    # Both prev values must be on the ladder; otherwise we can't judge.
    if prev_p1 not in _POINT_LADDER_IDX or prev_p2 not in _POINT_LADDER_IDX:
        return True, "OK"
    # Both curr values must also be on the ladder (AD handled above).
    if curr_p1 not in _POINT_LADDER_IDX or curr_p2 not in _POINT_LADDER_IDX:
        return True, "OK"

    pp1_idx = _POINT_LADDER_IDX[prev_p1]
    pp2_idx = _POINT_LADDER_IDX[prev_p2]
    cp1_idx = _POINT_LADDER_IDX[curr_p1]
    cp2_idx = _POINT_LADDER_IDX[curr_p2]

    # p1 gains one step, p2 unchanged
    p1_scored = (cp1_idx == pp1_idx + 1) and (curr_p2 == prev_p2)
    # p2 gains one step, p1 unchanged
    p2_scored = (cp2_idx == pp2_idx + 1) and (curr_p1 == prev_p1)

    if p1_scored or p2_scored:
        return True, "OK"

    return False, (
        f"illegal standard-ladder transition: "
        f"{prev.point_score!r} → {curr.point_score!r}"
    )


# ── Drift detection ───────────────────────────────────────────────────────────


_DRIFT_FIELDS: tuple[str, ...] = (
    "point_score",
    "game_score",
    "set_score",
    "server",
    "returner",
)


def is_drift(events: list[EventRow], index: int) -> bool:
    """Return True if events[index] is a drift duplicate (3+ identical consecutive readings).

    Compares ``point_score``, ``game_score``, ``set_score``, ``server``, and
    ``returner`` fields across three consecutive events.  Timestamps are
    deliberately *excluded* — a scoreboard that freezes on-screen will
    have different timestamps but identical score/player fields.

    Returns ``False`` when ``index < 2`` (not enough history to classify drift).

    Args:
        events: Full list of :class:`EventRow` objects in order.
        index:  Index of the event under examination.

    Returns:
        ``True`` if ``events[index]``, ``events[index-1]``, and
        ``events[index-2]`` share identical values for all tracked fields;
        ``False`` otherwise.

    Examples:
        >>> # Three identical rows ⇒ drift at index 2
        >>> e = EventRow(0, 0.0, 'A', 'B', '0-0', '0-1', '30-15', 'A', 'B')
        >>> is_drift([e, e, e], 2)
        True
        >>> is_drift([e, e, e], 1)   # index < 2
        False
    """
    if index < 2:
        return False

    def _key(e: EventRow) -> tuple[str, ...]:
        return tuple(getattr(e, f) for f in _DRIFT_FIELDS)

    return _key(events[index]) == _key(events[index - 1]) == _key(events[index - 2])


# ── Server-flip cross-check ────────────────────────────────────────────────────


_GAME_RESET_SCORES: frozenset[str] = frozenset({"-", "0-0", ""})


def check_server_flip(
    prev: EventRow,
    curr: EventRow,
    expected_server: str | None,
) -> tuple[str, str, str | None]:
    """Cross-check server at game boundaries.

    Detects a *game boundary* when ``curr.point_score`` resets to one of
    ``"-"``, ``"0-0"``, or ``""`` while ``prev.point_score`` was **not** one
    of those.  At such a boundary, if *expected_server* is provided and
    differs (case-insensitively) from ``curr.server``, the correction is
    applied: the expected server is returned and the former OCR-read server
    becomes the returner.

    No correction is applied when:
    * the event is not at a game boundary,
    * *expected_server* is ``None``,
    * ``curr.server`` already matches *expected_server*.

    Args:
        prev:            The preceding :class:`EventRow`.
        curr:            The current :class:`EventRow` to inspect.
        expected_server: Player name that the state machine expects to be
                         serving after the game boundary, or ``None`` to skip
                         the check.

    Returns:
        A three-tuple ``(server, returner, debug_flag)`` where *debug_flag*
        is ``"SERVER_FLIP_CORRECTED"`` when a correction was applied, and
        ``None`` otherwise.

    Examples:
        >>> # Boundary detected and OCR disagrees with expected server
        >>> prev = EventRow(0, 0.0, 'A', 'B', '0-0', '0-1', '40-30', 'A', 'B')
        >>> curr = EventRow(1, 5.0, 'A', 'B', '0-0', '1-1', '-',     'A', 'B')
        >>> check_server_flip(prev, curr, expected_server='B')
        ('B', 'A', 'SERVER_FLIP_CORRECTED')
    """
    is_boundary = (
        curr.point_score in _GAME_RESET_SCORES
        and prev.point_score not in _GAME_RESET_SCORES
    )

    if is_boundary and expected_server is not None:
        if not _names_match(expected_server, curr.server):
            # State-machine wins: swap to the expected server; the player
            # that OCR reported as server becomes the returner.
            return expected_server, curr.server, "SERVER_FLIP_CORRECTED"

    return curr.server, curr.returner, None


# ── Core conversion ────────────────────────────────────────────────────────────


def events_to_points(
    events: list[EventRow],
    lag_offset: float = DEFAULT_LAG_OFFSET,
    start_offset: float = DEFAULT_START_OFFSET,
    debug: bool = False,
) -> list[dict[str, str]]:
    """Convert a sorted list of event-stream rows into point-by-point rows.

    For sorted events ``E[0..N]``, row ``i`` is built as follows:

    ``Point_Start``
        ``"00:00"``                          for the first *emitted* row
        ``MM:SS(E[i].timestamp_sec + start_offset)``  otherwise

    ``Point_End``
        ``MM:SS(E[i+1].timestamp_sec - lag_offset)``
        clamped so it is at least ``Point_Start + 1 s`` when ``i < N``
        ``"{MM:SS(E[N].timestamp_sec)}(Placeholder because we arent testing on full video just a clip rn)"``
        when ``i == N`` (last event by index, no successor)

    ``Game_Score``
        Derived from ``E[i].point_score`` with the server's score first.

    ``Set_Score``
        Derived from ``E[i].game_score`` (games-in-set) with server's score first.

    **Hardening (M3):**

    * **Drift:** if three consecutive events share identical score/player
      fields (see :func:`is_drift`), the duplicate is skipped and a
      ``DRIFT_DETECTED`` warning is emitted to *stderr*.
    * **Invalid transitions:** illegal score moves (see
      :func:`validate_transition`) are skipped.  A single invalid event
      emits one warning; a second consecutive invalid emits a louder
      "re-anchoring" warning.  The counter resets on the next valid event.
    * **Server-flip:** at game boundaries, :func:`check_server_flip`
      compares the OCR-read server against the state-machine expectation.
      Mismatches are corrected and a warning is emitted; when *debug* is
      ``True`` the ``_debug`` column is set to ``"SERVER_FLIP_CORRECTED"``.

    Args:
        events:       Sorted list of ``EventRow`` objects.
        lag_offset:   Seconds to subtract from the next event's timestamp.
        start_offset: Seconds to add to the current event's timestamp.
        debug:        When ``True``, a ``"_debug"`` key is added to every
                      output dict with value ``"OK"`` or
                      ``"SERVER_FLIP_CORRECTED"``.

    Returns:
        List of row dicts.  Keys are the six output fieldnames; a seventh
        ``"_debug"`` key is present when *debug* is ``True``.  No blank,
        drift, or invalid-transition rows are included.
    """
    if not events:
        return []

    rows: list[dict[str, str]] = []
    n = len(events)
    rows_emitted: int = 0           # counts rows actually appended
    consecutive_invalid: int = 0    # consecutive illegal-transition counter
    expected_server: str | None = None  # state-machine server expectation

    for i, event in enumerate(events):
        # ── Drift detection ──────────────────────────────────────────────────
        if is_drift(events, i):
            print(
                f"WARNING: DRIFT_DETECTED at index {i} "
                f"(ts={event.timestamp_sec:.2f}s) — skipping duplicate row.",
                file=sys.stderr,
            )
            continue

        # ── Transition validation ─────────────────────────────────────────────
        if i > 0:
            valid, reason = validate_transition(events[i - 1], event)
            if not valid:
                consecutive_invalid += 1
                if consecutive_invalid == 1:
                    print(
                        f"WARNING: INVALID_TRANSITION at index {i} "
                        f"(ts={event.timestamp_sec:.2f}s): {reason} "
                        "— skipping row.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"WARNING: {consecutive_invalid} CONSECUTIVE INVALID "
                        f"TRANSITIONS — most recent at index {i} "
                        f"(ts={event.timestamp_sec:.2f}s): {reason}. "
                        "Re-anchoring on next valid reading.",
                        file=sys.stderr,
                    )
                continue  # skip this row; do NOT reset consecutive_invalid
            else:
                consecutive_invalid = 0

        # ── Server-flip cross-check ───────────────────────────────────────────
        prev_for_flip = events[i - 1] if i > 0 else event
        server, returner, flip_flag = check_server_flip(
            prev_for_flip, event, expected_server
        )
        debug_status = "OK"
        if flip_flag == "SERVER_FLIP_CORRECTED":
            print(
                f"WARNING: SERVER_FLIP_CORRECTED at index {i} "
                f"(ts={event.timestamp_sec:.2f}s) — "
                f"OCR reported '{event.server}', expected '{expected_server}'; "
                f"using '{server}'.",
                file=sys.stderr,
            )
            debug_status = "SERVER_FLIP_CORRECTED"

        # ── Update expected_server for next game boundary ─────────────────────
        # When at a game-start boundary, the current returner serves next game.
        if event.point_score in _GAME_RESET_SCORES:
            expected_server = returner
        elif expected_server is None:
            # Pre-first-boundary: initialise from first trusted OCR reading.
            expected_server = server

        # ── Timestamps ───────────────────────────────────────────────────────
        is_first = rows_emitted == 0   # first *emitted* row
        is_last = i == n - 1           # last event by index

        if is_first:
            point_start = "00:00"
            point_start_sec = 0.0
        else:
            if event.point_score in ("-", ""):
                # Game-boundary event: annotator uses raw timestamp (no look-ahead
                # needed — scoreboard already shows the new game state).
                point_start_sec = event.timestamp_sec
            else:
                point_start_sec = event.timestamp_sec + start_offset
            point_start = sec_to_mmss(point_start_sec)

        if is_last:
            # Placeholder: raw event timestamp, no offset
            placeholder_ts = sec_to_mmss(event.timestamp_sec)
            point_end = f"{placeholder_ts}{_PLACEHOLDER_SUFFIX}"
        else:
            next_event = events[i + 1]
            end_sec = next_event.timestamp_sec - lag_offset
            # Clamp: end must be at least 1 second after start
            min_end_sec = point_start_sec + 1.0
            end_sec = max(end_sec, min_end_sec)
            point_end = sec_to_mmss(end_sec)

        # ── Score fields ──────────────────────────────────────────────────────
        game_score = build_game_score(
            event.point_score,
            event.player1_name,
            server,
        )
        set_score = build_set_score(
            event.game_score,
            event.player1_name,
            server,
        )

        # ── Build output row ──────────────────────────────────────────────────
        row: dict[str, str] = {
            "Point_Start": point_start,
            "Point_End":   point_end,
            "Server":      normalize_name(server),
            "Returner":    normalize_name(returner),
            "Game_Score":  game_score,
            "Set_Score":   set_score,
        }
        if debug:
            row["_debug"] = debug_status

        rows.append(row)
        rows_emitted += 1

    return rows


# ── I/O ───────────────────────────────────────────────────────────────────────


def load_events(csv_path: str) -> list[EventRow]:
    """Load and parse event-stream CSV rows from ``scoreboard_ocr.py`` output.

    Rows that are entirely blank are silently skipped.  The returned list is
    sorted ascending by ``timestamp_sec``.

    Args:
        csv_path: Path to the event-stream CSV file.

    Returns:
        Sorted list of ``EventRow`` objects.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Event-stream CSV not found: '{csv_path}'")

    rows: list[EventRow] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for lineno, raw in enumerate(reader, start=2):  # 1-indexed; row 1 is header
            # Skip entirely blank rows (e.g., trailing comma-only rows)
            if not any(v.strip() for v in raw.values() if v is not None):
                continue

            try:
                row = EventRow(
                    frame_num=int(raw.get("frame_num") or 0),
                    timestamp_sec=float(raw.get("timestamp_sec") or 0.0),
                    player1_name=(raw.get("player1_name") or "").strip(),
                    player2_name=(raw.get("player2_name") or "").strip(),
                    set_score=(raw.get("set_score") or "0-0").strip(),
                    game_score=(raw.get("game_score") or "0-0").strip(),
                    point_score=(raw.get("point_score") or "0-0").strip(),
                    server=(raw.get("server") or "").strip(),
                    returner=(raw.get("returner") or "").strip(),
                )
                rows.append(row)
            except (ValueError, TypeError) as exc:
                print(
                    f"WARNING: Skipping malformed row at line {lineno}: {exc}",
                    file=sys.stderr,
                )
                continue

    rows.sort(key=lambda r: r.timestamp_sec)
    return rows


def write_points(
    point_rows: list[dict[str, str]],
    output_path: str,
    debug: bool = False,
) -> None:
    """Write point-by-point rows to a CSV file.

    Creates parent directories as needed.  Blank rows are never written.

    Args:
        point_rows:  List of row dicts (from :func:`events_to_points`).
        output_path: Destination CSV file path.
        debug:       When ``True``, appends a ``_debug`` column to the
                     output (7 columns total instead of 6).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = OUTPUT_FIELDNAMES + (["_debug"] if debug else [])

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in point_rows:
            writer.writerow(row)


# ── Offsets config ────────────────────────────────────────────────────────────


def load_offsets_config(config_path: str) -> dict[str, Any]:
    """Load a broadcaster-offsets JSON config file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Parsed dictionary, or empty dict if the file is missing / unreadable.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            print(
                f"WARNING: Offsets config at '{config_path}' is not a JSON "
                "object — ignoring.",
                file=sys.stderr,
            )
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"WARNING: Could not read offsets config '{config_path}': {exc} "
            "— using hardcoded defaults.",
            file=sys.stderr,
        )
        return {}


def resolve_offsets(
    config: dict[str, Any],
    broadcaster_key: str | None,
    explicit_lag: float | None,
    explicit_start: float | None,
) -> tuple[float, float]:
    """Resolve lag_offset and start_offset using a precedence chain.

    Precedence (highest → lowest):
    1. Explicit ``--broadcaster-lag`` / ``--start-offset`` CLI flag.
    2. Broadcaster-specific entry in the config (``config[broadcaster_key]``).
    3. ``config["default"]`` block.
    4. Hardcoded ``DEFAULT_LAG_OFFSET`` / ``DEFAULT_START_OFFSET``.

    Args:
        config:          Loaded config dict (may be empty).
        broadcaster_key: Key to look up in *config*, or ``None``.
        explicit_lag:    Value from ``--broadcaster-lag``, or ``None``.
        explicit_start:  Value from ``--start-offset``, or ``None``.

    Returns:
        Tuple ``(lag_offset, start_offset)`` in seconds.
    """
    # Start from hardcoded defaults
    lag = DEFAULT_LAG_OFFSET
    start = DEFAULT_START_OFFSET

    # Layer 3: config "default" block
    if "default" in config:
        default_block = config["default"]
        if "lag_offset" in default_block:
            lag = float(default_block["lag_offset"])
        if "start_offset" in default_block:
            start = float(default_block["start_offset"])

    # Layer 2: broadcaster-specific block
    if broadcaster_key and broadcaster_key in config:
        bcast_block = config[broadcaster_key]
        if "lag_offset" in bcast_block:
            lag = float(bcast_block["lag_offset"])
        if "start_offset" in bcast_block:
            start = float(bcast_block["start_offset"])
        print(
            f"point_builder: using calibrated offsets for broadcaster "
            f"'{broadcaster_key}' — lag={lag}s  start={start}s",
        )
    elif broadcaster_key:
        print(
            f"WARNING: Broadcaster key '{broadcaster_key}' not found in "
            "offsets config — falling back to 'default' block.",
            file=sys.stderr,
        )

    # Layer 1: explicit CLI flags override everything
    if explicit_lag is not None:
        lag = explicit_lag
    if explicit_start is not None:
        start = explicit_start

    return lag, start


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "point_builder — Convert a scoreboard event-stream CSV "
            "(one row per score change) into a point-by-point CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Event-stream CSV produced by scoreboard_ocr.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the point-by-point CSV.",
    )
    parser.add_argument(
        "--broadcaster-lag",
        type=float,
        default=None,          # None means "not explicitly set" — config wins
        dest="lag_offset",
        metavar="SECS",
        help=(
            "Seconds to subtract from the next event timestamp (Point_End). "
            "Overrides any value in --offsets-config. "
            f"Hardcoded default: {DEFAULT_LAG_OFFSET}s."
        ),
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        default=None,          # None means "not explicitly set" — config wins
        dest="start_offset",
        metavar="SECS",
        help=(
            "Seconds to add to the current event timestamp (Point_Start). "
            "Overrides any value in --offsets-config. "
            f"Hardcoded default: {DEFAULT_START_OFFSET}s."
        ),
    )
    parser.add_argument(
        "--offsets-config",
        default=None,
        dest="offsets_config",
        metavar="PATH",
        help=(
            "Path to the broadcaster offsets JSON config. "
            f"Auto-detected at '{DEFAULT_CONFIG_PATH}' if it exists."
        ),
    )
    parser.add_argument(
        "--broadcaster",
        default=None,
        dest="broadcaster",
        metavar="KEY",
        help=(
            "Broadcaster/clip key to look up in --offsets-config, "
            "e.g. 'test_vids/1'. Falls back to 'default' block if key missing."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Append a '_debug' column to the output CSV. Values: "
            "'OK', 'SERVER_FLIP_CORRECTED'. "
            "Invalid-transition and drift rows are always skipped (never emitted)."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point: ``python -m src.inference.point_builder``."""
    parser = _build_parser()
    args = parser.parse_args()

    # ── Resolve offsets config path ─────────────────────────────────────────
    if args.offsets_config is not None:
        config_path = args.offsets_config
    elif Path(DEFAULT_CONFIG_PATH).exists():
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = None

    config: dict[str, Any] = {}
    if config_path:
        config = load_offsets_config(config_path)

    lag_offset, start_offset = resolve_offsets(
        config=config,
        broadcaster_key=args.broadcaster,
        explicit_lag=args.lag_offset,
        explicit_start=args.start_offset,
    )

    # ── Load events ─────────────────────────────────────────────────────────
    try:
        events = load_events(args.input)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not events:
        print(
            f"ERROR: No valid event rows found in '{args.input}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    point_rows = events_to_points(
        events,
        lag_offset=lag_offset,
        start_offset=start_offset,
        debug=args.debug,
    )

    write_points(point_rows, args.output, debug=args.debug)

    print(
        f"point_builder: wrote {len(point_rows)} rows → {args.output} "
        f"(lag={lag_offset}s  start={start_offset}s"
        + ("  debug=ON" if args.debug else "")
        + ")"
    )


if __name__ == "__main__":
    main()
