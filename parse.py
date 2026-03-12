from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_RE_WEIGHT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>lbs?|kg)\b", re.IGNORECASE
)
_RE_PERCENT_1RM = re.compile(r"(?P<pct>\d+(?:\.\d+)?)%\s*1RM\b", re.IGNORECASE)
_RE_SETS_MULT = re.compile(r"^\s*(?P<count>\d+)\s*x\s*", re.IGNORECASE)
_RE_REPS_AND_NAME = re.compile(r"^\s*(?P<reps>\d+)\s+(?P<name>.+?)\s*$")
_RE_REPS_TOKEN = re.compile(r"(?P<reps>\d+)\s*reps?\b", re.IGNORECASE)
_RE_DURATION_TOKEN = re.compile(
    r"^\s*(?P<name>.+?)\s*,\s*(?P<dur>\d+\s*:\s*\d{2}|\d+(?:\.\d+)?)\s*(?P<unit>secs?|seconds?|mins?|minutes?)?\s*$",
    re.IGNORECASE,
)

_KG_TO_LB = 2.2046226218
_SINGULAR_LAST_WORD = {
    # common BTWB plurals (extend as you encounter new ones)
    "cleans": "clean",
    "jerks": "jerk",
    "presses": "press",
    "squats": "squat",
    "snatches": "snatch",
}


def _singularize_last_word(word: str) -> str:
    """
    Best-effort singularization for movement names.
    Only operates on the *last* word token (already stripped of punctuation).
    """
    w = word.lower()
    repl = _SINGULAR_LAST_WORD.get(w)
    if repl:
        return repl

    # carries -> carry
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"

    # presses -> press, snatches -> snatch, boxes -> box, etc.
    if len(w) > 3 and w.endswith("es"):
        base = w[:-2]
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base

    # cleans -> clean, squats -> squat, jerks -> jerk
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]

    return w


@dataclass(frozen=True)
class Component:
    sequence_index: int
    reps: int
    name: str


@dataclass(frozen=True)
class ParsedSetLine:
    raw_line: str
    set_multiplier: int
    weight_value: Optional[float]
    weight_unit: Optional[str]
    target_pct_1rm: Optional[float]
    components: tuple[Component, ...]

    @property
    def is_complex(self) -> bool:
        return len(self.components) > 1

    @property
    def reps_expr(self) -> str:
        if not self.components:
            return ""
        if len(self.components) == 1:
            return str(self.components[0].reps)
        return "+".join(str(c.reps) for c in self.components)

    @property
    def total_reps(self) -> int:
        return sum(c.reps for c in self.components)

    @property
    def movement_label(self) -> str:
        if not self.components:
            return ""
        if len(self.components) == 1:
            return self.components[0].name
        return " + ".join(c.name for c in self.components)


def _parse_weight_token(token: str) -> tuple[Optional[float], Optional[str]]:
    m = _RE_WEIGHT.search(token)
    if not m:
        return None, None
    value = float(m.group("value"))
    unit = m.group("unit").lower()
    if unit == "lbs":
        unit = "lb"
    return value, unit


def _parse_reps_token(token: str) -> Optional[int]:
    m = _RE_REPS_TOKEN.search(token)
    if not m:
        return None
    return int(m.group("reps"))


def _parse_duration_seconds(text: str) -> Optional[int]:
    """
    Parses tokens like "30 secs", "1 min", "2:30" into seconds.
    Returns None if no recognizable duration is found.
    """
    s = text.strip().lower()
    if not s:
        return None

    m = re.fullmatch(r"(?P<mm>\d+)\s*:\s*(?P<ss>\d{2})", s)
    if m:
        return int(m.group("mm")) * 60 + int(m.group("ss"))

    m = re.fullmatch(r"(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>secs?|seconds?)", s)
    if m:
        return int(float(m.group("n")))

    m = re.fullmatch(r"(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>mins?|minutes?)", s)
    if m:
        return int(float(m.group("n")) * 60)

    # If unit is omitted, interpret as seconds (common in "Hollow Hold, 30")
    m = re.fullmatch(r"(?P<n>\d+(?:\.\d+)?)", s)
    if m:
        return int(float(m.group("n")))

    return None


def _to_lbs(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if unit is None:
        return None
    u = unit.lower()
    if u in {"lb", "lbs"}:
        return float(value)
    if u == "kg":
        return float(value) * _KG_TO_LB
    return None


def _normalize_movement_name(name: str) -> str:
    """
    Normalize movement naming so plural/singular variants group together.
    Currently focuses on last-word plurals like:
      - "Back Squats" -> "Back Squat"
      - "Hang Squat Snatches" -> "Hang Squat Snatch"
    """
    s = re.sub(r"\s{2,}", " ", name.strip())
    if not s:
        return s

    # Normalize unicode fraction glyphs to ASCII tokens.
    # Examples:
    #   "1¼ front squat"   -> "1 1/4 front squat"
    #   "1 ¼ front squat"  -> "1 1/4 front squat"
    #   "1 1¼ front squat" -> "1 1/4 front squat"  (sometimes appears in exports)
    s = (
        s.replace("¼", " 1/4")
        .replace("½", " 1/2")
        .replace("¾", " 3/4")
    )
    # Fix occasional duplication like "1 1 1/4" -> "1 1/4"
    s = re.sub(r"\b(\d+)\s+\1\s+(1/4|1/2|3/4)\b", r"\1 \2", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Convert "with X second pause" suffixes into a "pause <movement>" prefix.
    # Examples:
    #   "squat clean with 3 second pause" -> "pause squat clean"
    #   "squat clean with a 3 sec pause"  -> "pause squat clean"
    pause_flag = False
    s2, n = re.subn(
        r"\bwith\s+(?:a\s+)?\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:s|sec|secs|second|seconds)?\s*pause\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    if n:
        pause_flag = True
        s = s2

    # Drop parenthetical time annotations like:
    #   "(3sec)", "(3 secs)", "(3 seconds)", "(10s)", "(3sec)s"
    # and also trailing comma durations like:
    #   ", 30 secs", ", 1 min", ", 3-5sec"
    # This keeps movement names consistent for grouping.
    s = re.sub(
        r"\(\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:s|sec|secs|second|seconds)\s*\)\s*s?",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r",\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes)\b\.?\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s{2,}", " ", s).strip()
    if pause_flag and not re.search(r"\bpause\b", s, flags=re.IGNORECASE):
        s = f"pause {s}".strip()

    # Fix common typos / token variants before singularization.
    # Example: "fronts squats" -> "front squats"
    s = re.sub(r"\bfronts\b", "front", s, flags=re.IGNORECASE)

    tokens = s.split(" ")
    last = tokens[-1]

    m = re.match(r"^(?P<core>[A-Za-z]+)(?P<punct>[^A-Za-z]*)$", last)
    if not m:
        return s.lower()

    core = m.group("core")
    punct = m.group("punct")
    tokens[-1] = f"{_singularize_last_word(core)}{punct}"

    # Canonical form: lowercased for consistent grouping/joins.
    return " ".join(tokens).lower()


def _parse_target_pct_1rm(text: str) -> tuple[str, Optional[float]]:
    """
    Extracts trailing or embedded ", 60% 1RM" style tokens.
    Returns (cleaned_text, pct_float_or_none).
    """
    m = _RE_PERCENT_1RM.search(text)
    if not m:
        return text, None
    pct = float(m.group("pct"))
    cleaned = _RE_PERCENT_1RM.sub("", text)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned.strip())
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, pct


def _parse_rest_seconds(header_line: str) -> Optional[int]:
    """
    Parses BTWB header lines like:
    - "Sets : rest 3 mins"
    - "Sets : rest 2:30"
    - "Sets : rest 1 min"
    - "Sets"
    """
    s = header_line.strip().lower()
    if not s.startswith("sets"):
        return None
    if "rest" not in s:
        return None

    # rest 2:30
    m = re.search(r"rest\s+(?P<mm>\d+)\s*:\s*(?P<ss>\d+)", s)
    if m:
        return int(m.group("mm")) * 60 + int(m.group("ss"))

    # rest 3 mins / 1 min
    m = re.search(r"rest\s+(?P<n>\d+(?:\.\d+)?)\s*(?:mins?|minutes?)\b", s)
    if m:
        return int(float(m.group("n")) * 60)

    # rest 30 sec(s)
    m = re.search(r"rest\s+(?P<n>\d+(?:\.\d+)?)\s*(?:secs?|seconds?)\b", s)
    if m:
        return int(float(m.group("n")))

    return None


def _split_components(expr: str) -> list[str]:
    # Normalize spacing around "+"
    expr = re.sub(r"\s*\+\s*", " + ", expr.strip())
    return [p.strip() for p in expr.split(" + ") if p.strip()]


def _parse_components(expr: str) -> tuple[Component, ...]:
    parts = _split_components(expr)
    components: list[Component] = []
    for i, part in enumerate(parts, start=1):
        m = _RE_REPS_AND_NAME.match(part)
        if m:
            reps = int(m.group("reps"))
            name = _normalize_movement_name(m.group("name"))
        else:
            # Sometimes BTWB (or user input) omits reps; assume 1.
            reps = 1
            name = _normalize_movement_name(part)
        components.append(Component(sequence_index=i, reps=reps, name=name))
    return tuple(components)


def parse_set_line(line: str) -> ParsedSetLine:
    """
    Parses a single performed set line, e.g.
      - "5 Back Squats | 90 lbs"
      - "1x [ 1 Snatch Balance + 2 Overhead Squats ] | 40 lbs"
      - "1x [ 3 Snatch Push Press + 1 Overhead Squat ], 65% 1RM | 55 lbs"
      - "1 4 Front Squat + 8 Back Squat, 94% 1RM | 85 lbs"
      - "15 GHD Hip Extensions"  (unweighted / no barbell tracking)
    """
    raw = line.rstrip("\n")
    if "|" in raw:
        left, right = (p.strip() for p in raw.split("|", 1))
        weight_value, weight_unit = _parse_weight_token(right)
    else:
        # Some BTWB "Sets" blocks include bodyweight/accessory movements without recorded weight,
        # e.g. "15 GHD Hip Extensions". Treat as a valid set line with unknown weight.
        left, right = raw.strip(), ""
        weight_value, weight_unit = None, None

    left, target_pct = _parse_target_pct_1rm(left)

    set_multiplier = 1
    m = _RE_SETS_MULT.match(left)
    if m:
        set_multiplier = int(m.group("count"))
        left = left[m.end() :].strip()

    # Bracketed complex: [ 1 A + 2 B ]
    # Must be checked before the "movement name | weight" fallback.
    bracket_match = re.search(r"\[\s*(?P<inside>.+?)\s*\]", left)
    if bracket_match:
        inside = bracket_match.group("inside").strip()
        components = _parse_components(inside)
        return ParsedSetLine(
            raw_line=raw,
            set_multiplier=set_multiplier,
            weight_value=weight_value,
            weight_unit=weight_unit,
            target_pct_1rm=target_pct,
            components=components,
        )

    # Alternate BTWB format:
    #   "Squat Clean | 8 reps, 53 lbs"
    # i.e. movement name on the left, reps on the right.
    if not re.match(r"^\s*\d", left):
        reps_on_right = _parse_reps_token(right)
        if reps_on_right is not None:
            components = (
                Component(
                    sequence_index=1,
                    reps=reps_on_right,
                    name=_normalize_movement_name(left),
                ),
            )
            return ParsedSetLine(
                raw_line=raw,
                set_multiplier=set_multiplier,
                weight_value=weight_value,
                weight_unit=weight_unit,
                target_pct_1rm=target_pct,
                components=components,
            )

        # Another common format: movement name (no reps) + weight.
        # Example: "Squat Clean with 3 second pause | 100 lbs"
        # Assume 1 rep when reps aren't explicitly provided.
        if weight_value is not None:
            components = (
                Component(
                    sequence_index=1,
                    reps=1,
                    name=_normalize_movement_name(left),
                ),
            )
            return ParsedSetLine(
                raw_line=raw,
                set_multiplier=set_multiplier,
                weight_value=weight_value,
                weight_unit=weight_unit,
                target_pct_1rm=target_pct,
                components=components,
            )

        # No weight pipe AND no "8 reps" token on the right: interpret as "<reps> <movement>"
        # e.g. "15 GHD Hip Extensions"
        m = _RE_REPS_AND_NAME.match(left)
        if m:
            reps = int(m.group("reps"))
            name = _normalize_movement_name(m.group("name"))
            components = (Component(sequence_index=1, reps=reps, name=name),)
            return ParsedSetLine(
                raw_line=raw,
                set_multiplier=set_multiplier,
                weight_value=weight_value,
                weight_unit=weight_unit,
                target_pct_1rm=target_pct,
                components=components,
            )

        # Time-based holds/accessories like "Hollow Hold, 30 secs"
        m = _RE_DURATION_TOKEN.match(left)
        if m:
            name = _normalize_movement_name(m.group("name"))
            dur = m.group("dur")
            unit = (m.group("unit") or "").strip()
            duration_seconds = _parse_duration_seconds(f"{dur} {unit}".strip())
            # We don't currently model time separately; store as a single "attempt" rep count
            # so the line is parseable and downstream can filter on weight_lbs being null.
            reps = 1 if duration_seconds is None else 1
            components = (Component(sequence_index=1, reps=reps, name=name),)
            return ParsedSetLine(
                raw_line=raw,
                set_multiplier=set_multiplier,
                weight_value=weight_value,
                weight_unit=weight_unit,
                target_pct_1rm=target_pct,
                components=components,
            )

    # Unbracketed but multi-part: "1 4 Front Squat + 8 Back Squat"
    # This is effectively "1x [4 FS + 8 BS]" (one complex), so strip the leading "1 ".
    m = _RE_REPS_AND_NAME.match(left)
    if not m:
        raise ValueError(f"Unable to parse reps/name: {raw!r}")
    first_reps = int(m.group("reps"))
    remainder = m.group("name").strip()

    # If remainder looks like a complex expression, treat it as such and ignore the leading "1"
    # (or more generally, treat first_reps as the "complex count", not a movement rep count).
    if "+" in remainder and re.match(r"^\s*\d+\s+\S", remainder):
        components = _parse_components(remainder)
    else:
        components = (
            Component(
                sequence_index=1,
                reps=first_reps,
                name=_normalize_movement_name(remainder),
            ),
        )

    return ParsedSetLine(
        raw_line=raw,
        set_multiplier=set_multiplier,
        weight_value=weight_value,
        weight_unit=weight_unit,
        target_pct_1rm=target_pct,
        components=components,
    )


def parse_sets_description(
    description: str,
    *,
    date: Optional[str] = None,
    event_id: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Parses a full BTWB "Sets" description block into:
      1) a single metadata row dict
      2) expanded per-lift rows (each performed set expanded, each complex component expanded)
    """
    lines = [ln.strip() for ln in description.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.fullmatch(r"-{5,}", ln)]
    if not lines:
        raise ValueError("Empty description")

    header = lines[0]
    rest_seconds = _parse_rest_seconds(header)

    # Summary is typically last line, but not guaranteed to be present.
    summary_line = lines[-1] if len(lines) >= 2 else None
    set_lines = lines[1:] if header.lower().startswith("sets") else lines

    # If the last line looks like a BTWB summary (contains Rx'd or starts with Completed or "123 lbs | ..."),
    # exclude it from set parsing.
    reported_total_value: Optional[float] = None
    reported_total_unit: Optional[str] = None
    if set_lines:
        maybe_summary = set_lines[-1]
        if "rx" in maybe_summary.lower() and "|" in maybe_summary:
            # "2500 lbs | ... | Rx'd" or "Completed | ... | Rx'd"
            first_token = maybe_summary.split("|", 1)[0].strip()
            if re.match(r"^\d", first_token):
                reported_total_value, reported_total_unit = _parse_weight_token(first_token)
            summary_line = maybe_summary
            set_lines = set_lines[:-1]

    parsed_lines: list[ParsedSetLine] = []
    for ln in set_lines:
        if not ln or ln.lower().startswith("sets"):
            continue
        try:
            parsed_lines.append(parse_set_line(ln))
        except Exception as e:
            ctx = (
                "Failed to parse a set line.\n\n"
                f"date={date!r}\n"
                f"event_id={event_id!r}\n"
                f"line={ln!r}\n\n"
                "Full description:\n"
                f"{description}"
            )
            raise ValueError(ctx) from e

    # Expand performed sets and emit per-lift rows
    block_id = f"{date}::{event_id}" if date is not None or event_id is not None else None

    expanded_set_instances: list[ParsedSetLine] = []
    for p in parsed_lines:
        expanded_set_instances.extend([p] * max(1, p.set_multiplier))

    raw_set_lines_unexpanded = [p.raw_line for p in parsed_lines]
    raw_set_lines = [p.raw_line for p in expanded_set_instances]

    lift_rows: list[dict[str, Any]] = []
    for set_index, p in enumerate(expanded_set_instances, start=1):
        for c in p.components:
            lift_rows.append(
                {
                    "block_id": block_id,
                    "date": date,
                    "event_id": event_id,
                    "set_index": set_index,
                    "sequence_index": c.sequence_index,
                    "lift_name": c.name,
                    "reps": c.reps,
                    "weight": p.weight_value,
                    "unit": p.weight_unit,
                    "target_pct_1rm": p.target_pct_1rm,
                    "is_complex": p.is_complex,
                    "movement_label": p.movement_label,
                    "raw_set_line": p.raw_line,
                }
            )

    # Block-level aggregates
    weight_unit = next((p.weight_unit for p in expanded_set_instances if p.weight_unit), None)
    movement_label = next((p.movement_label for p in parsed_lines if p.movement_label), "")
    is_complex = any(p.is_complex for p in parsed_lines)
    target_pcts = [p.target_pct_1rm for p in expanded_set_instances if p.target_pct_1rm is not None]
    unique_target_pcts = sorted({float(x) for x in target_pcts})

    set_count = len(expanded_set_instances)
    reps_per_set = [p.reps_expr for p in expanded_set_instances]
    weights_per_set = [p.weight_value for p in expanded_set_instances]
    pcts_per_set = [p.target_pct_1rm for p in expanded_set_instances]

    total_set_weight = sum((p.weight_value or 0.0) for p in expanded_set_instances)
    total_volume = sum((p.weight_value or 0.0) * p.total_reps for p in expanded_set_instances)

    meta_row: dict[str, Any] = {
        "block_id": block_id,
        "date": date,
        "event_id": event_id,
        "description": description,
        "raw_set_lines": raw_set_lines,
        "raw_set_lines_unexpanded": raw_set_lines_unexpanded,
        "rest_seconds": rest_seconds,
        "movement_label": movement_label,
        "is_complex": is_complex,
        "set_count": set_count,
        "reps_per_set": reps_per_set,
        "weights_per_set": weights_per_set,
        "unit": weight_unit,
        "target_pct_1rm_per_set": pcts_per_set,
        "target_pct_1rm_unique": unique_target_pcts,
        # In BTWB exports, the numeric summary line sometimes matches `total_set_weight` (complexes)
        # and sometimes matches `total_volume` (simple lifts). For analysis, `total_volume` is usually
        # the most meaningful "total weight" (weight * reps).
        "total_weight": total_volume,
        "total_set_weight": total_set_weight,
        "total_volume": total_volume,
        "reported_total": reported_total_value,
        "reported_total_unit": reported_total_unit,
        "raw_header": header,
        "raw_summary": summary_line,
    }
    return meta_row, lift_rows


def parse_wl_sets_records(
    records: Iterable[dict[str, Any]],
    *,
    date_key: str = "date",
    event_id_key: str = "event_id",
    description_key: str = "description",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Parses the `wl_sets` structure in your notebook (list of dicts with date/event_id/description)
    into:
      - metadata_df: one row per BTWB "Sets" block
      - lifts_df: one row per performed set * per lift component
    """
    meta_rows: list[dict[str, Any]] = []
    lift_rows: list[dict[str, Any]] = []

    for rec in records:
        desc = rec.get(description_key)
        if not isinstance(desc, str) or not desc.strip():
            continue
        meta, lifts = parse_sets_description(
            desc,
            date=str(rec.get(date_key)) if rec.get(date_key) is not None else None,
            event_id=str(rec.get(event_id_key)) if rec.get(event_id_key) is not None else None,
        )
        meta_rows.append(meta)
        lift_rows.extend(lifts)

    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is required for parse_wl_sets_records(). "
            "Install it (e.g. `pip install pandas`) or call parse_sets_description() directly."
        ) from e

    metadata_df = pd.DataFrame(meta_rows)
    lifts_df = pd.DataFrame(lift_rows)
    return metadata_df, lifts_df


def parse_wl_blocks_and_sets(
    records: Iterable[dict[str, Any]],
    *,
    date_key: str = "date",
    event_id_key: str = "event_id",
    description_key: str = "description",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Two DataFrames:

    - blocks_df: **one row per event_id**, containing the raw `description` and `raw_set_lines` (list).
    - sets_df: **one row per performed set line** (expanded for `Nx ...`), linked by event_id and
      ordered by `sequence_id` (1..N within event).
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is required for parse_wl_blocks_and_sets(). Install it (e.g. `pip install pandas`)."
        ) from e

    block_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []

    for rec in records:
        desc = rec.get(description_key)
        if not isinstance(desc, str) or not desc.strip():
            continue

        date = str(rec.get(date_key)) if rec.get(date_key) is not None else None
        event_id = str(rec.get(event_id_key)) if rec.get(event_id_key) is not None else None

        meta, _ = parse_sets_description(desc, date=date, event_id=event_id)
        block_rows.append(meta)

        # Use the already-expanded raw_set_lines so sequence_id matches attempt order.
        for sequence_id, raw_line in enumerate(meta.get("raw_set_lines", []) or [], start=1):
            p = parse_set_line(raw_line)
            set_rows.append(
                {
                    "block_id": meta.get("block_id"),
                    "date": date,
                    "event_id": event_id,
                    "sequence_id": sequence_id,
                    "raw_set_line": raw_line,
                    "movement_label": p.movement_label,
                    "reps_expr": p.reps_expr,
                    "is_complex": p.is_complex,
                    "weight": p.weight_value,
                    "unit": p.weight_unit,
                    "target_pct_1rm": p.target_pct_1rm,
                }
            )

    blocks_df = pd.DataFrame(block_rows)
    sets_df = pd.DataFrame(set_rows)

    # Enforce one row per event_id in blocks_df (if input has accidental duplicates).
    if not blocks_df.empty and "event_id" in blocks_df.columns:
        blocks_df = (
            blocks_df.sort_values(by=["date", "event_id"], kind="stable", na_position="last")
            .drop_duplicates(subset=["event_id"], keep="first")
            .reset_index(drop=True)
        )

    return blocks_df, sets_df


def parse_wl_metadata_and_sets(
    records: Iterable[dict[str, Any]],
    *,
    date_key: str = "date",
    event_id_key: str = "event_id",
    description_key: str = "description",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    EXACTLY the two frames you requested:

    1) metadata_df (one row per event_id):
       - event_id, date, movements (sorted unique list), rest_seconds, description

    2) sets_df (one row per set line within each description, expanded for `Nx ...`):
       - event_id, sequence_id, movement_label, is_complex,
         movements (list), repetitions, repetitions_expr, target_pct_1rm, weight_lbs, raw_set_line
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is required for parse_wl_metadata_and_sets(). Install it (e.g. `pip install pandas`)."
        ) from e

    meta_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []

    for rec in records:
        desc = rec.get(description_key)
        if not isinstance(desc, str) or not desc.strip():
            continue

        date = str(rec.get(date_key)) if rec.get(date_key) is not None else None
        event_id = str(rec.get(event_id_key)) if rec.get(event_id_key) is not None else None

        meta, _ = parse_sets_description(desc, date=date, event_id=event_id)

        # Collect all movement components in this description (normalized), for metadata.
        component_names: set[str] = set()
        for raw_line in meta.get("raw_set_lines", []) or []:
            try:
                p = parse_set_line(raw_line)
                for c in p.components:
                    if c.name:
                        component_names.add(c.name)
            except Exception:
                # parse_sets_description would've already raised earlier if this line was bad.
                pass
        movements = sorted(component_names)

        meta_rows.append(
            {
                "event_id": event_id,
                "date": date,
                "movements": movements,
                "rest_seconds": meta.get("rest_seconds"),
                "description": desc,
            }
        )

        for sequence_id, raw_line in enumerate(meta.get("raw_set_lines", []) or [], start=1):
            p = parse_set_line(raw_line)
            set_rows.append(
                {
                    "event_id": event_id,
                    "sequence_id": sequence_id,
                    "movement_label": p.movement_label,
                    "movements": [c.name for c in p.components if c.name],
                    "is_complex": p.is_complex,
                    "repetitions": p.total_reps,
                    "repetitions_expr": p.reps_expr,
                    "target_pct_1rm": p.target_pct_1rm,
                    "weight_lbs": _to_lbs(p.weight_value, p.weight_unit),
                    "raw_set_line": raw_line,
                }
            )

    metadata_df = pd.DataFrame(meta_rows)
    sets_df = pd.DataFrame(set_rows)

    if not metadata_df.empty:
        metadata_df = metadata_df.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

    return metadata_df, sets_df


def parse_wl_sets_records_df(
    records: Iterable[dict[str, Any]],
    *,
    date_key: str = "date",
    event_id_key: str = "event_id",
    description_key: str = "description",
) -> "pd.DataFrame":
    """
    Single DataFrame output (long format): one row per performed set * per lift component,
    with block-level metadata repeated onto each row.

    If a block has no parseable lift rows (rare), it contributes a single row with the
    block-level columns populated and lift-level columns null.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is required for parse_wl_sets_records_df(). Install it (e.g. `pip install pandas`)."
        ) from e

    rows: list[dict[str, Any]] = []

    for rec in records:
        desc = rec.get(description_key)
        if not isinstance(desc, str) or not desc.strip():
            continue

        date = str(rec.get(date_key)) if rec.get(date_key) is not None else None
        event_id = str(rec.get(event_id_key)) if rec.get(event_id_key) is not None else None

        meta, lifts = parse_sets_description(desc, date=date, event_id=event_id)

        if not lifts:
            rows.append(
                {
                    **meta,
                    "set_index": None,
                    "sequence_index": None,
                    "lift_name": None,
                    "reps": None,
                    "weight": None,
                    "target_pct_1rm": None,
                    "raw_set_line": None,
                }
            )
            continue

        # drop these to avoid duplicates / conflicts with lift row columns
        meta_for_join = dict(meta)
        meta_for_join.pop("unit", None)
        meta_for_join.pop("movement_label", None)
        meta_for_join.pop("is_complex", None)

        for lr in lifts:
            # lift rows already include date/event_id/block_id and lift details.
            # Add block-level metadata that isn't already on the lift rows.
            rows.append({**meta_for_join, **lr})

    df = pd.DataFrame(rows)
    return df


def parse_wl_sets_records_sets_df(
    records: Iterable[dict[str, Any]],
    *,
    date_key: str = "date",
    event_id_key: str = "event_id",
    description_key: str = "description",
) -> "pd.DataFrame":
    """
    Single DataFrame output (set-grain): **one row per performed set**.

    - Expands e.g. `4x [ ... ]` into 4 rows (set_index increments).
    - For complexes, stores the sequence as list columns (`sequence_lift_names`, `sequence_reps`).
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is required for parse_wl_sets_records_sets_df(). Install it (e.g. `pip install pandas`)."
        ) from e

    set_rows: list[dict[str, Any]] = []

    for rec in records:
        desc = rec.get(description_key)
        if not isinstance(desc, str) or not desc.strip():
            continue

        date = str(rec.get(date_key)) if rec.get(date_key) is not None else None
        event_id = str(rec.get(event_id_key)) if rec.get(event_id_key) is not None else None

        # Reuse the robust parsing + error context in parse_sets_description()
        meta, _ = parse_sets_description(desc, date=date, event_id=event_id)

        # Re-parse set lines into per-set instances for 1-row-per-set output
        # (We could refactor parse_sets_description to return these, but this keeps the API stable.)
        lines = [ln.strip() for ln in desc.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not re.fullmatch(r"-{5,}", ln)]
        if not lines:
            continue

        header = lines[0]
        rest_seconds = _parse_rest_seconds(header)
        set_lines = lines[1:] if header.lower().startswith("sets") else lines

        # Drop BTWB summary line if present
        if set_lines:
            maybe_summary = set_lines[-1]
            if "rx" in maybe_summary.lower() and "|" in maybe_summary:
                set_lines = set_lines[:-1]

        parsed_lines: list[ParsedSetLine] = []
        for ln in set_lines:
            if not ln or ln.lower().startswith("sets"):
                continue
            try:
                parsed_lines.append(parse_set_line(ln))
            except Exception as e:
                ctx = (
                    "Failed to parse a set line.\n\n"
                    f"date={date!r}\n"
                    f"event_id={event_id!r}\n"
                    f"line={ln!r}\n\n"
                    "Full description:\n"
                    f"{desc}"
                )
                raise ValueError(ctx) from e

        expanded: list[ParsedSetLine] = []
        for p in parsed_lines:
            expanded.extend([p] * max(1, p.set_multiplier))

        for set_index, p in enumerate(expanded, start=1):
            seq_names = [c.name for c in p.components]
            seq_reps = [c.reps for c in p.components]
            total_reps = sum(seq_reps)
            set_rows.append(
                {
                    # block identifiers
                    "block_id": meta.get("block_id"),
                    "date": date,
                    "event_id": event_id,
                    # block-level useful context
                    "rest_seconds": rest_seconds,
                    "raw_header": meta.get("raw_header"),
                    "raw_summary": meta.get("raw_summary"),
                    # set-level
                    "set_index": set_index,
                    "is_complex": p.is_complex,
                    "movement_label": p.movement_label,
                    "reps_expr": p.reps_expr,
                    "total_reps": total_reps,
                    "weight": p.weight_value,
                    "unit": p.weight_unit,
                    "target_pct_1rm": p.target_pct_1rm,
                    "set_volume": (p.weight_value or 0.0) * total_reps,
                    "sequence_lift_names": seq_names,
                    "sequence_reps": seq_reps,
                    "raw_set_line": p.raw_line,
                    # optional block totals repeated (handy for joins/pivots)
                    "block_set_count": meta.get("set_count"),
                    "block_total_weight": meta.get("total_weight"),
                    "block_target_pct_1rm_unique": meta.get("target_pct_1rm_unique"),
                }
            )

    return pd.DataFrame(set_rows)


def parse_wl_sets_records_groups_df(
    records: Iterable[dict[str, Any]],
    *,
    date_key: str = "date",
    event_id_key: str = "event_id",
    description_key: str = "description",
) -> "pd.DataFrame":
    """
    Single DataFrame output (group-grain): **one row per group** within each BTWB "Sets" block.

    A "group" is defined as: same movement (movement_label) AND same reps pattern (reps_expr).
    Example: all "5 Back Squats | ..." attempts in a block become one group row.

    Adds `group_max_weight` as the max weight across attempts in that group.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "pandas is required for parse_wl_sets_records_groups_df(). Install it (e.g. `pip install pandas`)."
        ) from e

    sets_df = parse_wl_sets_records_sets_df(
        records,
        date_key=date_key,
        event_id_key=event_id_key,
        description_key=description_key,
    )
    if sets_df.empty:
        return sets_df

    # Ensure stable ordering so list-aggregations preserve attempt order.
    sets_df = sets_df.sort_values(
        by=["block_id", "set_index"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    # Normalize movement label for grouping robustness (case/whitespace).
    movement_norm = (
        sets_df["movement_label"]
        .astype("string")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    sets_df = sets_df.assign(group_movement_key=movement_norm, group_reps_key=sets_df["reps_expr"])

    def _list_nonnull(s: "pd.Series") -> list[Any]:
        return [x for x in s.tolist() if pd.notna(x)]

    group_cols = ["block_id", "date", "event_id", "group_movement_key", "group_reps_key"]
    grouped = (
        sets_df.groupby(group_cols, dropna=False, sort=False)
        .agg(
            rest_seconds=("rest_seconds", "first"),
            unit=("unit", "first"),
            is_complex=("is_complex", "first"),
            movement_label=("movement_label", "first"),
            reps_expr=("reps_expr", "first"),
            attempt_set_indices=("set_index", _list_nonnull),
            attempt_weights=("weight", _list_nonnull),
            group_set_count=("set_index", "count"),
            group_max_weight=("weight", "max"),
            group_min_weight=("weight", "min"),
            group_mean_weight=("weight", "mean"),
            group_last_weight=("weight", "last"),
            target_pct_1rm_unique=("target_pct_1rm", lambda s: sorted({float(x) for x in s.dropna().tolist()})),
            raw_set_lines=("raw_set_line", _list_nonnull),
            raw_header=("raw_header", "first"),
            raw_summary=("raw_summary", "first"),
            block_set_count=("block_set_count", "first"),
            block_total_weight=("block_total_weight", "first"),
        )
        .reset_index()
    )

    # Nicety: drop normalized movement key and keep original label.
    grouped = grouped.drop(columns=["group_movement_key"]).rename(
        columns={"group_reps_key": "group_reps_key"}
    )
    return grouped

def parse_weightlifting_sets_from_btwb_workout_data(
    data: dict,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Filter the data to only include weightlifting sets, then parse the workouts
    into two frames: metadata of each grouping of sets and the individual sets.
    """

    wl_data = [
        {
            'date': day,
            'event_id': event_id,
            'description': description
        }
        for day, events in data.items()
        for event_id, description in events.items()
        if description.startswith('Set')
    ]
    metadata, sets = parse_wl_metadata_and_sets(wl_data)
    return metadata, sets
