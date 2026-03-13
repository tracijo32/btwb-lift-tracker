from pandas import DataFrame
from pathlib import Path

def load_withings_data(path_to_files: Path) -> dict:
    import glob, json
    files = glob.glob(str(path_to_files / 'year=????' / 'month=??.json'))
    data = [d for f in files for d in json.load(open(f))]
    return data

def load_btwb_workout_data(path_to_files: Path) -> dict:
    import glob, json
    files = glob.glob(str(path_to_files / 'year=????' / 'month=??.json'))
    data = {k:v  for f in files for k,v in json.load(open(f)).items()}
    return data

def _validate_data_frame(
    df: DataFrame,
    dtypes: dict[str, str]
) -> DataFrame:
    
    required_columns = dtypes.keys()
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")
    
    out = df.reindex(columns=required_columns).copy()

    # Pandas doesn't support generic list dtypes like "list[string]" in astype().
    # Keep list-like columns as-is (typically object dtype) while still casting
    # everything else.
    castable: dict[str, str] = {}
    for col, dtype in dtypes.items():
        if isinstance(dtype, str) and dtype.strip().lower().startswith("list["):
            continue
        castable[col] = dtype

    if castable:
        out = out.astype(castable)

    return out


def lookup_nested_obs_var(
    obs_var: dict,
    *levels,
    default: float,
) -> float:
    """
    Lookup an observation variance from a nested dictionary.

    Semantics per level:
      - Exact match wins
      - Else 'all'
      - Else 'other' (catch-all for keys not explicitly assigned)
      - If the current value is not a dict, it is treated as applying to all
        remaining lower levels.
    """

    node = obs_var

    for level in levels:
        if not isinstance(node, dict):
            return float(node)

        candidates: list[object] = []

        if level is not None:
            candidates.append(level)
            # Be forgiving about reps keys: allow int <-> numeric str matches.
            if isinstance(level, int):
                candidates.append(str(level))
            elif isinstance(level, str) and level.isdigit():
                candidates.append(int(level))

        candidates.extend(["all", "other"])

        matched = False
        for k in candidates:
            if k in node:
                node = node[k]
                matched = True
                break

        if not matched:
            return float(default)

    # If there are no more levels but we still have a dict, interpret it as:
    # - explicit 'all' if present, else explicit 'other' if present
    # - if exactly one scalar value exists, treat it as the default for this branch
    if isinstance(node, dict):
        if "all" in node and not isinstance(node["all"], dict):
            return float(node["all"])
        if "other" in node and not isinstance(node["other"], dict):
            return float(node["other"])

        scalar_values = [v for v in node.values() if not isinstance(v, dict)]
        if len(scalar_values) == 1:
            return float(scalar_values[0])

        return float(default)

    return float(node)

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Convert '#RRGGBB' or '#RGB' (optionally without '#') to 'rgba(r,g,b,a)'.
    """
    s = hex_color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Expected 3 or 6 hex digits, got {hex_color!r}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"