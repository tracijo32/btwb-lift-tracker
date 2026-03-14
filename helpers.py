from pandas import DataFrame
from pathlib import Path
import re
from typing import Any, Union


_MONTH_JSON_SUFFIX_RE = re.compile(r"(?:^|/)year=\d{4}/month=\d{2}\.json$")


def _is_gs_uri(s: str) -> bool:
    return s.startswith("gs://")


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """
    Returns (bucket, blob_name_or_prefix_without_leading_slash).
    Accepts:
      - gs://my-bucket
      - gs://my-bucket/some/prefix
      - gs://my-bucket/path/to/file.json
    """
    if not _is_gs_uri(uri):
        raise ValueError(f"Not a gs:// URI: {uri!r}")
    rest = uri[len("gs://") :]
    if not rest.strip():
        raise ValueError(f"Invalid gs:// URI (missing bucket): {uri!r}")
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def _iter_month_json_objects(source: Union[str, Path]) -> list[Any]:
    """
    Loads and returns parsed JSON objects from:
      - local directory containing year=####/month=##.json
      - local file path to a .json file
      - gs://bucket[/prefix] containing year=####/month=##.json
      - gs://bucket/path/to/file.json
    """
    import json

    src = str(source)
    if _is_gs_uri(src):
        bucket_name, blob_or_prefix = _parse_gs_uri(src)
        try:
            from google.cloud import storage  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Reading from GCS requires the optional dependency `google-cloud-storage`. "
                "Install it (e.g. `pip install google-cloud-storage`) and ensure "
                "Application Default Credentials are configured."
            ) from e

        client = storage.Client()

        # If the URI points at a single JSON file, load just that object.
        if blob_or_prefix and blob_or_prefix.lower().endswith(".json") and not blob_or_prefix.endswith("/"):
            blob = client.bucket(bucket_name).blob(blob_or_prefix)
            raw = blob.download_as_bytes()
            return [json.loads(raw)]

        # Otherwise treat as a prefix; list all matching month JSON blobs beneath it.
        prefix = (blob_or_prefix or "").lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        blobs = list(client.list_blobs(bucket_name, prefix=prefix))
        month_blobs = [b for b in blobs if _MONTH_JSON_SUFFIX_RE.search(getattr(b, "name", "") or "")]
        month_blobs.sort(key=lambda b: b.name)

        out: list[Any] = []
        for b in month_blobs:
            raw = b.download_as_bytes()
            out.append(json.loads(raw))
        return out

    # Local path handling
    p = Path(source)
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            return [json.load(f)]

    import glob

    files = sorted(glob.glob(str(p / "year=????" / "month=??.json")))
    out: list[Any] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out

def load_withings_data(path_to_files: Union[str, Path]) -> list[dict[str, Any]]:
    payloads = _iter_month_json_objects(path_to_files)
    # Withings monthly exports are lists; flatten them.
    data: list[dict[str, Any]] = []
    for month in payloads:
        if isinstance(month, list):
            data.extend([d for d in month if isinstance(d, dict)])
    return data

def load_btwb_workout_data(path_to_files: Union[str, Path]) -> dict[str, Any]:
    payloads = _iter_month_json_objects(path_to_files)
    # BTWB monthly exports are dicts keyed by day; merge them.
    data: dict[str, Any] = {}
    for month in payloads:
        if isinstance(month, dict):
            data.update(month)
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