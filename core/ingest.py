"""Loading files into DataFrames with friendly errors."""
from __future__ import annotations

import csv
import io

import pandas as pd

MAX_MB = 100


class IngestError(Exception):
    """Raised with a message that is safe to show directly to the user."""


def _detect_sep(sample: str) -> str:
    """Comma, semicolon, tab or pipe. Falls back to a comma (e.g. for a one-column file)."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_text(raw: bytes) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            sep = _detect_sep(raw[:20000].decode(enc, errors="ignore"))
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise IngestError(f"Could not decode the file: {last_err}")


def _clean_columns(cols) -> list[str]:
    """Strip names, name blank columns, and make duplicates unique ('a', 'a' -> 'a', 'a.1')."""
    out, seen = [], {}
    for i, c in enumerate(cols, 1):
        name = str(c).strip() or f"column_{i}"
        if name in seen:
            seen[name] += 1
            new = f"{name}.{seen[name]}"
            while new in seen:
                seen[name] += 1
                new = f"{name}.{seen[name]}"
            name = new
        seen[name] = 0
        out.append(name)
    return out


def list_sheets(file_bytes: bytes) -> list[str]:
    """Names of the sheets in an Excel file."""
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    except Exception as e:
        raise IngestError(f"Could not read the Excel file: {e}") from e


def load_file(file, name: str | None = None, sheet: str | None = None) -> pd.DataFrame:
    """Load a CSV/TSV/TXT/Excel upload (file-like object or bytes).

    For Excel files, `sheet` picks the sheet by name (default: the first one)."""
    name = (name or getattr(file, "name", "") or "").lower()
    raw = file.read() if hasattr(file, "read") else file

    if not raw:
        raise IngestError("The file is empty.")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise IngestError(f"The file is larger than {MAX_MB} MB.")

    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet if sheet else 0)
        elif name.endswith((".csv", ".tsv", ".txt")):
            df = _read_text(raw)
        else:
            raise IngestError("Unsupported file type. Please upload CSV, TSV, TXT or Excel.")
    except IngestError:
        raise
    except Exception as e:  # pandas parser errors, bad Excel files, ...
        raise IngestError(f"Could not read the file: {e}") from e

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise IngestError("The file has no rows or no columns.")

    df.columns = _clean_columns(df.columns)
    return df
