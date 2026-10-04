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


def _bad_line_numbers(raw: bytes, enc: str, sep: str) -> list[int]:
    """Line numbers of rows that have more fields than the header (e.g. an unquoted comma in a text value)."""
    reader = csv.reader(io.StringIO(raw.decode(enc, errors="replace"), newline=""), delimiter=sep)
    try:
        width = len(next(reader))
    except StopIteration:
        return []
    bad = []
    try:
        for row in reader:
            if len(row) > width:
                bad.append(reader.line_num)
    except csv.Error:
        pass
    return bad


def _is_number(v: str) -> bool:
    v = v.strip()
    if not v:
        return True
    try:
        float(v)
        return True
    except ValueError:
        return False


def _repair_rows(raw: bytes, enc: str, sep: str):
    """Fix rows that have extra fields because a text value contained the delimiter.

    For every too-long row, try merging each possible run of neighbouring fields back into one value and keep the
    merge that best fits the rows that parsed correctly (numeric columns stay numeric, categorical columns keep
    values already seen). Returns (DataFrame, [(file line, column name), ...]) or None if repair is not safe."""
    reader = csv.reader(io.StringIO(raw.decode(enc, errors="replace"), newline=""), delimiter=sep)
    try:
        header = next(reader)
    except StopIteration:
        return None
    width = len(header)
    rows = []
    try:
        for row in reader:
            rows.append((reader.line_num, row))
    except csv.Error:
        return None
    good = [r for _, r in rows if len(r) == width]
    bad = [(ln, r) for ln, r in rows if len(r) > width]
    if not bad or len(good) < 5 or len(bad) > 0.2 * len(rows):
        return None  # nothing to repair, too little evidence, or probably the wrong delimiter altogether

    numeric = []
    known = []
    multiword = []  # share of values holding several words: free-text columns are where stray commas appear
    for j in range(width):
        vals = [r[j].strip() for r in good if r[j].strip()]
        multiword.append(sum(" " in v for v in vals) / len(vals) if vals else 0.0)
        numeric.append(bool(vals) and sum(_is_number(v) for v in vals) / len(vals) >= 0.9)
        uniq = set(vals)
        known.append(uniq if (len(good) >= 20 and len(uniq) <= 50 and not numeric[j]) else set())

    fixed, repaired = {}, []
    for ln, row in bad:
        k = len(row) - width
        best, best_score, best_i = None, None, 0
        for i in range(width):
            cand = row[:i] + [sep.join(row[i:i + k + 1])] + row[i + k + 1:]
            score = 0.0
            for j, v in enumerate(cand):
                if numeric[j]:
                    score += 1 if _is_number(v) else -1
                elif known[j] and v.strip() in known[j]:
                    score += 1
            if numeric[i]:
                score -= 3  # never glue text into a numeric column
            score += 0.5 * multiword[i]  # tie-breaker: prefer free-text columns
            if best_score is None or score > best_score:
                best, best_score, best_i = cand, score, i
        fixed[ln] = best
        repaired.append((ln, str(header[best_i]).strip() or f"column_{best_i + 1}"))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for ln, row in rows:
        writer.writerow(fixed.get(ln, row))
    buf.seek(0)
    return pd.read_csv(buf), repaired


def _read_text(raw: bytes) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            sep = _detect_sep(raw[:20000].decode(enc, errors="ignore"))
            try:
                return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
            except pd.errors.ParserError:
                # Some rows have more fields than the header. First try to repair them automatically.
                fixed = _repair_rows(raw, enc, sep)
                if fixed is not None:
                    df, repaired = fixed
                    df.attrs["repaired_lines"] = repaired
                    return df
                # Not safely repairable: load everything else and report what was skipped.
                skipped = _bad_line_numbers(raw, enc, sep)
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
                df.attrs["skipped_lines"] = skipped
                return df
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
