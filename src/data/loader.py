"""Robust dataset loading for CSV and Excel files."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or is unusable."""


@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    file_name: str
    file_type: str  # "csv" or "excel"
    n_rows: int
    n_cols: int
    warnings: list[str]


def _detect_file_type(file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    raise DatasetLoadError(
        f"Unsupported file type for '{file_name}'. Only .csv, .xlsx and .xls are supported."
    )


def load_dataset(file_obj: Union[io.BytesIO, "io.IOBase"], file_name: str) -> LoadResult:
    """Load an uploaded CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_obj: a file-like object (e.g. Streamlit's UploadedFile)
    file_name: the original file name, used to detect the file type.

    Raises
    ------
    DatasetLoadError on any failure, with a human-readable message.
    """
    warnings: list[str] = []
    file_type = _detect_file_type(file_name)

    try:
        if file_type == "csv":
            # Try a couple of common encodings/separators defensively.
            try:
                df = pd.read_csv(file_obj)
            except UnicodeDecodeError:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding="latin-1")
                warnings.append("File was not UTF-8 encoded; loaded using latin-1 fallback.")
            except pd.errors.ParserError:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep=None, engine="python")
                warnings.append("Standard comma parsing failed; auto-detected delimiter instead.")
        else:
            df = pd.read_excel(file_obj)
    except DatasetLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any parser failure clearly
        raise DatasetLoadError(f"Could not read '{file_name}': {exc}") from exc

    if df is None or df.shape[0] == 0:
        raise DatasetLoadError(f"'{file_name}' loaded but contains 0 rows.")
    if df.shape[1] == 0:
        raise DatasetLoadError(f"'{file_name}' loaded but contains 0 columns.")

    # Normalize column names: strip whitespace (do NOT rename/guess meaning here).
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[1] != len(set(df.columns)):
        warnings.append("Duplicate column names detected after trimming whitespace; "
                         "later columns may overwrite earlier ones during mapping.")

    return LoadResult(
        dataframe=df,
        file_name=file_name,
        file_type=file_type,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        warnings=warnings,
    )


def load_sample_dataset(path: str) -> pd.DataFrame:
    """Load the application's built-in sample dataset from disk."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DatasetLoadError(
            f"Built-in sample dataset not found at {path}. "
            "Run scripts/generate_sample_data.py to create it."
        ) from exc
