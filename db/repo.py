"""Save and load datasets and pipelines. Every function takes a database session."""
from pathlib import Path

import pandas as pd

from db.models import Dataset, Pipeline

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # columns that mix numbers and text cannot be stored as-is: store those as text
        safe = df.copy()
        for c in safe.columns:
            if safe[c].dtype == object:
                safe[c] = safe[c].map(lambda v: v if pd.isna(v) else str(v))
        safe.to_parquet(path, index=False)


def save_dataset(session, name: str, df: pd.DataFrame, upload_dir=UPLOAD_DIR) -> int:
    ds = Dataset(name=name, n_rows=len(df), n_cols=df.shape[1])
    session.add(ds)
    session.flush()                       # gives the new row its id
    path = Path(upload_dir) / f"{ds.id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, path)
    ds.file_path = str(path)
    session.commit()
    return ds.id


def load_dataset(session, dataset_id: int) -> pd.DataFrame:
    ds = session.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"No dataset with id {dataset_id}")
    return pd.read_parquet(ds.file_path)


def list_datasets(session) -> list[Dataset]:
    return session.query(Dataset).order_by(Dataset.created_at.desc()).all()


def save_pipeline(session, dataset_id: int, name: str, steps: list[dict]) -> int:
    pl = Pipeline(dataset_id=dataset_id, name=name, steps=steps)
    session.add(pl)
    session.commit()
    return pl.id


def list_pipelines(session, dataset_id: int) -> list[Pipeline]:
    return (session.query(Pipeline).filter_by(dataset_id=dataset_id)
            .order_by(Pipeline.created_at.desc()).all())


def load_pipeline_steps(session, pipeline_id: int) -> list[dict]:
    pl = session.get(Pipeline, pipeline_id)
    if pl is None:
        raise ValueError(f"No pipeline with id {pipeline_id}")
    return pl.steps
