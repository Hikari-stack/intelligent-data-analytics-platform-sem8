"""Database tests run on a throwaway SQLite database, so MySQL is not needed to run them."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.clean import replay
from db.models import init_db
from db.repo import (list_datasets, list_pipelines, load_dataset, load_pipeline_steps,
                     save_dataset, save_pipeline)


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with sessionmaker(bind=engine)() as s:
        yield s


def test_dataset_round_trip(session, tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"], "c": [1.5, np.nan, 3.5]})
    ds_id = save_dataset(session, "demo.csv", df, upload_dir=tmp_path / "up")
    back = load_dataset(session, ds_id)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    row = list_datasets(session)[0]
    assert (row.name, row.n_rows, row.n_cols) == ("demo.csv", 3, 3)


def test_mixed_type_column_can_be_saved(session, tmp_path):
    df = pd.DataFrame({"code": [10, "abc", 20, "xyz"]})      # numbers and text in one column
    ds_id = save_dataset(session, "mixed.xlsx", df, upload_dir=tmp_path / "up")
    assert len(load_dataset(session, ds_id)) == 4


def test_pipeline_round_trip_and_replay(session, tmp_path):
    df = pd.DataFrame({"Age": [20.0, None, 40.0, 30.0]})
    ds_id = save_dataset(session, "ages", df, upload_dir=tmp_path / "up")
    steps = [{"op": "impute", "col": "Age", "method": "median", "reason": "demo"}]
    pl_id = save_pipeline(session, ds_id, "fill ages", steps)
    assert list_pipelines(session, ds_id)[0].name == "fill ages"
    cleaned = replay(load_dataset(session, ds_id), load_pipeline_steps(session, pl_id))
    assert cleaned["Age"].isna().sum() == 0


def test_unknown_ids_give_clear_errors(session):
    with pytest.raises(ValueError):
        load_dataset(session, 999)
    with pytest.raises(ValueError):
        load_pipeline_steps(session, 999)
