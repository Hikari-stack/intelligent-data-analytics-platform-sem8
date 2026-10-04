import io

from core.ingest import load_file


def test_rows_with_extra_fields_are_skipped_not_fatal():
    csv_text = "id,name,city\n1,Ann,Pune\n2,Bob,Delhi,India,Extra\n3,Cy,Goa\n"
    df = load_file(io.BytesIO(csv_text.encode()), "x.csv")
    assert list(df["id"]) == [1, 3]
    assert df.attrs["skipped_lines"] == [3]


def test_clean_file_has_no_skipped_lines():
    df = load_file(io.BytesIO(b"a,b\n1,2\n3,4\n"), "x.csv")
    assert not df.attrs.get("skipped_lines")
