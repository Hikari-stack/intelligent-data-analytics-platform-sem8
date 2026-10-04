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


def _big_csv(extra_row: str) -> bytes:
    lines = ["id,name,city,amount"]
    cities = ["Pune", "Delhi", "Goa"]
    for i in range(1, 31):
        lines.append(f"{i},Person{i},{cities[i % 3]},{i * 10}")
    lines.append(extra_row)
    return ("\n".join(lines) + "\n").encode()


def test_unquoted_comma_in_text_is_repaired_automatically():
    df = load_file(io.BytesIO(_big_csv("31,Smith, John,Pune,500")), "x.csv")
    assert len(df) == 31
    row = df[df["id"] == 31].iloc[0]
    assert row["name"] == "Smith, John" and row["city"] == "Pune" and row["amount"] == 500
    assert df.attrs["repaired_lines"] == [(32, "name")]
    assert not df.attrs.get("skipped_lines")


def test_comma_in_last_text_column_is_merged_into_that_column():
    notes = ["arrived on time", "box was dented", "customer was happy", "needs a follow up"]
    csv_bytes = ("id,name,notes\n" + "\n".join(f"{i},N{i},{notes[i % 4]}" for i in range(1, 25))
                 + "\n25,N25,late, and damaged\n").encode()
    df = load_file(io.BytesIO(csv_bytes), "x.csv")
    assert df.iloc[-1]["notes"] == "late, and damaged"
    assert df.iloc[-1]["name"] == "N25"
    assert df.dtypes["id"].kind in "iu"
