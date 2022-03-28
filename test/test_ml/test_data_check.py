def test_column_names(data):

    expected_columns = [
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "hours-per-week",
            "native-country",
            "salary"
    ]

    these_columns = data.columns.values

    assert list(expected_columns) == list(these_columns)
