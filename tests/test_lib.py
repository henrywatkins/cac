import pytest

from cac import cfeatures, one_hot_encode


def test_one_hot_encode():
    data = [
        ["apple", "cat", "cherry"],
        ["dog", "elephant", "fox"],
        ["green", "yellow", "whale"],
        ["car", "bus", "train"],
        ["orange", "orange", "purple"],
    ]
    result, classes = one_hot_encode(data)

    # Add your assertions here
    assert result.shape[0] == len(data)
    assert result.shape[1] == len(classes)
    assert result[0][0] == 1


def test_cfeatures():
    group_ids = [1, 2, 3, 1, 1]
    data = [
        ["apple", "banana", "cherry"],
        ["dog", "elephant", "fox"],
        ["green", "yellow", "blue"],
        ["car", "bus", "train"],
        ["red", "orange", "purple"],
    ]
    top_k = 1

    result = cfeatures(group_ids, data, top_k, show_values=True)

    # Add your assertions here
    assert isinstance(result, dict)
    assert len(result) == len(set(group_ids))
    assert result[1][0][0] == "train"
    assert result[2][0][1] == 1.0
