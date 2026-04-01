from pipeline import distribute_personas


def test_distribute_personas_equal_ratio():
    counts = distribute_personas(10, [1, 1])
    assert sum(counts) == 10
    assert counts[0] == counts[1]


def test_distribute_personas_3_to_1():
    counts = distribute_personas(20, [3, 1])
    assert sum(counts) == 20
    assert counts[0] > counts[1]


def test_distribute_personas_single_audience():
    counts = distribute_personas(15, [1])
    assert counts == [15]


def test_distribute_personas_minimum_one_per_audience():
    counts = distribute_personas(3, [10, 10, 10])
    assert all(c >= 1 for c in counts)
    assert sum(counts) == 3


def test_distribute_personas_three_audiences():
    counts = distribute_personas(20, [2, 1, 1])
    assert sum(counts) == 20
    assert counts[0] >= counts[1]
    assert counts[0] >= counts[2]
