import math

import pytest

from src.layer import Layer


@pytest.fixture
def layer():
    layer = Layer()

    layer.add_node("a", [1.0, 1.0])
    layer.add_node("b", [1.1, 1.1])
    layer.add_node("c", [-1.0, -1.0])

    layer.add_node("d", [7.0, 5.0])
    layer.add_node("e", [6.0, 4.0])
    layer.add_node("f", [6.0, 6.0])

    return layer


def test_distance(layer: Layer):

    x = [1.0, 1.0]
    y = [2.0, 3.0]

    expected = 2.2360679775
    assert math.isclose(expected, layer.distance(x, y))


def test_get_nearest(layer: Layer):

    assert layer._get_nearest([1.0, 1.0], layer.nodes.keys()) == "a"
    assert layer._get_nearest([7.1, 5.1], layer.nodes.keys()) == "d"


def test_get_furtherst(layer: Layer):

    assert layer._get_furthest([1.0, 1.0], layer.nodes.keys()) == "d"
    assert layer._get_furthest([7.1, 5.1], layer.nodes.keys()) == "c"


def test_get_value(layer: Layer):
    assert layer.get_value("a") == [1.0, 1.0]


def test_set_get_neighbors(layer: Layer):

    assert layer.get_neighbors("a") == []

    layer.set_neighbors("a", ["b", "c"])
    assert layer.get_neighbors("a") == ["b", "c"]


def test_add_node(layer: Layer):

    layer.add_node("z", [-1.0, -2.0])

    assert layer.get_neighbors("z") == []
    assert layer.get_value("z") == [-1.0, -2.0]


def test_connect(layer: Layer):

    layer.connect("a", "b")

    assert "a" in layer.get_neighbors("b")
    assert "b" in layer.get_neighbors("a")


def test_select_neighbors(layer: Layer):

    assert layer.select_neighbors([1.1, 1.1], ["a", "b", "c", "d"], 3) == [
        "b",
        "a",
        "c",
    ]

    assert layer.select_neighbors([7.1, 5.1], ["a", "b", "c", "d"], 2) == [
        "d",
        "b",
    ]


def test_select_neighbors_test(layer: Layer):

    layer.connect("a", "b")
    layer.connect("b", "e")
    layer.connect("d", "c")
    layer.connect("b", "c")

    neighbors = layer.select_neighbors_heuristic(
        [1.1, 1.1],
        ["a", "b", "c", "d"],
        2,
        extend_candidates=True,
        keep_pruned_connections=True,
    )

    assert "a" in neighbors
    assert "b" in neighbors


def test_search(layer: Layer):

    layer.connect("a", "b")
    layer.connect("b", "e")

    assert "e" in layer.search([6.1, 4.1], 2, "a")

    layer.connect("a", "f")

    assert "f" in layer.search([6.1, 4.1], 1, "a")

    layer.connect("c", "f")
    layer.connect("e", "c")
    layer.connect("b", "d")

    assert "d" in layer.search([6.1, 4.1], 3, "a")
    assert "e" in layer.search([6.1, 4.1], 2, "e")
    assert "d" in layer.search([7.0, 5.0], 1, "b")

    assert "d" in layer.search([6.1, 6.0], 1, "b")
    assert "f" in layer.search([6.1, 6.0], 5, "b")
