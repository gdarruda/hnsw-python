import uuid

import numpy as np
import pytest

from hnsw import HNSW


@pytest.fixture
def hnsw():
    return HNSW(5, 3, 1.0)


def test_get_level(hnsw: HNSW):
    assert 0 <= hnsw._get_level() <= 100


def test_create_layer(hnsw: HNSW):

    for _ in range(10):
        hnsw._create_layer()

    assert len(hnsw.layers) == 10


def test_insert(hnsw: HNSW):

    num_samples = 1_000
    samples = np.random.rand(1_000, 3).tolist()

    for sample in samples:
        hnsw.insert(str(uuid.uuid4()), sample)

    assert len(hnsw.layers) > 0

    for i, layer in enumerate(hnsw.layers):

        if i == 0:
            assert len(layer.edges) == num_samples
            assert len(layer.nodes) == num_samples
        else:
            assert len(layer.edges) <= num_samples
            assert len(layer.nodes) <= num_samples

        for key, values in layer.edges.items():
            assert key not in values
            assert 0 <= len(values) <= 5
