import numpy as np

from src.layer import Layer


class HNSW:

    def __init__(self, m_max: int, m_max0: int, ef_construction: int, m_l: float):
        self.layers: list[Layer] = []
        self.m_max = m_max
        self.m_max0 = m_max0
        self.m_l = m_l
        self.ef_construction = ef_construction
        self.entrypoint: str | None = None

    def _get_level(self) -> int:
        return int(np.floor(-np.log(np.random.uniform()) * self.m_l))

    def _create_layer(self):
        self.layers.append(Layer())

    def insert(self, key: str, value: list[float]):

        max_level = len(self.layers) - 1
        level = self._get_level()

        while max_level < level:
            self._create_layer()
            max_level = len(self.layers) - 1

        entrypoint: str = self.entrypoint

        for level in reversed(range(level, max_level)):
            entrypoint, *_ = self.layers[level].search(value, 1, entrypoint)

        for level in reversed(range(min(level, max_level) + 1)):

            layer = self.layers[level]
            layer.add_node(key, value)

            if len(layer.nodes) == 1:
                self.entrypoint = key
            else:
                neighbors = layer.select_neighbors_heuristic(
                    key,
                    layer.search(value, self.ef_construction, entrypoint),
                    self.m_max,
                )

                for neighbor in neighbors:

                    layer.connect(key, neighbor)
                    neighbor_neighbors = layer.get_neighbors(neighbor)

                    m_max = self.m_max0 if level == 0 else self.m_max

                    if len(neighbor_neighbors) > m_max:
                        layer.set_neighbors(
                            neighbor,
                            layer.select_neighbors_heuristic(
                                neighbor,
                                neighbor_neighbors,
                                m_max,
                            ),
                        )

    def search(
        self, query: list[float], k: int, ef_search: int
    ) -> list[tuple[str, list[float]]]:

        if self.entrypoint is None:
            return []

        entrypoint = self.entrypoint

        for layer in reversed(self.layers):
            keys = layer.search(query, ef_search, entrypoint)
            entrypoint, _ = min(
                [(key, Layer.distance(query, layer.nodes[key])) for key in keys],
                key=lambda x: x[1],
            )

        layer, *_ = self.layers
        keys = layer.search(query, ef_search, entrypoint)

        k_keys = sorted(
            [(key, Layer.distance(query, layer.nodes[key])) for key in keys],
            key=lambda x: x[1],
        )[:k]

        return [(key, layer.nodes[key]) for key, _ in k_keys]
