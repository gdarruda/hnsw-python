import numpy as np

from layer import Layer


class HNSW:

    def __init__(self, m_max: int, ef_construction: int):
        self.layers: list[Layer] = []
        self.m_max = m_max
        self.ef_construction = ef_construction
        self.entrypoint: str | None = None

    @staticmethod
    def _get_level() -> int:
        p = np.random.randint(0, 2)
        return 1 + HNSW._get_level() if p == 1 else 0

    def _create_layer(self):

        layer = Layer(self.m_max)
        self.layers.append(layer)

    def insert(self, key: str, value: list[float]):

        max_level = len(self.layers) - 1
        level = HNSW._get_level()

        while max_level < level:
            self._create_layer()
            max_level = len(self.layers) - 1

        entrypoint = self.entrypoint

        for level in reversed(range(level, max_level)):
            entrypoint, *_ = self.layers[level].search(value, 1, entrypoint)

        for level in reversed(range(min(level, max_level) + 1)):

            layer = self.layers[level]
            layer.add_node(key, value)

            if len(layer.nodes) == 1:
                self.entrypoint = key
            else:
                neighbors = layer.select_neighbors(
                    value,
                    layer.search(value, self.ef_construction, entrypoint),
                    self.m_max,
                )

                for neighbor in neighbors:

                    layer.connect(key, neighbor)
                    neighbor_neighbors = layer.get_neighbors(neighbor)

                    if len(neighbor_neighbors) > self.m_max:
                        layer.set_neighbors(
                            neighbor,
                            layer.select_neighbors(
                                layer.get_value(key),
                                neighbor_neighbors,
                                layer.m_max,
                            ),
                        )
