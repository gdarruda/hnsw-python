import sys
from collections.abc import Iterable
from itertools import product
from math import sqrt


class Layer:

    def __init__(self):
        self.nodes: dict[str, list[float]] = {}
        self.edges: dict[str, list[str]] = {}

    @staticmethod
    def _distance(x: list[float], y: list[float]) -> float:
        return sqrt(sum([(x - y) ** 2 for x, y in zip(x, y)]))

    def _get_nearest(self, query: list[float], nodes: Iterable[str]) -> str:

        nearest = ("", sys.float_info.max)

        for node in nodes:

            node_distance = Layer._distance(query, self.nodes[node])
            _, smaller_distance = nearest

            if node_distance < smaller_distance:
                nearest = (node, node_distance)

        return nearest[0]

    def _get_furthest(self, query: list[float], nodes: Iterable[str]) -> str:

        furthest = ("", -sys.float_info.max)

        for node in nodes:

            node_distance = Layer._distance(query, self.nodes[node])
            _, bigger_distance = furthest

            if node_distance > bigger_distance:
                furthest = (node, node_distance)

        return furthest[0]

    def get_value(self, key: str) -> list[float]:
        return self.nodes[key]

    def get_neighbors(self, key: str) -> list[str]:
        return self.edges[key]

    def set_neighbors(self, key: str, neighbors: list[str]):
        self.edges[key] = neighbors

    def add_node(self, key: str, value: list[float]):
        self.nodes[key] = value
        self.edges[key] = []

    def connect(self, x: str, y: str):
        self.edges[x].append(y)
        self.edges[y].append(x)

    def select_neighbors(
        self, value: list[float], candidates: list[str], m: int
    ) -> list[str]:

        best_candidates = sorted(
            [
                (Layer._distance(v, self.nodes[c]), c)
                for v, c in product([value], candidates)
            ],
            key=lambda x: x[0],
        )[:m]

        return [c for _, c in best_candidates]

    def search(
        self, query: list[float], elements_to_return: int, entrypoint: str
    ) -> Iterable[str]:

        visted = set([entrypoint])
        candidates = set([entrypoint])
        nearest_neighbors = set([entrypoint])

        while len(candidates) > 0:

            nearest_node = self._get_nearest(query, candidates)
            candidates.remove(nearest_node)

            furthest_node = self._get_furthest(query, nearest_neighbors)

            if Layer._distance(query, self.nodes[nearest_node]) > Layer._distance(
                query, self.nodes[furthest_node]
            ):
                break

            for node in self.edges[nearest_node]:

                if node not in visted:

                    visted.add(node)
                    furthest_element = self._get_furthest(query, nearest_neighbors)

                    if (
                        Layer._distance(query, self.nodes[node])
                        < Layer._distance(query, self.nodes[furthest_element])
                        or len(nearest_neighbors) < elements_to_return
                    ):

                        candidates.add(node)
                        nearest_neighbors.add(node)

                        if len(nearest_neighbors) > elements_to_return:
                            nearest_neighbors.remove(furthest_element)

        return nearest_neighbors
