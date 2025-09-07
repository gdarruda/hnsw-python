import sys
from collections.abc import Iterable
from itertools import product
from math import sqrt


class Layer:

    def __init__(self):
        self.nodes: dict[str, list[float]] = {}
        self.edges: dict[str, list[str]] = {}

    @staticmethod
    def distance(x: list[float], y: list[float]) -> float:
        return sqrt(sum([(x - y) ** 2 for x, y in zip(x, y)]))

    def _get_nearest(self, query: list[float], nodes: Iterable[str]) -> str:

        nearest = ("", sys.float_info.max)

        for node in nodes:

            node_distance = Layer.distance(query, self.nodes[node])
            _, smaller_distance = nearest

            if node_distance < smaller_distance:
                nearest = (node, node_distance)

        return nearest[0]

    def _get_furthest(self, query: list[float], nodes: Iterable[str]) -> str:

        furthest = ("", -sys.float_info.max)

        for node in nodes:

            node_distance = Layer.distance(query, self.nodes[node])
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

    def select_neighbors(self, key: str, candidates: list[str], m: int) -> list[str]:

        value = self.nodes[key]

        best_candidates = sorted(
            [
                (Layer.distance(v, self.nodes[c]), c)
                for v, c in product([value], candidates)
                if c != key
            ],
            key=lambda x: x[0],
        )

        return [c for _, c in best_candidates][:m]

    def select_neighbors_heuristic(
        self,
        key: str,
        candidates: list[str],
        m: int,
        extend_candidates: bool = True,
        keep_pruned_connections: bool = True,
    ) -> list[str]:

        value = self.nodes[key]
        neighbors = set([])
        working_candidates = set([c for c in candidates if c != key])

        if extend_candidates:
            for candidate in candidates:
                for candidate_neighbor in self.edges[candidate]:
                    if (
                        candidate_neighbor not in working_candidates
                        and candidate_neighbor != key
                    ):
                        working_candidates.add(candidate_neighbor)

        discarded_candidates = set([])

        while len(working_candidates) > 0 and len(neighbors) < m:

            nearest_wc = self._get_nearest(value, working_candidates)
            working_candidates.remove(nearest_wc)

            nearest_wc_distance = self.distance(self.nodes[nearest_wc], value)

            add_candidate = bool(
                sum(
                    [
                        nearest_wc_distance < self.distance(value, self.nodes[neighbor])
                        for neighbor in neighbors
                    ]
                )
            )

            if add_candidate or len(neighbors) == 0:
                neighbors.add(nearest_wc)
            else:
                discarded_candidates.add(nearest_wc)

        if keep_pruned_connections:
            while len(discarded_candidates) > 0 and len(neighbors) < m:
                nearest_candidate = self._get_nearest(value, discarded_candidates)
                discarded_candidates.remove(nearest_candidate)
                neighbors.add(nearest_candidate)

        return list(neighbors)

    def search(
        self, query: list[float], elements_to_return: int, entrypoint: str
    ) -> list[str]:

        visted = set([entrypoint])
        candidates = set([entrypoint])
        nearest_neighbors = set([entrypoint])

        while len(candidates) > 0:

            nearest_node = self._get_nearest(query, candidates)
            candidates.remove(nearest_node)

            furthest_node = self._get_furthest(query, nearest_neighbors)

            if Layer.distance(query, self.nodes[nearest_node]) > Layer.distance(
                query, self.nodes[furthest_node]
            ):
                break

            for node in self.edges[nearest_node]:

                if node not in visted:

                    visted.add(node)
                    furthest_element = self._get_furthest(query, nearest_neighbors)

                    if (
                        Layer.distance(query, self.nodes[node])
                        < Layer.distance(query, self.nodes[furthest_element])
                        or len(nearest_neighbors) < elements_to_return
                    ):

                        candidates.add(node)
                        nearest_neighbors.add(node)

                        if len(nearest_neighbors) > elements_to_return:
                            nearest_neighbors.remove(furthest_element)

        return list(nearest_neighbors)
