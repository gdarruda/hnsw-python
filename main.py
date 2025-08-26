import numpy as np

from hnsw import HNSW


def main():
    hnsw = HNSW(3, 3)
    hnsw.insert("a", [1, 1])
    hnsw.insert("b", [2, 1])
    hnsw.insert("c", [2, 2])
    hnsw.insert("d", [5, 5])
    hnsw.insert("e", [6, 5])
    hnsw.insert("f", [8, 3])


if __name__ == "__main__":
    main()
