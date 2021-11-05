#!/usr/bin/env python3

import random
import numpy as np
from typing import (TypeVar, Hashable, Tuple)
import networkx as nx

NT = TypeVar('Node', bound=Hashable)


class SpatialGraphWorld:
    def __init__(self,
                 pos: np.ndarray,
                 edges: Tuple[Tuple[int, int], ...]):
        self._pos = pos
        self._edges = edges
        G = nx.Graph()
        for i in range(len(pos)):
            G.add_node(i)
        for i, j in edges:
            G.add_edge(i, j)
        self._G = G

    def _prev(self, a: NT) -> Tuple[NT, ...]:
        return list(self._G.neighbors(a))

    def _next(self, a: NT) -> Tuple[NT, ...]:
        return list(self._G.neighbors(a))

    def _cost(self, a: NT, b: NT) -> float:
        return np.linalg.norm(self._pos[a] - self._pos[b])

    def _heuristic(self, a: NT, b: NT) -> float:
        return np.linalg.norm(self._pos[a] - self._pos[b])


def _generate_graph(rng: np.random.Generator, n: int = 8,
                    p: float = 0.1):
    # Make a sensible looking thing with
    # spatial graph, i suppose...
    # -> (1) generate spatial coordinates
    pos = rng.uniform(size=(n, 3))

    # Make a sensible list of allowed edges.
    edges = rng.choice(2, size=(n, n), replace=True, p=[1 - p, p])
    ii = np.arange(n)
    edges[ii, ii] = 0  # no self loops
    edges = np.argwhere(edges)

    return (pos, edges)


def add_and_remove_edges(rng: np.random.Generator,
                         G, p_new_connection, p_remove_connection):
    """for each node, add a new connection to random other node, with prob
    p_new_connection, remove a connection, with prob p_remove_connection.

    operates on G in-place

    Reference:
        https://stackoverflow.com/a/42653330
    """
    new_edges = []
    rem_edges = []

    for node in G.nodes():
        # find the other nodes this one is connected to
        connected = [to for (fr, to) in G.edges(node)]
        # and find the remainder of nodes, which are candidates for new edges
        unconnected = [n for n in G.nodes() if not n in connected]

        # probabilistically add a random edge
        if len(unconnected):  # only try if new edge is possible
            if rng.uniform() < p_new_connection:
                new = rng.choice(unconnected)
                G.add_edge(node, new)
                # print "\tnew edge:\t {} -- {}".format(node, new)
                new_edges.append((node, new))
                # book-keeping, in case both add and remove done in same cycle
                unconnected.remove(new)
                connected.append(new)

        # probabilistically remove a random edge
        if len(connected):  # only try if an edge exists to remove
            if rng.uniform() < p_remove_connection:
                remove = rng.choice(connected)
                G.remove_edge(node, remove)
                # print "\tedge removed:\t {} -- {}".format(node, remove)
                rem_edges.append((node, remove))
                # book-keeping, in case lists are important later?
                connected.remove(remove)
                unconnected.append(remove)
    return rem_edges, new_edges


def create_updates(rng: np.random.Generator,
                   world: SpatialGraphWorld,
                   init: int,
                   goal: int):
    i0 = np.max(list(world._G.nodes)) + 1
    di = rng.integers(16)
    # print(F'di={di}')
    for i in range(i0, i0 + di):
        # print(F'new_node = {i}')
        world._G.add_node(i)
    world._pos = np.concatenate([
        world._pos,
        rng.uniform(size=(di, world._pos.shape[-1]))])
    rem_edges, new_edges = add_and_remove_edges(rng,
                                                world._G,
                                                0.8,0.5
                                                )
    out = {}
    out['edges'] = []
    for u, v in rem_edges:
        out['edges'].append((u, v, world._cost(u, v)))
        out['edges'].append((v, u, world._cost(v, u)))

    for u, v in new_edges:
        out['edges'].append((u, v, np.inf))
        out['edges'].append((v, u, np.inf))

    return out


def main():
    rng = np.random.default_rng(0)
    pos, edges = _generate_graph(rng)
    world = SpatialGraphWorld(pos, edges)

    n = len(pos)
    for i in range(n):
        print(world._next(i))

    for i, j in rng.integers(0, n, size=(1, 2)):
        print(world._heuristic(i, j))

    for i, j in rng.integers(0, n, size=(1, 2)):
        print(world._cost(i, j))

    print(create_updates(rng, world, 0, 0))

    ## NOTE(ycho): List of (u, v, w_uv)
    #adjlist = [
    #    (0, 1, 0.2),
    #    (1, 2, 0.2),
    #    (1, 3, 0.3),
    #    (3, 4, 0.3),
    #    (2, 4, 0.3)
    #]


if __name__ == '__main__':
    main()
