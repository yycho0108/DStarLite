#!/usr/bin/env python3

import cv2
import numpy as np
import itertools
from typing import Hashable, Callable, Tuple
from collections import defaultdict, deque
import heapq
from queue import PriorityQueue

from dataclasses import dataclass, field
from tqdm.auto import tqdm

from grid_world import GridWorld2D, create_updates


class DStarQueue:
    def __init__(self):
        # self.q = []
        # self.q = PriorityQueue()
        self.q = {}

    def insert(self, node: Hashable, key: Tuple[float, float]):
        # heapq.heappush(self.q, (key, node))
        # self.q.put((key, node))
        # self.q.append((key, node))
        self.q[node] = key

    def update(self, node: Hashable, key: Tuple[float, float]):
        # heapq.heapreplace
        # self.q.u
        self.q[node] = key

    def remove(self, node: Hashable):
        self.q.pop(node)

    def top_key(self):
        return min(self.q.values())

    def top(self):
        node, key = min(self.q.items(), key=(lambda e: e[1]))
        return node

    def __contains__(self, node):
        return (node in self.q)

#class DStarQueue:
#    """
#    A* priority queue.
#    """
#
#    def __init__(self):
#        """
#
#        Args:
#            max_cost: The max cost, beyond which to suppress insertion.
#            weight:   The weight to apply to heuristic for weighted A*.
#        """
#        self._q = []
#
#    def insert(self, node: Hashable, key: Tuple[float, float]):
#        """Add an element to the priority queue."""
#        heapq.heappush(self._q, (key, node))
#
#    def pop(self) -> Tuple[NodeType, float, float]:
#        """Pop the top item and return that item."""
#        # NOTE(ycho): `heappop` pops the *smallest* item off the heap.
#        key, node = heapq.heappop(self._q)
#        return (x, g, f)
#
#    def top_key(self):
#        return self._q[0][0]
#
#    def top(self):
#        return self._q[0][1]
#
#    def __contains__(self, node):
#        # Why?
#        return (node in self._q)
#
#    def size(self):
#        """Size of the priority queue; |OpenSet|"""
#        return len(self._q)


class DStar:
    def __init__(self,
                 heuristic_fn: Callable[[Hashable, Hashable], float],
                 prev_fn: Callable[[Hashable], Tuple[Hashable, ...]],
                 next_fn: Callable[[Hashable], Tuple[Hashable, ...]],
                 cost_fn: Callable[[Hashable, Hashable], float]
                 ):
        self._h = heuristic_fn
        self._p = prev_fn
        self._n = next_fn
        self._c = cost_fn

        self._k_m = 0
        self._g = {}
        self._rhs = {}
        self._U = None

    def _key(self, ref: Hashable, node: Hashable):
        """Compute the lexicographical key by which nodes will be sorted."""

        g = self._g[node]
        rhs = self._rhs[node]

        k2 = min(g, rhs)
        k1 = k2 + self._h(ref, node) + self._k_m
        return (k1, k2)

    def _initialize(self, init: Hashable, goal: Hashable):

        k_m = 0
        inf = float('inf')
        rhs = defaultdict(lambda: inf)
        g = defaultdict(lambda: inf)
        rhs[goal] = 0.0
        U = DStarQueue()
        U.insert(goal, (self._h(init, goal), 0))

        # Reset state-specific variables.
        self._k_m = k_m
        self._rhs = rhs
        self._g = g
        self._U = U

    def _update_vertex(self, ref: Hashable, u: Hashable):
        # NOTE(ycho): Unpack state variables bound to class instance.
        rhs, g, U = self._rhs, self._g, self._U

        if g[u] != rhs[u]:
            key = self._key(ref, u)
            if u in U:
                U.update(u, key)
            else:
                U.insert(u, key)
        elif (u in U):
            U.remove(u)

    def _compute_shortest_path(self, ref: Hashable, goal: Hashable,
                               on_expand):
        # NOTE(ycho): Unpack state variables bound to class instance.
        rhs, g, U = self._rhs, self._g, self._U

        while (U.top_key() < self._key(ref, ref)) or (rhs[ref] != g[ref]):
            u = U.top()
            k_old = U.top_key()
            on_expand(u)
            k_new = self._key(ref, u)
            if k_old < k_new:
                # Update `u` which is top item of U
                # this can be done with heapreplace(...)
                U.update(u, k_new)
            elif g[u] > rhs[u]:
                # g > rhs, locally overconsistent
                g[u] = rhs[u]
                U.remove(u)

                for s in self._p(u):
                    if (s != goal):
                        rhs[s] = min(rhs[s], g[u] + self._c(s, u))
                    # Update `u` which is *not* top item of U
                    # can't be done with heapreplace(...)
                    self._update_vertex(ref, s)  # update `s`
            else:
                # locally underconsistent
                g_old = g[u]
                g[u] = np.inf
                for s in [u] + self._p(u):
                    if (rhs[s] == self._c(s, u) + g_old and s != goal):
                        rhs[s] = min([g[s1] + self._c(s, s1)
                                      for s1 in self._n(s)])
                    self._update_vertex(ref, s)  # update `s`

    def plan(self, init: Hashable, goal: Hashable,
             on_expand):
        s_last = init  # s_last, L#29
        self._initialize(init, goal)

        try:
            self._compute_shortest_path(init, goal, on_expand)
        except ValueError:
            yield None, None
            return
        g = self._g

        s_start = init
        while s_start != goal:
            # Determine best next state.
            s_nexts = self._n(s_start)
            if len(s_nexts) <= 0:
                yield (None, None)
                return
            costs = [self._c(s_start, s) + g[s] for s in s_nexts]
            s_start = s_nexts[np.argmin(costs)]

            # Option#1 : yield current action
            updates = {}
            yield (s_start, updates)

            # if any edge costs changed ... somehow.
            # What would this mean in our case?
            # --> a new object has appeared; meaning
            # 1) the expected revealed volume has changed, and
            # 2) a particular CCG(?) has acquired a new object
            self._k_m += self._h(s_last, s_start)
            s_last = s_start

            # FIXME(ycho): consider graph-level transforms!
            # I guess it doesn't happen every day, but it could happen
            # self._g, self._rhs, self._U = updates['transform'](
            # self._g, self._rhs, self._U)

            if 'edges' in updates:
                for (u, v, c_old) in tqdm(updates['edges']):
                    # NOTE(ycho): assume that after `updates`,
                    # self._c() will always compute the updated cost function.
                    # Since we're operating over an implicitly defined graph,
                    # we'd have to make this assumption.
                    # u = tuple(u)
                    # v = tuple(v)
                    c_new = self._c(u, v)
                    if c_old > c_new:
                        if u != goal:
                            self._rhs[u] = min(
                                self._rhs[u],
                                c_new + self._g[v])
                    elif self._rhs[u] == c_old + self._g[v]:
                        if u != goal:
                            self._rhs[u] = min(
                                (self._c(u, s) + self._g[s] for s in
                                 self._n(u)),
                                default=np.inf)
                    self._update_vertex(s_start, u)

                if len(updates) > 0:
                    # TODO(ycho): what if goal also changed?
                    # (does it happen?)
                    try:
                        self._compute_shortest_path(s_start, goal, on_expand)
                    except ValueError:
                        yield None, None
                        return
