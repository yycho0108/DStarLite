#!/usr/bin/env python3

import cv2
import numpy as np
import itertools
from typing import Hashable, Callable, Tuple
from collections import defaultdict, deque
# import heapq
from queue import PriorityQueue

from dataclasses import dataclass, field
from tqdm.auto import tqdm

#@dataclass(order=True)
#class DStarEntry:
#    key: Tuple[float,float]
#    item: Any=field(compare=False)


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
        self._compute_shortest_path(init, goal, on_expand)
        g = self._g

        s_start = init
        while s_start != goal:
            # Determine best next state.
            s_nexts = self._n(s_start)
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
                    u = tuple(u)
                    v = tuple(v)
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
                    self._compute_shortest_path(s_start, goal, on_expand)


class GridWorld2D:
    def __init__(self, m: np.ndarray, v_obs: int = 0):
        self.m = m
        self.v_obs = v_obs

        # NOTE(ycho): up-down-left-right
        # self.delta = list(itertools.product([-1, 1], repeat=2))
        self.delta = [[-1, 0],
                      [0, -1],
                      [0, 1],
                      [1, 0]]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]):
        # return np.linalg.norm((a[0] - b[0], a[1] - b[1]))
        return np.abs(np.subtract(a, b)).sum()

    def _prev(self, a: Tuple[int, int]):
        out = [(a[0] + d[0], a[1] + d[1]) for d in self.delta]
        # out-of-bounds
        out = [
            o for o in out if (
                0 <= o[0] and 0 <= o[1] and o[0] < self.m.shape[0] and o[1] < self.m.shape[1])]
        # obstacle
        out = [o for o in out if self.m[o[0], o[1]] != self.v_obs]
        return out

    def _next(self, a: Tuple[int, int]):
        out = [(a[0] + d[0], a[1] + d[1]) for d in self.delta]
        # out-of-bounds
        out = [
            o for o in out if (
                0 <= o[0] and 0 <= o[1] and o[0] < self.m.shape[0] and o[1] < self.m.shape[1])]
        # obstacle
        out = [o for o in out if self.m[o[0], o[1]] != self.v_obs]
        return out

    def _cost(self, a: Tuple[int, int], b: Tuple[int, int]):
        if np.any([
                np.less(a, 0).any(),
                np.greater_equal(a, self.m.shape).any(),
                np.less(b, 0).any(),
                np.greater_equal(b, self.m.shape).any()]):
            return 100000
        if self.m[b[0], b[1]] == self.v_obs:
            return 100000
        return np.linalg.norm((a[0] - b[0], a[1] - b[1]))


def create_updates(world,
        init,
        goal,
                   v_obs: int = 0,
                   p: float = 0.3):
    m = world.m

    # (1) apply random flips on the map with probability `p`.
    flip = np.random.choice([0,255], m.shape, replace=True, p=[
                            1 - p, p]).astype(np.uint8)
    if True:
        #flip[init[0]-10:init[0]+10,
        #        init[1]-10:init[1]+10] = False

        #flip[:init[0] - 10] = False
        #flip[init[0] + 10:] = False
        #flip[:, :init[1] - 10] = False
        #flip[:, init[1] + 10:] = False

        flip[:init[0] - 10] = False
        flip[init[0] + 10:] = False
        flip[:, :init[1] - 10] = False
        flip[:, init[1] + 10:] = False
    m2 = m ^ flip

    pos = np.argwhere(flip)

    #if True:
    #    pos - init

    #idx = np.where(m2)
    #print(np.shape(idx))

    #msk = (m[idx] == v_obs)

    ## (1) all `now-occupied` nodes will have edge costs=inf.
    #pos = np.transpose(idx)
    #pos[msk]

    out = {'edges': []}
    for p in pos:
        for d in world.delta:
            # src, dst, c_old
            out['edges'].append((p, p + d, world._cost(p, p + d)))
            out['edges'].append((p + d, p, world._cost(p + d, p)))

    # Update the map ...
    world.m = m2

    return out


def main():
    # m = np.random.choice([0, 1], size=(256, 256), replace=True)
    m = cv2.imread(
        '/home/jamiecho/Repos/NextBestView/src/opensearch/data/map.png',
        cv2.IMREAD_GRAYSCALE)

    factor = 2
    m = cv2.erode(m, np.ones(3), iterations=3)
    m = cv2.resize(
        m,
        dsize=None,
        fx=1 / factor,
        fy=1 / factor,
        interpolation=cv2.INTER_NEAREST)

    _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    world = GridWorld2D(m)

    init = (124 // factor, 107 // factor)
    goal = (276 // factor, 107 // factor)
    dstar = DStar(world._heuristic, world._prev, world._next, world._cost)

    # m2 = m.copy()
    vis = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)

    def on_expand(node: Hashable):
        # cv2.circle(vis, node[::-1], 1, (255, 0, 0))
        vis[node[0], node[1]] = (255, 0, 0)
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        cv2.imshow('vis', vis)
        cv2.waitKey(1)

    prev = init
    mod = 1
    for (action, updates) in dstar.plan(init, goal, on_expand):
        cv2.line(vis, init[::-1], action[::-1], (0, 0, 255),)
        cv2.imshow('vis', vis)
        cv2.waitKey(100)
        init = action
        if mod:
            us = create_updates(world, init, goal)
            updates.update(us)
            mod = False

            # NOTE(ycho): reset vis.
            vis = cv2.cvtColor(world.m, cv2.COLOR_GRAY2RGB)


if __name__ == '__main__':
    main()
