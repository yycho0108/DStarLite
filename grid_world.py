#!/usr/bin/env python3

import numpy as np
import cv2
import itertools
from typing import Hashable, Callable, Tuple
from dataclasses import dataclass, field


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
    flip = np.random.choice([0, 255], m.shape, replace=True, p=[
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
            u = tuple(p)
            v = tuple(p+d)
            out['edges'].append((u, v, world._cost(p, p + d)))
            out['edges'].append((v, u, world._cost(p + d, p)))

    # Update the map ...
    world.m = m2

    return out
