#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Hashable
from dstarlite import DStar, DStarQueue
from grid_world import GridWorld2D, create_updates


def main():
    # m = np.random.choice([0, 1], size=(256, 256), replace=True)
    m = cv2.imread(
        '/home/jamiecho/Repos/Ravel/objsearch/opensearch/src/opensearch/data/map.png',
        cv2.IMREAD_GRAYSCALE)

    factor = 2
    # m = cv2.erode(m, np.ones(3), iterations=9)
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
    cv2.circle(vis, init[::-1], 3, (255, 0, 0), -1)  # blue=init
    cv2.circle(vis, goal[::-1], 3, (0, 0, 255), -1)  # red=goal

    def on_expand(node: Hashable):
        # cv2.circle(vis, node[::-1], 1, (255, 0, 0))
        vis[node[0], node[1]] = (0, 255, 0)

        cv2.circle(vis, init[::-1], 3, (255, 0, 0), -1)  # blue=init
        cv2.circle(vis, goal[::-1], 3, (0, 0, 255), -1)  # red=goal

        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        cv2.imshow('vis', vis)
        cv2.waitKey(1)

    prev = init
    mod = 1
    for (action, updates) in dstar.plan(init, goal, on_expand):
        cv2.line(vis, init[::-1], action[::-1], (128, 0, 128),)
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
