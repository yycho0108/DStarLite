#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Hashable
from dstarlite import DStar, DStarQueue
# from grid_world import GridWorld2D, create_updates
from graph_world import SpatialGraphWorld, _generate_graph, create_updates
import networkx as nx
from matplotlib import pyplot as plt


def main():
    draw:bool = True

    # seed: int = 9263
    seed = None
    if seed is None:
        pass

    found = False
    while not found:
        seed = np.random.randint(2**16 - 1)
        # seed = 9910
        # seed:int=10230
        # seed:int=59715
        # seed:int = 36998
        seed:int = 44778
        print(F'seed={seed}')

        rng = np.random.default_rng(seed)

        pos, edges = _generate_graph(rng, n=16, p=0.1)
        world = SpatialGraphWorld(pos, edges)
        dstar = DStar(world._heuristic, world._prev, world._next, world._cost)

        # FIXME(ycho): select `init`/`goal` from a single
        # connected component inside this graph.
        init, goal = rng.choice(len(pos), 2, replace=False)

        def on_expand(*args, **kwds):
            pass

        print(F'From {init}')
        print(F'To   {goal}')
        # print(F'edges={edges}')

        mod = 1
        it = 0

        if draw:
            #nx.draw_networkx(world._G, pos=world._pos[...,:2],
            #        edge_color='r', style='dotted')
            nx.draw_networkx_nodes(world._G, pos=world._pos[...,:2])
            nx.draw_networkx_labels(world._G, pos=world._pos[...,:2])
            nx.draw_networkx_edges(world._G, pos=world._pos[...,:2] - 0.01,
                    edge_color='r', style='dashed')

        actions = [init]
        for (action, updates) in dstar.plan(init, goal, on_expand):
            if action is None:
                found = False
                break
            it += 1
            print('action', action)
            if action >= 16:
                found = True
            actions.append(action)
            init = action
            if mod:
                us = create_updates(rng, world, init, goal)
                # print('updates', us)
                updates.update(us)
                mod = False
                if draw:
                    nx.draw_networkx_nodes(world._G, pos=world._pos[...,:2])
                    nx.draw_networkx_labels(world._G, pos=world._pos[...,:2])
                    nx.draw_networkx_edges(world._G, pos=world._pos[...,:2] + 0.01,
                            edge_color='b', style='dashed')
            if it > 200:
                found = False
                break
        found = (found and it >= 6)
        if found:
            if draw:
                nx.draw_networkx_edges(world._G, pos=world._pos[...,:2],
                        edge_color='k', width=2, style='solid', edgelist=list(zip(actions[:-1], actions[1:])))
                plt.axis('equal')
                plt.show()
            break


if __name__ == '__main__':
    main()
