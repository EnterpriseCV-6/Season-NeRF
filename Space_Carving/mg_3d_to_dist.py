import numpy as np
from maxflow import fastmin

def graph_cut(world, start=0, end = -1, height = 1/3):
    pts = np.arange(0, world.shape[2])
    pts = (np.abs(pts.reshape([-1, 1]) - pts.reshape([1, -1])))
    if end == -1:
        end = world.shape[2]-1
    slope = height / (end - start)
    pts = (pts - start) * slope
    pts[pts > height] = height
    pts[pts < 0] = 0

    # pts = np.linspace(0, 1, world.shape[2])
    # pts = (np.abs(pts.reshape([-1, 1]) - pts.reshape([1, -1])) / 3)
    # print(pts)
    adj = 0

    adj_cost = pts

    world_reshaped = world#np.reshape(world, [world.shape[0]*world.shape[1], world.shape[2]])
    init_energy_cost = fastmin.energy_of_grid_labeling(world_reshaped, adj_cost, np.argmin(world, 2))
    new_labels = fastmin.aexpansion_grid(world_reshaped, adj_cost, labels=np.argmin(world, 2))
    final_energy_cost = fastmin.energy_of_grid_labeling(world_reshaped, adj_cost, new_labels)
    return np.array(new_labels).reshape([world.shape[0], world.shape[1]]), init_energy_cost, final_energy_cost, adj

def greedy_H_map(H_map):
    return np.argmax(H_map, 2) / H_map.shape[2]

def expected_H_map(H_map, eps = 1e-8):
    return (np.sum(H_map * np.linspace(0, 1, H_map.shape[2]).reshape([1,1,H_map.shape[2]]), 2) + eps) / (np.sum(H_map, 2) + eps)

def energy_min_H_map(H_map, start = 0, end = -1, h = 1/3):
    world = H_map
    world_adj = -world
    world_adj -= np.min(world_adj)
    best_labels, init_energy, final_energy, adj = graph_cut(world_adj, start, end, h)
    print("Initial Energy:", init_energy)
    print("Final Energy:", final_energy)
    return best_labels / H_map.shape[2]

def scale(map, bounds_LLA):
    return map * 1. * (bounds_LLA[2,1] - bounds_LLA[2,0]) + bounds_LLA[2,0]