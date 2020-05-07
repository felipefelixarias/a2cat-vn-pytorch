import pickle
import os

file = os.path.join('/home','felipe', '.visual_navigation', 'scenes', 'thor-cached-212-174.pkl')
#file = os.path.join('../thor-cached-212-174.pkl')

def load_graph(file):
    
    if isinstance(file, str):
        with open(file, 'rb') as f:
            graph = pickle.load(f)
    else:
        graph = pickle.load(file)
    if not hasattr(graph, 'graph') or graph.graph is None:
        graph.graph, graph.optimal_actions = compute_shortest_path_data(graph.maze)
    return graph


with open(file, 'rb') as f:
        graph = load_graph(f)

# Load data
observations = graph._observations
depths = graph._depths
segmentations = graph._segmentations

import numpy as np
print(observations.shape)
print(np.max(observations))
print(np.min(observations))

print(depths.shape)
print(np.max(depths))
print(np.min(depths))

print(segmentations.shape)
print(np.max(segmentations))
print(np.min(segmentations))
