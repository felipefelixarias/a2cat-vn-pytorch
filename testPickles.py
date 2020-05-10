import pickle
import os

file = os.path.join('~/.visual_navigation', 'scenes', 'thor-cached-212-174.pkl') #('thor-cached-208-174', [(6, 3, 1)])
#file = os.path.join('../thor-cached-208-174.pkl')

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


            tasks = [('thor-cached-212-174', [(10, 8, 2)]),
                ('thor-cached-227-174', [(10, 5, 0)]),
                ('thor-cached-301-174', [(2, 9, 0)]),
                ('thor-cached-308-174', [(5, 6, 3)])

# Load data
observations = graph._observations
depths = graph._depths
segmentations = graph._segmentations

import numpy as np
obs=observations[10][8][2]
print(obs.shape)
print(np.max(obs))
print(np.min(obs))

fp_obs = obs[:,:,:3]
tp_obs = obs[:,:,-3:]
print(fp_obs.shape)
print(tp_obs.shape)

from PIL import Image
img = Image.fromarray(fp_obs, 'RGB')
img.show()
img = Image.fromarray(tp_obs, 'RGB')
img.show()


print(observations.shape)
print(np.max(observations))
print(np.min(observations))

print(depths.shape)
print(np.max(depths))
print(np.min(depths))

print(segmentations.shape)
print(np.max(segmentations))
print(np.min(segmentations))

file = os.path.join('~/.visual_navigation', 'scenes', 'thor-cached-227-174.pkl') #('thor-cached-208-174', [(6, 3, 1)])
with open(file, 'rb') as f:
        graph = load_graph(f)

# Load data
observations = graph._observations
depths = graph._depths
segmentations = graph._segmentations

import numpy as np
obs=observations[10][5][0]
print(obs.shape)
print(np.max(obs))
print(np.min(obs))
print(observations.shape)
print(np.max(observations))
print(np.min(observations))
fp_obs = obs[:,:,:3]
tp_obs = obs[:,:,-3:]
print(fp_obs.shape)
print(tp_obs.shape)

from PIL import Image
img = Image.fromarray(fp_obs, 'RGB')
img.show()
img = Image.fromarray(tp_obs, 'RGB')
img.show()
print(depths.shape)
print(np.max(depths))
print(np.min(depths))

print(segmentations.shape)
print(np.max(segmentations))
print(np.min(segmentations))

file = os.path.join('~/.visual_navigation', 'scenes', 'thor-cached-308-174.pkl') #('thor-cached-208-174', [(6, 3, 1)])
with open(file, 'rb') as f:
        graph = load_graph(f)

# Load data
observations = graph._observations
depths = graph._depths
segmentations = graph._segmentations

import numpy as np
obs=observations[5][6][3]
print(obs.shape)
print(np.max(obs))
print(np.min(obs))
fp_obs = obs[:,:,:3]
tp_obs = obs[:,:,-3:]
print(fp_obs.shape)
print(tp_obs.shape)

from PIL import Image
img = Image.fromarray(fp_obs, 'RGB')
img.show()
img = Image.fromarray(tp_obs, 'RGB')
img.show()
print(observations.shape)
print(np.max(observations))
print(np.min(observations))

print(depths.shape)
print(np.max(depths))
print(np.min(depths))

print(segmentations.shape)
print(np.max(segmentations))
print(np.min(segmentations))

file = os.path.join('~/.visual_navigation', 'scenes', 'thor-cached-301-174.pkl') #('thor-cached-208-174', [(6, 3, 1)])
with open(file, 'rb') as f:
        graph = load_graph(f)

# Load data
observations = graph._observations
depths = graph._depths
segmentations = graph._segmentations

import numpy as np
obs=observations[2][9][0]
print(obs.shape)
print(np.max(obs))
print(np.min(obs))
fp_obs = obs[:,:,:3]
tp_obs = obs[:,:,-3:]
print(fp_obs.shape)
print(tp_obs.shape)

from PIL import Image
img = Image.fromarray(fp_obs, 'RGB')
img.show()
img = Image.fromarray(tp_obs, 'RGB')
img.show()
print(observations.shape)
print(np.max(observations))
print(np.min(observations))

print(depths.shape)
print(np.max(depths))
print(np.min(depths))

print(segmentations.shape)
print(np.max(segmentations))
print(np.min(segmentations))