import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from graph.thor_graph import GridWorldReconstructor
from graph.util import dump_graph

graph = GridWorldReconstructor(screen_size = (84, 84)).reconstruct()
with open('./scenes/kitchen-84.pkl', 'wb+') as f:
    dump_graph(graph, f)