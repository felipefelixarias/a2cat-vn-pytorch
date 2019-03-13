import os
from os.path import expanduser
from graph.util import load_graph, dump_graph

def thor_generator(scene, screen_size, goal, seed = 1):
    def _thunk():
        import ai2thor.controller
        import graph.thor_graph
        reconstructor = graph.thor_graph.GridWorldReconstructor(scene, screen_size = screen_size, seed = seed)
        graph = reconstructor.reconstruct()
        graph.goal = goal
        return graph
    return _thunk


graph_generators = {
    'kitchen-224': thor_generator('FloorPlan28', (224, 224,), (7, 0)),
    'kitchen-84': thor_generator('FloorPlan28', (84, 84,), (7, 0))
}

def _to_pascal(text):
    return ''.join(map(lambda x: x.capitalize(), text.split('-')))

def available_scenes():
    return [(_to_pascal(x), x) for x in graph_generators.keys()]

def get_graph(graph):
    home = expanduser("~")
    basepath = os.path.join(home, '.visual_navigation', 'scenes')
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    filename = os.path.join(basepath, '%s.pkl' % graph)
    if not os.path.exists(filename):
        graph = graph_generators.get(graph)()        
        with open(filename, 'wb+') as f:
            dump_graph(graph, f)
            f.flush()

    with open(filename, 'rb') as f:
        graph = load_graph(f)

    return graph
