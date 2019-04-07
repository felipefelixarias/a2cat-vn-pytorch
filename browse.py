import environments

if __name__ == '__main__':
    #from graph.util import load_graph
    #env = environments.make('AuxiliaryGraph-v0', graph_file = '/home/jonas/.visual_navigation/scenes/thor-cached-225.pkl')
    #env.unwrapped.browse().show()

    # env = environments.make('CachedThor-v0', goals = [], h5_file_path = 'test.h5') #, goals = [], scenes = 311)
    # env.unwrapped.browse().show()

    env = environments.make('AuxiliaryThor-v1', goals=[], scenes=311)
    env.unwrapped.browse().show()