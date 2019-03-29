import environments

if __name__ == '__main__':
    env = environments.make('CachedThor-v0', goals = [], h5_file_path = 'test.h5') #, goals = [], scenes = 311)
    env.unwrapped.browse().show()