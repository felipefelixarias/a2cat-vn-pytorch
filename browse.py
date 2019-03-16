import environments

if __name__ == '__main__':
    env = environments.make('ContinuousGoalThor-v0', goals = [], scenes = 311)
    env.unwrapped.browse().show()