import numpy as np
from operator import add

def direction_to_change(direction):
        if direction == 0:
            return (1, 0)
        elif direction == 1:
            return (0, 1)
        elif direction == 2:
            return (-1, 0)
        elif direction == 3:
            return (0, -1)
        return None

def step(state, action):
    if action == 0:
        change = direction_to_change(state[2])
        return (state[0] + change[0], state[1] + change[1], state[2])
    elif action == 2:
        change = direction_to_change(state[2])
        return (state[0] - change[0], state[1] - change[1], state[2])
    elif action == 1:
        return (state[0], state[1], (state[2] + 1) % 4)
    elif action == 3:
        return (state[0], state[1], (state[2] + 3) % 4)

def enumerate_positions(maze):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y]:
                yield (x, y)

def find_state(maze):
    return next(enumerate_positions) + (0,)

def is_valid_state(maze, state):
    return state[0] >= 0 and state[1] >= 0 and state[0] < maze.shape[0] and state[1] < maze.shape[1] and maze[state[0], state[1]]

def compute_shortest_path_data(maze):
    distances = np.ndarray(maze.shape + maze.shape, dtype = np.int32)
    actions = np.ndarray(maze.shape + maze.shape + (4,), dtype = np.bool)
    distances.fill(-1)
    actions.fill(False)
    def fill_shortest_path(goal, position, distance, from_direction):
        if distances[position + goal] != -1 and distances[position + goal] < distance:
            return

        if distances[position + goal] == distance:
            actions[position + goal + ((from_direction + 2) % 4,)] = True
            return            
        
        actions[position + goal] = False
        actions[position + goal + ((from_direction + 2) % 4,)] = True
        distances[position + goal] = distance

        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(0))), distance + 1, 0)
        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(1))), distance + 1, 1)
        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(2))), distance + 1, 2)
        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(3))), distance + 1, 3)
        

    for goal in enumerate_positions(maze):
        fill_shortest_path(goal, goal, 0, 0)
        actions[goal + goal] = False

    return distances, actions

