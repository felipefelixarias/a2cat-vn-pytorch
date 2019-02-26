import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np

def display_q(trainer, fig = None):
    env = trainer.env

    old_state = env.unwrapped.state

    if hasattr(env.unwrapped, 'graph_number'):
        old_selected_graph = env.unwrapped.graph_number
        maze = env.unwrapped.graphs[0].maze
        env.unwrapped.graph_number = 0
        is_multi = True
    else:
        maze = env.unwrapped.graph.maze
        is_multi = False
    shape = maze.shape    
    Q = np.zeros(shape + (4,), dtype = np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            env.reset()
            env.unwrapped.state = (y, x)
            observation, _, _, _ = env.step(None)
            Q[y, x,:] = trainer._q(observation[None])[0]
    env.reset()
    env.unwrapped.state = old_state

    if is_multi:
        env.unwrapped.graph_number = old_selected_graph
                
    V = np.max(Q, 2)
    policy = np.argmax(Q, axis = 2)
    show_policy_and_value(V, policy, maze, fig)
    return

def display_policy_value(a2ctrainer, fig = None):
    env = a2ctrainer.validation_env
    if hasattr(env.unwrapped, 'graph_number'):
        old_selected_graph = env.unwrapped.graph_number
        maze = env.unwrapped.graphs[0].maze
        env.unwrapped.graph_number = 0
        is_multi = True
    else:
        maze = env.unwrapped.graph.maze
        is_multi = False
    old_state = env.unwrapped.state
    shape = maze.shape
    policy = np.zeros(shape, dtype = np.int32)
    value = np.zeros(shape, dtype = np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            env.reset()
            env.unwrapped.state = (y, x)
            obs, _, _, _ = env.step(None)
            policy[y, x], value[y, x] = a2ctrainer._eval_pv(obs)

    env.reset()
    if is_multi:
        env.unwrapped.graph_number = old_selected_graph
    env.unwrapped.state = old_state
    show_policy_and_value(value, policy, maze, fig)
    return

def show_policy_and_value(v, policy, maze, fig):
    policy_symbols = {0: '↓',
                        1: '→',
                        2: '↑',
                        3: '←'}
    mask = np.invert(maze.astype(np.bool))

    # to center the heatmap around zero
    plt.figure(num = fig.number)
    maxval = max(np.abs(np.min(v)), np.abs(np.max(v)))
    ax = sns.heatmap(v, annot=True, mask=mask, fmt='.3f', square=1, linewidth=1., cmap='coolwarm', vmin=-maxval,
                         vmax=maxval)
    #for i, j in zip(*np.where(self.terminals == 1)):
    #    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=3))

    if policy is not None:
        policy = policy.reshape([-1])
        for t, pol in zip(ax.texts, policy[:][(~mask).flat]):
            t.set_text(policy_symbols[pol])
            t.set_size('xx-large')

