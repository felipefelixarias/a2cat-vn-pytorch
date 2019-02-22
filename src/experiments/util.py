import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np

def display_q(trainer):
    env = trainer.env

    maze = env.unwrapped.graph.maze
    shape = maze.shape
    Q = np.zeros(shape + (4,), dtype = np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            observation = env.unwrapped.observe((y, x))
            observation = observation.reshape([-1])
            Q[y, x,:] = trainer._q(observation[None])[0]
                
    V = np.max(Q, 2)
    policy = np.argmax(Q, axis = 2)
    show_policy_and_value(V, policy, maze)
    return

    print(V)

    min_v = np.min(V)
    max_v = np.max(V)
    V = (V - min_v) / (max_v - min_v)

    # Remove maze
    V[env.unwrapped.graph.maze == 0] = 1

    img = plt.imshow(V, interpolation='none', cmap='gray')
    plt.show()

def show_policy_and_value(v, policy, maze):
    policy_symbols = {0: '↓',
                        1: '→',
                        2: '↑',
                        3: '←'}
    mask = np.invert(maze.astype(np.bool))

    # to center the heatmap around zero
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

    plt.show()

