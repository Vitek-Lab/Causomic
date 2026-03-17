from notears import utils
import numpy as np
from notears.linear import notears_linear
import networkx as nx
import matplotlib.pyplot as plt

utils.set_random_seed(1)

n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)

G = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
pos = nx.spring_layout(G, seed=1)

plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', arrowsize=15, node_size=400)
if 'W_true' in globals():
    edge_labels = {(u, v): f"{W_true[u, v]:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("B_true DAG")
plt.show()

X = utils.simulate_linear_sem(W_true, n, sem_type)


edges = np.transpose(np.nonzero(B_true))
m = len(edges)
if m == 0:
    print("No edges in B_true to plot.")
else:
    cols = 4
    rows = (m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()
    for k, (i, j) in enumerate(edges):
        ax = axes[k]
        ax.scatter(X[:, i], X[:, j], s=10, alpha=0.6)
        ax.set_xlabel(f'X[:,{i}] (parent)')
        ax.set_ylabel(f'X[:,{j}] (child)')
        w = W_true[i, j] if 'W_true' in globals() else None
        if w is not None:
            ax.set_title(f'{i} -> {j}, w={w:.2f}')
        else:
            ax.set_title(f'{i} -> {j}')
    for k in range(m, len(axes)):
        fig.delaxes(axes[k])
    plt.tight_layout()
    plt.show()

W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
assert utils.is_dag(W_est)
acc = utils.count_accuracy(B_true, W_est != 0)
print(acc)