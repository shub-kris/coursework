import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


def kdtree_search(tree, queries):
    queries = queries.reshape(1, -1)
    dist, ind = tree.query(queries, 1)


def generate_data(n, dim):
    X = np.random.rand(n, dim)
    return X


def find_distance(q, X):
    start_time = time.time()
    dist = (q - X) ** 2
    dist = np.sum(dist, axis=1)
    min_dist = np.min(dist)


def plot_time_analysis(times, dims, name):
    # plot to file
    plt.clf()
    plt.plot(dims, times)
    plt.title("Query Times")
    plt.xlabel("dimension (D)")
    plt.ylabel("time (ms)")
    plt.savefig(f"{name}.png", bbox_inches="tight")


def main():
    KDTree_time = []
    brute_time = []
    N = np.power(2, 10)
    dim = [d for d in range(1, 500, 10)]

    # kd-tree search
    for D in dim:
        X = generate_data(N, D)
        start_time = time.time()
        tree = KDTree(X)
        for x in X:
            kdtree_search(tree, x)
        #     find_distance(x, X)
        elapsed_time = time.time() - start_time
        KDTree_time.append(elapsed_time * 1000.0)

    # Brute-force search
    for D in dim:
        X = generate_data(N, D)
        start_time = time.time()
        for x in X:
            find_distance(x, X)
        elapsed_time = time.time() - start_time
        brute_time.append(elapsed_time * 1000.0)

    plot_time_analysis(KDTree_time, dim, "KDTree")
    plot_time_analysis(brute_time, dim, "Brute_Force")


if __name__ == "__main__":
    main()


# def time_analysis():

#     N = np.power(2, 10)
#     dim = [i for i in range(1, 501, 10)]
#     time_dict = {}
#     for i in dim:
#         start_time = time.time()
#         X = generate_data(N, i)
#         for x in X:
#             find_distance(x, X)
#         elapsed_time = time.time() - start_time
#         time_dict[i] = elapsed_time
#         print(f"Dimension: {i}, time_taken: {elapsed_time}")
