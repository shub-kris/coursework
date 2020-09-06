import os
import gzip
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score

"""
see: https://github.com/zalandoresearch/fashion-mnist
"""


def load_mnist(path, kind="train", each=1):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images[::each, :], labels[::each]


def kdtree_search(tree, train_labels, test_data, k):
    """
	It returns the labels for the nearest neighbours for each test_img ( return a list of arrays)
	"""
    topk_labels = []
    for x in test_data:
        dist, ind = tree.query(x.reshape(1, -1), k)
        pred_labels = train_labels[ind][0]
        topk_labels.append(pred_labels)

    return topk_labels


def topk_accuracy(topk_labels, test_labels):
    accuracy = np.array(
        [
            1.0 if true_label in topk_list else 0.0
            for true_label, topk_list in zip(test_labels, topk_labels)
        ]
    ).mean()
    return accuracy


def plot_accuracy(topk, k):
    plt.plot(k, topk, "o--")
    plt.xticks(k)
    plt.xlabel("k")
    plt.ylabel("Top-k-Accuracy")
    plt.savefig("Topk.png", bbox_inches="tight")


def main():
    train_img, train_label = load_mnist(".", kind="train", each=10)
    test_img, test_label = load_mnist(".", kind="t10k", each=10)
    cnt = np.array(np.unique(train_label, return_counts=True)).T
    topk = []
    K = [i for i in range(1, 11)]
    for k in K:
        tree = KDTree(train_img)
        topk_labels = kdtree_search(tree, train_label, test_img, k)
        topk.append(topk_accuracy(topk_labels, test_label))
    plot_accuracy(topk, K)
    ind = np.where((train_label == 2) | (train_label == 6))
    ps_train_img, ps_train_label = train_img[ind], train_label[ind]

    ind = np.where((test_label == 2) | (test_label == 6))
    ps_test_img, ps_test_label = test_img[ind], test_label[ind]

    k = 1
    tree = KDTree(ps_train_img)
    pred_label = kdtree_search(tree, ps_train_label, ps_test_img, 1)
    pred_label = np.concatenate(pred_label, axis=0)

    precision = precision_score(
        ps_test_label, pred_label, pos_label=2
    )  # Please check changing pos_label = 6 changes value of precision and recall
    recall = recall_score(ps_test_label, pred_label, pos_label=2)
    print(f"Precision: {precision}, Recall: {recall}")


if __name__ == "__main__":
    main()
