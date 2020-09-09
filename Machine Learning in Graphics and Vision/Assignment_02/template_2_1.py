from __future__ import print_function
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-5


def log(x):
    return np.log(x + eps)


class LogisticRegression:
    def __init__(self):
        self.weights = np.array([])
        self.losses = []
        self.lr = 1e-1
        self.max_iter = 10

    def init_weights(self, dim):
        # uniform initialization of weights
        self.weights = np.ones((dim, 1)) / dim

    def predict_proba(self, features):
        """
        Exercise 1a: Compute the probability of assigning a class to each feature of an image
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
        Returns:
            prob (np.array): probabilities [N] of N examples
        """
        # TODO: INSERT
        temp = features @ self.weights
        prob = 1 / (1 + np.exp(-temp))
        return prob

    def predict(self, features):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
        Returns:
            pred (np.array): predictions [N] of N examples
        """
        prob = self.predict_proba(features)
        # decision boundary at 0.5
        pred = np.array([1.0 if x >= 0.5 else 0.0 for x in prob])[:, np.newaxis]
        return pred

    def compute_loss(self, features, labels):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
        Returns:
            loss (scalar): loss of the current model
        """
        examples = len(labels)

        """
        Exercise 1b:    Compute the loss for the features of all input images
                        NOTE: Don't forget to remove the first quit() command in the main program!

        HINT: Use the provided log function to avoid nans with large learning rate
        """
        prob = self.predict(features)
        loss = -(labels * log(prob) + (1 - labels) * log(1 - prob))  # TODO: REPLACE

        return loss.sum() / examples  # Why not loss.mean() instead of doing this

    def score(self, pred, labels):
        """
        Args:
            pred (np.array): predictions [N, 1] of N examples
            labels (np.array): labels [N, 1] of N examples
        Returns:
            score (scalar): accuracy of the predicted labels
        """
        diff = pred - labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

    def update_weights(self, features, labels, lr):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
            lr (scalar): learning rate scales the gradients
        """
        examples = len(labels)

        """
        Exercise 1c:    Compute the gradients given the features of all input images
                        NOTE: Don't forget to remove the second quit() command in the main program!
        """
        gradient = 0  # TODO: REPLACE
        prob = self.predict(features)
        gradient = features.T @ (prob - labels)
        # gradient = features.T @ (-labels * (1 - prob) + (1 - labels) * prob)

        # update weights
        self.weights -= lr * gradient / examples

    def fit(self, features, labels):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
        """
        # gradient descent
        for i in range(self.max_iter):
            # update weights using the gradients
            self.update_weights(features, labels, self.lr)

            # compute loss
            loss = self.compute_loss(features, labels)
            self.losses.append(loss)

            # print current loss
            print("Iteration {}\t Loss {}".format(i, loss))


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


# load fashion mnist
train_img, train_label = load_mnist(".", kind="train", each=1)
test_img, test_label = load_mnist(".", kind="t10k", each=1)
train_img = train_img.astype(np.float) / 255.0
test_img = test_img.astype(np.float) / 255.0

# label definition of fashion mnist
labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# consider only the classes 'Pullover' and 'Coat'
labels_mask = [2, 4]
train_mask = np.zeros(len(train_label), dtype=bool)
test_mask = np.zeros(len(test_label), dtype=bool)
train_mask[(train_label == labels_mask[0]) | (train_label == labels_mask[1])] = 1
test_mask[(test_label == labels_mask[0]) | (test_label == labels_mask[1])] = 1

# classification of Pullover
train_img = train_img[train_mask, :]
test_img = test_img[test_mask, :]
train_label = np.array(
    [1.0 if x == labels_mask[0] else 0.0 for x in train_label[train_mask]]
)[:, np.newaxis]
test_label = np.array(
    [1.0 if x == labels_mask[0] else 0.0 for x in test_label[test_mask]]
)[:, np.newaxis]


# init logistic regression
logreg = LogisticRegression()
logreg.init_weights(train_img.shape[1])
logreg.lr = 1e-2
logreg.max_iter = 10

accs = []

# testing without training
y_pred = logreg.predict(test_img)
score = logreg.score(y_pred, test_label)
accs.append(score)
print(
    "Accuracy of initial logistic regression classifier on test set: {:.2f}".format(
        score
    )
)

# quit() ### Exercise 1b: Remove exit ###

# compute initialization loss
loss = logreg.compute_loss(train_img, train_label)
print("Initialization loss {}".format(loss))

# quit() ### Exercise 1c: Remove exit ###

"""
Exercise 1d: Plot the cross entropy loss for t=0 and t=1
"""
# TODO: Insert

# compute test error after max_iter

for i in range(0, 100):
    # training
    logreg.fit(train_img, train_label)

    # testing
    y_pred = logreg.predict(test_img)
    score = logreg.score(y_pred, test_label)
    accs.append(score)
    print(
        "Accuracy of logistic regression classifier on test set: {:.2f}".format(score)
    )

"""
Exercise 1e: Plot the learning curves (losses and accs) using different learning rates (1e-4,1e-3,1e-2,1e-1,1e-0)
"""
losses = logreg.losses
# TODO: INSERT

"""
Exercise 1f: Plot the optimized weights and weights.*img (.* denotes element-wise multiplication)
"""
# TODO: INSERT
