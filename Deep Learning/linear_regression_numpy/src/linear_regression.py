''''
TODO fill in your names:
group member 1: Shubham Krishna
group member 2:

'''

import matplotlib.pyplot as plt
import mnist as mnist
import numpy as np

# Load the raw MNIST
X_train, y_train = mnist.read(dataset='training')
X_test, y_test = mnist.read(dataset='testing')

# split eval data from train data:
eval_data_size = 10000
train_data_size = 50000
test_data_size = 10000

X_eval = X_train[0:10000, :, :]
y_eval = y_train[0:10000]
X_train = X_train[10000:, :, :]
y_train = y_train[10000:]
# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Evaluation data shape: ', X_eval.shape)
print('Evaluation labels shape: ', y_eval.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Reshape the image data into rows
# Datatype float allows you to subtract images (is otherwise uint8)
X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('float')
X_eval = np.reshape(X_eval, (X_eval.shape[0], -1)).astype('float')
X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype('float')
print("x shapes:")
print(X_train.shape, X_eval.shape, X_test.shape)
# normalize train data from range 0 to 255 to range 0 to 1
X_train = X_train / 255
X_eval = X_eval / 255
X_test = X_test / 255


# transform to y to one hot encoded vectors:
# each row is one y vector
def make_one_hot(v):
    """
    :param v: vector of the length of the dataset containing class labels from 0 to 9
    :return: a matrix of dim(length dataset,10), where the index of the corresponding label is set to one.
    """
    num_classes = 10
    v_one_hot = np.zeros((len(v), num_classes))
    for i,j in enumerate(v):
        v_one_hot[i][j] = 1
    # TODO calculate one hot encoded y vectors
    return v_one_hot


y_train = make_one_hot(y_train)
y_eval = make_one_hot(y_eval)
y_test = make_one_hot(y_test)
print("y shapes:")
print(y_train.shape, y_eval.shape, y_test.shape)

# TODO for task d adapt the following parameters to achieve better results
batch_size = 100
epochs = 10
learning_rate = 0.01

# usually one would use a random weight initialization, but for reproducible results we use fixed weights
# Don't change these parameters
W = np.ones((784, 10)) * 0.01
b = np.ones((10)) * 0.01


def get_next_batch(iteration, batch_size, data, label):
    X = data[iteration * batch_size:(iteration + 1) * batch_size, :]
    y = label[iteration * batch_size:(iteration + 1) * batch_size, :]
    return X, y



def get_loss(y_hat, y):
    """
    :param y_hat: dim(batch_size,10)
    :param y: dim(batch_size,10)
    :return:
    """
    l = 0
    for row_y, row_y_hat in zip(y, y_hat):
    # TODO calc the loss:
        l = l +  np.sum((row_y - row_y_hat)**2)
    return l / batch_size


def get_accuracy(y_hat, y):
    """
    the accuracy for one image is one if the maximum of y_hat has the same index as the 1 in y
    :param y_hat:  dim(batch_size,10)
    :param y: dim(batch_size,10)
    :return: mean accuracy
    """
    acc = 0
    for row_y, row_y_hat in zip(y, y_hat):
    # TODO calc the accuracy:
        if(np.argmax(row_y) == np.argmax(row_y_hat)):
            acc = acc + 1
    return acc / batch_size


def do_network_inference(x):  # over whole batch
    """
    :param x: dim(batchsize,784)
    :return: dim(batchsize,10)
    """
    # TODO calculate y_hat without using a loop, note that the numpy vector addition has some special features that can be used.
    y_hat = np.matmul(x,W) + b
    return y_hat


def get_delta_weights(y_hat, y, x_batch):
    """
    :param y_hat:  dim(batchsize,10)
    :param y:  dim(batchsize,10)
    :param x_batch: dim(batchsize,784)
    :return: dim(784,10)
    """
    # TODO calculate delta_w with a sum over outer products
    y_dif_transformed = np.transpose(np.expand_dims(y_hat - y, axis = -1), (0, 2, 1))
    x_transformed = np.expand_dims(x_batch, axis = -1)
    #print(f'X: {np.sum(x_transformed)}')
    delta_w = np.sum(2 * np.matmul(x_transformed, y_dif_transformed), axis=0)
    #print(f'dW: {np.sum(delta_w)}')    
    #print(delta_w)
    return delta_w / batch_size


def get_delta_biases(y_hat, y):
    """
    :param y_hat:  dim(batchsize,10)
    :param y:  dim(batchsize,10)
    :return:  dim(10)
    """
    # TODO calculate delta_b
    delta_b = np.zeros(10)
    for i,j in zip(y_hat,y):
        delta_b = delta_b + 2 * (i - j)
    return delta_b / batch_size


def do_parameter_update(delta_w, delta_b, W, b):
    """
    :param delta_w: dim(748,10)
    :param delta_b: dim(10)
    :param W: dim(748,10)
    :param b: dim(10)
    """
    # TODO update W and b
    W = W - learning_rate * delta_w
    b = b - learning_rate * delta_b
    return W, b
    

# do training and evaluation
mean_eval_losses = []
mean_train_losses = []
mean_eval_accs = []
mean_train_accs = []

for epoch in range(epochs):
    # training
    mean_train_loss_per_epoch = 0
    mean_train_acc_per_epoch = 0
    for i in range(train_data_size // batch_size):
        x, y = get_next_batch(i, batch_size, X_train, y_train)
        y_hat = do_network_inference(x)
        train_loss = get_loss(y_hat, y)
        train_accuracy = get_accuracy(y_hat, y)
        delta_w = get_delta_weights(y_hat, y, x)
        delta_b = get_delta_biases(y_hat, y)
        W, b = do_parameter_update(delta_w, delta_b, W, b)
        # print(f'dif: {np.sum(w_old-W)}')
        mean_train_loss_per_epoch += train_loss
        mean_train_acc_per_epoch += train_accuracy
        # print("epoch: {0:d} \t iteration {1:d} \t train loss: {2:f}".format(epoch, i,train_loss))
    mean_train_loss_per_epoch = mean_train_loss_per_epoch / ((train_data_size // batch_size))
    mean_train_acc_per_epoch = mean_train_acc_per_epoch / ((train_data_size // batch_size))
    print("epoch:{0:d} \t mean train loss: {1:f} \t mean train acc: {2:f}".format(epoch,mean_train_loss_per_epoch,
                                                                              mean_train_acc_per_epoch))
    # evaluation:
    mean_eval_loss_per_epoch = 0
    mean_eval_acc_per_epoch = 0
    # TODO calculate the evaluation loss and accuracy
    for i in range(eval_data_size // batch_size):
        x, y = get_next_batch(i, batch_size, X_eval, y_eval)
        y_hat = do_network_inference(x)
        eval_loss = get_loss(y_hat, y)
        eval_accuracy = get_accuracy(y_hat, y)
        mean_eval_loss_per_epoch += eval_loss
        mean_eval_acc_per_epoch += eval_accuracy
    
    mean_eval_loss_per_epoch = mean_eval_loss_per_epoch / (eval_data_size // batch_size)
    mean_eval_acc_per_epoch = mean_eval_acc_per_epoch / ((eval_data_size // batch_size))
    print("epoch:{0:d} \t mean eval loss: {1:f} \t mean eval acc: {2:f}".format(epoch,mean_eval_loss_per_epoch,
                                                                            mean_eval_acc_per_epoch))
    mean_eval_losses.append(mean_eval_loss_per_epoch)
    mean_train_losses.append(mean_train_loss_per_epoch)
    mean_eval_accs.append(mean_eval_acc_per_epoch)
    mean_train_accs.append(mean_train_acc_per_epoch)
# testing
mean_test_loss_per_epoch = 0
mean_test_acc_per_epoch = 0
# TODO calculate the test loss and accuracy
for i in range(test_data_size // batch_size):
    x, y = get_next_batch(i, batch_size, X_test, y_test)
    y_hat = do_network_inference(x)
    test_loss = get_loss(y_hat, y)
    test_accuracy = get_accuracy(y_hat, y)
    mean_test_loss_per_epoch += test_loss
    mean_test_acc_per_epoch += test_accuracy

mean_test_loss_per_epoch = mean_test_loss_per_epoch / (test_data_size // batch_size)
mean_test_acc_per_epoch = mean_test_acc_per_epoch / ((test_data_size // batch_size))
print("final test loss: {0:f} \t final test acc: {1:f}".format(mean_test_loss_per_epoch, mean_test_acc_per_epoch))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(epochs), mean_train_losses, "r", label="train loss")
ax1.plot(range(epochs), mean_eval_losses, "b", label="eval loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(range(epochs), mean_train_accs, "r", label="train acc")
ax2.plot(range(epochs), mean_eval_accs, "b", label="eval acc")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()
plt.show()
