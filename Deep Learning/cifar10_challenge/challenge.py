import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

"""
DNN challenge example code.
Authors: Shubham Krishna, Bozidar Antic
"""

def train_augment(x: tf.Tensor, y: tf.Tensor):
    """ apply augmentations to image x """
    x = tf.image.random_flip_left_right(x)
    return x, y


def get_model(c_out, input_shape):
    model = tf.keras.Sequential()
    model.add(Conv2D(filters = 32,kernel_size = (3,3),padding= 'same',input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters = 32,padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides = None,padding = 'valid',data_format = None))
    
    model.add(Conv2D(filters= 64, padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters= 64, padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides = None,padding = 'valid',data_format = None))
    
    model.add(Conv2D(filters= 128, padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters= 128, padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides = None,padding = 'valid',data_format = None))
    
    model.add(Conv2D(filters= 256, padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters= 256, padding = 'same', kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides = None,padding = 'valid',data_format = None))
    
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(c_out))
 
    return model


def main():
    parser = argparse.ArgumentParser("dnn_challenge")
    parser.add_argument('--save_dir', type=str, default='./log/')
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    args.save_dir = os.path.expanduser(args.save_dir)
    args.data_dir = os.path.expanduser(args.data_dir)


    # load data
    eval_data_size = 5000
    (x_train, y_train), (x_test) = np.load(args.data_dir + "/WS1920_challenge_data_set.npy", allow_pickle=True)

    x_train = np.expand_dims(x_train, 4).astype('float32') / 255
    x_eval = x_train[0:eval_data_size, ...]
    x_train = x_train[eval_data_size:, ...]
    y_eval = y_train[0:eval_data_size, ...]
    y_train = y_train[eval_data_size:, ...]
    x_test = np.expand_dims(x_test, 4).astype('float32') / 255
    num_classes = np.max(y_train) + 1

    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(train_augment).batch(
        args.batch_size).prefetch(2)
    eval_set = tf.data.Dataset.from_tensor_slices((x_eval, y_eval)).batch(args.batch_size).prefetch(2)
    test_set = tf.data.Dataset.from_tensor_slices(x_test).batch(args.batch_size).prefetch(2)

    model = get_model(num_classes, [32, 32, 1])
    model.summary()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.003)
    #tf.keras.optimizers.Adam(learning_rate = 0.003)
    #tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.8)

    # tensorboard writer
    logdir = args.save_dir + "/tb/%d/" % time.time()
    writer = tf.summary.create_file_writer(logdir)  # Needed for Tensorboard logging

    @tf.function
    def graph_trace_function(x, y):
        with tf.GradientTape():
            logits = model(x, training=True)
            loss_value = loss(y, logits)
            # when we add gradients here the graph gets quite uninterpretable
        return loss_value

    # TODO use a tf file writer in combination with tf.summary.trace_on() tf.summary.trace_export()
    #  graph_trace_function() and zero tensor inputs to save the graph

    for e in range(args.epochs):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_loss = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(train_set):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss(y, logits)

            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            train_accuracy.update_state(y, logits)
            train_loss.update_state(loss_value)

        tf.print("-" * 50, output_stream=sys.stdout)
        eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        eval_loss = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(eval_set):
            logits = model(x, training=False)
            loss_value = loss(y, logits)
            eval_accuracy.update_state(y, logits)
            eval_loss.update_state(loss_value)

        tf.print("epoch {0:d} \ntrain_loss: {1:2.5f} \ntrain_accuracy: {2:2.5f}".format(e, train_loss.result(),
                                                                                          train_accuracy.result()),
                 output_stream=sys.stdout)
        tf.print("eval_loss: {0:2.5f} \neval_accuracy: {1:2.5f}".format(eval_loss.result(),
                                                                         eval_accuracy.result()),
                 output_stream=sys.stdout)

    # predict labels
    predicted = []
    for x in test_set:
        y_ = model(x, training=False).numpy()
        predicted.append(y_)
    predicted = np.concatenate(predicted, axis=0)
    predicted = np.argmax(predicted, axis=1).astype('int32')
    predicted = np.expand_dims(predicted, 1)
    indices = np.expand_dims(np.arange(len(predicted)), 1)
    predicted = np.concatenate([indices, predicted], axis=1).astype('int32')
    path = args.save_dir + str(int(time.time())) + '_predictions.csv'
    np.savetxt(path, predicted, delimiter=",", header='Id,Category', fmt='%d')
    print("saved predictions as: " + path)


if __name__ == '__main__':
    main()
