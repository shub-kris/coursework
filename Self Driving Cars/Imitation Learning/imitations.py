import os
import numpy as np
import gym
from pyglet.window import key
import time


def load_imitations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert imitations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    # idx_file = os.path.join(data_folder,"count.npy")    # I don't think there is any file like this count.npy is analogous to how many files present in directory
    # assert os.path.exists(idx_file), "file doesn't exist: %s" % idx_file  # Checks if file is present or not
    # idx = np.load(idx_file)                                         # Loads the idx_file and returns a list or numpy array not an integer

    observations = []
    actions = []
    for i in range(599):  # So how does it work?? range() takes integer as an argument
        # if i % max(1, int(idx / 10)) == 0:                  # Why?
        # print("preloading data %d/%d",(i, idx - 1))
        observations.append(
            np.load(os.path.join(data_folder, "observation_%05d.npy" % i))
        )
        actions.append(np.load(os.path.join(data_folder, "action_%05d.npy" % i)))
    # for i in range(1533):
    # observations.append(np.load(os.path.join(data_folder,"myobservation_%05d.npy" % i)))
    # actions.append(np.load(os.path.join(data_folder,"myaction_%05d.npy" % i)))
    # for i in range(101):
    # observations.append(np.load(os.path.join(data_folder,"errorobservation_%05d.npy" % i)))
    # actions.append(np.load(os.path.join(data_folder,"erroraction_%05d.npy" % i)))
    # for i in range(631):
    # observations.append(np.load(os.path.join(data_folder,"goodobservation_%05d.npy" % i)))
    # actions.append(np.load(os.path.join(data_folder,"goodaction_%05d.npy" % i)))
    # for i in range(461):
    # observations.append(np.load(os.path.join(data_folder,"gobservation_%05d.npy" % i)))
    # actions.append(np.load(os.path.join(data_folder,"gaction_%05d.npy" % i)))

    return observations, actions


def save_imitations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    for i in range(len(actions)):
        np.save(os.path.join(data_folder, "gobservation_%05d" % i), observations[i])
        np.save(os.path.join(data_folder, "gaction_%05d" % i), actions[i])


class ControlStatus:
    """
    Class to keep track of key presses while recording imitations.
    """

    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.start = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.ESCAPE:
            self.quit = True
        if k == key.SPACE:
            self.stop = True
        if k == key.TAB:
            self.save = True
        if k == key.B:
            self.start = True
        if k == key.LEFT:
            self.steer = -1.0
        if k == key.RIGHT:
            self.steer = +1.0
        if k == key.UP:
            self.accelerate = +0.5
        if k == key.DOWN:
            self.brake = +0.8

    def key_release(self, k, mod):
        if k == key.LEFT and self.steer < 0.0:
            self.steer = 0.0
        if k == key.RIGHT and self.steer > 0.0:
            self.steer = 0.0
        if k == key.UP:
            self.accelerate = 0.0
        if k == key.DOWN:
            self.brake = 0.0


def record_imitations(imitations_folder):
    """
    Function to record own imitations by driving the car in the gym car-racing
    environment.
    imitations_folder:  python string, the path to where the recorded imitations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    env = gym.make("CarRacing-v0").env
    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            if status.start:
                observations.append(observation.copy())
                actions.append(
                    np.array([status.steer, status.accelerate, status.brake])
                )
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step(
                [status.steer, status.accelerate, status.brake]
            )
            total_reward += reward
            env.render()
            time.sleep(0.20)

        if status.save:
            save_imitations(imitations_folder, actions, observations)
            status.save = False
            status.start = False

        status.stop = False
        env.close()
