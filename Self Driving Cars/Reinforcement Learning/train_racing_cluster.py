from pyvirtualdisplay import Display
import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent in headless mode on the cluster
    """
    display = Display(visible=0, size=(800, 600))
    display.start()
    env = gym.make("CarRacing-v0")
    deepq.learn(env)
    env.close()
    display.stop()


if __name__ == "__main__":
    main()
