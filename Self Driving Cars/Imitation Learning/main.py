import os
import sys
import numpy as np
import torch
import gym

from training import train
from imitations import record_imitations

directory = "" 
trained_network_file = os.path.join(directory, 'data/train.t7')
imitations_folder = os.path.join(directory, 'data/teacher')


def evaluate(): 
    """
    """
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    env = gym.make('CarRacing-v0')
    # you can set it to torch.device('cuda') in case you have a gpu
    device = torch.device('cuda')
    infer_action = infer_action.to(device)

    for episode in range(5):
        observation = env.reset()

        reward_per_episode = 0
        for t in range(500):
            env.render()
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            observation, reward, done, info = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))


def calculate_score_for_leaderboard():
    """
    Evaluate the performance of the network. This is the function to be used for
    the final ranking on the course-wide leader-board, only with a different set
    of seeds. Better not change it.
    """
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    env = gym.make('CarRacing-v0')
    # you can set it to torch.device('cuda') in case you have a gpu
    device = torch.device('cuda')

    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]
    total_reward = 0

    for episode in range(10):
        env.seed(seeds[episode])
        observation = env.reset()

        reward_per_episode = 0
        for t in range(600):
            env.render()
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            observation, reward, done, info = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        total_reward += reward_per_episode

    print('---------------------------')
    print(' total score: %f' % (total_reward / 10))
    print('---------------------------')


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        train(imitations_folder, trained_network_file)
    elif sys.argv[1] == "teach":
        record_imitations(imitations_folder)
    elif sys.argv[1] == "test":
        evaluate()
    elif sys.argv[1] == "score":
        calculate_score_for_leaderboard()
    else:
        print('This command is not supported, valid options are: train, teach, '
              'test and score.')
