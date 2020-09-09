import numpy as np
import torch
import torch.optim as optim
from action import get_action_set, select_exploratory_action, select_greedy_action
from learning import perform_qlearning_step, update_target_net
from model import DQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training


def evaluate(env, load_path="agent.pt"):
    """ Evaluate a trained model and compute your leaderboard scores
	
	NO CHANGES SHOULD BE MADE TO THIS FUNCTION
	
    Parameters
    -------
    env: gym.Env
        environment to evaluate on
    load_path: str
        path to load the model (.pt) from
    """
    episode_rewards = [0.0]
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These are not the final evaluation seeds, do not overfit on these tracks!
    seeds = [
        22597174,
        68545857,
        75568192,
        91140053,
        86018367,
        49636746,
        66759182,
        91294619,
        84274995,
        31531469,
    ]

    # Build & load network
    policy_net = DQN(action_size, device).to(device)
    checkpoint = torch.load(load_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    # Iterate over a number of evaluation episodes
    for i in range(10):
        env.seed(seeds[i])
        obs, done = env.reset(), False
        obs = get_state(obs)
        t = 0

        # Run each episode until episode has terminated or 600 time steps have been reached
        while not done and t < 600:
            env.render()
            action_id = select_greedy_action(obs, policy_net, action_size)
            # print(f'Action seleted: {action_id}')
            action = actions[action_id]
            obs, rew, done, _ = env.step(action)
            obs = get_state(obs)
            episode_rewards[-1] += rew
            t += 1
        print("episode %d \t reward %f" % (i, episode_rewards[-1]))
        episode_rewards.append(0.0)

    print("---------------------------")
    print(" total score: %f" % np.mean(np.array(episode_rewards)))
    print("---------------------------")


def learn(
    env,
    lr=1e-4,
    total_timesteps=100000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    train_freq=1,
    action_repeat=4,
    batch_size=32,
    learning_starts=1000,
    gamma=0.99,
    target_network_update_freq=500,
    model_identifier="agent",
    checkpoints=[100000],
):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    checkpoints: list of ints
        number of timesteps after which model should be saved
    """
    episode_rewards = [0.0]
    training_losses = []
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build networks
    policy_net = DQN(action_size, device).to(device)
    target_net = DQN(action_size, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(
        schedule_timesteps=int(exploration_fraction * total_timesteps),
        initial_p=1.0,
        final_p=exploration_final_eps,
    )

    # Initialize environment and get first state
    obs = get_state(env.reset())

    # Iterate over the total number of time steps
    for t in range(total_timesteps):

        # Select action
        action_id = select_exploratory_action(
            obs, policy_net, action_size, exploration, t
        )
        env_action = actions[action_id]
        # Perform action fram_skip-times
        for f in range(action_repeat):
            new_obs, rew, done, _ = env.step(env_action)
            episode_rewards[-1] += rew
            if done:
                break

        # Store transition in the replay buffer.
        new_obs = get_state(new_obs)
        replay_buffer.add(obs, action_id, rew, new_obs, float(done))
        obs = new_obs

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            obs = get_state(env.reset())
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(
                policy_net,
                target_net,
                optimizer,
                replay_buffer,
                batch_size,
                gamma,
                device,
            )
            training_losses.append(loss)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)

        # save models and plots on checkpoint timesteps
        if (t + 1) in checkpoints:
            torch.save(
                policy_net.state_dict(), model_identifier + "c" + str(t + 1) + ".pt"
            )
            visualize_training(
                episode_rewards, training_losses, model_identifier + "c" + str(t + 1)
            )

    # Save the trained policy network
    torch.save(policy_net.state_dict(), model_identifier + ".pt")

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier)
