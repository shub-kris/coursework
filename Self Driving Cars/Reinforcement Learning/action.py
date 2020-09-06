import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """
    with torch.no_grad():
        actions = policy_net(state).max(1)[1].view(1,
                1).to('cpu').numpy()[0,0]
    
    return actions

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """
    eps = exploration.value(t)
    r = random.random()
    if r > eps:
        return select_greedy_action(state, policy_net, action_size)
    else:
        action_number = random.randrange(action_size)
        return action_number


def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    # 2c)
    # return [[-1.5, 0.05, 0], [1.5, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0], [0, 0, 0,]]
    # return [[-1.5, 0.05, 0], [1.5, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
    # return [[-1.5, 0.05, 0], [1.5, 0.05, 0], [-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
