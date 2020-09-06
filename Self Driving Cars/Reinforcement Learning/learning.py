import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    # sample form replay buffer
    bs, ba, br, bs1, bd = replay_buffer.sample(batch_size)  
    bs, ba, br, bd = torch.tensor(bs, device=device), torch.tensor(ba,
            device=device), torch.tensor(br, device=device), torch.tensor(bd, device=device)
    # compute Q(s_t, a)
    q = policy_net(bs).gather(1, ba.unsqueeze(1))
    # doulbe Q-learning
    q1_max_ind = policy_net(bs1).max(1)[1]
    maxq1 = target_net(bs1).gather(1, q1_max_ind.unsqueeze(1)).detach().squeeze(-1)
    # regular Q-learning
    # compute max Q(s_{t+1}, a)
    # ques = target_net(bs1)
    # maxq1 = ques.max(1)[0].detach()

    # masking
    maxq1 *= torch.tensor([0. if item else 1. for item in bd], device=device)
    # compute target
    target = br + gamma * maxq1
    # compute loss
    loss = torch.nn.functional.smooth_l1_loss(q, target.unsqueeze(1))
    optimizer.zero_grad()
    # caluculate gradients
    loss.backward()
    # clip gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # optimize model
    optimizer.step()

    return loss

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """
    target_net.load_state_dict(policy_net.state_dict())
