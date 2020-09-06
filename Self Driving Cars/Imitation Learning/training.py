import torch
import random
import time
import numpy as np
from network import ClassificationNetwork
from imitations import load_imitations
import torch

def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()      #Logic for which action to take
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-4,weight_decay = 0.0001)    # weight_decay  
    observations, actions = load_imitations(data_folder) #Loading data
    
    observations = [torch.Tensor(observation) for observation in observations] #Converts every observation numpy array into torch tensor and append to list
    
     #Data Augmentation (Flipped the images left to eight)
    for i in range(len(actions)):
        if actions[i][0] != 0:    #Flipping only right images
            observations.append(torch.Tensor(np.fliplr(observations[i]).copy()))
            actions.append(actions[i] * [-1,1,1])
    
    actions = [torch.Tensor(action) for action in actions]    #Converts every action numpy array into torch tensor
    
    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]  # Iterates over (observation, classes from classification network) (List of Tuples)
    gpu = torch.device('cuda')

    nr_epochs = 100
    batch_size = 64
    number_of_classes = 9  # needs to be changed
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)
        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):  #batch_ix gives only index value
            batch_in.append(batch[0].to(gpu)) #sends tensor to GPU the observation tensor
            batch_gt.append(batch[1].to(gpu)) #the inferred classes tensor to GPU

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:   # When everything has been appended into one list
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))            #stacks / append all rows into one and then reshapes 
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                batch_out = infer_action(batch_in)        #Forward Propagation    
                loss = cross_entropy_loss(batch_out, batch_gt)  #Calculates loss

                optimizer.zero_grad()    # So that from previous epoch doesn't take any values
                loss.backward()    #backpropagation
                optimizer.step()    #Updating the weights
                total_loss += loss  #Updating the loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    epsilon = 0.00001
    
    # calculating weighted cross entropy loss
    loss = (batch_gt * torch.log(batch_out + epsilon) + (1 - batch_gt) * torch.log( 1 - batch_out + epsilon))
    loss = (torch.Tensor([0.25,0.6,0.25,0.25,0.45,0.15,0.1,0.25,0.1]).to(torch.device('cuda')) * loss)
    return -torch.mean(torch.sum(loss, dim = 1), dim = 0)