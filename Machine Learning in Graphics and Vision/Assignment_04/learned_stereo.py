#!/usr/bin/env python
"""Script used to train a neural network for stereo matching
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo_batch_provider import KITTIDataset, PatchProvider
from matplotlib import pyplot as plt
from handcrafted_stereo import visualize_disparity, add_padding


class StereoMatchingNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """
        super().__init__()
        gpu = torch.device("cuda")
        filter_number = 16

        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 4.2 a))
        # -------------------------------------
        #######################################
        self.conv1 = nn.Conv2d(1, filter_number, 3).to(gpu)
        self.conv2 = nn.Conv2d(filter_number, filter_number, 3).to(gpu)
        self.conv3 = nn.Conv2d(filter_number, filter_number, 3).to(gpu)
        self.conv4 = nn.Conv2d(filter_number, filter_number, 3).to(gpu)

    def forward(self, X):
        """ The forward pass of the network. Returns the features for a given image patch.

        ---------
        Arguments:
            X (tensor): image patch of shape (batch_size, height, width, n_channels)

        Returns:
        ---------
            features (tensor): predicted normalized features of the input image patch X,
                               shape (batch_size, height - 8, width - 8, n_features)
        """
        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 4.2 a))
        # -------------------------------------
        #######################################
        X = X.permute(0, 3, 1, 2)
        x = self.conv1(X)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.normalize(x, dim=1, p=2)

        x = x.permute(0, 2, 3, 1)

        return x


def save_loss_plot(loss, out_file="./output/learned_stereo/loss.png"):
    """ Saves the list of loss values as a plot to the file out_file.

    ---------    
    Arguments:
        loss (list): list of loss values for all iterations
        out_file (str): output file path

    Returns:
    ---------
        0
    """
    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 4.2 c))
    # -------------------------------------
    #######################################
    plt.plot(loss)
    plt.title("Loss Plot")
    plt.savefig(out_file)

    return 0


def save_output_files(out_dir, disparity, image_number=0, max_disparity=50):
    """ Saves the output files.

    ---------
    Arguments:
        out_dir (str): output directory
        disparity (numpy array): disparity values
        image_number (int): image number for naming conventions
        max_disparity (int): maximum disparity for the visualizations
    """
    # Create output direction's
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save disparty to file
    disp_dir = os.path.join(out_dir, "disparities")
    if not os.path.exists(disp_dir):
        os.makedirs(disp_dir)
    out_file = os.path.join(disp_dir, "%04d_disparity.npy" % image_number)
    np.save(out_file, disparity.numpy())

    # Save visualization to file
    vis_dir = os.path.join(out_dir, "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # Define output file name and title for the plot
    out_file_vis = os.path.join(vis_dir, "%04d_vis.png" % image_number)
    title = "Disparity map for image %04d with a Siamese Neural Network" % image_number

    # Visualize the disparty and save it to a file
    visualize_disparity(
        disparity.numpy().astype(np.uint8),
        title=title,
        out_file=out_file_vis,
        max_disparity=max_disparity,
    )


def hinge_loss(score_pos, score_neg, label):
    """ Computes the hinge loss for the similarity of a positive and a negative example.

    Arguments:
    ---------
        score_pos (tensor): similarity score of the positive example
        score_neg (tensor): similarity score of the negative example
        label: the true labels

    Returns:
    --------
        avg_loss: the mean loss over the patch and the mini batch
        acc: the accuracy of the prediction
    """
    # Calculate the hinge loss max(0, margin + s_neg - s_pos)
    loss = torch.max(0.2 + score_neg - score_pos, torch.zeros_like(score_pos))

    # Obtain the mean over the patch and the mini batch
    avg_loss = torch.mean(loss)

    # Calculate the accuracy
    similarity = torch.stack([score_pos, score_neg], dim=1)
    labels = torch.argmax(label, dim=1)
    predictions = torch.argmax(similarity, dim=1)
    acc = torch.mean((labels == predictions).float())

    return avg_loss, acc


def calculate_similarity_score(infer_similarity_metric, Xl, Xr):
    """Computes the similarity score for two stereo image patches.

    Arguments:
    ---------
        infer_similarity_metric:  pytorch module object
        Xl: tensor holding the left image patch
        Xr: tensor holding the right image patch

    Returns:
    --------
        score (tensor): the similarity score of both image patches which is the dot product of their features
    """
    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 4.2 b))
    # -------------------------------------
    #######################################
    Fl = infer_similarity_metric(Xl)
    Fr = infer_similarity_metric(Xr)
    # z = torch.sum(Fl * Fr,dim=(1,2,3))
    y = torch.einsum("abcd,abcd->a", Fl, Fr)
    return y


def training_loop(
    infer_similarity_metric, patches, optimizer, iterations=1000, batch_size=128
):
    """ Runs the training loop of the siamese network.
    
    ---------    
    Arguments:
        infer_similarity_metric (obj): pytorch module
        patches (obj): patch provider object
        optimizer (obj): optimizer object
        iterations (int): number of iterations to perform
        batch_size (int): batch size
    """

    loss_list = []
    try:
        print("Starting training loop.")
        for idx, batch in zip(range(iterations), patches.iterate_batches(batch_size)):
            # Extract the batches and labels
            Xl, Xr_pos, Xr_neg = batch
            label = torch.eye(2).cuda()[[0] * len(Xl)]  # label is always [1, 0]

            # calculate the similarity score
            score_pos = calculate_similarity_score(infer_similarity_metric, Xl, Xr_pos)
            score_neg = calculate_similarity_score(infer_similarity_metric, Xl, Xr_neg)
            # compute the loss and accuracy
            loss, acc = hinge_loss(score_pos, score_neg, label)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()
            # let the optimizer perform one step and update the weights
            optimizer.step()

            # Append loss to list
            loss_list.append(loss.item())

            if idx % 50 == 0:
                print("Loss (%04d it):%.04f \tAccuracy: %0.3f" % (idx, loss, acc))
    finally:
        patches.stop()
        with torch.no_grad():
            save_loss_plot(loss_list)
        print("Finished training!")


def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50):
    """ Computes the disparity of the stereo image pair.

    Arguments:
    ---------
        infer_similarity_metric:  pytorch module object
        img_l: tensor holding the left image
        img_r: tensor holding the right image
        max_disparity (int): maximum disparity

    Returns:
    --------
        D: tensor holding the disparity
    """
    # get the image features by applying the similarity metric
    Fl = infer_similarity_metric(img_l[None])
    Fr = infer_similarity_metric(img_r[None])

    # images of shape B x H x W x C
    B, H, W, C = Fl.shape
    # Initialize the disparity
    disparity = torch.zeros((B, H, W)).int()
    # Initialize current similarity to -infimum
    current_similarity = torch.ones((B, H, W)) * -np.inf

    # Loop over all possible disparity values
    Fr_shifted = Fr
    for d in range(max_disparity + 1):
        if d > 0:
            # initialize shifted right image
            Fr_shifted = torch.zeros_like(Fr)
            # insert values which are shifted to the right by d
            Fr_shifted[:, :, d:] = Fr[:, :, :-d]

        # Calculate similarities
        sim_d = torch.sum(Fl * Fr_shifted, dim=3)
        # Check where similarity for disparity d is better than current one
        indices_pos = sim_d > current_similarity
        # Enter new similarity values
        current_similarity[indices_pos] = sim_d[indices_pos]
        # Enter new disparity values
        disparity[indices_pos] = d

    return disparity


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Exercise_04 for the Machine Learning Course in Graphics and Vision 2020"
        )
    )
    parser.add_argument(
        "action",
        choices=["train", "evaluate"],
        help="Choose the action to perform {train, evaluate}",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./KITTI_2015_subset",
        help="Path to the KITTI 2015 subset folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/learned_stereo",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="The number of iterations (default 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="The batch size used for training (default 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="The learning rate used for training (default 3e-4)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=9,
        help="The patch size the siamese network is using during training (default 9)",
    )
    parser.add_argument(
        "--max-disparity",
        type=int,
        default=50,
        help="The maximum disparity that is tested (default 50)",
    )

    args = parser.parse_args(argv)

    # Fix random seed for reproducibility
    np.random.seed(7)
    torch.manual_seed(7)

    # Shortcuts for directories
    input_dir = args.input_dir
    out_dir = args.output_dir
    model_out_dir = os.path.join(out_dir, "model")
    model_file = os.path.join(model_out_dir, "model.t7")

    # Other shortcuts
    patch_size = args.patch_size
    padding = patch_size // 2
    max_disparity = args.max_disparity

    # Check if output directory exists and if not create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    # Perform the requested action
    if args.action == "train":
        # Create dataloader for KITTI training set
        dataset = KITTIDataset(
            os.path.join(input_dir, "data_scene_flow/training/"),
            os.path.join(input_dir, "data_scene_flow/training/disp_noc_0"),
        )
        # Load patch provider
        patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))

        # Initialize the network
        infer_similarity_metric = StereoMatchingNetwork()
        optimizer = torch.optim.SGD(
            infer_similarity_metric.parameters(), lr=args.lr, momentum=0.9
        )

        # Start training loop
        training_loop(
            infer_similarity_metric,
            patches,
            optimizer,
            iterations=args.iterations,
            batch_size=args.batch_size,
        )
        # Save checkpoint
        torch.save(infer_similarity_metric, model_file)
        print("Saved model checkpoint.")

    elif args.action == "evaluate":
        # Check that model file exists
        if not os.path.exists(model_file):
            raise UserWarning("No model file found. Can not perform evaluation.")
        # Load the model
        infer_similarity_metric = torch.load(model_file, map_location="cpu")
        infer_similarity_metric.eval()
        print("Successfully loaded \n\t%s" % model_file)

        # Load KITTI test split
        dataset = KITTIDataset(os.path.join(input_dir, "data_scene_flow/testing/"))
        # Loop over test images
        for i in range(len(dataset)):
            print("Processing %d image" % i)
            # Load images and add padding
            img_l, img_r = dataset[i]
            img_l, img_r = add_padding(img_l, padding), add_padding(img_r, padding)
            img_l, img_r = torch.Tensor(img_l), torch.Tensor(img_r)

            disparity_map = compute_disparity_CNN(
                infer_similarity_metric, img_l, img_r, max_disparity=max_disparity
            )
            save_output_files(
                out_dir, disparity_map[0], image_number=i, max_disparity=max_disparity
            )


if __name__ == "__main__":
    main(sys.argv[1:])
