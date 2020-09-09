#!/usr/bin/env python
import argparse
import os
import sys

import imageio
import numpy as np
from skimage.transform import rescale
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def disparity_to_depth(disparity, baseline, focal_length):
    """ Calculates the depth from disparity.

    Arguments:
    ----------
        disparity (numpy array): disparity of size (H, W)
        baseline (float): baseline length
        focal_length (float): focal length
    
    Returns:
    --------
        depth (numpy array): array of depth values of size (H, W)
    """
    depth = np.full(disparity.shape, np.inf)

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 4.3 a))
    # -------------------------------------
    #######################################

    return depth


def img_to_3D_coordinates(img_x, img_y, focal_length, Z, cx, cy):
    """ Transforms image x- and y-coordinates to the respective coordinates X and Y in 3D space.

    Arguments:
    ----------
    img_x (numpy array): image x-coordinates of size (H, W)
    img_y (numpy array): image y-coordinates of size (H, W)
    focal_length (float): focal length
    Z (numpy array): 3D space Z coordinates of size (H, W)
        These are the depth values
    cx (float): x-coordinate of principal point 
    cy (float): y-coordinate of principal point

    Returns:
    ----------
    X (numpy array): respective x-coordinates in 3D space of size (H, W)
    Y (numpy array): respective y-coordinates in 3D space of size (H, W)
    Z (numpy array): respective z-coordinates in 3D space of size (H, W) 
        (same as the input array Z)

    """
    X = np.zeros_like(img_x)
    Y = np.zeros_like(img_y)

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 4.3 b))
    # -------------------------------------
    #######################################

    return X, Y, Z


def disparity_to_pointcloud(img, disparity, maximum_depth=50):
    """ Returns 3D points and their color value from the disparity and RGB-image input.

    Arguments:
    ----------
    img (numpy array): RGB image of size (H, W, 3)
    disparity (numpy array): disparity map of size (H, W)
    maximum_depth (int): maximum depth value

    Returns:
    ----------
    points (numpy array): array containing the xyz coordinates of the points
        of size (T, 3)
    colors (numpy array): array containing the RGB color values of the points
        of size (T, 3)
    """
    H, W = img.shape[:2]

    # Define constants
    baseline = 0.54 / 2
    focal_length = 722 / 2
    cx = 609.5593 / 2
    cy = 172.85 / 2

    # Calculate depth and make sure values lie between 0 and maximum_depth
    depth = disparity_to_depth(disparity, baseline, focal_length)
    depth = np.clip(depth, 0.0, maximum_depth)

    # Get image x- and y-coordinates
    img_x, img_y = np.mgrid[0:H, 0:W]
    # Transform them to 3D space coordinates X and Y
    X, Y, Z = img_to_3D_coordinates(img_x, img_y, focal_length, depth, cx, cy)

    # Stack X, Y and Z(=depth) value
    points = np.stack([X, Y, Z], axis=-1)
    # Save respective RGB values in colors array
    colors = img[img_x, img_y, :]

    # Reshape and remove points with depth <= 0
    mask = points[:, :, -1] > 0
    points = points[mask]
    colors = colors[mask]

    return points, colors


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Exercise_4_3 for the Machine Learning Course in Graphics"
            " and Vision 2018"
        )
    )
    parser.add_argument(
        "--input-disparity",
        help="Path to the disparity map",
        type=str,
        default="./examples/disparity.npy",
    )
    parser.add_argument(
        "--input-image",
        help="Path to the input image",
        type=str,
        default="./examples/input_image.png",
    )
    parser.add_argument(
        "--output-dir",
        help="Where the output files should be saved.",
        type=str,
        default="./output/pointcloud",
    )
    parser.add_argument(
        "--interactive",
        help="Whether to show an interactive plot",
        action="store_true",
    )
    parser.add_argument(
        "--half-resolution",
        help="Whether half resolution images should be used",
        type=bool,
        default=True,
    )

    args = parser.parse_args(argv)

    # Shortcuts
    disp = args.input_disparity
    img = args.input_image
    output_dir = args.output_dir
    half_resolution = args.half_resolution
    interactive = args.interactive

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # As we have calculated the disparity on half resolution, we have to downsample the input images accordingly
    if half_resolution:
        img = imageio.imread(img).astype(np.float32) / 255.0
        img = rescale(img, 0.5, anti_aliasing=True, mode="reflect", multichannel=True)

    # Get disparity
    disp = np.load(disp)

    # Create point cloud for depth
    points, colors = disparity_to_pointcloud(img, disp)

    # Plot point cloud
    fig = plt.figure(figsize=(15, 5))
    ax = Axes3D(fig)
    plt.axis("equal")
    H = img.shape[0]
    ax.scatter(points[:, 1], points[:, 2], H - points[:, 0], c=colors, s=1)
    ax.view_init(25, -75)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    if interactive:
        plt.show()
    else:
        out_file = os.path.join(output_dir, "pointcloud.png")
        plt.savefig(out_file, dpi=200)
        plt.close()
        print("Saved figure to %s." % out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
