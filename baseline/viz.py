import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib


cm = matplotlib.colormaps.get_cmap("tab20")
def_colors = cm.colors
cus_colors = ["k"] + [def_colors[i] for i in range(1, 20)] + ["w"]
cmap = ListedColormap(colors=cus_colors, name="agri", N=21)


def get_rgb(x: torch.Tensor, batch_index: int = 0, t_show: int = 0) -> np.ndarray:
    """
    Retrieves an RGB image from a Sentinel-2 time series.

    This function extracts the RGB (Red, Green, Blue) channels from a Sentinel-2
    time series, normalizes the values between 0 and 1, and prepares the image
    for display by rearranging the axes to match the standard image format.

    Args:
        x (torch.Tensor): The input time series data with shape
            (batch_size, time_steps, channels, height, width). The expected
            channels are ordered as [Blue, Green, Red, ...].
        batch_index (int, optional): Index of the batch to extract the image
            from. Defaults to 0.
        t_show (int, optional): The time step index to show. Defaults to 0.

    Returns:
        np.ndarray: A normalized RGB image as a NumPy array with shape
        (height, width, 3), ready for visualization.

    Example:
        >>> rgb_image = get_rgb(x, batch_index=0, t_show=2)
        >>> plt.imshow(rgb_image)
        >>> plt.show()

    Notes:
        - The RGB image is created from the Red, Green, and Blue bands (channels 2, 1, 0).
        - The pixel values are normalized to the range [0, 1].
        - The channel axes are rearranged to make the image displayable by matplotlib.
    """

    im = x[batch_index, t_show, [2, 1, 0]].cpu().numpy()
    mx = im.max(axis=(1, 2))
    mi = im.min(axis=(1, 2))
    im = (im - mi[:, None, None]) / (mx - mi)[:, None, None]
    im = im.swapaxes(0, 2).swapaxes(0, 1)
    im = np.clip(im, a_max=1, a_min=0)
    return im


def plot_s2_and_labels(x: dict, y: torch.tensor, bid: int = 0, t_show: int = 0) -> None:
    """
    Plots a Sentinel-2 observation and its corresponding semantic labels side by side.

    This function generates a 1x2 subplot displaying:
    1. A single Sentinel-2 observation at a specified time step.
    2. The corresponding semantic labels.

    Args:
        x (dict): A dictionary containing Sentinel-2 time series data under the key `x["S2"]`,
            where the expected shape is (batch_size, time_steps, channels, height, width).
        y (torch.Tensor): The ground truth semantic labels with shape
            (batch_size, height, width) or (batch_size, 1, height, width).
        bid (int, optional): Index of the batch to plot. Defaults to 0.

    Returns:
        None: This function displays the plot and does not return any value.

    Example:
        >>> plot_s2_and_labels(x, y, bid=0)

    Notes:
        - The Sentinel-2 observation is shown as an RGB image using bands 2, 1, and 0.
        - Semantic labels are displayed using a custom colormap (`cmap`) with values ranging
          from 0 to 20.
        - Make sure that `x["S2"]` is a dictionary key containing the time series data.
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))

    # Plot the Sentinel-2 RGB image
    axes[0].imshow(get_rgb(x["S2"], batch_index=bid, t_show=t_show))
    axes[0].set_title(f"One S2 observation (n°{t_show})")

    # Plot the semantic labels
    axes[1].imshow(y[bid].squeeze(), cmap=cmap, vmin=0, vmax=20)
    axes[1].set_title("Semantic labels.")

    # Show the plot
    plt.show()


def plot_sequence_grid(x, bid=0, N=5, M=10):
    """
    Plots a sequence of images from the batch 'x' in an N x M grid layout.

    Parameters:
    x (dict): The input batch containing image sequences (e.g., x["S2"]).
    bid (int): The batch index to plot, in [0, BATCHSIZE-1]. Default is 0.
    N (int): Number of rows in the subplot grid. Default is 5.
    M (int): Number of columns in the subplot grid. Default is 10.

    Raises:
    ValueError: If SEQUENCE_LENGTH exceeds the total number of subplots (N * M).
    """
    SEQUENCE_LENGTH = x["S2"].shape[1]

    # Calculate the total number of subplots needed
    total_subplots = N * M

    # Ensure that N * M is enough for SEQUENCE_LENGTH
    if SEQUENCE_LENGTH > total_subplots:
        raise ValueError(
            f"SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) is greater than N * M ({N * M})"
        )

    # Create a figure with N rows and M columns
    fig, axes = plt.subplots(
        N, M, figsize=(M * 4, N * 4)
    )  # Adjust figsize based on the grid

    # Flatten the axes array to make it easier to index
    axes = axes.flatten()

    # Loop over the sequence and plot the images
    for i in range(SEQUENCE_LENGTH):
        axes[i].imshow(get_rgb(x["S2"], batch_index=bid, t_show=i))
        axes[i].set_title(f"Observation {i}")

    # If there are unused subplots, hide them
    for i in range(SEQUENCE_LENGTH, total_subplots):
        axes[i].axis("off")  # Hide any unused subplots

    plt.tight_layout()
    plt.show()
