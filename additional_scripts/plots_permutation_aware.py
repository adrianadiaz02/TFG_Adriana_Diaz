"""
This script processes action probability matrices to estimate transcripts and generate 
permutation-aware priors. It provides functionalities to visualize these matrices and 
save the results as images.
"""

__author__ = 'Adriana Díaz Soley '
__date__ = 'April 2024'


import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from imageio import imread


def plot_to_image(figure):
    """
    Converts a Matplotlib figure to an image.

    Parameters:
    figure (matplotlib.figure.Figure): The figure to convert.

    Returns:
    np.ndarray: The image array.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = imread(buf)
    return image.transpose(2, 0, 1)


def plot_confusion_matrix(q, transcript=None):
    """
    Plots a confusion matrix with optional transcript annotations.

    Parameters:
    q (np.ndarray): The confusion matrix data.
    transcript (list, optional): The transcript data to annotate the plot.

    Returns:
    matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(q, ax=ax, cbar=True)

    if transcript is not None:
        for action, frame in transcript:
            ax.add_patch(plt.Rectangle((action - 1, frame), 1, 1, fill=False, edgecolor='#39FF14', lw=3))

    return fig


def estimate_transcripts_basic(q, num_actions):
    """
    Estimate transcripts by finding the frame with the maximum probability for each action.

    Parameters:
    q (torch.Tensor): Probability tensor of shape (num_frames, num_actions).
    num_actions (int): Number of actions.

    Returns:
    list: Sorted list of (action, frame) pairs.
    """
    frame_indices = torch.argmax(q, dim=0)  # Find max along rows, resulting in [num_actions]
    
    action_frame_pairs = [(i + 1, int(frame_indices[i])) for i in range(num_actions)]
    action_frame_pairs.sort(key=lambda x: x[1])  # Sort by frame index

    return action_frame_pairs


def estimate_transcripts(q, num_actions, min_separation=50, max_relative_difference=0.7):
    """
    Estimate transcripts allowing actions to appear multiple times based on significant probability peaks.

    Parameters:
    q (torch.Tensor): Probability tensor of shape (num_frames, num_actions).
    num_actions (int): Number of actions.
    min_separation (int): Minimum separation between peaks for the same action.
    max_relative_difference (float): Maximum relative difference for considering secondary peaks.

    Returns:
    list: List of (action, frame) pairs representing the estimated transcript.
    """
    q = q.cpu()
    action_frame_pairs = []

    for action_index in range(num_actions):
        probabilities = q[:, action_index].numpy()
        primary_peak_index = np.argmax(probabilities)
        primary_peak_value = probabilities[primary_peak_index]

        action_frame_pairs.append((int(action_index + 1), int(primary_peak_index), primary_peak_value))

        secondary_peak_index = -1
        secondary_peak_value = -1
        for i, value in enumerate(probabilities):
            if i != primary_peak_index and value > secondary_peak_value:
                secondary_peak_index = i
                secondary_peak_value = value

        if secondary_peak_index != -1:
            distance = abs(primary_peak_index - secondary_peak_index)
            relative_difference = secondary_peak_value / primary_peak_value

            if distance >= min_separation and relative_difference >= max_relative_difference:
                action_frame_pairs.append((int(action_index + 1), int(secondary_peak_index), secondary_peak_value))

    action_frame_pairs.sort(key=lambda x: x[2], reverse=True)
    action_frame_pairs = action_frame_pairs[:num_actions]
    action_frame_pairs.sort(key=lambda x: x[1])

    return [(action, frame) for action, frame, _ in action_frame_pairs]


def create_cost_matrix(num_frames, num_clusters, transcript):
    """
    Creates a cost matrix based on the distance between frames and assigned frames in the transcript.

    Parameters:
    num_frames (int): Number of frames.
    num_clusters (int): Number of clusters (actions).
    transcript (list): List of (action, frame) pairs.

    Returns:
    np.ndarray: Cost matrix.
    """
    cost_matrix = np.full((num_frames, num_clusters), np.inf)
    action_to_frame = {action: frame for action, frame in transcript}

    for i in range(num_frames):
        for j in range(1, num_clusters + 1):
            if j in action_to_frame:
                assigned_frame = action_to_frame[j]
                cost_matrix[i, j - 1] = abs(i - assigned_frame)

    return cost_matrix


def gaussian_transform(cost_matrix, sigma):
    """
    Applies a Gaussian function to the cost matrix to create a probability matrix.

    Parameters:
    cost_matrix (np.ndarray): The cost matrix.
    sigma (float): The standard deviation for the Gaussian function.

    Returns:
    np.ndarray: The transformed probability matrix.
    """
    gaussian_matrix = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-cost_matrix/(2 * sigma ** 2))
    return gaussian_matrix * 10


def main():
    save_Q_plot = False
    save_Mt_computed_plot = False
    save_Mt_from_train = True
    save_Ma_plot = True

    if save_Q_plot:
        q_values = np.load("additional_outputs/q_values_last_video_last_epoch_bf_debug_cereals_permut_2train_sigma6_scale.npy")
        q_tensor = torch.tensor(q_values, dtype=torch.float32)

        num_actions = q_values.shape[1]
        num_frames = q_values.shape[0]
        transcript = estimate_transcripts_basic(q_tensor, num_actions)
        print(transcript)
        
        fig = plot_confusion_matrix(q_values, transcript)
        plt.title("Frame-level pseudo-label codes Q with maximums in green")
        plt.xlabel("Action Index")
        plt.ylabel("Frame Number")
        
        fig.savefig("additional_outputs/q_matrix_cereals1.png")
        print("Confusion matrix plot saved.")

    if save_Mt_computed_plot:
        cost_matrix = create_cost_matrix(num_frames, num_actions, transcript)
        Mt = gaussian_transform(cost_matrix, 6)

        fig = plot_confusion_matrix(Mt)
        plt.title("Permutation aware prior distribution")
        plt.xlabel("Action Index")
        plt.ylabel("Frame Number")

        fig.savefig("additional_outputs/Mt_permutation_prior_cereals1.png")


    if save_Mt_from_train:
        Mt2 = np.load("additional_outputs/Mt2_bf_debug_scrambledegg_permut_2train_sigma6_scale.npy")

        fig = plot_confusion_matrix(Mt2)
        plt.title("Permutation aware prior distribution")
        plt.xlabel("Action Index")
        plt.ylabel("Frame Number")

        if save_Mt_from_train:
            fig.savefig("additional_outputs/Mt_permutation_prior_scrambledegg2.png")


    if save_Ma_plot:
        Ma = np.load("additional_outputs/p_gauss_Ma_bf_debug_scrambledegg_permut_2train_sigma6_scale.npy")
        fig = plot_confusion_matrix(Ma)
        plt.title("Fixed-order prior distribution")
        plt.xlabel("Action Index")
        plt.ylabel("Frame Number")

        fig.savefig("additional_outputs/Ma_fixed_order_prior_scrmabledegg.png")

if __name__ == "__main__":
    main()
