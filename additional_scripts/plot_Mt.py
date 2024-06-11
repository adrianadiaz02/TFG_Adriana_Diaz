"""
This script computes a permutation-aware prior for a given set of video transcripts. 
It processes estimated transcripts, generates a prior distribution, normalizes it 
against a reference prior, and saves the resulting heatmap as an image.
"""

__author__ = 'Adriana Díaz Soley '
__date__ = 'April 2024'

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import json
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
    Plots a confusion matrix.

    Parameters:
    q (np.ndarray): The confusion matrix data.
    transcript (list, optional): The transcript data to annotate the plot.

    Returns:
    matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(q, ax=ax, cbar=True)
    return fig


def cost_paper(i, j, n, m):
    """
    Computes the cost for aligning frames based on a paper's method.

    Parameters:
    i (int): Frame index.
    j (int): Action index.
    n (int): Total number of frames.
    m (int): Total number of actions.

    Returns:
    float: The computed cost.
    """
    return ((i - j) ** 2) / (2 * (n / m) ** 2)


def gaussian(cost, sigma):
    """
    Applies a Gaussian function to the cost.

    Parameters:
    cost (np.ndarray): The cost matrix.
    sigma (float): The standard deviation for the Gaussian function.

    Returns:
    np.ndarray: The Gaussian-applied cost matrix.
    """
    return np.exp(-cost / (2 * sigma ** 2))


def compute_permutation_aware_prior(video_ids, estimated_transcripts_all, num_clusters, num_frames, sigma):
    """
    Computes the permutation-aware prior for the current batch using the estimated transcripts.

    Parameters:
    video_ids (list): List of video IDs.
    estimated_transcripts_all (dict): Estimated transcripts for all videos.
    num_clusters (int): Number of clusters (actions).
    num_frames (int): Number of frames.
    sigma (float): Standard deviation for the Gaussian function.

    Returns:
    np.ndarray: The permutation-aware prior.
    """
    cost_matrix = np.zeros((num_frames, num_clusters))
    
    for video_id in video_ids:
        video_name = video_id.split('/')[-1]
        transcript = estimated_transcripts_all.get(video_name, [])
        print(f"Transcript for {video_name}: {transcript}")
        
        # Calculate midpoints of the action segments
        action_positions = {}
        for idx, (action, start_frame) in enumerate(transcript):
            if idx < len(transcript) - 1:
                next_start_frame = transcript[idx + 1][1]
            else:
                next_start_frame = num_frames  # Assume the last action goes until the end of the frames
            mid_point = (start_frame + next_start_frame) // 2
            action_positions[action] = mid_point

        for j in range(num_clusters):
            if j + 1 in action_positions:
                frame_position = action_positions[j + 1]
                for i in range(num_frames):
                    cost_matrix[i, j] = cost_paper(i, frame_position, num_frames, num_clusters)

    p_permutation_aware = gaussian(cost_matrix, sigma)
    return p_permutation_aware


def normalize_prior(prior, reference_prior):
    """
    Normalizes the permutation-aware prior to match the reference prior.

    Parameters:
    prior (np.ndarray): The permutation-aware prior.
    reference_prior (np.ndarray): The reference fixed-order prior.

    Returns:
    np.ndarray: The normalized permutation-aware prior.
    """
    return prior * (reference_prior.sum() / prior.sum())


def main():
    """
    Main function to compute and save the permutation-aware prior.
    """
    estimated_transcripts_path = "/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs/bf_debug_scrambledegg_permut_NUEVO/estimated_transcripts_scrambledegg.json"
    reference_prior_path = "/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/additional_outputs/p_gauss_Ma_bf_debug_scrambledegg_permut_2train_sigma6_scale.npy"
    num_clusters = 12
    video_ids = ["P48_webcam02_P48_scrambledegg"]
    num_frames = 512 
    sigma = 1  

    # Load estimated transcripts
    with open(estimated_transcripts_path, 'r') as file:
        estimated_transcripts_all = json.load(file)

    # Load the reference fixed-order prior
    reference_prior = np.load(reference_prior_path)

    # Compute the permutation-aware prior M_t
    M_t = compute_permutation_aware_prior(video_ids, estimated_transcripts_all, num_clusters, num_frames, sigma)

    # Normalize the permutation-aware prior to match the reference prior
    M_t = normalize_prior(M_t, reference_prior)

    # Plot and save the permutation-aware prior
    fig = plot_confusion_matrix(M_t, transcript=estimated_transcripts_all[video_ids[0].split('/')[-1]])
    plt.title("Permutation aware prior distribution")
    plt.xlabel("Action Index")
    plt.ylabel("Frame Number")
    fig.savefig("additional_outputs/Mt_nuevo_permutation_prior_scramb3.png")

    # Uncomment to plot and save the fixed-order prior for comparison
    """
    fig_fixed_order = plot_confusion_matrix(reference_prior)
    plt.title("Fixed-order prior distribution")
    plt.xlabel("Action Index")
    plt.ylabel("Frame Number")
    fig_fixed_order.savefig("additional_outputs/fixed_order_prior_scramb.png")
    """

if __name__ == "__main__":
    main()
