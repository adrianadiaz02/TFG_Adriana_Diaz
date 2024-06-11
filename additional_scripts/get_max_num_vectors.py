"""
This script processes feature vector files within a specified directory structure. It determines 
the number of feature vectors in each file, identifies the file with the maximum number of feature
vectors for each activity, and saves this information to a file.
"""

__author__ = 'Adriana Díaz Soley '
__date__ = 'April 2024'

import os


def get_num_vectors_video(filename):
    """
    Reads a file containing feature vectors and returns the number of vectors and their dimensions.
    
    Parameters:
    filename (str): Path to the file containing feature vectors.
    
    Returns:
    int: Number of feature vectors in the file.
    """
    with open(filename, "r") as file:
        lines = file.readlines()
        number_of_vectors = len(lines)  # Number of feature vectors in the file
        vector_dimensions = len(lines[0].split())  # Dimensions of each vector (space-separated)
    
    last_part = os.path.basename(filename)
    # Uncomment the following lines to print debug information
    # print(last_part)
    # print(f"Number of feature vectors: {number_of_vectors}") 
    # print(f"Dimensions of each vector: {vector_dimensions}") # Feature dimension (same for all videos)
    
    return number_of_vectors


def get_max_feature_vector_count(activities_folder_path):
    """
    Determines the file with the maximum number of feature vectors for each activity in the specified directory.
    
    Parameters:
    activities_folder_path (str): Path to the main activities folder.
    
    Returns:
    dict: A dictionary where keys are activity names and values are the highest number of feature vectors found in any file of that activity.
    """
    # Dictionary to store the activity and the highest number of feature vectors
    activity_max_vectors = {}

    # List all folders in the activities folder path
    for activity in os.listdir(activities_folder_path):
        activity_path = os.path.join(activities_folder_path, activity)
        print(activity_path)  # Uncomment to print the current activity path for debugging

        if os.path.isdir(activity_path):
            # Initialize the max count for this activity with 0
            activity_max_vectors[activity] = 0

            # List all files in the activity directory
            for filename in os.listdir(activity_path):
                file_path = os.path.join(activity_path, filename)
                number_of_vectors = get_num_vectors_video(file_path)

                # Update the max count if the current file has more vectors
                if number_of_vectors > activity_max_vectors[activity]:
                    activity_max_vectors[activity] = number_of_vectors

        print(activity_max_vectors[activity])  # Uncomment to print the max vectors for current activity for debugging

    return activity_max_vectors


def save_dictionary_to_file(dictionary, filename):
    """
    Saves a dictionary to a file.
    
    Parameters:
    dictionary (dict): The dictionary to save.
    filename (str): Path to the output file.
    """
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

# Specify the path to the activity folder
activity_folder_path = "/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/data/features/s1"

# Get the maximum number of feature vectors for each activity
activity_max_feature_vectors = get_max_feature_vector_count(activity_folder_path)

# Print the result
for activity, max_count in activity_max_feature_vectors.items():
    print(f"{activity}: {max_count}")

# Path to save the dictionary file
dictionary_filename = "/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/Dictionary_activity_max_vectors.txt"

# Call the function to save the dictionary
save_dictionary_to_file(activity_max_feature_vectors, dictionary_filename)
