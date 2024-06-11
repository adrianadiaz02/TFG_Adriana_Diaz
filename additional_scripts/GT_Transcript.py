"""
This script reads action files from a specified directory for a particular activity. 
It processes each file to generate transcripts, maps video IDs to filenames, and identifies 
repeated actions within the videos. The results are saved to JSON files and printed.
"""

__author__ = 'Adriana Díaz Soley '
__date__ = 'April 2024'


import os
import json

def save_to_json(data, file_name):
    """
    Saves data to a JSON file.

    Parameters:
    data (dict): The data to save.
    file_name (str): The name of the JSON file.
    """
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, separators=(',', ':'))


def read_and_process_files(directory, activity):
    """
    Reads and processes action files from a directory for a specified activity.

    Parameters:
    directory (str): The path to the directory containing action files.
    activity (str): The activity to filter files by.

    Returns:
    tuple: Containing transcripts, video ID to filename map, repeated actions report, and total video count.
    """
    action_index = {}
    transcript = {}
    video_id_to_filename = {}  # Dictionary to map video IDs to filenames
    video_id = 0 

    global_action_count = 1  # Assign a unique number to each action
    repeated_actions_report = {}  
    total_video_count = 0

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(activity):
            total_video_count += 1

            full_path = os.path.join(directory, filename)
            with open(full_path, 'r') as file:
                actions = file.read().strip().split('\n')
                
            last_action = None
            current_action_number = None
            video_transcript = []
            action_counts = {}

            # Track where each action starts and assign a unique number if new
            for index, action in enumerate(actions):
                if action != last_action:
                    if action not in action_index:
                        action_index[action] = global_action_count
                        global_action_count += 1
                    last_action = action
                    current_action_number = action_index[action]
                    video_transcript.append([current_action_number, index])

                    # Count the occurrences of each action
                    if action in action_counts:
                        action_counts[action] += 1
                    else:
                        action_counts[action] = 1
            
            # Save the transcript
            transcript[filename] = video_transcript
            video_id_for_file = "video_" + str(video_id)
            video_id_to_filename[video_id_for_file] = filename  # Map the current video ID to the filename
            video_id += 1  # Increment the video ID for the next file

            # Identify and record repeated actions, excluding "SIL"
            repeated_actions = {action: count for action, count in action_counts.items() if count > 1 and action != "SIL"}
            if repeated_actions:
                repeated_actions_report[filename] = repeated_actions
    
    return transcript, video_id_to_filename, repeated_actions_report, total_video_count


def main():
    """
    Main function to process files and generate reports.
    """
    directory = 'data/groundTruth'
    activity = "pancake"
    
    # Process the files in the specified directory for the given activity
    transcripts, video_id_map, repeated_actions_report, total_count = read_and_process_files(directory, activity)
    
    # Save the transcripts to a JSON file
    save_to_json(transcripts, f'transcripts_GT_{activity}.json')
    save_to_json(video_id_map, f'video_id_map_{activity}.json')
    print("Transcripts saved to transcripts_GT_", activity)

    # Print and report repeated actions if any are found
    if repeated_actions_report:
        print("Repeated actions found in the following videos:")
        for video, actions in repeated_actions_report.items():
            for action, count in actions.items():
                print(f"  Action: {action}, Repeated: {count} times")

        # Count and report the number of videos with repeated actions
        num_videos_with_repeated_actions = len(repeated_actions_report)
        print(f"\nResults for activity: {activity}")
        print(f"Total number of videos with repeated actions: {num_videos_with_repeated_actions}")
        print(f"Total number of videos: {total_count}")
        print(f"Proportion of videos with repetition: {num_videos_with_repeated_actions/total_count}")
    else:
        print("No repeated actions found in any video.")

if __name__ == "__main__":
    main()
