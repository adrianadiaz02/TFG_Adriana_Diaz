import os
import json

def save_to_json(data, file_name):
    # Save the dictionary in a compact form without newlines or indentation
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, separators=(',', ':'))

def read_and_process_files(directory, activity):
    action_index = {}
    transcript = {}
    video_id_to_filename = {}  # Dictionary to map video IDs to filenames
    video_id = 0  # Initialize video ID

    # Assign a unique number to each action
    global_action_count = 1
    
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(activity):
            full_path = os.path.join(directory, filename)
            with open(full_path, 'r') as file:
                actions = file.read().strip().split('\n')
                
            last_action = None
            current_action_number = None
            video_transcript = []
            
            # Track where each action starts and assign a unique number if new
            for index, action in enumerate(actions):
                if action != last_action:
                    if action not in action_index:
                        action_index[action] = global_action_count
                        global_action_count += 1
                    last_action = action
                    current_action_number = action_index[action]
                    video_transcript.append([current_action_number, index])
            
            transcript[filename] = video_transcript
            video_id_for_file = "video_" + str(video_id)
            video_id_to_filename[video_id_for_file] = filename  # Map the current video ID to the filename
            video_id += 1  # Increment the video ID for the next file
    
    return transcript, video_id_to_filename


def main():
    directory = 'data/groundTruth'
    activity = "cereals"
    transcripts, video_id_map = read_and_process_files(directory, activity)
    save_to_json(transcripts, f'transcripts_GT_{activity}.json')
    save_to_json(video_id_map, f'video_id_map_{activity}.json')
    print("Transcripts saved to transcripts_GT_", activity)

if __name__ == "__main__":
    main()
