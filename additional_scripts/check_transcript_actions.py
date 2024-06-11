"""
This script reads a JSON file containing video transcripts, checks for repeated actions within each video, 
and prints information about videos that have repeated actions.
"""

__author__ = 'Adriana Díaz Soley '
__date__ = 'April 2024'

import json

# Read the JSON file
transcript = "/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs/bf_debug_salat_imp_relabel/estimated_transcripts_salat.json"
with open(transcript) as f:
    data = json.load(f)


# Check for repeated actions in each video and print the information if found
for video, actions in data.items():
    action_set = set()  # Using a set to track unique actions in this video
    repeated_action = None
    for action, _ in actions:
        if action in action_set:
            repeated_action = action
            #break  # No need to continue checking if we already found a repeated action
        else:
            action_set.add(action)

    if repeated_action:
        print("Videos with repeated actions:")
        print(f'"{video}": {actions}')